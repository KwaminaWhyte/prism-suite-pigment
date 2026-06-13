//! The eframe application: panels, tools, and the GPU canvas. Owns document
//! state and drives GPU uploads/readbacks via `frame.wgpu_render_state()`.

use std::collections::HashSet;

use eframe::egui_wgpu;
use eframe::wgpu;
use half::f16;
use prism_core::adjust::{Adjustment, CurvePoints};
use prism_core::color::{linear_to_srgb, srgb_to_linear};
use prism_core::fill::flood_fill_mask;
use prism_core::histogram::{self, Histogram};
use prism_core::layer::{TextDef, VectorDef};
use prism_core::raster::{self, CombineMode};
use prism_core::shape::{self, ShapeKind};
use prism_core::{BlendMode, Document, Layer, LayerId, LayerKind, Size};
use prism_io::document_file::{self, DocMeta, LayerMeta, LayerPixels};
use prism_io::document_file::{
    BevelStyle, ColorOverlayStyle, GlowStyle, GradientOverlayStyle, LayerStyles, ShadowStyle,
    StrokeStyle,
};
use prism_io::resize::{resize_rgba_f32, Quality};
use prism_io::text::{self, TextAlign};
use prism_io::LoadedImage;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::SystemTime;

use crate::canvas::{CanvasGpu, CanvasPaint, Dab, LayerDraw, SelectionOp, ViewTransform};
use crate::path::WorkPath;

/// Window-menu panel visibility (the canvas is always shown).
#[derive(Clone, Copy)]
pub(crate) struct PanelVis {
    pub tool_options: bool,
    pub tools: bool,
    pub properties: bool,
}

impl Default for PanelVis {
    fn default() -> Self {
        Self {
            tool_options: true,
            tools: true,
            properties: true,
        }
    }
}

impl PanelVis {
    fn all_shown(&self) -> bool {
        self.tool_options && self.tools && self.properties
    }
    fn show_all(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod panel_tests {
    use super::PanelVis;

    #[test]
    fn default_all_shown_and_reset() {
        let mut p = PanelVis::default();
        assert!(p.all_shown());
        p.tools = false;
        assert!(!p.all_shown());
        p.show_all();
        assert!(p.all_shown() && p.tool_options && p.properties);
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tool {
    Move,      // pan the view (hand)
    MoveLayer, // translate the active layer
    Brush,
    Eraser,
    Clone,       // clone stamp: Alt-click sets source, drag stamps from it
    Heal,        // healing brush: Alt-click source, brush region, gradient-domain heal on release
    SpotHeal,    // spot heal: brush a blemish, auto-source + heal on release (no manual source)
    ContentFill, // content-aware fill: brush a region, PatchMatch-synthesize from surroundings
    Patch,       // patch: lasso a region, drag it onto a source area, gradient-domain seamless-clone
    Dodge,       // dodge (lighten) / burn (darken with Alt) — brushed soft tonal adjust
    Liquify,     // mesh warp: push/twirl/pucker/bloat via a displacement field
    Detail,      // detail brush: saturate/desaturate (sponge), blur, sharpen
    Fill,
    Eyedropper,
    SelectRect,
    SelectEllipse,
    Lasso,
    MagicWand,
    Transform,
    Crop,
    Text,
    Pen,         // pen: click to add anchors, click-drag for Bézier handles
    DirectSelect, // direct-select: drag anchors/handles of the work path
    ShapeRect,
    ShapeEllipse,
    Gradient,
}

/// Layers + pixels staged for GPU upload on the next frame.
struct PendingUpload {
    size: Size,
    layers: Vec<(LayerId, Vec<u8>)>, // RGBA16F LE bytes per layer
}

#[derive(Clone, Copy, PartialEq)]
enum LayerAction {
    None,
    Delete(LayerId),
    MoveUp(LayerId),
    MoveDown(LayerId),
}

/// 8-bit sRGB RGBA -> linear-premultiplied f16 bytes (the GPU layer format).
fn rgba8_to_f16_bytes(rgba8: &[u8]) -> Vec<u8> {
    let mut o = Vec::with_capacity(rgba8.len() * 2);
    for px in rgba8.chunks_exact(4) {
        let a = px[3] as f32 / 255.0;
        let ch = [
            srgb_to_linear(px[0] as f32 / 255.0) * a,
            srgb_to_linear(px[1] as f32 / 255.0) * a,
            srgb_to_linear(px[2] as f32 / 255.0) * a,
            a,
        ];
        for &c in &ch {
            o.extend_from_slice(&f16::from_f32(c).to_le_bytes());
        }
    }
    o
}

fn image_to_f16_bytes(img: &LoadedImage) -> Vec<u8> {
    rgba8_to_f16_bytes(&img.rgba8)
}

fn f16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect()
}

fn f32_to_f16_bytes(px: &[f32]) -> Vec<u8> {
    let mut o = Vec::with_capacity(px.len() * 2);
    for &c in px {
        o.extend_from_slice(&f16::from_f32(c).to_le_bytes());
    }
    o
}

/// Run a closure with mutable access to the GPU canvas state + device/queue.
fn with_gpu<R>(
    frame: &mut eframe::Frame,
    f: impl FnOnce(&mut CanvasGpu, &wgpu::Device, &wgpu::Queue) -> R,
) -> Option<R> {
    let rs = frame.wgpu_render_state()?;
    let mut rend = rs.renderer.write();
    let gpu = rend.callback_resources.get_mut::<CanvasGpu>()?;
    Some(f(gpu, &rs.device, &rs.queue))
}

pub struct PigmentApp {
    doc: Document,
    view: ViewTransform,
    background_id: LayerId,
    pending: Option<PendingUpload>,

    tool: Tool,
    brush_color: egui::Color32,
    brush_size: f32,
    brush_hardness: f32,
    brush_opacity: f32,
    /// Pressure → dab size. Hardware pressure isn't exposed through eframe today,
    /// so we drive it from stroke velocity (faster = thinner) when enabled.
    speed_dynamics: bool,
    min_size_scale: f32,
    fill_tolerance: f32,
    fill_contiguous: bool,
    sample_all: bool,
    needs_fit: bool,

    stroke_last: Option<egui::Vec2>,
    stroke_residual: f32,
    stroke_dirty: Option<(egui::Vec2, egui::Vec2)>,
    wet_active: bool,
    /// Clone Stamp / Healing Brush source anchor (doc px), set by Alt-click.
    clone_source: Option<egui::Vec2>,
    /// destAnchor − sourceAnchor for the active clone/heal stroke (doc px).
    clone_offset: [f32; 2],
    /// Healing Brush coverage (canvas-sized, doc px); accumulated over a stroke,
    /// solved on release.
    heal_mask: Vec<bool>,
    /// Patch tool: the lasso-defined region being transplanted (in-progress
    /// freehand points in doc px while drawing; committed as `patch_region`).
    patch_points: Vec<egui::Vec2>,
    /// Patch tool: the committed destination region as a 0/1 mask (canvas-sized),
    /// and the centroid the user grabs to drag it onto a source area.
    patch_region: Vec<bool>,
    patch_anchor: Option<egui::Vec2>,
    /// Patch tool: live drag offset (source = region translated by −offset).
    patch_offset: egui::Vec2,
    /// Patch tool: "Source" mode (drag the selection onto the texture to use as
    /// fill; PS-style) vs "Destination" mode (drag a sampled patch onto the sel).
    patch_source_mode: bool,
    /// Dodge/Burn soft coverage (0..1, canvas-sized); accumulated over a stroke,
    /// applied on release.
    tone_mask: Vec<f32>,
    /// Liquify: frozen straight-linear source captured at stroke start, the
    /// accumulating displacement field, and the active warp mode.
    liquify_src: Vec<f32>,
    liquify_disp: Vec<[f32; 2]>,
    liquify_mode: u8, // 0 push, 1 twirl, 2 pucker, 3 bloat
    detail_mode: u8,  // 0 saturate, 1 desaturate, 2 blur, 3 sharpen
    undo_count: u32,
    redo_count: u32,
    // Compositing dirty tracking.
    force_composite: bool,
    last_fingerprint: u64,

    // Selection.
    sel_drag_start: Option<egui::Vec2>,
    sel_op_pending: Option<SelectionOp>,
    selection_active: bool,
    sel_base: Vec<f32>,            // selection snapshot at op start (for combine)
    sel_mode: CombineMode,         // current modifier-derived combine mode
    lasso_points: Vec<egui::Vec2>, // in-progress lasso (document px)
    feather_radius: f32,

    // Pen tool / work paths (vector overlay, not part of the GPU composite).
    /// The current editable work path (anchors in document px).
    work_path: WorkPath,
    /// Pen drag in progress on the just-added anchor: drag sets its handles.
    pen_dragging: bool,
    /// Direct-select grab: which anchor and which part is being dragged.
    /// `part`: 0 = on-curve point (moves whole anchor), 1 = h_in, 2 = h_out.
    pen_grab: Option<(usize, u8)>,

    // Move/transform of the active layer.
    xform_active: bool,
    xform_translate: egui::Vec2, // document px
    xform_scale: f32,

    // Clipboard (full-canvas RGBA f32, selection-masked) + resize dialog state.
    clipboard: Option<Vec<f32>>,
    clipboard_size: Size,
    resize_w: u32,
    resize_h: u32,

    // Layer masks.
    masked_layers: HashSet<LayerId>,
    edit_mask: bool,
    /// Per-layer Blend-If luma ranges [this_black, this_white, under_black, under_white].
    blend_if: HashMap<LayerId, [f32; 4]>,
    /// Layers clipped to the layer directly below (clipping mask).
    clipped_layers: HashSet<LayerId>,
    /// Per-layer outer-stroke style: (straight rgba, width in doc px).
    layer_strokes: HashMap<LayerId, ([f32; 4], f32)>,
    /// Per-layer drop-shadow style: (straight rgba, offset px [dx,dy], blur px).
    layer_shadows: HashMap<LayerId, ([f32; 4], [f32; 2], f32)>,
    /// Per-layer color-overlay style: straight rgba (a = strength).
    layer_overlays: HashMap<LayerId, [f32; 4]>,
    /// Per-layer inner-shadow style: (straight rgba, offset px [dx,dy], blur px).
    layer_inner_shadows: HashMap<LayerId, ([f32; 4], [f32; 2], f32)>,
    /// Per-layer outer-glow style: (straight rgba, size px).
    layer_outer_glows: HashMap<LayerId, ([f32; 4], f32)>,
    /// Per-layer inner-glow style: (straight rgba, size px).
    layer_inner_glows: HashMap<LayerId, ([f32; 4], f32)>,
    /// Per-layer gradient-overlay style: (color0 rgb, color1 rgb, angle deg, opacity).
    layer_grad_overlays: HashMap<LayerId, ([f32; 4], [f32; 4], f32, f32)>,
    /// Per-layer bevel-&-emboss style (Inner Bevel). Tuple:
    /// (highlight rgba, shadow rgba, size px, soften px, angle deg, altitude deg).
    #[allow(clippy::type_complexity)]
    layer_bevels: HashMap<LayerId, ([f32; 4], [f32; 4], f32, f32, f32, f32)>,
    /// Panel visibility (Window menu show/hide).
    panels: PanelVis,

    // Filters.
    filter_radius: f32,
    filter_amount: f32,
    filter_block: f32,
    // Sharpen family (Phase 8).
    high_pass_radius: f32, // Gaussian radius for High Pass (px)
    high_pass_amount: f32, // detail gain (1 = identity high pass)
    // Blur family (Phase 8).
    motion_angle: f32,    // degrees
    motion_distance: f32, // taps each side
    box_radius: f32,
    radial_amount: f32, // spin: degrees; zoom: percent
    radial_samples: u32,
    radial_spin: bool, // true = spin, false = zoom
    // Tilt-Shift (Blur Gallery).
    tilt_center: f32,    // focus line position (0..1 of canvas height)
    tilt_half_band: f32, // sharp band half-width (px)
    tilt_feather: f32,   // blur ramp width past the band (px)
    tilt_radius: f32,    // max blur radius (px)
    tilt_angle: f32,     // band tilt (degrees)
    // Iris Blur (Blur Gallery): sharp inside an ellipse, blur outside.
    iris_rx: f32,      // ellipse x radius (px)
    iris_ry: f32,      // ellipse y radius (px)
    iris_feather: f32, // feather (normalized fraction of the ellipse radius)
    iris_radius: f32,  // max blur radius (px)
    // Spin Blur (Blur Gallery): rotational motion blur about the canvas center.
    spin_angle: f32,   // blur angle (degrees)
    spin_samples: u32, // tap count (quality)
    // Distort family (Phase 8).
    twirl_angle: f32,       // degrees
    twirl_radius: f32,      // pixels
    pinch_amount: f32,      // signed: + pinch, - bulge
    pinch_radius: f32,      // pixels
    ripple_amplitude: f32,  // pixels
    ripple_wavelength: f32, // pixels
    polar_to_polar: bool,   // true = rect->polar, false = polar->rect
    // Stylize family (Phase 8).
    edge_width: f32,        // Sobel sampling step (px) for Find Edges
    emboss_angle: f32,      // degrees
    emboss_amount: f32,     // relief gain
    emboss_width: f32,      // Sobel sampling step (px)
    glow_brightness: f32,   // glowing-edges brightness gain
    glow_width: f32,        // Sobel sampling step (px)
    diffuse_amount: f32,    // max neighbour displacement (px)
    diffuse_seed: f32,      // deterministic scramble seed
    oil_paint_radius: f32,  // oil-paint (Kuwahara) quadrant half-size (px)
    posterize_levels: u32,  // destructive posterize level count (2..=255)
    threshold_level: f32,   // destructive threshold luma cutoff (0..1)
    // Noise family (Phase 8).
    noise_amount: f32,      // add-noise strength (0..1)
    noise_mono: bool,       // monochromatic (same noise on R/G/B)
    noise_gaussian: bool,   // gaussian (true) vs uniform (false)
    noise_seed: f32,        // deterministic noise seed
    median_radius: f32,     // median / dust window radius (px)
    dust_threshold: f32,    // dust & scratches threshold (0..1)
    // Pixelate family (Phase 8).
    mosaic_cell: f32,       // mosaic / crystallize cell size (px)
    crystallize_seed: f32,  // crystallize jitter seed
    halftone_cell: f32,     // color-halftone dot-screen cell size (px)
    halftone_angle: f32,    // color-halftone screen angle (deg)
    mezzotint_amount: f32,  // mezzotint threshold bias (0..1)
    mezzotint_seed: f32,    // mezzotint dither seed
    // Render family (Phase 8).
    clouds_seed: f32,       // clouds / difference-clouds fBm seed
    clouds_scale: f32,      // base feature size (px)
    clouds_roughness: f32,  // per-octave amplitude falloff (0..1)
    clouds_octaves: u32,    // fBm octave count

    hist: Option<Histogram>,

    // Phase 4: generated (text/vector) layers re-rasterize when their def changes.
    gen_fp: HashMap<LayerId, u64>,
    /// Accumulated placement offset (doc px) of each generated (text/vector)
    /// layer. Generated layers carry no position in their definition — their
    /// pixels are rasterized at the canvas origin — so a Move/Transform bake's
    /// translate is recorded here and re-applied on every re-rasterize, keeping
    /// moved text/shapes in place when a property (family/size/color/align)
    /// changes. Not persisted: a reloaded `.pigment` reconstructs the layer from
    /// pixels, which already carry the placement.
    gen_offset: HashMap<LayerId, egui::Vec2>,
    shape_drag: Option<LayerId>,
    grad_start: Option<egui::Vec2>,
    /// Multi-stop gradient editor state (stops, geometry, dither, presets).
    gradient: gradient::GradientEditor,
    /// A "fill layer with gradient" request (start→end axis in doc px), applied
    /// on the next frame so the GPU readback/upload runs with `frame` in hand.
    grad_fill_pending: Option<(egui::Vec2, egui::Vec2)>,

    /// The current pattern tile (captured from a selection / whole layer via
    /// **Edit ▸ Define Pattern**), used by **Fill with Pattern**. Session-scoped
    /// (not persisted to `.pigment`).
    pattern: Option<pattern::Pattern>,
    /// Pattern-fill scale (the tile is drawn `scale×` its native size).
    pattern_scale: f32,
    /// Pattern-fill origin offset (doc px).
    pattern_offset: (f32, f32),
    /// A "fill with pattern" request, applied on the next frame so the GPU
    /// readback/upload runs with `frame` in hand (mirrors `grad_fill_pending`).
    pat_fill_pending: bool,

    // Dynamic-Link: layers placed from a `.contour` file with a live link. Each
    // frame we stat the source file; a newer mtime triggers a re-rasterize +
    // re-upload to the SAME layer id. (path, last-seen modified time.)
    linked_contours: HashMap<LayerId, (PathBuf, SystemTime)>,

    /// Font families available to text layers, enumerated once from the system
    /// font database (via `prism_io::text::available_families`) and cached
    /// because scanning the DB is expensive. Populated lazily the first time the
    /// text-layer font chooser is shown.
    font_families: Vec<String>,

    /// Named layer comps: snapshots of per-layer appearance (visibility,
    /// opacity, blend mode) the user can create, restore, rename, and delete.
    /// Persisted in the `.pigment` document. See the `comps` module for the pure
    /// capture/restore model.
    comps: Vec<comps::LayerComp>,
    /// Scratch buffer for the "new comp" name field in the Layers panel.
    new_comp_name: String,

    /// Per-layer **smart-filter stacks** (non-destructive, re-editable filters
    /// applied on top of each layer's stored source pixels), keyed by the stable
    /// [`LayerId`]. Absent / empty for layers with no smart filters. Persisted in
    /// the `.pigment` document; see the `smart_filter` module for the pure model
    /// and `canvas::CanvasGpu::reapply_smart_filters` for the GPU re-apply.
    smart_filters: std::collections::HashMap<LayerId, smart_filter::SmartFilterStack>,
}

/// Whether the active layer's live affine must be sent to the GPU this frame.
///
/// True while a Move/Transform drag is in progress (`active`), and *also* on the
/// drag-stop frame that bakes it (`bake`): `drag_stopped` clears `active` before
/// we get here, so without the `bake` term the bake frame would send no affine,
/// the GPU would clear `xform_layer`, and `bake_transform` would be a no-op —
/// snapping the layer (text included) back to its origin.
fn send_layer_xform(active: bool, bake: bool) -> bool {
    active || bake
}

/// Build the uv-space layer-from-canvas affine for a translate + uniform scale
/// about the canvas center. Returns (2x2 matrix [a,b,c,d], offset).
fn compute_xform(translate: egui::Vec2, scale: f32, size: Size) -> ([f32; 4], [f32; 2]) {
    let inv = 1.0 / scale.max(1e-3);
    let tx = translate.x / size.width as f32;
    let ty = translate.y / size.height as f32;
    let m = [inv, 0.0, 0.0, inv];
    let off = [0.5 - (0.5 + tx) * inv, 0.5 - (0.5 + ty) * inv];
    (m, off)
}

#[cfg(test)]
mod xform_tests {
    use super::*;

    // The drag-stop (bake) frame must still send the affine even though the
    // drag is no longer "active" — otherwise the bake is a no-op and the moved
    // layer (text included) snaps back to its origin.
    #[test]
    fn affine_sent_on_bake_frame() {
        // Mid-drag: active, not yet baking.
        assert!(send_layer_xform(true, false));
        // Drag-stop / bake frame: active was just cleared, but bake is set.
        assert!(send_layer_xform(false, true));
        // Both set (defensive).
        assert!(send_layer_xform(true, true));
        // Idle frame: nothing to send.
        assert!(!send_layer_xform(false, false));
    }

    // A pure +half-width translate maps to a layer-from-canvas offset of -0.5.
    #[test]
    fn translate_maps_to_uv_offset() {
        let size = Size::new(100, 50);
        let (m, off) = compute_xform(egui::vec2(50.0, 0.0), 1.0, size);
        assert_eq!(m, [1.0, 0.0, 0.0, 1.0]);
        assert!((off[0] - -0.5).abs() < 1e-6, "off.x = {}", off[0]);
        assert!(off[1].abs() < 1e-6, "off.y = {}", off[1]);
    }
}

#[cfg(test)]
mod first_load_tests {
    use super::*;

    // The default document staged on startup must carry a real, varying gradient
    // in its background layer's upload buffer. If this were empty/uniform the
    // canvas would come up blank on first load (the bug we fixed). This is the
    // exact byte buffer the `pending` upload pushes to the GPU on the first frame.
    #[test]
    fn default_background_upload_is_a_real_gradient() {
        let placeholder = prism_io::placeholder(Size::new(64, 64));
        let bytes = image_to_f16_bytes(&placeholder);

        // Non-empty: four f16 channels per pixel, two bytes each.
        let expected_len = (placeholder.size.width * placeholder.size.height) as usize * 4 * 2;
        assert_eq!(bytes.len(), expected_len, "background buffer is empty/wrong size");

        // The placeholder ramps red across X and green down Y, so the buffer must
        // not be a single flat color — otherwise nothing would be visible as a
        // gradient even once composited.
        let px = f16_bytes_to_f32(&bytes);
        let first = &px[0..4];
        assert!(
            px.chunks_exact(4).any(|p| p[0..4] != *first),
            "default background is uniform, not a gradient"
        );
    }
}

impl PigmentApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let render_state = cc
            .wgpu_render_state
            .as_ref()
            .expect("pigment requires the wgpu backend");
        let gpu = CanvasGpu::new(&render_state.device, render_state.target_format);
        render_state.renderer.write().callback_resources.insert(gpu);

        crate::icons::install(&cc.egui_ctx);
        crate::theme::apply(&cc.egui_ctx);

        let placeholder = prism_io::placeholder(Size::new(1280, 800));
        let doc = Document::new(placeholder.size);
        let background_id = doc.layers.layers[0].id;
        let pending = Some(PendingUpload {
            size: placeholder.size,
            layers: vec![(background_id, image_to_f16_bytes(&placeholder))],
        });
        Self {
            doc,
            view: ViewTransform::default(),
            background_id,
            pending,
            tool: Tool::Brush,
            brush_color: egui::Color32::from_rgb(20, 120, 230),
            brush_size: 40.0,
            brush_hardness: 0.5,
            brush_opacity: 1.0,
            speed_dynamics: false,
            min_size_scale: 0.35,
            fill_tolerance: 0.1,
            fill_contiguous: true,
            sample_all: false,
            needs_fit: true,
            stroke_last: None,
            stroke_residual: 0.0,
            stroke_dirty: None,
            wet_active: false,
            clone_source: None,
            clone_offset: [0.0; 2],
            heal_mask: Vec::new(),
            patch_points: Vec::new(),
            patch_region: Vec::new(),
            patch_anchor: None,
            patch_offset: egui::Vec2::ZERO,
            patch_source_mode: true,
            tone_mask: Vec::new(),
            liquify_src: Vec::new(),
            liquify_disp: Vec::new(),
            liquify_mode: 0,
            detail_mode: 0,
            undo_count: 0,
            redo_count: 0,
            force_composite: true,
            last_fingerprint: 0,
            sel_drag_start: None,
            sel_op_pending: None,
            selection_active: false,
            sel_base: Vec::new(),
            sel_mode: CombineMode::Replace,
            lasso_points: Vec::new(),
            feather_radius: 2.0,
            work_path: WorkPath::new(),
            pen_dragging: false,
            pen_grab: None,
            xform_active: false,
            xform_translate: egui::Vec2::ZERO,
            xform_scale: 1.0,
            clipboard: None,
            clipboard_size: Size::new(1, 1),
            resize_w: 1280,
            resize_h: 800,
            masked_layers: HashSet::new(),
            blend_if: HashMap::new(),
            clipped_layers: HashSet::new(),
            layer_strokes: HashMap::new(),
            layer_shadows: HashMap::new(),
            layer_overlays: HashMap::new(),
            layer_inner_shadows: HashMap::new(),
            layer_outer_glows: HashMap::new(),
            layer_inner_glows: HashMap::new(),
            layer_grad_overlays: HashMap::new(),
            layer_bevels: HashMap::new(),
            panels: PanelVis::default(),
            edit_mask: false,
            filter_radius: 4.0,
            filter_amount: 1.0,
            filter_block: 8.0,
            high_pass_radius: 3.0,
            high_pass_amount: 1.0,
            motion_angle: 0.0,
            motion_distance: 12.0,
            box_radius: 4.0,
            radial_amount: 15.0,
            radial_samples: 24,
            radial_spin: true,
            tilt_center: 0.5,
            tilt_half_band: 60.0,
            tilt_feather: 120.0,
            tilt_radius: 12.0,
            tilt_angle: 0.0,
            iris_rx: 200.0,
            iris_ry: 150.0,
            iris_feather: 0.5,
            iris_radius: 14.0,
            spin_angle: 15.0,
            spin_samples: 24,
            twirl_angle: 90.0,
            twirl_radius: 200.0,
            pinch_amount: 0.5,
            pinch_radius: 200.0,
            ripple_amplitude: 8.0,
            ripple_wavelength: 40.0,
            polar_to_polar: true,
            edge_width: 1.0,
            emboss_angle: 135.0,
            emboss_amount: 4.0,
            emboss_width: 1.0,
            glow_brightness: 6.0,
            glow_width: 1.0,
            diffuse_amount: 4.0,
            diffuse_seed: 1.0,
            oil_paint_radius: 3.0,
            posterize_levels: 4,
            threshold_level: 0.5,
            noise_amount: 0.1,
            noise_mono: false,
            noise_gaussian: true,
            noise_seed: 1.0,
            median_radius: 1.0,
            dust_threshold: 0.1,
            mosaic_cell: 8.0,
            crystallize_seed: 1.0,
            halftone_cell: 8.0,
            halftone_angle: 45.0,
            mezzotint_amount: 0.5,
            mezzotint_seed: 1.0,
            clouds_seed: 1.0,
            clouds_scale: 32.0,
            clouds_roughness: 0.5,
            clouds_octaves: 5,
            hist: None,
            gen_fp: HashMap::new(),
            gen_offset: HashMap::new(),
            shape_drag: None,
            grad_start: None,
            gradient: gradient::GradientEditor::default(),
            grad_fill_pending: None,
            pattern: None,
            pattern_scale: 1.0,
            pattern_offset: (0.0, 0.0),
            pat_fill_pending: false,
            linked_contours: HashMap::new(),
            font_families: Vec::new(),
            comps: Vec::new(),
            new_comp_name: String::new(),
            smart_filters: HashMap::new(),
        }
    }
}

mod adjustments;
pub(crate) mod camera_raw;
mod comps;
mod edit;
mod gradient;
mod io;
mod pattern;
mod retouch;
pub(crate) mod smart_filter;
mod state;
mod view;

pub(crate) use adjustments::adjustment_ui;

fn screen_to_doc(p: egui::Pos2, doc_rect: egui::Rect, size: Size) -> egui::Vec2 {
    let u = (p.x - doc_rect.min.x) / doc_rect.width().max(1e-3);
    let v = (p.y - doc_rect.min.y) / doc_rect.height().max(1e-3);
    egui::vec2(u * size.width as f32, v * size.height as f32)
}

fn doc_to_screen(p: egui::Vec2, doc_rect: egui::Rect, size: Size) -> egui::Pos2 {
    doc_rect.min
        + egui::vec2(
            p.x / size.width as f32 * doc_rect.width(),
            p.y / size.height as f32 * doc_rect.height(),
        )
}

/// Shift = add, Alt = subtract, Shift+Alt = intersect, else replace.
fn mode_from_modifiers(m: egui::Modifiers) -> CombineMode {
    match (m.shift, m.alt) {
        (true, false) => CombineMode::Add,
        (false, true) => CombineMode::Subtract,
        (true, true) => CombineMode::Intersect,
        _ => CombineMode::Replace,
    }
}

/// Rasterize a rectangle/ellipse marquee to a 0/1 selection mask.
fn shape_mask(rect: [f32; 4], ellipse: bool, w: u32, h: u32) -> Vec<f32> {
    let [rx, ry, rw, rh] = rect;
    let (cx, cy) = (rx + rw * 0.5, ry + rh * 0.5);
    let (hx, hy) = ((rw * 0.5).max(1e-3), (rh * 0.5).max(1e-3));
    let mut m = vec![0.0; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
            let inside = if ellipse {
                let (dx, dy) = ((px - cx) / hx, (py - cy) / hy);
                dx * dx + dy * dy <= 1.0
            } else {
                px >= rx && px <= rx + rw && py >= ry && py <= ry + rh
            };
            if inside {
                m[(y * w + x) as usize] = 1.0;
            }
        }
    }
    m
}

/// Translate a boolean region mask by `(dx, dy)` doc px (rounded), dropping
/// pixels that fall off-canvas. Used by the Patch tool to relocate the
/// destination region in Destination mode.
fn translate_mask(mask: &[bool], w: u32, h: u32, dx: f32, dy: f32) -> Vec<bool> {
    let (dx, dy) = (dx.round() as i64, dy.round() as i64);
    let (wi, hi) = (w as i64, h as i64);
    let mut out = vec![false; (w * h) as usize];
    for y in 0..hi {
        for x in 0..wi {
            if !mask[(y * wi + x) as usize] {
                continue;
            }
            let (nx, ny) = (x + dx, y + dy);
            if nx >= 0 && ny >= 0 && nx < wi && ny < hi {
                out[(ny * wi + nx) as usize] = true;
            }
        }
    }
    out
}

/// Copy a `dw`x`dh` window of an RGBA-f32 image, sampling src at (x+ox, y+oy);
/// out-of-bounds reads are transparent. Used for crop and canvas resize.
fn reposition(src: &[f32], sw: u32, sh: u32, dw: u32, dh: u32, ox: i32, oy: i32) -> Vec<f32> {
    let mut dst = vec![0.0; (dw * dh * 4) as usize];
    for y in 0..dh {
        for x in 0..dw {
            let sx = x as i32 + ox;
            let sy = y as i32 + oy;
            if sx >= 0 && sx < sw as i32 && sy >= 0 && sy < sh as i32 {
                let si = ((sy as u32 * sw + sx as u32) * 4) as usize;
                let di = ((y * dw + x) * 4) as usize;
                dst[di..di + 4].copy_from_slice(&src[si..si + 4]);
            }
        }
    }
    dst
}

/// Mirror an RGBA-f32 image horizontally or vertically in place-by-copy.
fn flip(src: &[f32], w: u32, h: u32, horizontal: bool) -> Vec<f32> {
    let mut dst = vec![0.0; src.len()];
    for y in 0..h {
        for x in 0..w {
            let (sx, sy) = if horizontal {
                (w - 1 - x, y)
            } else {
                (x, h - 1 - y)
            };
            let si = ((sy * w + sx) * 4) as usize;
            let di = ((y * w + x) * 4) as usize;
            dst[di..di + 4].copy_from_slice(&src[si..si + 4]);
        }
    }
    dst
}

fn srgba_to_color(c: [f32; 4]) -> egui::Color32 {
    egui::Color32::from_rgba_unmultiplied(
        (c[0] * 255.0) as u8,
        (c[1] * 255.0) as u8,
        (c[2] * 255.0) as u8,
        (c[3] * 255.0) as u8,
    )
}

fn color_to_srgba(c: egui::Color32) -> [f32; 4] {
    [
        c.r() as f32 / 255.0,
        c.g() as f32 / 255.0,
        c.b() as f32 / 255.0,
        c.a() as f32 / 255.0,
    ]
}

fn clamp_seed(p: egui::Vec2, size: Size) -> Option<(u32, u32)> {
    if p.x < 0.0 || p.y < 0.0 || p.x >= size.width as f32 || p.y >= size.height as f32 {
        return None;
    }
    Some((p.x as u32, p.y as u32))
}
