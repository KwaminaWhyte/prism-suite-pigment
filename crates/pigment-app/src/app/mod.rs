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
use prism_io::resize::{resize_rgba_f32, Quality};
use prism_io::text::{self, TextAlign};
use prism_io::LoadedImage;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::SystemTime;

use crate::canvas::{CanvasGpu, CanvasPaint, Dab, LayerDraw, SelectionOp, ViewTransform};

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
    /// Panel visibility (Window menu show/hide).
    panels: PanelVis,

    // Filters.
    filter_radius: f32,
    filter_amount: f32,
    filter_block: f32,

    hist: Option<Histogram>,

    // Phase 4: generated (text/vector) layers re-rasterize when their def changes.
    gen_fp: HashMap<LayerId, u64>,
    shape_drag: Option<LayerId>,
    grad_start: Option<egui::Vec2>,

    // Dynamic-Link: layers placed from a `.contour` file with a live link. Each
    // frame we stat the source file; a newer mtime triggers a re-rasterize +
    // re-upload to the SAME layer id. (path, last-seen modified time.)
    linked_contours: HashMap<LayerId, (PathBuf, SystemTime)>,
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
            panels: PanelVis::default(),
            edit_mask: false,
            filter_radius: 4.0,
            filter_amount: 1.0,
            filter_block: 8.0,
            hist: None,
            gen_fp: HashMap::new(),
            shape_drag: None,
            grad_start: None,
            linked_contours: HashMap::new(),
        }
    }
}

mod adjustments;
mod edit;
mod io;
mod retouch;
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
