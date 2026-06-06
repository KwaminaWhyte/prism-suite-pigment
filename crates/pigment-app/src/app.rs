//! The eframe application: panels, tools, and the GPU canvas. Owns document
//! state and drives GPU uploads/readbacks via `frame.wgpu_render_state()`.

use std::collections::HashSet;

use eframe::egui_wgpu;
use eframe::wgpu;
use half::f16;
use pigment_core::adjust::Adjustment;
use pigment_core::color::{linear_to_srgb, srgb_to_linear};
use pigment_core::fill::flood_fill_mask;
use pigment_core::histogram::{self, Histogram};
use pigment_core::layer::{TextDef, VectorDef};
use pigment_core::raster::{self, CombineMode};
use pigment_core::shape::{self, ShapeKind};
use pigment_core::{BlendMode, Document, Layer, LayerId, LayerKind, Size};
use pigment_io::document_file::{self, DocMeta, LayerMeta, LayerPixels};
use pigment_io::resize::{resize_rgba_f32, Quality};
use pigment_io::text::{self, TextAlign};
use pigment_io::LoadedImage;
use std::collections::HashMap;

use crate::canvas::{CanvasGpu, CanvasPaint, Dab, LayerDraw, SelectionOp, ViewTransform};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tool {
    Move,      // pan the view (hand)
    MoveLayer, // translate the active layer
    Brush,
    Eraser,
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

    // Filters.
    filter_radius: f32,
    filter_amount: f32,
    filter_block: f32,

    hist: Option<Histogram>,

    // Phase 4: generated (text/vector) layers re-rasterize when their def changes.
    gen_fp: HashMap<LayerId, u64>,
    shape_drag: Option<LayerId>,
    grad_start: Option<egui::Vec2>,
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

        let placeholder = pigment_io::placeholder(Size::new(1280, 800));
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
            edit_mask: false,
            filter_radius: 4.0,
            filter_amount: 1.0,
            filter_block: 8.0,
            hist: None,
            gen_fp: HashMap::new(),
            shape_drag: None,
            grad_start: None,
        }
    }

    /// Re-rasterize any text/vector layer whose definition changed, uploading
    /// the result to its layer texture (keeps them editable after creation).
    fn sync_generated_layers(&mut self, frame: &mut eframe::Frame) {
        use std::hash::{Hash, Hasher};
        let (w, h) = (self.doc.size.width, self.doc.size.height);
        let mut jobs: Vec<(LayerId, Vec<u8>)> = Vec::new();
        for l in &self.doc.layers.layers {
            let fp = match &l.kind {
                LayerKind::Text(t) => {
                    let mut hsh = std::collections::hash_map::DefaultHasher::new();
                    t.text.hash(&mut hsh);
                    t.font_px.to_bits().hash(&mut hsh);
                    t.align.hash(&mut hsh);
                    for c in t.color {
                        c.to_bits().hash(&mut hsh);
                    }
                    (w, h).hash(&mut hsh);
                    hsh.finish()
                }
                LayerKind::Vector(v) => {
                    let mut hsh = std::collections::hash_map::DefaultHasher::new();
                    v.kind.hash(&mut hsh);
                    for x in v.rect {
                        x.to_bits().hash(&mut hsh);
                    }
                    for c in v.color {
                        c.to_bits().hash(&mut hsh);
                    }
                    (w, h).hash(&mut hsh);
                    hsh.finish()
                }
                _ => continue,
            };
            if self.gen_fp.get(&l.id) == Some(&fp) {
                continue;
            }
            self.gen_fp.insert(l.id, fp);
            let px = match &l.kind {
                LayerKind::Text(t) => {
                    let align = match t.align {
                        1 => TextAlign::Center,
                        2 => TextAlign::Right,
                        _ => TextAlign::Left,
                    };
                    text::render_text(&t.text, t.font_px, t.color, w, h, align)
                }
                LayerKind::Vector(v) => {
                    let kind = if v.kind == 1 {
                        ShapeKind::Ellipse
                    } else {
                        ShapeKind::Rectangle
                    };
                    let c = v.color;
                    let lin = [
                        srgb_to_linear(c[0]),
                        srgb_to_linear(c[1]),
                        srgb_to_linear(c[2]),
                        c[3],
                    ];
                    shape::fill_shape(kind, v.rect, lin, w, h)
                }
                _ => continue,
            };
            jobs.push((l.id, f32_to_f16_bytes(&px)));
        }
        if jobs.is_empty() {
            return;
        }
        with_gpu(frame, |gpu, device, queue| {
            for (id, bytes) in &jobs {
                gpu.ensure_layer(device, *id);
                gpu.upload_layer(queue, *id, bytes);
            }
        });
        self.force_composite = true;
    }

    fn refresh_histogram(&mut self, frame: &mut eframe::Frame) {
        let order = self.layer_order();
        let comp = with_gpu(frame, |gpu, d, q| {
            let p = gpu.composite_now(d, q, &order);
            gpu.read_composite(d, q, p)
        })
        .flatten();
        self.force_composite = true;
        if let Some(b) = comp {
            self.hist = Some(histogram::histogram(&f16_bytes_to_f32(&b), 256));
        }
    }

    /// Fill the active layer with a foreground→transparent linear gradient,
    /// composited over its existing pixels.
    fn do_gradient(&mut self, frame: &mut eframe::Frame, p0: egui::Vec2, p1: egui::Vec2) {
        let (w, h) = (self.doc.size.width, self.doc.size.height);
        let active = self.active_id();
        let c = self.brush_color;
        let c0 = [
            srgb_to_linear(c.r() as f32 / 255.0),
            srgb_to_linear(c.g() as f32 / 255.0),
            srgb_to_linear(c.b() as f32 / 255.0),
            1.0,
        ];
        let grad = shape::linear_gradient((p0.x, p0.y), (p1.x, p1.y), c0, [0.0; 4], w, h);
        with_gpu(frame, |gpu, d, q| {
            gpu.begin_command_now(d, q, active, "Gradient");
            let Some(b) = gpu.read_layer(d, q, active) else {
                return;
            };
            let mut base = f16_bytes_to_f32(&b);
            for i in 0..(w * h) as usize {
                let ga = grad[i * 4 + 3];
                for c in 0..4 {
                    base[i * 4 + c] = grad[i * 4 + c] + base[i * 4 + c] * (1.0 - ga);
                }
            }
            gpu.upload_layer(q, active, &f32_to_f16_bytes(&base));
        });
        self.force_composite = true;
    }

    fn do_filter(&mut self, frame: &mut eframe::Frame, kind: u32, radius: f32, amount: f32) {
        let active = self.active_id();
        with_gpu(frame, |gpu, d, q| {
            gpu.apply_filter(d, q, active, kind, radius, amount)
        });
        self.force_composite = true;
    }

    fn add_mask(&mut self, frame: &mut eframe::Frame, from_selection: bool) {
        let active = self.active_id();
        let values = if from_selection && self.selection_active {
            Some(self.read_selection(frame))
        } else {
            None
        };
        with_gpu(frame, |gpu, d, q| {
            gpu.set_mask(d, q, active, values.as_deref())
        });
        self.masked_layers.insert(active);
        self.force_composite = true;
    }

    fn delete_mask(&mut self, frame: &mut eframe::Frame) {
        let active = self.active_id();
        with_gpu(frame, |gpu, _, _| gpu.delete_mask(active));
        self.masked_layers.remove(&active);
        self.edit_mask = false;
        self.force_composite = true;
    }

    /// Read every layer to CPU f32, map each through `map`, recreate the canvas
    /// at `new_size`, and re-upload. Used by resize/canvas/crop.
    fn rebuild_document(
        &mut self,
        frame: &mut eframe::Frame,
        new_size: Size,
        map: impl Fn(&[f32]) -> Vec<f32>,
    ) {
        let ids: Vec<LayerId> = self.doc.layers.layers.iter().map(|l| l.id).collect();
        let layers_px = with_gpu(frame, |gpu, d, q| {
            ids.iter()
                .filter_map(|id| {
                    gpu.read_layer(d, q, *id)
                        .map(|b| (*id, f16_bytes_to_f32(&b)))
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
        let mapped: Vec<(LayerId, Vec<u8>)> = layers_px
            .iter()
            .map(|(id, px)| (*id, f32_to_f16_bytes(&map(px))))
            .collect();
        self.doc.size = new_size;
        with_gpu(frame, |gpu, d, q| {
            gpu.ensure_canvas(d, new_size);
            for (id, bytes) in &mapped {
                gpu.ensure_layer(d, *id);
                gpu.upload_layer(q, *id, bytes);
            }
        });
        self.selection_active = false;
        self.force_composite = true;
        self.needs_fit = true;
    }

    fn resize_image(&mut self, frame: &mut eframe::Frame, nw: u32, nh: u32) {
        let old = self.doc.size;
        let q = if nw < old.width {
            Quality::Lanczos3
        } else {
            Quality::Bicubic
        };
        self.rebuild_document(frame, Size::new(nw, nh), move |px| {
            resize_rgba_f32(px, old.width, old.height, nw, nh, q)
        });
    }

    fn resize_canvas(&mut self, frame: &mut eframe::Frame, nw: u32, nh: u32) {
        let old = self.doc.size;
        self.rebuild_document(frame, Size::new(nw, nh), move |px| {
            reposition(px, old.width, old.height, nw, nh, 0, 0)
        });
    }

    fn crop_to_selection(&mut self, frame: &mut eframe::Frame) {
        if !self.selection_active {
            return;
        }
        let (w, h) = (self.doc.size.width, self.doc.size.height);
        let sel = self.read_selection(frame);
        let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0u32, 0u32);
        for y in 0..h {
            for x in 0..w {
                if sel[(y * w + x) as usize] > 0.5 {
                    x0 = x0.min(x);
                    y0 = y0.min(y);
                    x1 = x1.max(x + 1);
                    y1 = y1.max(y + 1);
                }
            }
        }
        if x1 <= x0 || y1 <= y0 {
            return;
        }
        let (rw, rh, ox, oy) = (x1 - x0, y1 - y0, x0 as i32, y0 as i32);
        self.rebuild_document(frame, Size::new(rw, rh), move |px| {
            reposition(px, w, h, rw, rh, ox, oy)
        });
    }

    fn flip_active(&mut self, frame: &mut eframe::Frame, horizontal: bool) {
        let active = self.active_id();
        let (w, h) = (self.doc.size.width, self.doc.size.height);
        with_gpu(frame, |gpu, d, q| {
            gpu.begin_command_now(d, q, active, "Flip");
            if let Some(b) = gpu.read_layer(d, q, active) {
                let f = flip(&f16_bytes_to_f32(&b), w, h, horizontal);
                gpu.upload_layer(q, active, &f32_to_f16_bytes(&f));
            }
        });
        self.force_composite = true;
    }

    /// Copy the active layer (masked by the selection) into the clipboard.
    fn copy_selection(&mut self, frame: &mut eframe::Frame) {
        let active = self.active_id();
        let n = (self.doc.size.width * self.doc.size.height) as usize;
        let has_sel = self.selection_active;
        let res = with_gpu(frame, |gpu, d, q| {
            let px = gpu.read_layer(d, q, active)?;
            let sel = if has_sel {
                gpu.read_selection(d, q)
            } else {
                None
            };
            Some((f16_bytes_to_f32(&px), sel))
        })
        .flatten();
        let Some((mut px, sel)) = res else { return };
        if let Some(s) = sel {
            for i in 0..n {
                let a = s[i];
                for c in 0..4 {
                    px[i * 4 + c] *= a;
                }
            }
        }
        self.clipboard = Some(px);
        self.clipboard_size = self.doc.size;
    }

    fn cut_selection(&mut self, frame: &mut eframe::Frame) {
        self.copy_selection(frame);
        if !self.selection_active {
            return;
        }
        let active = self.active_id();
        let n = (self.doc.size.width * self.doc.size.height) as usize;
        with_gpu(frame, |gpu, d, q| {
            gpu.begin_command_now(d, q, active, "Cut");
            let (Some(b), Some(sel)) = (gpu.read_layer(d, q, active), gpu.read_selection(d, q))
            else {
                return;
            };
            let mut px = f16_bytes_to_f32(&b);
            for i in 0..n {
                if sel[i] > 0.5 {
                    for c in 0..4 {
                        px[i * 4 + c] = 0.0;
                    }
                }
            }
            gpu.upload_layer(q, active, &f32_to_f16_bytes(&px));
        });
        self.force_composite = true;
    }

    fn paste(&mut self, frame: &mut eframe::Frame) {
        let Some(mut cb) = self.clipboard.clone() else {
            return;
        };
        if self.clipboard_size != self.doc.size {
            cb = reposition(
                &cb,
                self.clipboard_size.width,
                self.clipboard_size.height,
                self.doc.size.width,
                self.doc.size.height,
                0,
                0,
            );
        }
        let id = self.doc.layers.add_raster("Pasted");
        self.doc.active_layer = Some(id);
        with_gpu(frame, |gpu, d, q| {
            gpu.ensure_layer(d, id);
            gpu.upload_layer(q, id, &f32_to_f16_bytes(&cb));
        });
        self.force_composite = true;
    }

    fn layer_from_selection(&mut self, frame: &mut eframe::Frame) {
        self.copy_selection(frame);
        self.paste(frame);
    }

    /// Read the current GPU selection mask (zeros if none active).
    fn read_selection(&mut self, frame: &mut eframe::Frame) -> Vec<f32> {
        let n = (self.doc.size.width * self.doc.size.height) as usize;
        if !self.selection_active {
            return vec![0.0; n];
        }
        with_gpu(frame, |gpu, device, queue| {
            gpu.read_selection(device, queue)
        })
        .flatten()
        .unwrap_or_else(|| vec![0.0; n])
    }

    /// Upload a CPU selection mask and mark a selection active.
    fn set_selection(&mut self, frame: &mut eframe::Frame, mask: &[f32]) {
        with_gpu(frame, |gpu, _, queue| gpu.upload_selection(queue, mask));
        self.selection_active = true;
    }

    /// Read → transform → upload the selection (feather / grow / shrink).
    fn map_selection(
        &mut self,
        frame: &mut eframe::Frame,
        f: impl Fn(&[f32], u32, u32) -> Vec<f32>,
    ) {
        if !self.selection_active {
            return;
        }
        let (w, h) = (self.doc.size.width, self.doc.size.height);
        let cur = self.read_selection(frame);
        let next = f(&cur, w, h);
        self.set_selection(frame, &next);
    }

    /// Combine a freshly-computed mask with the op's base per the active mode.
    fn commit_selection(&mut self, frame: &mut eframe::Frame, new_mask: Vec<f32>) {
        let combined = raster::combine(&self.sel_base, &new_mask, self.sel_mode);
        self.set_selection(frame, &combined);
    }

    /// Grow the current stroke's dirty bounding box by a dab at `p` of radius `r`.
    fn expand_dirty(&mut self, p: egui::Vec2, r: f32) {
        let lo = egui::vec2(p.x - r, p.y - r);
        let hi = egui::vec2(p.x + r, p.y + r);
        self.stroke_dirty = Some(match self.stroke_dirty {
            None => (lo, hi),
            Some((a, b)) => (
                egui::vec2(a.x.min(lo.x), a.y.min(lo.y)),
                egui::vec2(b.x.max(hi.x), b.y.max(hi.y)),
            ),
        });
    }

    /// The stroke's dirty box as a clamped `[x, y, w, h]` (whole canvas if unset).
    fn dirty_rect(&self) -> [u32; 4] {
        let (w, h) = (self.doc.size.width, self.doc.size.height);
        match self.stroke_dirty {
            Some((mn, mx)) => {
                let x = mn.x.floor().clamp(0.0, w as f32) as u32;
                let y = mn.y.floor().clamp(0.0, h as f32) as u32;
                let rw = ((mx.x.ceil().clamp(0.0, w as f32) as u32).saturating_sub(x)).max(1);
                let rh = ((mx.y.ceil().clamp(0.0, h as f32) as u32).saturating_sub(y)).max(1);
                [x, y, rw, rh]
            }
            None => [0, 0, w, h],
        }
    }

    fn layer_fingerprint(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.active_id().0.hash(&mut h);
        for l in &self.doc.layers.layers {
            l.id.0.hash(&mut h);
            l.visible.hash(&mut h);
            l.blend.shader_id().hash(&mut h);
            l.opacity.to_bits().hash(&mut h);
            if let LayerKind::Adjustment(a) = &l.kind {
                let (k, p) = a.encode();
                k.hash(&mut h);
                for v in p {
                    v.to_bits().hash(&mut h);
                }
            }
        }
        h.finish()
    }

    fn active_id(&self) -> LayerId {
        self.doc.active_layer.unwrap_or(self.background_id)
    }

    fn layer_order(&self) -> Vec<LayerDraw> {
        self.doc
            .layers
            .layers
            .iter()
            .map(|l| {
                let (adjust_kind, adjust) = match &l.kind {
                    LayerKind::Adjustment(a) => a.encode(),
                    _ => (0, [0.0; 4]),
                };
                LayerDraw {
                    id: l.id,
                    opacity: l.opacity,
                    blend: l.blend.shader_id(),
                    visible: l.visible,
                    adjust_kind,
                    adjust,
                }
            })
            .collect()
    }

    fn open_image(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Images", pigment_io::SUPPORTED_EXTENSIONS)
            .pick_file()
        else {
            return;
        };
        match pigment_io::load_image(&path) {
            Ok(img) => {
                self.doc = Document::new(img.size);
                self.background_id = self.doc.layers.layers[0].id;
                self.pending = Some(PendingUpload {
                    size: img.size,
                    layers: vec![(self.background_id, image_to_f16_bytes(&img))],
                });
                self.needs_fit = true;
            }
            Err(e) => log::error!("open image failed: {e}"),
        }
    }

    fn open_pigment(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Pigment", &["pigment"])
            .pick_file()
        else {
            return;
        };
        let (meta, pixels) = match document_file::load_document(&path) {
            Ok(v) => v,
            Err(e) => {
                log::error!("open .pigment failed: {e}");
                return;
            }
        };
        let size = Size::new(meta.width, meta.height);
        let mut doc = Document::new(size);
        doc.layers.layers.clear();
        let mut staged = Vec::new();
        for (i, lm) in meta.layers.iter().enumerate() {
            let id = doc.layers.alloc_id();
            let mut layer = Layer::raster(id, lm.name.clone());
            layer.opacity = lm.opacity;
            layer.visible = lm.visible;
            layer.blend = BlendMode::from_shader_id(lm.blend);
            doc.layers.layers.push(layer);
            let bytes = pixels.get(i).map(|p| p.rgba16f.clone()).unwrap_or_default();
            staged.push((id, bytes));
        }
        self.background_id = doc
            .layers
            .layers
            .first()
            .map(|l| l.id)
            .unwrap_or(LayerId(0));
        doc.active_layer = doc.layers.layers.last().map(|l| l.id);
        self.doc = doc;
        self.pending = Some(PendingUpload {
            size,
            layers: staged,
        });
        self.needs_fit = true;
    }

    fn open_psd(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Photoshop", &["psd"])
            .pick_file()
        else {
            return;
        };
        let doc_psd = match pigment_io::psd_import::load_psd(&path) {
            Ok(d) => d,
            Err(e) => {
                log::error!("open .psd failed: {e}");
                return;
            }
        };
        let size = Size::new(doc_psd.width.max(1), doc_psd.height.max(1));
        let mut doc = Document::new(size);
        doc.layers.layers.clear();
        let mut staged = Vec::new();
        for pl in &doc_psd.layers {
            let id = doc.layers.alloc_id();
            let mut layer = Layer::raster(id, pl.name.clone());
            layer.opacity = pl.opacity;
            layer.visible = pl.visible;
            layer.blend = BlendMode::from_shader_id(pl.blend);
            doc.layers.layers.push(layer);
            staged.push((id, rgba8_to_f16_bytes(&pl.rgba8)));
        }
        if staged.is_empty() {
            let id = doc.layers.add_raster("Background");
            staged.push((id, vec![0u8; (size.width * size.height * 8) as usize]));
        }
        self.background_id = doc
            .layers
            .layers
            .first()
            .map(|l| l.id)
            .unwrap_or(LayerId(0));
        doc.active_layer = doc.layers.layers.last().map(|l| l.id);
        self.doc = doc;
        self.pending = Some(PendingUpload {
            size,
            layers: staged,
        });
        self.masked_layers.clear();
        self.needs_fit = true;
    }

    fn open_exr(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("OpenEXR", &["exr"])
            .pick_file()
        else {
            return;
        };
        let (size, rgba) = match pigment_io::exr_io::load_exr(&path) {
            Ok(v) => v,
            Err(e) => {
                log::error!("open .exr failed: {e}");
                return;
            }
        };
        // EXR is linear straight RGBA -> premultiply to f16.
        let mut bytes = Vec::with_capacity(rgba.len() * 2);
        for px in rgba.chunks_exact(4) {
            let a = px[3];
            for c in 0..3 {
                bytes.extend_from_slice(&f16::from_f32(px[c] * a).to_le_bytes());
            }
            bytes.extend_from_slice(&f16::from_f32(a).to_le_bytes());
        }
        self.doc = Document::new(size);
        self.background_id = self.doc.layers.layers[0].id;
        self.pending = Some(PendingUpload {
            size,
            layers: vec![(self.background_id, bytes)],
        });
        self.masked_layers.clear();
        self.needs_fit = true;
    }

    fn export_image(&mut self, frame: &mut eframe::Frame) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter(
                "Image",
                &["png", "jpg", "jpeg", "webp", "tif", "tiff", "bmp"],
            )
            .set_file_name("export.png")
            .save_file()
        else {
            return;
        };
        let (w, h) = (self.doc.size.width, self.doc.size.height);
        let order = self.layer_order();
        let comp = with_gpu(frame, |gpu, d, q| {
            let p = gpu.composite_now(d, q, &order);
            gpu.read_composite(d, q, p)
        })
        .flatten();
        self.force_composite = true;
        let Some(bytes) = comp else { return };
        let f = f16_bytes_to_f32(&bytes);
        // Composite is linear premultiplied -> straight sRGB 8-bit.
        let mut rgba8 = Vec::with_capacity((w * h * 4) as usize);
        for px in f.chunks_exact(4) {
            let a = px[3];
            let inv = if a > 1e-5 { 1.0 / a } else { 0.0 };
            let to8 =
                |lin: f32| (linear_to_srgb((lin * inv).clamp(0.0, 1.0)) * 255.0).round() as u8;
            rgba8.push(to8(px[0]));
            rgba8.push(to8(px[1]));
            rgba8.push(to8(px[2]));
            rgba8.push((a.clamp(0.0, 1.0) * 255.0).round() as u8);
        }
        if let Err(e) = pigment_io::export::save_rgba8(&path, &rgba8, w, h) {
            log::error!("export failed: {e}");
        }
    }

    fn save_pigment(&mut self, frame: &mut eframe::Frame) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Pigment", &["pigment"])
            .set_file_name("untitled.pigment")
            .save_file()
        else {
            return;
        };
        let meta = DocMeta {
            width: self.doc.size.width,
            height: self.doc.size.height,
            layers: self
                .doc
                .layers
                .layers
                .iter()
                .map(|l| LayerMeta {
                    id: l.id.0,
                    name: l.name.clone(),
                    blend: l.blend.shader_id(),
                    opacity: l.opacity,
                    visible: l.visible,
                })
                .collect(),
        };
        let ids: Vec<LayerId> = self.doc.layers.layers.iter().map(|l| l.id).collect();
        let pixels = with_gpu(frame, |gpu, device, queue| {
            ids.iter()
                .filter_map(|id| {
                    gpu.read_layer(device, queue, *id).map(|b| LayerPixels {
                        id: id.0,
                        rgba16f: b,
                    })
                })
                .collect::<Vec<_>>()
        });
        let Some(pixels) = pixels else { return };
        if let Err(e) = document_file::save_document(&path, &meta, &pixels) {
            log::error!("save failed: {e}");
        }
    }

    fn fit(&mut self, viewport: egui::Rect) {
        let img = egui::vec2(self.doc.size.width as f32, self.doc.size.height as f32);
        if img.x <= 0.0 || img.y <= 0.0 {
            return;
        }
        let scale = (viewport.width() / img.x).min(viewport.height() / img.y) * 0.95;
        self.view = ViewTransform {
            pan: egui::Vec2::ZERO,
            zoom: scale.clamp(0.02, 64.0),
        };
    }

    fn dab_at(&self, doc_pos: egui::Vec2, alpha: f32, size_scale: f32) -> Dab {
        let c = self.brush_color;
        Dab {
            center: [doc_pos.x, doc_pos.y],
            radius: (self.brush_size * 0.5 * size_scale).max(0.5),
            hardness: self.brush_hardness.clamp(0.0, 0.99),
            color: [
                srgb_to_linear(c.r() as f32 / 255.0),
                srgb_to_linear(c.g() as f32 / 255.0),
                srgb_to_linear(c.b() as f32 / 255.0),
                alpha,
            ],
        }
    }

    /// Bucket fill the active layer from `seed` (document pixels).
    fn do_fill(&mut self, frame: &mut eframe::Frame, seed: (u32, u32)) {
        let (w, h) = (self.doc.size.width, self.doc.size.height);
        let active = self.active_id();
        let c = self.brush_color;
        let a = self.brush_opacity;
        let fill = [
            srgb_to_linear(c.r() as f32 / 255.0) * a,
            srgb_to_linear(c.g() as f32 / 255.0) * a,
            srgb_to_linear(c.b() as f32 / 255.0) * a,
            a,
        ];
        let tol = self.fill_tolerance;
        let contiguous = self.fill_contiguous;
        let sample_all = self.sample_all;
        let has_sel = self.selection_active;
        let order = self.layer_order();
        with_gpu(frame, |gpu, device, queue| {
            gpu.begin_command_now(device, queue, active, "Fill");
            // Sample source: the composite (all layers) or just the active layer.
            let sample = if sample_all {
                let is_ping = gpu.composite_now(device, queue, &order);
                gpu.read_composite(device, queue, is_ping)
            } else {
                gpu.read_layer(device, queue, active)
            };
            let Some(sample) = sample else { return };
            let sbuf = f16_bytes_to_f32(&sample);
            let mask = flood_fill_mask(&sbuf, w, h, seed.0, seed.1, tol, contiguous);
            // Constrain the fill to the active selection, if any.
            let sel = if has_sel {
                gpu.read_selection(device, queue)
            } else {
                None
            };
            // Write target is always the active layer.
            let Some(active_bytes) = gpu.read_layer(device, queue, active) else {
                return;
            };
            let mut abuf = f16_bytes_to_f32(&active_bytes);
            for (i, &m) in mask.iter().enumerate() {
                let selected = sel.as_ref().is_none_or(|s| s[i] > 0.5);
                if m && selected {
                    let o = i * 4;
                    abuf[o..o + 4].copy_from_slice(&fill);
                }
            }
            gpu.upload_layer(queue, active, &f32_to_f16_bytes(&abuf));
        });
        self.force_composite = true;
    }

    /// Magic wand: flood the composite at `seed` within tolerance → selection.
    fn do_magic_wand(&mut self, frame: &mut eframe::Frame, seed: (u32, u32)) {
        let (w, h) = (self.doc.size.width, self.doc.size.height);
        let order = self.layer_order();
        let (tol, contiguous) = (self.fill_tolerance, self.fill_contiguous);
        let comp = with_gpu(frame, |gpu, device, queue| {
            let p = gpu.composite_now(device, queue, &order);
            gpu.read_composite(device, queue, p)
        })
        .flatten();
        self.force_composite = true;
        let Some(bytes) = comp else { return };
        let buf = f16_bytes_to_f32(&bytes);
        let mask_b = flood_fill_mask(&buf, w, h, seed.0, seed.1, tol, contiguous);
        let mask: Vec<f32> = mask_b.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
        self.commit_selection(frame, mask);
    }

    fn do_eyedrop(&mut self, frame: &mut eframe::Frame, seed: (u32, u32)) {
        let active = self.active_id();
        let sample_all = self.sample_all;
        let order = self.layer_order();
        let px = with_gpu(frame, |gpu, device, queue| {
            if sample_all {
                let is_ping = gpu.composite_now(device, queue, &order);
                gpu.read_composite_pixel(device, queue, is_ping, seed.0, seed.1)
            } else {
                gpu.read_pixel(device, queue, active, seed.0, seed.1)
            }
        })
        .flatten();
        if let Some(p) = px {
            let a = p[3];
            if a > 0.01 {
                let to8 =
                    |lin: f32| (linear_to_srgb(lin / a).clamp(0.0, 1.0) * 255.0).round() as u8;
                self.brush_color = egui::Color32::from_rgb(to8(p[0]), to8(p[1]), to8(p[2]));
            }
        }
        // Sample-all eyedrop runs composite_now (touches ping/pong); recomposite.
        if self.sample_all {
            self.force_composite = true;
        }
    }
}

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

fn adjustment_ui(ui: &mut egui::Ui, adj: &mut Adjustment) {
    match adj {
        Adjustment::BrightnessContrast {
            brightness,
            contrast,
        } => {
            ui.add(egui::Slider::new(brightness, -0.5..=0.5).text("brightness"));
            ui.add(egui::Slider::new(contrast, -0.5..=1.0).text("contrast"));
        }
        Adjustment::Levels {
            in_black,
            in_white,
            gamma,
        } => {
            ui.add(egui::Slider::new(in_black, 0.0..=1.0).text("black"));
            ui.add(egui::Slider::new(in_white, 0.0..=1.0).text("white"));
            ui.add(egui::Slider::new(gamma, 0.1..=4.0).text("gamma"));
        }
        Adjustment::HueSaturation {
            hue,
            saturation,
            lightness,
        } => {
            ui.add(egui::Slider::new(hue, -180.0..=180.0).text("hue"));
            ui.add(egui::Slider::new(saturation, -1.0..=1.0).text("saturation"));
            ui.add(egui::Slider::new(lightness, -0.5..=0.5).text("lightness"));
        }
        Adjustment::Exposure { stops } => {
            ui.add(egui::Slider::new(stops, -3.0..=3.0).text("stops"));
        }
        Adjustment::Threshold { level } => {
            ui.add(egui::Slider::new(level, 0.0..=1.0).text("level"));
        }
        Adjustment::Invert | Adjustment::BlackWhite => {
            ui.label("(no parameters)");
        }
    }
}

fn clamp_seed(p: egui::Vec2, size: Size) -> Option<(u32, u32)> {
    if p.x < 0.0 || p.y < 0.0 || p.x >= size.width as f32 || p.y >= size.height as f32 {
        return None;
    }
    Some((p.x as u32, p.y as u32))
}

impl eframe::App for PigmentApp {
    fn ui(&mut self, root: &mut egui::Ui, frame: &mut eframe::Frame) {
        // Process any staged GPU uploads (new/opened document) before painting.
        if let Some(pend) = self.pending.take() {
            with_gpu(frame, |gpu, device, queue| {
                gpu.ensure_canvas(device, pend.size);
                for (id, bytes) in &pend.layers {
                    gpu.ensure_layer(device, *id);
                    gpu.upload_layer(queue, *id, bytes);
                }
            });
            self.force_composite = true;
        }

        egui::TopBottomPanel::top("menu").show_inside(root, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Open image…").clicked() {
                        self.open_image();
                        ui.close_menu();
                    }
                    if ui.button("Open .pigment…").clicked() {
                        self.open_pigment();
                        ui.close_menu();
                    }
                    if ui.button("Open .psd…").clicked() {
                        self.open_psd();
                        ui.close_menu();
                    }
                    if ui.button("Open .exr…").clicked() {
                        self.open_exr();
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Save .pigment…").clicked() {
                        self.save_pigment(frame);
                        ui.close_menu();
                    }
                    if ui.button("Export image…").clicked() {
                        self.export_image(frame);
                        ui.close_menu();
                    }
                });
                ui.menu_button("Edit", |ui| {
                    if ui.button("Undo").clicked() {
                        self.undo_count += 1;
                        ui.close_menu();
                    }
                    if ui.button("Redo").clicked() {
                        self.redo_count += 1;
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Cut").clicked() {
                        self.cut_selection(frame);
                        ui.close_menu();
                    }
                    if ui.button("Copy").clicked() {
                        self.copy_selection(frame);
                        ui.close_menu();
                    }
                    if ui
                        .add_enabled(self.clipboard.is_some(), egui::Button::new("Paste"))
                        .clicked()
                    {
                        self.paste(frame);
                        ui.close_menu();
                    }
                    if ui.button("Layer from selection").clicked() {
                        self.layer_from_selection(frame);
                        ui.close_menu();
                    }
                });
                ui.menu_button("Image", |ui| {
                    ui.horizontal(|ui| {
                        ui.add(egui::DragValue::new(&mut self.resize_w).range(1..=16384));
                        ui.label("×");
                        ui.add(egui::DragValue::new(&mut self.resize_h).range(1..=16384));
                    });
                    if ui.button("Resize image (resample)").clicked() {
                        self.resize_image(frame, self.resize_w, self.resize_h);
                        ui.close_menu();
                    }
                    if ui.button("Canvas size (no resample)").clicked() {
                        self.resize_canvas(frame, self.resize_w, self.resize_h);
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Crop to selection").clicked() {
                        self.crop_to_selection(frame);
                        ui.close_menu();
                    }
                    if ui.button("Flip layer horizontal").clicked() {
                        self.flip_active(frame, true);
                        ui.close_menu();
                    }
                    if ui.button("Flip layer vertical").clicked() {
                        self.flip_active(frame, false);
                        ui.close_menu();
                    }
                });
                ui.menu_button("Filter", |ui| {
                    ui.add(
                        egui::Slider::new(&mut self.filter_radius, 0.0..=40.0).text("blur radius"),
                    );
                    if ui.button("Gaussian blur").clicked() {
                        self.do_filter(frame, 1, self.filter_radius, 0.0);
                        ui.close_menu();
                    }
                    ui.separator();
                    ui.add(
                        egui::Slider::new(&mut self.filter_amount, 0.0..=4.0)
                            .text("sharpen amount"),
                    );
                    if ui.button("Sharpen").clicked() {
                        self.do_filter(frame, 2, 0.0, self.filter_amount);
                        ui.close_menu();
                    }
                    ui.separator();
                    ui.add(
                        egui::Slider::new(&mut self.filter_block, 1.0..=40.0).text("pixel size"),
                    );
                    if ui.button("Pixelate").clicked() {
                        self.do_filter(frame, 3, self.filter_block, 0.0);
                        ui.close_menu();
                    }
                });
                ui.menu_button("Select", |ui| {
                    if ui.button("All").clicked() {
                        self.sel_op_pending = Some(SelectionOp::All);
                        self.selection_active = false;
                        ui.close_menu();
                    }
                    if ui.button("None").clicked() {
                        self.sel_op_pending = Some(SelectionOp::None);
                        self.selection_active = true;
                        ui.close_menu();
                    }
                    if ui.button("Invert").clicked() {
                        self.sel_op_pending = Some(SelectionOp::Invert);
                        self.selection_active = true;
                        ui.close_menu();
                    }
                    ui.separator();
                    ui.add(
                        egui::Slider::new(&mut self.feather_radius, 0.0..=30.0).text("feather px"),
                    );
                    if ui.button("Feather").clicked() {
                        let r = self.feather_radius;
                        self.map_selection(frame, move |m, w, h| raster::feather(m, w, h, r));
                        ui.close_menu();
                    }
                    if ui.button("Grow 1px").clicked() {
                        self.map_selection(frame, |m, w, h| raster::grow_shrink(m, w, h, 1));
                        ui.close_menu();
                    }
                    if ui.button("Shrink 1px").clicked() {
                        self.map_selection(frame, |m, w, h| raster::grow_shrink(m, w, h, -1));
                        ui.close_menu();
                    }
                });
                ui.menu_button("View", |ui| {
                    if ui.button("Fit to screen").clicked() {
                        self.needs_fit = true;
                        ui.close_menu();
                    }
                    if ui.button("100%").clicked() {
                        self.view = ViewTransform::default();
                        ui.close_menu();
                    }
                });
                ui.separator();
                ui.label(format!("{} %", (self.view.zoom * 100.0).round() as i32));
            });
        });

        egui::SidePanel::left("tools")
            .exact_width(48.0)
            .show_inside(root, |ui| {
                ui.add_space(6.0);
                use crate::icons;
                for (tool, icon, name) in [
                    (Tool::Move, icons::PAN, "Pan view"),
                    (Tool::MoveLayer, icons::MOVE, "Move layer"),
                    (Tool::Brush, icons::BRUSH, "Brush"),
                    (Tool::Eraser, icons::ERASER, "Eraser"),
                    (Tool::Fill, icons::FILL, "Bucket fill"),
                    (Tool::Eyedropper, icons::EYEDROPPER, "Eyedropper"),
                    (Tool::SelectRect, icons::RECT_SELECT, "Rectangle select"),
                    (Tool::SelectEllipse, icons::ELLIPSE_SELECT, "Ellipse select"),
                    (Tool::Lasso, icons::LASSO, "Lasso select"),
                    (Tool::MagicWand, icons::MAGIC_WAND, "Magic wand"),
                    (Tool::Transform, icons::TRANSFORM, "Transform"),
                    (Tool::Crop, icons::CROP, "Crop"),
                    (Tool::Text, icons::TEXT, "Text"),
                    (Tool::ShapeRect, icons::SHAPE, "Rectangle shape"),
                    (Tool::ShapeEllipse, icons::ELLIPSE_SELECT, "Ellipse shape"),
                    (Tool::Gradient, icons::GRADIENT, "Gradient"),
                ] {
                    let btn = egui::SelectableLabel::new(
                        self.tool == tool,
                        egui::RichText::new(icon).size(20.0),
                    );
                    if ui
                        .add_sized([36.0, 30.0], btn)
                        .on_hover_text(name)
                        .clicked()
                    {
                        self.tool = tool;
                    }
                }
                ui.add_space(6.0);
                ui.separator();
                ui.vertical_centered(|ui| ui.color_edit_button_srgba(&mut self.brush_color));
            });

        egui::SidePanel::right("panels")
            .default_width(250.0)
            .show_inside(root, |ui| {
                ui.heading("Brush");
                ui.add(egui::Slider::new(&mut self.brush_size, 1.0..=400.0).text("size"));
                ui.add(egui::Slider::new(&mut self.brush_hardness, 0.0..=0.99).text("hardness"));
                ui.add(egui::Slider::new(&mut self.brush_opacity, 0.0..=1.0).text("opacity"));
                ui.checkbox(&mut self.speed_dynamics, "speed → size")
                    .on_hover_text(
                        "Faster strokes paint thinner (stylus pressure isn't exposed by eframe)",
                    );
                if self.speed_dynamics {
                    ui.add(
                        egui::Slider::new(&mut self.min_size_scale, 0.05..=1.0).text("min size"),
                    );
                }
                if self.tool == Tool::Fill {
                    ui.add(
                        egui::Slider::new(&mut self.fill_tolerance, 0.0..=1.0).text("tolerance"),
                    );
                    ui.checkbox(&mut self.fill_contiguous, "contiguous");
                }
                if matches!(self.tool, Tool::Fill | Tool::Eyedropper) {
                    ui.checkbox(&mut self.sample_all, "sample all layers");
                }
                if matches!(self.tool, Tool::MoveLayer | Tool::Transform) {
                    ui.label("Drag to move the active layer.");
                    if self.tool == Tool::Transform {
                        ui.label("Shift+drag to scale.");
                    }
                }
                if matches!(
                    self.tool,
                    Tool::SelectRect | Tool::SelectEllipse | Tool::Lasso | Tool::MagicWand
                ) {
                    ui.label("Shift: add · Alt: subtract.");
                }

                ui.separator();
                egui::CollapsingHeader::new("Histogram").show(ui, |ui| {
                    if ui.button("Refresh").clicked() {
                        self.refresh_histogram(frame);
                    }
                    if let Some(h) = &self.hist {
                        let (rect, _) = ui.allocate_exact_size(
                            egui::vec2(ui.available_width(), 64.0),
                            egui::Sense::hover(),
                        );
                        let painter = ui.painter_at(rect);
                        painter.rect_filled(rect, 2.0, egui::Color32::from_gray(18));
                        let max = h.luma.iter().copied().max().unwrap_or(1).max(1) as f32;
                        let n = h.luma.len().max(1);
                        for (i, &c) in h.luma.iter().enumerate() {
                            let x = rect.left() + rect.width() * i as f32 / n as f32;
                            let bh = rect.height() * (c as f32 / max);
                            painter.line_segment(
                                [
                                    egui::pos2(x, rect.bottom()),
                                    egui::pos2(x, rect.bottom() - bh),
                                ],
                                egui::Stroke::new(1.0, egui::Color32::from_gray(200)),
                            );
                        }
                    }
                });

                ui.separator();
                let (undos, redos) =
                    with_gpu(frame, |gpu, _, _| gpu.history_labels()).unwrap_or_default();
                egui::CollapsingHeader::new(format!(
                    "History  ({} / {})",
                    undos.len(),
                    redos.len()
                ))
                .show(ui, |ui| {
                    // Future states (redoable), furthest first.
                    for (i, l) in redos.iter().enumerate().rev() {
                        if ui.small_button(format!("redo  {l}")).clicked() {
                            self.redo_count += (i + 1) as u32;
                        }
                    }
                    ui.label("—— now ——");
                    // Past states (undoable), newest first.
                    for (i, l) in undos.iter().rev().enumerate() {
                        if ui.small_button(format!("undo  {l}")).clicked() {
                            self.undo_count += (i + 1) as u32;
                        }
                    }
                });

                ui.separator();
                ui.horizontal(|ui| {
                    ui.heading("Layers");
                    if ui
                        .button(egui::RichText::new(crate::icons::PLUS_LAYER).size(18.0))
                        .on_hover_text("New layer")
                        .clicked()
                    {
                        let id = self
                            .doc
                            .layers
                            .add_raster(format!("Layer {}", self.doc.layers.layers.len()));
                        self.doc.active_layer = Some(id);
                    }
                    ui.menu_button("Adj", |ui| {
                        for adj in Adjustment::DEFAULTS {
                            if ui.button(adj.name()).clicked() {
                                let id = self.doc.layers.add_adjustment(adj);
                                self.doc.active_layer = Some(id);
                                ui.close_menu();
                            }
                        }
                    })
                    .response
                    .on_hover_text("New adjustment layer");
                    ui.menu_button("Mask", |ui| {
                        let has = self.masked_layers.contains(&self.active_id());
                        if ui.button("Add white mask").clicked() {
                            self.add_mask(frame, false);
                            ui.close_menu();
                        }
                        if ui
                            .add_enabled(
                                self.selection_active,
                                egui::Button::new("Add from selection"),
                            )
                            .clicked()
                        {
                            self.add_mask(frame, true);
                            ui.close_menu();
                        }
                        if ui
                            .add_enabled(has, egui::Button::new("Delete mask"))
                            .clicked()
                        {
                            self.delete_mask(frame);
                            ui.close_menu();
                        }
                        ui.add_enabled(
                            has,
                            egui::Checkbox::new(
                                &mut self.edit_mask,
                                "Edit mask (brush=reveal, eraser=hide)",
                            ),
                        );
                    });
                });
                ui.separator();

                let active = self.active_id();
                let mut action = LayerAction::None;
                let ids: Vec<LayerId> = self.doc.layers.layers.iter().rev().map(|l| l.id).collect();
                for id in ids {
                    let layer = self.doc.layers.get_mut(id).unwrap();
                    let is_active = id == active;
                    egui::Frame::NONE
                        .fill(if is_active {
                            egui::Color32::from_rgb(50, 70, 100)
                        } else {
                            egui::Color32::TRANSPARENT
                        })
                        .inner_margin(4.0)
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                let eye = if layer.visible {
                                    crate::icons::EYE
                                } else {
                                    " "
                                };
                                if ui
                                    .selectable_label(
                                        layer.visible,
                                        egui::RichText::new(eye).size(15.0),
                                    )
                                    .on_hover_text("Toggle visibility")
                                    .clicked()
                                {
                                    layer.visible = !layer.visible;
                                }
                                ui.add(
                                    egui::TextEdit::singleline(&mut layer.name).desired_width(90.0),
                                );
                                if ui
                                    .small_button(crate::icons::ARROW_UP)
                                    .on_hover_text("Move up")
                                    .clicked()
                                {
                                    action = LayerAction::MoveUp(id);
                                }
                                if ui
                                    .small_button(crate::icons::ARROW_DOWN)
                                    .on_hover_text("Move down")
                                    .clicked()
                                {
                                    action = LayerAction::MoveDown(id);
                                }
                                if ui
                                    .small_button(crate::icons::TRASH)
                                    .on_hover_text("Delete layer")
                                    .clicked()
                                {
                                    action = LayerAction::Delete(id);
                                }
                            });
                            let is_adjustment = matches!(layer.kind, LayerKind::Adjustment(_));
                            ui.horizontal(|ui| {
                                if ui.selectable_label(is_active, "active").clicked() {
                                    self.doc.active_layer = Some(id);
                                }
                                if !is_adjustment {
                                    egui::ComboBox::from_id_salt(("blend", id.0))
                                        .selected_text(format!("{:?}", layer.blend))
                                        .width(120.0)
                                        .show_ui(ui, |ui| {
                                            for mode in BlendMode::ALL {
                                                ui.selectable_value(
                                                    &mut layer.blend,
                                                    mode,
                                                    format!("{mode:?}"),
                                                );
                                            }
                                        });
                                }
                            });
                            ui.add(
                                egui::Slider::new(&mut layer.opacity, 0.0..=1.0)
                                    .show_value(false)
                                    .text("opacity"),
                            );
                            match &mut layer.kind {
                                LayerKind::Adjustment(adj) => adjustment_ui(ui, adj),
                                LayerKind::Text(t) => {
                                    ui.add(
                                        egui::TextEdit::singleline(&mut t.text)
                                            .desired_width(150.0),
                                    );
                                    ui.add(
                                        egui::Slider::new(&mut t.font_px, 6.0..=300.0).text("size"),
                                    );
                                    let mut col = srgba_to_color(t.color);
                                    if ui.color_edit_button_srgba(&mut col).changed() {
                                        t.color = color_to_srgba(col);
                                    }
                                    ui.horizontal(|ui| {
                                        ui.selectable_value(&mut t.align, 0, "L");
                                        ui.selectable_value(&mut t.align, 1, "C");
                                        ui.selectable_value(&mut t.align, 2, "R");
                                    });
                                }
                                LayerKind::Vector(v) => {
                                    let mut col = srgba_to_color(v.color);
                                    if ui.color_edit_button_srgba(&mut col).changed() {
                                        v.color = color_to_srgba(col);
                                    }
                                }
                                _ => {}
                            }
                        });
                    ui.separator();
                }

                // Apply structural layer changes after the loop.
                let ls = &mut self.doc.layers.layers;
                match action {
                    LayerAction::None => {}
                    LayerAction::Delete(id) => {
                        if ls.len() > 1 {
                            ls.retain(|l| l.id != id);
                            with_gpu(frame, |gpu, _, _| gpu.drop_layer(id));
                            if self.doc.active_layer == Some(id) {
                                self.doc.active_layer = ls.last().map(|l| l.id);
                            }
                            self.background_id =
                                ls.first().map(|l| l.id).unwrap_or(self.background_id);
                        }
                    }
                    LayerAction::MoveUp(id) => {
                        if let Some(i) = ls.iter().position(|l| l.id == id) {
                            if i + 1 < ls.len() {
                                ls.swap(i, i + 1);
                            }
                        }
                    }
                    LayerAction::MoveDown(id) => {
                        if let Some(i) = ls.iter().position(|l| l.id == id) {
                            if i > 0 {
                                ls.swap(i, i - 1);
                            }
                        }
                    }
                }
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(egui::Color32::from_gray(30)))
            .show_inside(root, |ui| {
                let rect = ui.available_rect_before_wrap();
                let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());

                if self.needs_fit {
                    self.fit(rect);
                    self.needs_fit = false;
                }

                let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                if scroll != 0.0 {
                    if let Some(cursor) = response.hover_pos() {
                        self.view
                            .zoom_to((scroll * 0.0015).exp(), cursor - rect.center());
                    }
                }

                let (mut undo, mut redo) = (self.undo_count, self.redo_count);
                self.undo_count = 0;
                self.redo_count = 0;
                let (mut do_copy, mut do_cut, mut do_paste) = (false, false, false);
                ui.input(|i| {
                    if i.modifiers.command {
                        if i.key_pressed(egui::Key::Z) {
                            if i.modifiers.shift {
                                redo += 1;
                            } else {
                                undo += 1;
                            }
                        }
                        if i.key_pressed(egui::Key::Y) {
                            redo += 1;
                        }
                        do_copy |= i.key_pressed(egui::Key::C);
                        do_cut |= i.key_pressed(egui::Key::X);
                        do_paste |= i.key_pressed(egui::Key::V);
                    }
                });
                if do_copy {
                    self.copy_selection(frame);
                }
                if do_cut {
                    self.cut_selection(frame);
                }
                if do_paste {
                    self.paste(frame);
                }

                let img = egui::vec2(self.doc.size.width as f32, self.doc.size.height as f32);
                let doc_rect = egui::Rect::from_center_size(
                    rect.center() + self.view.pan,
                    img * self.view.zoom,
                );

                let mut dabs: Vec<Dab> = Vec::new();
                let mut begin_command = false;
                let mut commit_command = false;
                let mut wet_begin = false;
                let mut wet_end = false;
                let mut bake = false;
                let erase = self.tool == Tool::Eraser;
                let paint_mask = self.edit_mask && self.masked_layers.contains(&self.active_id());
                let dirty_radius = self.brush_size * 0.5 + 1.0; // max dab extent
                                                                // Brush paints full-coverage dabs into the wet layer (opacity is
                                                                // applied when flattening). Eraser paints directly at its strength.
                let dab_alpha = if erase { self.brush_opacity } else { 1.0 };
                match self.tool {
                    Tool::Move => {
                        if response.dragged() {
                            self.view.pan += response.drag_delta();
                        }
                    }
                    Tool::MoveLayer | Tool::Transform => {
                        let allow_scale = self.tool == Tool::Transform;
                        if response.drag_started() {
                            self.xform_active = true;
                            self.xform_translate = egui::Vec2::ZERO;
                            self.xform_scale = 1.0;
                            begin_command = true;
                        }
                        if response.dragged() {
                            if allow_scale && ui.input(|i| i.modifiers.shift) {
                                self.xform_scale = (self.xform_scale
                                    * (1.0 - response.drag_delta().y * 0.005))
                                    .clamp(0.05, 20.0);
                            } else {
                                self.xform_translate +=
                                    response.drag_delta() / self.view.zoom.max(1e-3);
                            }
                            ui.ctx().request_repaint();
                        }
                        if response.drag_stopped() && self.xform_active {
                            bake = true;
                            commit_command = true;
                            self.xform_active = false;
                        }
                    }
                    Tool::Crop => {}
                    Tool::Text => {
                        if response.clicked() {
                            let id = self.doc.layers.add_text(TextDef::default());
                            self.doc.active_layer = Some(id);
                        }
                    }
                    Tool::ShapeRect | Tool::ShapeEllipse => {
                        let kind = if self.tool == Tool::ShapeEllipse {
                            1
                        } else {
                            0
                        };
                        if response.drag_started() {
                            if let Some(p) = response.interact_pointer_pos() {
                                let s = screen_to_doc(p, doc_rect, self.doc.size);
                                self.sel_drag_start = Some(s);
                                let c = self.brush_color;
                                let color = [
                                    c.r() as f32 / 255.0,
                                    c.g() as f32 / 255.0,
                                    c.b() as f32 / 255.0,
                                    1.0,
                                ];
                                let id = self.doc.layers.add_vector(
                                    "Shape",
                                    VectorDef {
                                        kind,
                                        rect: [s.x, s.y, 0.0, 0.0],
                                        color,
                                    },
                                );
                                self.doc.active_layer = Some(id);
                                self.shape_drag = Some(id);
                            }
                        }
                        if response.dragged() {
                            if let (Some(s), Some(p), Some(id)) = (
                                self.sel_drag_start,
                                response.interact_pointer_pos(),
                                self.shape_drag,
                            ) {
                                let cur = screen_to_doc(p, doc_rect, self.doc.size);
                                let rect = [
                                    s.x.min(cur.x),
                                    s.y.min(cur.y),
                                    (s.x - cur.x).abs(),
                                    (s.y - cur.y).abs(),
                                ];
                                if let Some(LayerKind::Vector(v)) =
                                    self.doc.layers.get_mut(id).map(|l| &mut l.kind)
                                {
                                    v.rect = rect;
                                }
                            }
                            ui.ctx().request_repaint();
                        }
                        if response.drag_stopped() {
                            self.shape_drag = None;
                            self.sel_drag_start = None;
                        }
                    }
                    Tool::Gradient => {
                        if response.drag_started() {
                            if let Some(p) = response.interact_pointer_pos() {
                                self.grad_start = Some(screen_to_doc(p, doc_rect, self.doc.size));
                            }
                        }
                        if response.dragged() {
                            ui.ctx().request_repaint();
                        }
                        if response.drag_stopped() {
                            if let (Some(s), Some(p)) =
                                (self.grad_start.take(), response.interact_pointer_pos())
                            {
                                let cur = screen_to_doc(p, doc_rect, self.doc.size);
                                self.do_gradient(frame, s, cur);
                            }
                        }
                        // Guide line.
                        if let (Some(s), Some(p)) = (self.grad_start, response.hover_pos()) {
                            let a = doc_to_screen(s, doc_rect, self.doc.size);
                            ui.painter().add(egui::Shape::line_segment(
                                [a, p],
                                egui::Stroke::new(1.5, egui::Color32::WHITE),
                            ));
                        }
                    }
                    Tool::Brush | Tool::Eraser => {
                        let spacing = (self.brush_size * 0.15).max(0.75);
                        let dt = ui.input(|i| i.stable_dt).max(1e-3);
                        if let Some(p) = response.interact_pointer_pos() {
                            let cur = screen_to_doc(p, doc_rect, self.doc.size);
                            match self.stroke_last {
                                None => {
                                    begin_command = !paint_mask; // mask paint isn't undoable yet
                                    self.stroke_dirty = None;
                                    self.expand_dirty(cur, dirty_radius);
                                    if !erase && !paint_mask {
                                        wet_begin = true;
                                        self.wet_active = true;
                                    }
                                    dabs.push(self.dab_at(cur, dab_alpha, 1.0));
                                    self.stroke_last = Some(cur);
                                    self.stroke_residual = 0.0;
                                }
                                Some(last) => {
                                    let seg = cur - last;
                                    let dist = seg.length();
                                    if dist > 1e-3 {
                                        // Velocity → size: faster strokes taper thinner.
                                        let scale = if self.speed_dynamics {
                                            const SPEED_MAX: f32 = 2500.0; // doc px/sec
                                            let n = (dist / dt / SPEED_MAX).clamp(0.0, 1.0);
                                            1.0 - n * (1.0 - self.min_size_scale)
                                        } else {
                                            1.0
                                        };
                                        let dir = seg / dist;
                                        let mut t = self.stroke_residual;
                                        while t <= dist {
                                            dabs.push(self.dab_at(
                                                last + dir * t,
                                                dab_alpha,
                                                scale,
                                            ));
                                            t += spacing;
                                        }
                                        self.stroke_residual = t - dist;
                                        self.stroke_last = Some(cur);
                                        self.expand_dirty(cur, dirty_radius);
                                    }
                                }
                            }
                            ui.ctx().request_repaint();
                        }
                        if response.drag_stopped()
                            || (self.stroke_last.is_some()
                                && response.interact_pointer_pos().is_none())
                        {
                            if self.wet_active {
                                wet_end = true;
                                self.wet_active = false;
                            }
                            commit_command = true;
                            self.stroke_last = None;
                            self.stroke_residual = 0.0;
                        }
                    }
                    Tool::Fill => {
                        if response.clicked() {
                            if let Some(p) = response.interact_pointer_pos() {
                                let d = screen_to_doc(p, doc_rect, self.doc.size);
                                if let Some(seed) = clamp_seed(d, self.doc.size) {
                                    self.do_fill(frame, seed);
                                }
                            }
                        }
                    }
                    Tool::Eyedropper => {
                        if response.clicked() {
                            if let Some(p) = response.interact_pointer_pos() {
                                let d = screen_to_doc(p, doc_rect, self.doc.size);
                                if let Some(seed) = clamp_seed(d, self.doc.size) {
                                    self.do_eyedrop(frame, seed);
                                }
                            }
                        }
                    }
                    Tool::SelectRect | Tool::SelectEllipse => {
                        let ellipse = self.tool == Tool::SelectEllipse;
                        if response.drag_started() {
                            if let Some(p) = response.interact_pointer_pos() {
                                self.sel_drag_start =
                                    Some(screen_to_doc(p, doc_rect, self.doc.size));
                                self.sel_mode = mode_from_modifiers(ui.input(|i| i.modifiers));
                                self.sel_base = self.read_selection(frame);
                            }
                        }
                        if response.dragged() {
                            if let (Some(start), Some(p)) =
                                (self.sel_drag_start, response.interact_pointer_pos())
                            {
                                let cur = screen_to_doc(p, doc_rect, self.doc.size);
                                let rect = [
                                    start.x.min(cur.x),
                                    start.y.min(cur.y),
                                    (start.x - cur.x).abs(),
                                    (start.y - cur.y).abs(),
                                ];
                                let (w, h) = (self.doc.size.width, self.doc.size.height);
                                let shape = shape_mask(rect, ellipse, w, h);
                                self.commit_selection(frame, shape);
                            }
                            ui.ctx().request_repaint();
                        }
                        if response.drag_stopped() {
                            self.sel_drag_start = None;
                        }
                    }
                    Tool::Lasso => {
                        if response.drag_started() {
                            self.lasso_points.clear();
                            self.sel_mode = mode_from_modifiers(ui.input(|i| i.modifiers));
                            self.sel_base = self.read_selection(frame);
                            if let Some(p) = response.interact_pointer_pos() {
                                self.lasso_points
                                    .push(screen_to_doc(p, doc_rect, self.doc.size));
                            }
                        }
                        if response.dragged() {
                            if let Some(p) = response.interact_pointer_pos() {
                                let d = screen_to_doc(p, doc_rect, self.doc.size);
                                if self
                                    .lasso_points
                                    .last()
                                    .is_none_or(|l| (*l - d).length() > 2.0)
                                {
                                    self.lasso_points.push(d);
                                }
                            }
                            ui.ctx().request_repaint();
                        }
                        if response.drag_stopped() {
                            if self.lasso_points.len() >= 3 {
                                let (w, h) = (self.doc.size.width, self.doc.size.height);
                                let pts: Vec<(f32, f32)> =
                                    self.lasso_points.iter().map(|v| (v.x, v.y)).collect();
                                let mask = raster::polygon_mask(&pts, w, h);
                                self.commit_selection(frame, mask);
                            }
                            self.lasso_points.clear();
                        }
                        // Draw the in-progress lasso path.
                        if self.lasso_points.len() >= 2 {
                            let pts: Vec<egui::Pos2> = self
                                .lasso_points
                                .iter()
                                .map(|v| doc_to_screen(*v, doc_rect, self.doc.size))
                                .collect();
                            ui.painter().add(egui::Shape::line(
                                pts,
                                egui::Stroke::new(1.5, egui::Color32::WHITE),
                            ));
                        }
                    }
                    Tool::MagicWand => {
                        if response.clicked() {
                            if let Some(p) = response.interact_pointer_pos() {
                                let d = screen_to_doc(p, doc_rect, self.doc.size);
                                if let Some(seed) = clamp_seed(d, self.doc.size) {
                                    self.sel_mode = mode_from_modifiers(ui.input(|i| i.modifiers));
                                    self.sel_base = self.read_selection(frame);
                                    self.do_magic_wand(frame, seed);
                                }
                            }
                        }
                    }
                }
                // Painting into a mask reveals (white); the eraser hides.
                if paint_mask && !erase {
                    for d in &mut dabs {
                        d.color = [1.0, 1.0, 1.0, 1.0];
                    }
                }

                // Re-rasterize edited text/vector layers before compositing.
                self.sync_generated_layers(frame);

                let layers = self.layer_order();

                // Recomposite only when content/structure changed (not on pan/zoom).
                let fp = self.layer_fingerprint();
                let dirty = !dabs.is_empty()
                    || wet_begin
                    || wet_end
                    || bake
                    || self.xform_active
                    || begin_command
                    || undo > 0
                    || redo > 0
                    || self.force_composite
                    || fp != self.last_fingerprint;
                let command_label = match self.tool {
                    Tool::Eraser => "Erase",
                    Tool::MoveLayer => "Move",
                    Tool::Transform => "Transform",
                    _ => "Brush",
                };
                let xform = if self.xform_active {
                    Some(compute_xform(
                        self.xform_translate,
                        self.xform_scale,
                        self.doc.size,
                    ))
                } else {
                    None
                };
                self.last_fingerprint = fp;
                self.force_composite = false;

                // Keep marching ants animating while a selection is active.
                if self.selection_active {
                    ui.ctx().request_repaint();
                }
                let time = ui.input(|i| i.time) as f32;

                ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                    rect,
                    CanvasPaint {
                        doc_rect,
                        checker_pts: 10.0,
                        canvas_size: self.doc.size,
                        layers,
                        active_id: self.active_id(),
                        dabs,
                        erase,
                        begin_command,
                        command_label: command_label.into(),
                        commit_command,
                        dirty_rect: self.dirty_rect(),
                        undo,
                        redo,
                        wet_begin,
                        wet_end,
                        wet_opacity: self.brush_opacity,
                        paint_into_wet: self.wet_active,
                        paint_mask,
                        dirty,
                        selection_op: self.sel_op_pending.take(),
                        time,
                        xform,
                        bake,
                    },
                ));
            });
    }
}
