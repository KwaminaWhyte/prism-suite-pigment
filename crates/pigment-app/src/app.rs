//! The eframe application: panels, tools, and the GPU canvas. Owns document
//! state and drives GPU uploads/readbacks via `frame.wgpu_render_state()`.

use eframe::egui_wgpu;
use eframe::wgpu;
use half::f16;
use pigment_core::color::{linear_to_srgb, srgb_to_linear};
use pigment_core::fill::flood_fill_mask;
use pigment_core::raster::{self, CombineMode};
use pigment_core::{BlendMode, Document, Layer, LayerId, Size};
use pigment_io::document_file::{self, DocMeta, LayerMeta, LayerPixels};
use pigment_io::LoadedImage;

use crate::canvas::{CanvasGpu, CanvasPaint, Dab, LayerDraw, SelectionOp, ViewTransform};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tool {
    Move,        // pan the view (hand)
    MoveLayer,   // translate the active layer
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

fn image_to_f16_bytes(img: &LoadedImage) -> Vec<u8> {
    let mut o = Vec::with_capacity(img.rgba8.len() * 2);
    for px in img.rgba8.chunks_exact(4) {
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
    sel_base: Vec<f32>,        // selection snapshot at op start (for combine)
    sel_mode: CombineMode,     // current modifier-derived combine mode
    lasso_points: Vec<egui::Vec2>, // in-progress lasso (document px)
    feather_radius: f32,

    // Move/transform of the active layer.
    xform_active: bool,
    xform_translate: egui::Vec2, // document px
    xform_scale: f32,
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
        }
    }

    /// Read the current GPU selection mask (zeros if none active).
    fn read_selection(&mut self, frame: &mut eframe::Frame) -> Vec<f32> {
        let n = (self.doc.size.width * self.doc.size.height) as usize;
        if !self.selection_active {
            return vec![0.0; n];
        }
        with_gpu(frame, |gpu, device, queue| gpu.read_selection(device, queue))
            .flatten()
            .unwrap_or_else(|| vec![0.0; n])
    }

    /// Upload a CPU selection mask and mark a selection active.
    fn set_selection(&mut self, frame: &mut eframe::Frame, mask: &[f32]) {
        with_gpu(frame, |gpu, _, queue| gpu.upload_selection(queue, mask));
        self.selection_active = true;
    }

    /// Read → transform → upload the selection (feather / grow / shrink).
    fn map_selection(&mut self, frame: &mut eframe::Frame, f: impl Fn(&[f32], u32, u32) -> Vec<f32>) {
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
            .map(|l| LayerDraw {
                id: l.id,
                opacity: l.opacity,
                blend: l.blend.shader_id(),
                visible: l.visible,
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
        let Some(path) = rfd::FileDialog::new().add_filter("Pigment", &["pigment"]).pick_file()
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
        self.background_id = doc.layers.layers.first().map(|l| l.id).unwrap_or(LayerId(0));
        doc.active_layer = doc.layers.layers.last().map(|l| l.id);
        self.doc = doc;
        self.pending = Some(PendingUpload { size, layers: staged });
        self.needs_fit = true;
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
                    gpu.read_layer(device, queue, *id)
                        .map(|b| LayerPixels { id: id.0, rgba16f: b })
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
        self.view = ViewTransform { pan: egui::Vec2::ZERO, zoom: scale.clamp(0.02, 64.0) };
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
            let sel = if has_sel { gpu.read_selection(device, queue) } else { None };
            // Write target is always the active layer.
            let Some(active_bytes) = gpu.read_layer(device, queue, active) else { return };
            let mut abuf = f16_bytes_to_f32(&active_bytes);
            for (i, &m) in mask.iter().enumerate() {
                let selected = sel.as_ref().map_or(true, |s| s[i] > 0.5);
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
                let to8 = |lin: f32| (linear_to_srgb(lin / a).clamp(0.0, 1.0) * 255.0).round() as u8;
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
                    if ui.button("Save .pigment…").clicked() {
                        self.save_pigment(frame);
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
                    ui.add(egui::Slider::new(&mut self.feather_radius, 0.0..=30.0).text("feather px"));
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

        egui::SidePanel::left("tools").exact_width(48.0).show_inside(root, |ui| {
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
            ] {
                let btn = egui::SelectableLabel::new(
                    self.tool == tool,
                    egui::RichText::new(icon).size(20.0),
                );
                if ui.add_sized([36.0, 30.0], btn).on_hover_text(name).clicked() {
                    self.tool = tool;
                }
            }
            ui.add_space(6.0);
            ui.separator();
            ui.vertical_centered(|ui| ui.color_edit_button_srgba(&mut self.brush_color));
        });

        egui::SidePanel::right("panels").default_width(250.0).show_inside(root, |ui| {
            ui.heading("Brush");
            ui.add(egui::Slider::new(&mut self.brush_size, 1.0..=400.0).text("size"));
            ui.add(egui::Slider::new(&mut self.brush_hardness, 0.0..=0.99).text("hardness"));
            ui.add(egui::Slider::new(&mut self.brush_opacity, 0.0..=1.0).text("opacity"));
            ui.checkbox(&mut self.speed_dynamics, "speed → size")
                .on_hover_text("Faster strokes paint thinner (stylus pressure isn't exposed by eframe)");
            if self.speed_dynamics {
                ui.add(egui::Slider::new(&mut self.min_size_scale, 0.05..=1.0).text("min size"));
            }
            if self.tool == Tool::Fill {
                ui.add(egui::Slider::new(&mut self.fill_tolerance, 0.0..=1.0).text("tolerance"));
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
            if matches!(self.tool, Tool::SelectRect | Tool::SelectEllipse | Tool::Lasso | Tool::MagicWand) {
                ui.label("Shift: add · Alt: subtract.");
            }

            ui.separator();
            let (undos, redos) = with_gpu(frame, |gpu, _, _| gpu.history_labels()).unwrap_or_default();
            egui::CollapsingHeader::new(format!("History  ({} / {})", undos.len(), redos.len()))
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
                    let id = self.doc.layers.add_raster(format!("Layer {}", self.doc.layers.layers.len()));
                    self.doc.active_layer = Some(id);
                }
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
                            let eye = if layer.visible { crate::icons::EYE } else { " " };
                            if ui
                                .selectable_label(layer.visible, egui::RichText::new(eye).size(15.0))
                                .on_hover_text("Toggle visibility")
                                .clicked()
                            {
                                layer.visible = !layer.visible;
                            }
                            ui.add(egui::TextEdit::singleline(&mut layer.name).desired_width(90.0));
                            if ui.small_button(crate::icons::ARROW_UP).on_hover_text("Move up").clicked() {
                                action = LayerAction::MoveUp(id);
                            }
                            if ui.small_button(crate::icons::ARROW_DOWN).on_hover_text("Move down").clicked() {
                                action = LayerAction::MoveDown(id);
                            }
                            if ui.small_button(crate::icons::TRASH).on_hover_text("Delete layer").clicked() {
                                action = LayerAction::Delete(id);
                            }
                        });
                        ui.horizontal(|ui| {
                            if ui.selectable_label(is_active, "active").clicked() {
                                self.doc.active_layer = Some(id);
                            }
                            egui::ComboBox::from_id_salt(("blend", id.0))
                                .selected_text(format!("{:?}", layer.blend))
                                .width(120.0)
                                .show_ui(ui, |ui| {
                                    for mode in BlendMode::ALL_SEPARABLE {
                                        ui.selectable_value(&mut layer.blend, mode, format!("{mode:?}"));
                                    }
                                });
                        });
                        ui.add(
                            egui::Slider::new(&mut layer.opacity, 0.0..=1.0)
                                .show_value(false)
                                .text("opacity"),
                        );
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
                        self.background_id = ls.first().map(|l| l.id).unwrap_or(self.background_id);
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
                        self.view.zoom_to((scroll * 0.0015).exp(), cursor - rect.center());
                    }
                }

                let (mut undo, mut redo) = (self.undo_count, self.redo_count);
                self.undo_count = 0;
                self.redo_count = 0;
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
                    }
                });

                let img = egui::vec2(self.doc.size.width as f32, self.doc.size.height as f32);
                let doc_rect =
                    egui::Rect::from_center_size(rect.center() + self.view.pan, img * self.view.zoom);

                let mut dabs: Vec<Dab> = Vec::new();
                let mut begin_command = false;
                let mut commit_command = false;
                let mut wet_begin = false;
                let mut wet_end = false;
                let mut bake = false;
                let erase = self.tool == Tool::Eraser;
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
                    // Implemented in a later Phase 2 wave.
                    Tool::Crop => {}
                    Tool::Brush | Tool::Eraser => {
                        let spacing = (self.brush_size * 0.15).max(0.75);
                        let dt = ui.input(|i| i.stable_dt).max(1e-3);
                        if let Some(p) = response.interact_pointer_pos() {
                            let cur = screen_to_doc(p, doc_rect, self.doc.size);
                            match self.stroke_last {
                                None => {
                                    begin_command = true;
                                    self.stroke_dirty = None;
                                    self.expand_dirty(cur, dirty_radius);
                                    if !erase {
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
                                            dabs.push(self.dab_at(last + dir * t, dab_alpha, scale));
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
                            || (self.stroke_last.is_some() && response.interact_pointer_pos().is_none())
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
                                self.sel_drag_start = Some(screen_to_doc(p, doc_rect, self.doc.size));
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
                                self.lasso_points.push(screen_to_doc(p, doc_rect, self.doc.size));
                            }
                        }
                        if response.dragged() {
                            if let Some(p) = response.interact_pointer_pos() {
                                let d = screen_to_doc(p, doc_rect, self.doc.size);
                                if self.lasso_points.last().is_none_or(|l| (*l - d).length() > 2.0) {
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
                    Some(compute_xform(self.xform_translate, self.xform_scale, self.doc.size))
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
