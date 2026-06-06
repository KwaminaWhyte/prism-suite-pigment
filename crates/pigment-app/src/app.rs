//! The eframe application: panels, tools, and the GPU canvas. Owns document
//! state and drives GPU uploads/readbacks via `frame.wgpu_render_state()`.

use eframe::egui_wgpu;
use eframe::wgpu;
use half::f16;
use pigment_core::color::{linear_to_srgb, srgb_to_linear};
use pigment_core::fill::flood_fill_mask;
use pigment_core::{BlendMode, Document, Layer, LayerId, Size};
use pigment_io::document_file::{self, DocMeta, LayerMeta, LayerPixels};
use pigment_io::LoadedImage;

use crate::canvas::{CanvasGpu, CanvasPaint, Dab, LayerDraw, ViewTransform};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tool {
    Move,
    Brush,
    Eraser,
    Fill,
    Eyedropper,
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
    fill_tolerance: f32,
    fill_contiguous: bool,
    sample_all: bool,
    needs_fit: bool,

    stroke_last: Option<egui::Vec2>,
    stroke_residual: f32,
    undo_count: u32,
    redo_count: u32,
}

impl PigmentApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let render_state = cc
            .wgpu_render_state
            .as_ref()
            .expect("pigment requires the wgpu backend");
        let gpu = CanvasGpu::new(&render_state.device, render_state.target_format);
        render_state.renderer.write().callback_resources.insert(gpu);

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
            fill_tolerance: 0.1,
            fill_contiguous: true,
            sample_all: false,
            needs_fit: true,
            stroke_last: None,
            stroke_residual: 0.0,
            undo_count: 0,
            redo_count: 0,
        }
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

    fn dab_at(&self, doc_pos: egui::Vec2) -> Dab {
        let c = self.brush_color;
        Dab {
            center: [doc_pos.x, doc_pos.y],
            radius: (self.brush_size * 0.5).max(0.5),
            hardness: self.brush_hardness.clamp(0.0, 0.99),
            color: [
                srgb_to_linear(c.r() as f32 / 255.0),
                srgb_to_linear(c.g() as f32 / 255.0),
                srgb_to_linear(c.b() as f32 / 255.0),
                self.brush_opacity,
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
            // Write target is always the active layer.
            let Some(active_bytes) = gpu.read_layer(device, queue, active) else { return };
            let mut abuf = f16_bytes_to_f32(&active_bytes);
            for (i, &m) in mask.iter().enumerate() {
                if m {
                    let o = i * 4;
                    abuf[o..o + 4].copy_from_slice(&fill);
                }
            }
            gpu.upload_layer(queue, active, &f32_to_f16_bytes(&abuf));
        });
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
    }
}

fn screen_to_doc(p: egui::Pos2, doc_rect: egui::Rect, size: Size) -> egui::Vec2 {
    let u = (p.x - doc_rect.min.x) / doc_rect.width().max(1e-3);
    let v = (p.y - doc_rect.min.y) / doc_rect.height().max(1e-3);
    egui::vec2(u * size.width as f32, v * size.height as f32)
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

        egui::SidePanel::left("tools").exact_width(52.0).show_inside(root, |ui| {
            ui.add_space(4.0);
            for (tool, glyph) in [
                (Tool::Move, "✋"),
                (Tool::Brush, "🖌"),
                (Tool::Eraser, "▱"),
                (Tool::Fill, "🪣"),
                (Tool::Eyedropper, "⛏"),
            ] {
                if ui.selectable_label(self.tool == tool, glyph).clicked() {
                    self.tool = tool;
                }
            }
            ui.separator();
            ui.color_edit_button_srgba(&mut self.brush_color);
        });

        egui::SidePanel::right("panels").default_width(250.0).show_inside(root, |ui| {
            ui.heading("Brush");
            ui.add(egui::Slider::new(&mut self.brush_size, 1.0..=400.0).text("size"));
            ui.add(egui::Slider::new(&mut self.brush_hardness, 0.0..=0.99).text("hardness"));
            ui.add(egui::Slider::new(&mut self.brush_opacity, 0.0..=1.0).text("opacity"));
            if self.tool == Tool::Fill {
                ui.add(egui::Slider::new(&mut self.fill_tolerance, 0.0..=1.0).text("tolerance"));
                ui.checkbox(&mut self.fill_contiguous, "contiguous");
            }
            if matches!(self.tool, Tool::Fill | Tool::Eyedropper) {
                ui.checkbox(&mut self.sample_all, "sample all layers");
            }

            ui.separator();
            let (undos, redos) = with_gpu(frame, |gpu, _, _| gpu.history_labels()).unwrap_or_default();
            egui::CollapsingHeader::new(format!("History  ({} / {})", undos.len(), redos.len()))
                .show(ui, |ui| {
                    // Future states (redoable), furthest first.
                    for (i, l) in redos.iter().enumerate().rev() {
                        if ui.small_button(format!("↷ {l}")).clicked() {
                            self.redo_count += (i + 1) as u32;
                        }
                    }
                    ui.label("● now");
                    // Past states (undoable), newest first.
                    for (i, l) in undos.iter().rev().enumerate() {
                        if ui.small_button(format!("↶ {l}")).clicked() {
                            self.undo_count += (i + 1) as u32;
                        }
                    }
                });

            ui.separator();
            ui.horizontal(|ui| {
                ui.heading("Layers");
                if ui.button("＋").on_hover_text("New layer").clicked() {
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
                            ui.checkbox(&mut layer.visible, "");
                            ui.add(egui::TextEdit::singleline(&mut layer.name).desired_width(110.0));
                            if ui.small_button("▲").clicked() {
                                action = LayerAction::MoveUp(id);
                            }
                            if ui.small_button("▼").clicked() {
                                action = LayerAction::MoveDown(id);
                            }
                            if ui.small_button("🗑").clicked() {
                                action = LayerAction::Delete(id);
                            }
                        });
                        ui.horizontal(|ui| {
                            if ui.selectable_label(is_active, "●").clicked() {
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
                let erase = self.tool == Tool::Eraser;
                match self.tool {
                    Tool::Move => {
                        if response.dragged() {
                            self.view.pan += response.drag_delta();
                        }
                    }
                    Tool::Brush | Tool::Eraser => {
                        let spacing = (self.brush_size * 0.15).max(0.75);
                        if let Some(p) = response.interact_pointer_pos() {
                            let cur = screen_to_doc(p, doc_rect, self.doc.size);
                            match self.stroke_last {
                                None => {
                                    begin_command = true;
                                    dabs.push(self.dab_at(cur));
                                    self.stroke_last = Some(cur);
                                    self.stroke_residual = 0.0;
                                }
                                Some(last) => {
                                    let seg = cur - last;
                                    let dist = seg.length();
                                    if dist > 1e-3 {
                                        let dir = seg / dist;
                                        let mut t = self.stroke_residual;
                                        while t <= dist {
                                            dabs.push(self.dab_at(last + dir * t));
                                            t += spacing;
                                        }
                                        self.stroke_residual = t - dist;
                                        self.stroke_last = Some(cur);
                                    }
                                }
                            }
                            ui.ctx().request_repaint();
                        }
                        if response.drag_stopped()
                            || (self.stroke_last.is_some() && response.interact_pointer_pos().is_none())
                        {
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
                }

                let layers = self.layer_order();

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
                        command_label: if erase { "Erase".into() } else { "Brush".into() },
                        undo,
                        redo,
                    },
                ));
            });
    }
}
