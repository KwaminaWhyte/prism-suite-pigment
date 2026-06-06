//! The eframe application: panel layout, brush input, and the GPU canvas.

use std::sync::Arc;

use eframe::egui_wgpu;
use half::f16;
use pigment_core::color::srgb_to_linear;
use pigment_core::{BlendMode, Document, LayerId, Size};
use pigment_io::LoadedImage;

use crate::canvas::{CanvasGpu, CanvasPaint, Dab, LayerDraw, ViewTransform};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tool {
    Move,
    Brush,
    Eraser,
    Eyedropper,
}

/// Convert an 8-bit sRGB image to linear-light premultiplied f16 for the GPU.
fn to_linear_premul_f16(img: &LoadedImage) -> Vec<f16> {
    let mut out = Vec::with_capacity(img.rgba8.len());
    for px in img.rgba8.chunks_exact(4) {
        let a = px[3] as f32 / 255.0;
        out.push(f16::from_f32(srgb_to_linear(px[0] as f32 / 255.0) * a));
        out.push(f16::from_f32(srgb_to_linear(px[1] as f32 / 255.0) * a));
        out.push(f16::from_f32(srgb_to_linear(px[2] as f32 / 255.0) * a));
        out.push(f16::from_f32(a));
    }
    out
}

pub struct PigmentApp {
    doc: Document,
    view: ViewTransform,
    background_id: LayerId,
    image_f16: Arc<Vec<f16>>,
    image_gen: u64,

    tool: Tool,
    brush_color: egui::Color32,
    brush_size: f32,    // diameter, document px
    brush_hardness: f32,
    brush_opacity: f32,
    needs_fit: bool,

    // Active stroke state.
    stroke_last: Option<egui::Vec2>,
    stroke_residual: f32,
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
        Self {
            doc,
            view: ViewTransform::default(),
            background_id,
            image_f16: Arc::new(to_linear_premul_f16(&placeholder)),
            image_gen: 0,
            tool: Tool::Brush,
            brush_color: egui::Color32::from_rgb(20, 120, 230),
            brush_size: 40.0,
            brush_hardness: 0.5,
            brush_opacity: 1.0,
            needs_fit: true,
            stroke_last: None,
            stroke_residual: 0.0,
        }
    }

    fn open_file(&mut self) {
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
                self.image_f16 = Arc::new(to_linear_premul_f16(&img));
                self.image_gen += 1;
                self.needs_fit = true;
            }
            Err(e) => log::error!("open failed: {e}"),
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

    fn active_id(&self) -> LayerId {
        self.doc.active_layer.unwrap_or(self.background_id)
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
}

fn screen_to_doc(p: egui::Pos2, doc_rect: egui::Rect, size: Size) -> egui::Vec2 {
    let u = (p.x - doc_rect.min.x) / doc_rect.width().max(1e-3);
    let v = (p.y - doc_rect.min.y) / doc_rect.height().max(1e-3);
    egui::vec2(u * size.width as f32, v * size.height as f32)
}

impl eframe::App for PigmentApp {
    fn ui(&mut self, root: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("menu").show_inside(root, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Open…").clicked() {
                        self.open_file();
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
                (Tool::Eyedropper, "⛏"),
            ] {
                if ui.selectable_label(self.tool == tool, glyph).clicked() {
                    self.tool = tool;
                }
            }
            ui.separator();
            ui.label("Brush");
            ui.color_edit_button_srgba(&mut self.brush_color);
        });

        egui::SidePanel::right("panels").default_width(240.0).show_inside(root, |ui| {
            ui.heading("Brush");
            ui.add(egui::Slider::new(&mut self.brush_size, 1.0..=400.0).text("size"));
            ui.add(egui::Slider::new(&mut self.brush_hardness, 0.0..=0.99).text("hardness"));
            ui.add(egui::Slider::new(&mut self.brush_opacity, 0.0..=1.0).text("opacity"));

            ui.separator();
            ui.horizontal(|ui| {
                ui.heading("Layers");
                if ui.button("＋").on_hover_text("New layer").clicked() {
                    let id = self.doc.layers.add_raster(format!(
                        "Layer {}",
                        self.doc.layers.layers.len()
                    ));
                    self.doc.active_layer = Some(id);
                }
            });
            ui.separator();

            let active = self.active_id();
            // Top layer first in the UI.
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
                            if ui.selectable_label(is_active, &layer.name).clicked() {
                                self.doc.active_layer = Some(id);
                            }
                        });
                        ui.add(
                            egui::Slider::new(&mut layer.opacity, 0.0..=1.0)
                                .show_value(false)
                                .text("opacity"),
                        );
                        egui::ComboBox::from_id_salt(("blend", id.0))
                            .selected_text(format!("{:?}", layer.blend))
                            .show_ui(ui, |ui| {
                                for mode in BlendMode::ALL_SEPARABLE {
                                    ui.selectable_value(&mut layer.blend, mode, format!("{mode:?}"));
                                }
                            });
                    });
                ui.separator();
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

                // Cursor-anchored zoom (always available).
                let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                if scroll != 0.0 {
                    if let Some(cursor) = response.hover_pos() {
                        self.view.zoom_to((scroll * 0.0015).exp(), cursor - rect.center());
                    }
                }

                // Document quad in screen points.
                let img = egui::vec2(self.doc.size.width as f32, self.doc.size.height as f32);
                let doc_rect = egui::Rect::from_center_size(
                    rect.center() + self.view.pan,
                    img * self.view.zoom,
                );

                let mut dabs: Vec<Dab> = Vec::new();
                match self.tool {
                    Tool::Move => {
                        if response.dragged() {
                            self.view.pan += response.drag_delta();
                        }
                    }
                    Tool::Brush => {
                        let spacing = (self.brush_size * 0.15).max(0.75);
                        if let Some(p) = response.interact_pointer_pos() {
                            let cur = screen_to_doc(p, doc_rect, self.doc.size);
                            match self.stroke_last {
                                None => {
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
                        if response.drag_stopped() || (!response.dragged() && self.stroke_last.is_some() && response.interact_pointer_pos().is_none()) {
                            self.stroke_last = None;
                            self.stroke_residual = 0.0;
                        }
                    }
                    _ => {}
                }

                let layers: Vec<LayerDraw> = self
                    .doc
                    .layers
                    .layers
                    .iter()
                    .map(|l| LayerDraw {
                        id: l.id,
                        opacity: l.opacity,
                        blend: l.blend.shader_id(),
                        visible: l.visible,
                    })
                    .collect();

                ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                    rect,
                    CanvasPaint {
                        doc_rect,
                        checker_pts: 10.0,
                        canvas_size: self.doc.size,
                        image: Some((self.image_gen, self.background_id, self.image_f16.clone())),
                        layers,
                        active_id: self.active_id(),
                        dabs,
                    },
                ));
            });
    }
}
