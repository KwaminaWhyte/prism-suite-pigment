//! The eframe application: panel layout + the GPU canvas viewport.

use std::sync::Arc;

use eframe::egui_wgpu;
use pigment_core::{Document, Size};
use pigment_io::LoadedImage;

use crate::canvas::{CanvasPaint, CanvasRenderer, ViewTransform};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tool {
    Move,
    Brush,
    Eraser,
    Eyedropper,
}

pub struct PigmentApp {
    doc: Document,
    view: ViewTransform,
    image: Arc<LoadedImage>,
    image_gen: u64,
    tool: Tool,
    needs_fit: bool,
}

impl PigmentApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Register long-lived GPU resources in egui's callback resource map.
        let render_state = cc
            .wgpu_render_state
            .as_ref()
            .expect("pigment requires the wgpu backend");
        let renderer = CanvasRenderer::new(&render_state.device, render_state.target_format);
        render_state
            .renderer
            .write()
            .callback_resources
            .insert(renderer);

        let placeholder = pigment_io::placeholder(Size::new(1280, 800));
        Self {
            doc: Document::new(placeholder.size),
            view: ViewTransform::default(),
            image: Arc::new(placeholder),
            image_gen: 0,
            tool: Tool::Brush,
            needs_fit: true,
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
                self.image = Arc::new(img);
                self.image_gen += 1;
                self.needs_fit = true;
            }
            Err(e) => log::error!("open failed: {e}"),
        }
    }

    /// Center the document and pick a zoom that fits it inside `viewport`.
    fn fit(&mut self, viewport: egui::Rect) {
        let img = egui::vec2(self.image.size.width as f32, self.image.size.height as f32);
        if img.x <= 0.0 || img.y <= 0.0 {
            return;
        }
        let scale = (viewport.width() / img.x).min(viewport.height() / img.y) * 0.95;
        self.view = ViewTransform { pan: egui::Vec2::ZERO, zoom: scale.clamp(0.02, 64.0) };
    }
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

        egui::SidePanel::left("tools").exact_width(48.0).show_inside(root, |ui| {
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
        });

        egui::SidePanel::right("layers").default_width(220.0).show_inside(root, |ui| {
            ui.heading("Layers");
            ui.separator();
            for layer in self.doc.layers.layers.iter_mut().rev() {
                ui.horizontal(|ui| {
                    ui.checkbox(&mut layer.visible, "");
                    ui.label(&layer.name);
                });
                ui.add(
                    egui::Slider::new(&mut layer.opacity, 0.0..=1.0)
                        .show_value(false)
                        .text("opacity"),
                );
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

                // Pan with drag.
                if response.dragged() {
                    self.view.pan += response.drag_delta();
                }
                // Cursor-anchored zoom on scroll.
                let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                if scroll != 0.0 {
                    if let Some(cursor) = response.hover_pos() {
                        let anchor = cursor - rect.center();
                        self.view.zoom_to((scroll * 0.0015).exp(), anchor);
                    }
                }

                // Document quad in screen points.
                let img = egui::vec2(
                    self.image.size.width as f32,
                    self.image.size.height as f32,
                );
                let display = img * self.view.zoom;
                let center = rect.center() + self.view.pan;
                let doc_rect =
                    egui::Rect::from_center_size(center, display);

                ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                    rect,
                    CanvasPaint {
                        doc_rect,
                        checker_pts: 10.0,
                        image: Some((self.image_gen, self.image.clone())),
                    },
                ));
            });
    }
}
