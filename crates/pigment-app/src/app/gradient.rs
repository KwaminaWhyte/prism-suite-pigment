//! Gradient editor: editable multi-stop color/opacity rails, geometry type,
//! dithering, and saved presets. The editor stores stop colors in **sRGB**
//! (what the user sees / picks); it converts to a linear-working-space
//! [`prism_core::Gradient`] for the fill, matching the rest of the pixel
//! pipeline. The actual sampling/rasterization/dither math lives in the shared,
//! app-agnostic `prism_core::gradient` module.

use super::*;
use prism_core::gradient::{ColorStop, Gradient, GradientType, OpacityStop};

/// An editable color stop (sRGB) for the UI.
#[derive(Clone, Copy)]
pub(crate) struct EditColorStop {
    pub pos: f32,
    pub color: egui::Color32, // sRGB, ignores alpha (opacity is its own rail)
}

/// An editable opacity stop for the UI.
#[derive(Clone, Copy)]
pub(crate) struct EditOpacityStop {
    pub pos: f32,
    pub alpha: f32,
}

/// A named gradient preset (used to populate the preset list).
pub(crate) struct GradientPreset {
    pub name: &'static str,
    pub colors: &'static [(f32, [u8; 3])], // (pos, sRGB rgb)
    pub opacities: &'static [(f32, f32)],  // (pos, alpha)
}

/// The full gradient-editor state owned by the app.
pub(crate) struct GradientEditor {
    pub color_stops: Vec<EditColorStop>,
    pub opacity_stops: Vec<EditOpacityStop>,
    pub kind: GradientType,
    pub dither: bool,
    /// When true, a single click/drag fills the whole layer (or selection) with
    /// the gradient laid out across the canvas; when false, the drag start→end
    /// defines the gradient axis (the gradient tool).
    pub fill_layer: bool,
    /// Selected color stop index (for editing in the panel).
    pub sel_color: usize,
    /// Selected opacity stop index.
    pub sel_opacity: usize,
}

impl Default for GradientEditor {
    fn default() -> Self {
        // Foreground (blue, the default brush color) → transparent, the classic
        // gradient-tool default. Color rail is two solid blue stops; opacity 1→0.
        Self {
            color_stops: vec![
                EditColorStop {
                    pos: 0.0,
                    color: egui::Color32::from_rgb(20, 120, 230),
                },
                EditColorStop {
                    pos: 1.0,
                    color: egui::Color32::from_rgb(20, 120, 230),
                },
            ],
            opacity_stops: vec![
                EditOpacityStop {
                    pos: 0.0,
                    alpha: 1.0,
                },
                EditOpacityStop {
                    pos: 1.0,
                    alpha: 0.0,
                },
            ],
            kind: GradientType::Linear,
            dither: true,
            fill_layer: false,
            sel_color: 0,
            sel_opacity: 0,
        }
    }
}

/// Built-in presets, Photoshop-style.
pub(crate) const PRESETS: &[GradientPreset] = &[
    GradientPreset {
        name: "Foreground→Transparent",
        colors: &[(0.0, [20, 120, 230]), (1.0, [20, 120, 230])],
        opacities: &[(0.0, 1.0), (1.0, 0.0)],
    },
    GradientPreset {
        name: "Black→White",
        colors: &[(0.0, [0, 0, 0]), (1.0, [255, 255, 255])],
        opacities: &[(0.0, 1.0), (1.0, 1.0)],
    },
    GradientPreset {
        name: "Spectrum",
        colors: &[
            (0.0, [255, 0, 0]),
            (0.25, [255, 255, 0]),
            (0.5, [0, 255, 0]),
            (0.75, [0, 128, 255]),
            (1.0, [180, 0, 255]),
        ],
        opacities: &[(0.0, 1.0), (1.0, 1.0)],
    },
    GradientPreset {
        name: "Sunset",
        colors: &[
            (0.0, [255, 94, 58]),
            (0.5, [255, 175, 64]),
            (1.0, [88, 28, 135]),
        ],
        opacities: &[(0.0, 1.0), (1.0, 1.0)],
    },
];

impl GradientEditor {
    /// Apply a preset by name (no-op if unknown).
    pub fn load_preset(&mut self, name: &str) {
        if let Some(p) = PRESETS.iter().find(|p| p.name == name) {
            self.color_stops = p
                .colors
                .iter()
                .map(|&(pos, [r, g, b])| EditColorStop {
                    pos,
                    color: egui::Color32::from_rgb(r, g, b),
                })
                .collect();
            self.opacity_stops = p
                .opacities
                .iter()
                .map(|&(pos, alpha)| EditOpacityStop { pos, alpha })
                .collect();
            self.sel_color = 0;
            self.sel_opacity = 0;
        }
    }

    /// Convert the editor (sRGB) to a shared linear-working-space [`Gradient`].
    pub fn to_core(&self) -> Gradient {
        let color_stops = self
            .color_stops
            .iter()
            .map(|s| {
                ColorStop::new(
                    s.pos,
                    [
                        srgb_to_linear(s.color.r() as f32 / 255.0),
                        srgb_to_linear(s.color.g() as f32 / 255.0),
                        srgb_to_linear(s.color.b() as f32 / 255.0),
                    ],
                )
            })
            .collect();
        let opacity_stops = self
            .opacity_stops
            .iter()
            .map(|s| OpacityStop::new(s.pos, s.alpha))
            .collect();
        Gradient {
            color_stops,
            opacity_stops,
            kind: self.kind,
            dither: self.dither,
        }
    }
}

impl PigmentApp {
    /// Render the gradient-editor controls into the tool-options bar.
    pub(crate) fn gradient_options_ui(&mut self, ui: &mut egui::Ui) {
        let g = &mut self.gradient;
        ui.separator();
        // Geometry type.
        egui::ComboBox::from_label("type")
            .selected_text(match g.kind {
                GradientType::Linear => "Linear",
                GradientType::Radial => "Radial",
                GradientType::Angle => "Angle",
                GradientType::Reflected => "Reflected",
                GradientType::Diamond => "Diamond",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut g.kind, GradientType::Linear, "Linear");
                ui.selectable_value(&mut g.kind, GradientType::Radial, "Radial");
                ui.selectable_value(&mut g.kind, GradientType::Angle, "Angle");
                ui.selectable_value(&mut g.kind, GradientType::Reflected, "Reflected");
                ui.selectable_value(&mut g.kind, GradientType::Diamond, "Diamond");
            });
        ui.checkbox(&mut g.dither, "dither");
        ui.checkbox(&mut g.fill_layer, "fill layer");

        // Presets.
        ui.separator();
        egui::ComboBox::from_label("preset")
            .selected_text("Presets")
            .show_ui(ui, |ui| {
                for p in PRESETS {
                    if ui.selectable_label(false, p.name).clicked() {
                        g.load_preset(p.name);
                    }
                }
            });

        // Color stops.
        ui.separator();
        ui.label("color stops");
        let mut remove_color: Option<usize> = None;
        for i in 0..g.color_stops.len() {
            ui.horizontal(|ui| {
                let s = &mut g.color_stops[i];
                let mut rgb = [
                    s.color.r() as f32 / 255.0,
                    s.color.g() as f32 / 255.0,
                    s.color.b() as f32 / 255.0,
                ];
                if ui.color_edit_button_rgb(&mut rgb).changed() {
                    s.color = egui::Color32::from_rgb(
                        (rgb[0] * 255.0) as u8,
                        (rgb[1] * 255.0) as u8,
                        (rgb[2] * 255.0) as u8,
                    );
                }
                ui.add(egui::Slider::new(&mut s.pos, 0.0..=1.0).text("pos"));
                if g.color_stops.len() > 1 && ui.small_button("✕").clicked() {
                    remove_color = Some(i);
                }
            });
        }
        if ui.small_button("+ color stop").clicked() {
            let last = g.color_stops.last().copied().unwrap_or(EditColorStop {
                pos: 1.0,
                color: egui::Color32::WHITE,
            });
            g.color_stops.push(EditColorStop {
                pos: (last.pos + 0.5).fract(),
                color: last.color,
            });
        }
        if let Some(i) = remove_color {
            g.color_stops.remove(i);
        }

        // Opacity stops.
        ui.separator();
        ui.label("opacity stops");
        let mut remove_op: Option<usize> = None;
        for i in 0..g.opacity_stops.len() {
            ui.horizontal(|ui| {
                let s = &mut g.opacity_stops[i];
                ui.add(egui::Slider::new(&mut s.alpha, 0.0..=1.0).text("α"));
                ui.add(egui::Slider::new(&mut s.pos, 0.0..=1.0).text("pos"));
                if g.opacity_stops.len() > 1 && ui.small_button("✕").clicked() {
                    remove_op = Some(i);
                }
            });
        }
        if ui.small_button("+ opacity stop").clicked() {
            let last = g.opacity_stops.last().copied().unwrap_or(EditOpacityStop {
                pos: 1.0,
                alpha: 1.0,
            });
            g.opacity_stops.push(EditOpacityStop {
                pos: (last.pos + 0.5).fract(),
                alpha: last.alpha,
            });
        }
        if let Some(i) = remove_op {
            g.opacity_stops.remove(i);
        }

        if g.fill_layer && ui.button("Fill layer with gradient").clicked() {
            let (w, h) = (self.doc.size.width as f32, self.doc.size.height as f32);
            // A top-left → bottom-right axis spanning the canvas; geometry types
            // reinterpret the axis (radial uses |length| as the radius, etc.).
            self.grad_fill_pending = Some((egui::vec2(0.0, h * 0.5), egui::vec2(w, h * 0.5)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn editor_to_core_converts_srgb_to_linear() {
        let ed = GradientEditor {
            color_stops: vec![
                EditColorStop {
                    pos: 0.0,
                    color: egui::Color32::from_rgb(0, 0, 0),
                },
                EditColorStop {
                    pos: 1.0,
                    color: egui::Color32::from_rgb(255, 255, 255),
                },
            ],
            opacity_stops: vec![
                EditOpacityStop {
                    pos: 0.0,
                    alpha: 1.0,
                },
                EditOpacityStop {
                    pos: 1.0,
                    alpha: 0.25,
                },
            ],
            kind: GradientType::Radial,
            dither: false,
            fill_layer: true,
            sel_color: 0,
            sel_opacity: 0,
        };
        let g = ed.to_core();
        // Black stays 0, white stays 1 in linear; geometry + dither carried over.
        assert_eq!(g.color_stops.len(), 2);
        assert!((g.color_stops[0].color[0]).abs() < 1e-4);
        assert!((g.color_stops[1].color[0] - 1.0).abs() < 1e-4);
        assert!((g.opacity_stops[1].alpha - 0.25).abs() < 1e-4);
        assert_eq!(g.kind, GradientType::Radial);
        assert!(!g.dither);
    }

    #[test]
    fn mid_gray_srgb_maps_to_linear() {
        // sRGB 0.5 → ~0.214 linear (the gamma boundary owned by prism-color).
        let ed = GradientEditor {
            color_stops: vec![EditColorStop {
                pos: 0.0,
                color: egui::Color32::from_rgb(188, 188, 188), // ~0.737 sRGB
            }],
            ..GradientEditor::default()
        };
        let g = ed.to_core();
        let expected = srgb_to_linear(188.0 / 255.0);
        assert!((g.color_stops[0].color[0] - expected).abs() < 1e-4);
    }

    #[test]
    fn load_preset_replaces_stops() {
        let mut ed = GradientEditor::default();
        ed.load_preset("Spectrum");
        assert_eq!(ed.color_stops.len(), 5);
        ed.load_preset("Black→White");
        assert_eq!(ed.color_stops.len(), 2);
        assert_eq!(ed.color_stops[0].color, egui::Color32::from_rgb(0, 0, 0));
        // Unknown preset is a no-op.
        ed.load_preset("nope");
        assert_eq!(ed.color_stops.len(), 2);
    }
}
