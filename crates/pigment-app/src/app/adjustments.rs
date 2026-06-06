use super::*;

pub(crate) fn adjustment_ui(ui: &mut egui::Ui, adj: &mut Adjustment) {
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
        Adjustment::Curves(cp) => {
            curve_editor(ui, cp);
        }
        Adjustment::Vibrance { amount } => {
            ui.add(egui::Slider::new(amount, -1.0..=1.0).text("vibrance"));
        }
        Adjustment::PhotoFilter { color, density } => {
            ui.horizontal(|ui| {
                ui.label("color");
                ui.color_edit_button_rgb(color);
            });
            ui.add(egui::Slider::new(density, 0.0..=1.0).text("density"));
        }
        Adjustment::Posterize { levels } => {
            let mut l = *levels as f32;
            if ui
                .add(egui::Slider::new(&mut l, 2.0..=32.0).text("levels"))
                .changed()
            {
                *levels = l.round() as u32;
            }
        }
        Adjustment::GradientMap { low, high } => {
            ui.horizontal(|ui| {
                ui.label("shadows");
                ui.color_edit_button_rgb(low);
            });
            ui.horizontal(|ui| {
                ui.label("highlights");
                ui.color_edit_button_rgb(high);
            });
        }
        Adjustment::Invert | Adjustment::BlackWhite => {
            ui.label("(no parameters)");
        }
    }
}

/// Draggable tone-curve editor for a Curves adjustment. A channel selector picks
/// the composite (RGB) or per-channel curve; drag a knot to move it, click empty
/// canvas to add one, double-click a knot to delete it. Endpoints stay pinned in
/// x (only their output moves). Curve points live in `[0,1]` with y = output up.
fn curve_editor(ui: &mut egui::Ui, cp: &mut CurvePoints) {
    let chan_id = ui.id().with("curve_chan");
    let mut chan: usize = ui.data(|d| d.get_temp(chan_id)).unwrap_or(0);
    ui.horizontal(|ui| {
        for (i, name) in ["RGB", "R", "G", "B"].iter().enumerate() {
            if ui.selectable_label(chan == i, *name).clicked() {
                chan = i;
            }
        }
    });
    ui.data_mut(|d| d.insert_temp(chan_id, chan));
    let pts = match chan {
        1 => &mut cp.r,
        2 => &mut cp.g,
        3 => &mut cp.b,
        _ => &mut cp.rgb,
    };

    let side = ui.available_width().clamp(120.0, 240.0);
    let (resp, painter) =
        ui.allocate_painter(egui::vec2(side, side), egui::Sense::click_and_drag());
    let rect = resp.rect;
    let to_screen = |p: (f32, f32)| {
        egui::pos2(
            rect.left() + p.0 * rect.width(),
            rect.bottom() - p.1 * rect.height(),
        )
    };
    let to_norm = |pos: egui::Pos2| {
        (
            ((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0),
            ((rect.bottom() - pos.y) / rect.height()).clamp(0.0, 1.0),
        )
    };

    // Background + grid + identity diagonal.
    painter.rect_filled(rect, 4.0, egui::Color32::from_gray(24));
    let grid = egui::Stroke::new(1.0, egui::Color32::from_gray(48));
    for i in 1..4 {
        let t = i as f32 / 4.0;
        painter.line_segment([to_screen((t, 0.0)), to_screen((t, 1.0))], grid);
        painter.line_segment([to_screen((0.0, t)), to_screen((1.0, t))], grid);
    }
    painter.line_segment(
        [to_screen((0.0, 0.0)), to_screen((1.0, 1.0))],
        egui::Stroke::new(1.0, egui::Color32::from_gray(64)),
    );

    // Drag/click interaction.
    let drag_id = ui.id().with("curve_drag");
    let mut dragging: Option<usize> = ui.data(|d| d.get_temp::<Option<usize>>(drag_id)).flatten();
    if resp.drag_started() {
        if let Some(pos) = resp.interact_pointer_pos() {
            // Grab the nearest knot within ~12px, else add a new one at the pointer.
            let mut best = None;
            let mut best_d = 12.0f32;
            for (i, &p) in pts.iter().enumerate() {
                let d = to_screen(p).distance(pos);
                if d < best_d {
                    best_d = d;
                    best = Some(i);
                }
            }
            dragging = Some(best.unwrap_or_else(|| {
                let (x, y) = to_norm(pos);
                pts.push((x, y));
                pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                pts.iter().position(|p| p.0 == x && p.1 == y).unwrap_or(0)
            }));
        }
    }
    if resp.dragged() {
        if let (Some(idx), Some(pos)) = (dragging, resp.interact_pointer_pos()) {
            let (mut x, y) = to_norm(pos);
            // Pin endpoint x; interior knots can't cross neighbours.
            if idx == 0 {
                x = 0.0;
            } else if idx == pts.len() - 1 {
                x = 1.0;
            } else {
                let lo = pts[idx - 1].0 + 1e-3;
                let hi = pts[idx + 1].0 - 1e-3;
                x = x.clamp(lo, hi);
            }
            pts[idx] = (x, y);
        }
    }
    if resp.drag_stopped() {
        dragging = None;
    }
    ui.data_mut(|d| d.insert_temp(drag_id, dragging));

    // Double-click a knot to delete it (keep at least the two endpoints).
    if resp.double_clicked() {
        if let Some(pos) = resp.interact_pointer_pos() {
            if pts.len() > 2 {
                if let Some((i, _)) = pts
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != 0 && *i != pts.len() - 1)
                    .map(|(i, p)| (i, to_screen(*p).distance(pos)))
                    .filter(|(_, d)| *d < 12.0)
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                {
                    pts.remove(i);
                }
            }
        }
    }

    // Spline + knots.
    let lut = prism_core::curve::build_lut(pts, 128);
    let line: Vec<egui::Pos2> = (0..128)
        .map(|i| to_screen((i as f32 / 127.0, lut[i])))
        .collect();
    painter.add(egui::Shape::line(
        line,
        egui::Stroke::new(1.5, egui::Color32::from_rgb(120, 200, 255)),
    ));
    for &p in pts.iter() {
        painter.circle(
            to_screen(p),
            3.5,
            egui::Color32::WHITE,
            egui::Stroke::new(1.0, egui::Color32::from_gray(40)),
        );
    }
}
