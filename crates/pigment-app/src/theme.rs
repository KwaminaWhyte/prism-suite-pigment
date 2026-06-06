//! Modern dark theme for Pigment.
//!
//! Installs a clean, flat-but-polished dark look: rounded widget corners, a teal
//! accent for selected/active state, layered panel backgrounds (side panels a
//! touch lighter than the central canvas frame), comfortable spacing, and
//! slightly larger heading text. Call once at startup with `cc.egui_ctx`.

use egui::{
    style::{Selection, WidgetVisuals, Widgets},
    Color32, Context, CornerRadius, FontFamily, FontId, Margin, Stroke, Style, Vec2, Visuals,
};

// --- Palette ----------------------------------------------------------------

const BG_BASE: Color32 = Color32::from_rgb(0x1a, 0x1c, 0x1f); // central / canvas frame
const BG_PANEL: Color32 = Color32::from_rgb(0x23, 0x26, 0x2b); // side panels (lighter)
const BG_WIDGET: Color32 = Color32::from_rgb(0x2c, 0x30, 0x36); // inactive widget fill
const BG_HOVER: Color32 = Color32::from_rgb(0x36, 0x3b, 0x42); // hovered widget fill
const BG_ACTIVE: Color32 = Color32::from_rgb(0x3f, 0x45, 0x4d); // pressed widget fill
const BG_EXTREME: Color32 = Color32::from_rgb(0x12, 0x13, 0x15); // text edits, scrollbars
const FAINT: Color32 = Color32::from_rgb(0x21, 0x24, 0x28); // striped rows

const STROKE_SUBTLE: Color32 = Color32::from_rgb(0x3a, 0x3f, 0x46);
const STROKE_STRONG: Color32 = Color32::from_rgb(0x50, 0x57, 0x60);

const TEXT: Color32 = Color32::from_rgb(0xe4, 0xe6, 0xe9);
const TEXT_MUTED: Color32 = Color32::from_rgb(0x9a, 0xa1, 0xab);

const ACCENT: Color32 = Color32::from_rgb(0x2d, 0xb6, 0xa8); // teal
const ACCENT_TEXT: Color32 = Color32::from_rgb(0xf2, 0xff, 0xfd);

const WARN: Color32 = Color32::from_rgb(0xe6, 0xa1, 0x3c);
const ERROR: Color32 = Color32::from_rgb(0xe5, 0x5b, 0x5b);

const RADIUS: u8 = 6;

/// Apply Pigment's modern dark theme to the given egui context.
pub fn apply(ctx: &Context) {
    let mut style = Style::default();

    style.visuals = visuals();
    tune_spacing(&mut style);
    tune_text(&mut style);

    ctx.set_style(style);
}

fn widget(bg_fill: Color32, weak_bg_fill: Color32, stroke: Color32, fg: Color32) -> WidgetVisuals {
    WidgetVisuals {
        bg_fill,
        weak_bg_fill,
        bg_stroke: Stroke::new(1.0, stroke),
        corner_radius: CornerRadius::same(RADIUS),
        fg_stroke: Stroke::new(1.0, fg),
        expansion: 0.0,
    }
}

fn visuals() -> Visuals {
    let mut v = Visuals::dark();

    v.dark_mode = true;
    v.panel_fill = BG_PANEL;
    v.window_fill = BG_PANEL;
    v.faint_bg_color = FAINT;
    v.extreme_bg_color = BG_EXTREME;
    v.code_bg_color = BG_EXTREME;

    v.override_text_color = Some(TEXT);
    v.weak_text_color = Some(TEXT_MUTED);
    v.hyperlink_color = ACCENT;
    v.warn_fg_color = WARN;
    v.error_fg_color = ERROR;

    v.window_corner_radius = CornerRadius::same(8);
    v.menu_corner_radius = CornerRadius::same(8);
    v.window_stroke = Stroke::new(1.0, STROKE_SUBTLE);

    v.button_frame = true;
    v.slider_trailing_fill = true;

    v.widgets = Widgets {
        // Windows / labels / non-interactive frames. bg_fill here is the central
        // (canvas) background so the central panel reads darker than side panels.
        noninteractive: widget(BG_BASE, BG_BASE, STROKE_SUBTLE, TEXT_MUTED),
        inactive: widget(BG_WIDGET, BG_WIDGET, STROKE_SUBTLE, TEXT),
        hovered: widget(BG_HOVER, BG_HOVER, STROKE_STRONG, TEXT),
        active: widget(BG_ACTIVE, BG_ACTIVE, ACCENT, ACCENT_TEXT),
        open: widget(BG_WIDGET, BG_WIDGET, STROKE_STRONG, TEXT),
    };

    v.selection = Selection {
        bg_fill: ACCENT.gamma_multiply(0.55),
        stroke: Stroke::new(1.0, ACCENT_TEXT),
    };

    v
}

fn tune_spacing(style: &mut Style) {
    let s = &mut style.spacing;
    s.item_spacing = Vec2::new(8.0, 7.0);
    s.button_padding = Vec2::new(9.0, 5.0);
    s.window_margin = Margin::same(10);
    s.menu_margin = Margin::same(6);
    s.indent = 18.0;
    s.interact_size.y = 26.0;
    s.scroll.bar_width = 9.0;
}

fn tune_text(style: &mut Style) {
    use egui::TextStyle::*;
    style.text_styles = [
        (Heading, FontId::new(22.0, FontFamily::Proportional)),
        (Body, FontId::new(14.0, FontFamily::Proportional)),
        (Button, FontId::new(14.0, FontFamily::Proportional)),
        (Small, FontId::new(11.0, FontFamily::Proportional)),
        (Monospace, FontId::new(13.0, FontFamily::Monospace)),
    ]
    .into();
}
