//! Icon font for Pigment's toolbar and UI.
//!
//! Uses [`egui-phosphor`] (Phosphor icons, MIT-licensed) which ships a TTF and
//! glyph constants compatible with egui 0.34. We register the font into both the
//! Proportional and Monospace families so glyphs render inline with regular text
//! (e.g. `"{ICON} Brush"`), then re-export the codepoints under tool-oriented
//! names so call sites read clearly.

use egui_phosphor::regular as ph;

/// Merge the Phosphor icon font into the context's font definitions.
///
/// Appends the icon font to both the Proportional and Monospace families (as a
/// fallback after the default text fonts) so icon glyphs render alongside text.
/// Call once at startup with `cc.egui_ctx`.
pub fn install(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();

    // Insert the Phosphor TTF under a known key.
    fonts.font_data.insert(
        "phosphor".to_owned(),
        std::sync::Arc::new(egui_phosphor::Variant::Regular.font_data()),
    );

    // Append as a fallback (lowest priority) for both built-in families so the
    // default text fonts win for normal characters and Phosphor supplies icons.
    for family in [
        egui::FontFamily::Proportional,
        egui::FontFamily::Monospace,
    ] {
        fonts
            .families
            .entry(family)
            .or_default()
            .push("phosphor".to_owned());
    }

    ctx.set_fonts(fonts);
}

// --- Tool / action glyphs (re-exported Phosphor codepoints) -----------------

/// Pan / hand tool.
pub const PAN: &str = ph::HAND;
/// Brush / paint tool.
pub const BRUSH: &str = ph::PAINT_BRUSH;
/// Eraser tool.
pub const ERASER: &str = ph::ERASER;
/// Fill / paint-bucket tool.
pub const FILL: &str = ph::PAINT_BUCKET;
/// Eyedropper / color-pick tool.
pub const EYEDROPPER: &str = ph::EYEDROPPER;
/// Rectangle (marquee) selection tool.
pub const RECT_SELECT: &str = ph::RECTANGLE_DASHED;
/// Ellipse / oval selection tool.
pub const ELLIPSE_SELECT: &str = ph::CIRCLE_DASHED;
/// Lasso (freehand) selection tool.
pub const LASSO: &str = ph::LASSO;
/// Magic-wand selection tool.
pub const MAGIC_WAND: &str = ph::MAGIC_WAND;
/// Move / four-way arrows tool.
pub const MOVE: &str = ph::ARROWS_OUT_CARDINAL;
/// Crop tool.
pub const CROP: &str = ph::CROP;
/// Free transform / scale tool.
pub const TRANSFORM: &str = ph::ARROWS_OUT;

// --- Layer / panel actions --------------------------------------------------

/// Add a new layer.
pub const PLUS_LAYER: &str = ph::STACK_PLUS;
/// Delete / trash.
pub const TRASH: &str = ph::TRASH;
/// Move item up.
pub const ARROW_UP: &str = ph::ARROW_UP;
/// Move item down.
pub const ARROW_DOWN: &str = ph::ARROW_DOWN;
/// Visibility toggle (eye).
pub const EYE: &str = ph::EYE;

// --- History ----------------------------------------------------------------

/// Undo.
pub const UNDO: &str = ph::ARROW_COUNTER_CLOCKWISE;
/// Redo.
pub const REDO: &str = ph::ARROW_CLOCKWISE;
