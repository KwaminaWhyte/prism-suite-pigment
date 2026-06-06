//! Text rasterization into a linear-light premultiplied RGBA f32 layer.
//!
//! Uses `cosmic-text` for shaping + glyph coverage, then converts the supplied
//! straight sRGB color into linear light and premultiplies by glyph coverage so
//! the result can be uploaded directly as a layer.

use cosmic_text::{
    Align, Attrs, Buffer, Color, FontSystem, Metrics, Shaping, SwashCache,
};

/// Horizontal text alignment within the target buffer.
#[derive(Clone, Copy)]
pub enum TextAlign {
    Left,
    Center,
    Right,
}

impl TextAlign {
    fn to_cosmic(self) -> Align {
        match self {
            TextAlign::Left => Align::Left,
            TextAlign::Center => Align::Center,
            TextAlign::Right => Align::Right,
        }
    }
}

/// Standard sRGB transfer function (gamma-decode): straight sRGB -> linear.
#[inline]
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Rasterize `text` into a `width`x`height` RGBA buffer of **linear-light
/// premultiplied f32** (len == width*height*4), suitable to upload as a layer.
/// - `font_px`: font size in pixels.
/// - `color`: straight sRGB RGBA in 0..=1 (alpha multiplies glyph coverage).
/// - Text is laid out starting near the top-left with the given alignment and
///   wraps to `width`. Glyph coverage (0..1) becomes alpha; rgb = color.rgb in
///   linear light, premultiplied by (coverage*color.a).
/// - Empty text or zero dims => all-zero buffer.
pub fn render_text(
    text: &str,
    font_px: f32,
    color: [f32; 4],
    width: u32,
    height: u32,
    align: TextAlign,
) -> Vec<f32> {
    let px_count = (width as usize) * (height as usize);
    let mut out = vec![0.0f32; px_count * 4];

    if text.is_empty() || width == 0 || height == 0 {
        return out;
    }

    // Pre-decode the destination color to linear light. The glyph coverage and
    // the caller's alpha both modulate the final premultiplied result.
    let lin_r = srgb_to_linear(color[0].clamp(0.0, 1.0));
    let lin_g = srgb_to_linear(color[1].clamp(0.0, 1.0));
    let lin_b = srgb_to_linear(color[2].clamp(0.0, 1.0));
    let dst_a = color[3].clamp(0.0, 1.0);

    // A `FontSystem` is not cheap; created per call for simplicity per spec.
    let mut font_system = FontSystem::new();
    let mut swash_cache = SwashCache::new();

    let metrics = Metrics::new(font_px, font_px * 1.2);
    let mut buffer = Buffer::new(&mut font_system, metrics);

    buffer.set_size(Some(width as f32), Some(height as f32));

    let attrs = Attrs::new();
    buffer.set_text(text, &attrs, Shaping::Advanced, Some(align.to_cosmic()));
    buffer.shape_until_scroll(&mut font_system, false);

    let w = width as i32;
    let h = height as i32;

    // The draw callback hands us a rectangle (px,py,rw,rh) filled with `c`,
    // where `c.a()` is glyph coverage in 0..=255. We override rgb with the
    // caller's color and premultiply by coverage * dst_a in linear light.
    buffer.draw(
        &mut font_system,
        &mut swash_cache,
        Color::rgba(255, 255, 255, 255),
        |px, py, rw, rh, c| {
            let coverage = c.a() as f32 / 255.0;
            if coverage <= 0.0 {
                return;
            }
            let final_a = coverage * dst_a;
            if final_a <= 0.0 {
                return;
            }
            let pr = lin_r * final_a;
            let pg = lin_g * final_a;
            let pb = lin_b * final_a;

            for dy in 0..rh as i32 {
                let y = py + dy;
                if y < 0 || y >= h {
                    continue;
                }
                for dx in 0..rw as i32 {
                    let x = px + dx;
                    if x < 0 || x >= w {
                        continue;
                    }
                    let idx = ((y as usize) * (width as usize) + (x as usize)) * 4;
                    // Source-over accumulate (handles overlapping glyph rects).
                    let inv = 1.0 - final_a;
                    out[idx] = pr + out[idx] * inv;
                    out[idx + 1] = pg + out[idx + 1] * inv;
                    out[idx + 2] = pb + out[idx + 2] * inv;
                    out[idx + 3] = final_a + out[idx + 3] * inv;
                }
            }
        },
    );

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_glyph_alpha() {
        let buf = render_text("A", 48.0, [1.0, 1.0, 1.0, 1.0], 64, 64, TextAlign::Left);
        assert_eq!(buf.len(), 64 * 64 * 4);

        let lit = buf.iter().skip(3).step_by(4).filter(|&&a| a > 0.1).count();
        assert!(lit > 0, "expected some pixels with alpha > 0.1, got {lit}");
    }

    #[test]
    fn empty_text_all_zero() {
        let buf = render_text("", 48.0, [1.0, 1.0, 1.0, 1.0], 64, 64, TextAlign::Left);
        assert_eq!(buf.len(), 64 * 64 * 4);
        assert!(buf.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn zero_dims_empty_buffer() {
        let buf = render_text("A", 48.0, [1.0, 1.0, 1.0, 1.0], 0, 64, TextAlign::Left);
        assert_eq!(buf.len(), 0);
        assert!(buf.iter().all(|&v| v == 0.0));

        let buf2 = render_text("A", 48.0, [1.0, 1.0, 1.0, 1.0], 64, 0, TextAlign::Center);
        assert_eq!(buf2.len(), 0);
    }
}
