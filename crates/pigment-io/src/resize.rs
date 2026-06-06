//! High-quality image resampling for the editor's linear-premultiplied RGBA
//! `f32` representation.
//!
//! Backed by [`fast_image_resize`] v6 (SIMD convolution). Pixels are 4-channel
//! interleaved `f32`, row-major. The data is already premultiplied, so we feed
//! the channels straight through with alpha multiply/divide disabled
//! (`use_alpha(false)`); re-premultiplying here would darken edges.

use fast_image_resize::images::Image;
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};

/// Resampling quality.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quality {
    /// Nearest-neighbor. Fast, blocky; no new colors.
    Nearest,
    /// Bilinear (linear convolution). Cheap smoothing.
    Bilinear,
    /// Bicubic (Catmull-Rom). Good default for upscale.
    Bicubic,
    /// Lanczos-3. Sharpest; good default for downscale.
    Lanczos3,
}

impl Quality {
    fn alg(self) -> ResizeAlg {
        match self {
            Quality::Nearest => ResizeAlg::Nearest,
            Quality::Bilinear => ResizeAlg::Convolution(FilterType::Bilinear),
            Quality::Bicubic => ResizeAlg::Convolution(FilterType::CatmullRom),
            Quality::Lanczos3 => ResizeAlg::Convolution(FilterType::Lanczos3),
        }
    }
}

/// Resize an interleaved RGBA `f32` image (`src.len() == src_w*src_h*4`) to
/// `dst_w x dst_h`. Returns a new `Vec` of len `dst_w*dst_h*4`.
///
/// Pixels are treated as already premultiplied (alpha is **not** re-applied).
/// Prefer [`Quality::Lanczos3`] for downscale and [`Quality::Bicubic`] for
/// upscale.
///
/// # Panics
/// Panics if `src.len() != src_w*src_h*4`, or if any dimension is zero.
pub fn resize_rgba_f32(
    src: &[f32],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    quality: Quality,
) -> Vec<f32> {
    let expected = src_w as usize * src_h as usize * 4;
    assert_eq!(
        src.len(),
        expected,
        "src len {} != src_w*src_h*4 ({expected})",
        src.len()
    );
    assert!(
        src_w > 0 && src_h > 0 && dst_w > 0 && dst_h > 0,
        "dimensions must be non-zero"
    );

    // Fast path: identical size -> straight copy.
    if src_w == dst_w && src_h == dst_h {
        return src.to_vec();
    }

    // Reinterpret &[f32] as bytes; fast_image_resize works on byte buffers and
    // groups them into F32x4 pixels (16 bytes each). `from_vec_u8` takes
    // ownership, so build an owned byte copy of the source.
    let src_bytes: Vec<u8> = bytemuck::cast_slice(src).to_vec();
    let src_img = Image::from_vec_u8(src_w, src_h, src_bytes, PixelType::F32x4)
        .expect("valid source image buffer");

    let mut dst_img = Image::new(dst_w, dst_h, PixelType::F32x4);

    let opts = ResizeOptions::new()
        .resize_alg(quality.alg())
        // Already premultiplied — do not let the crate multiply/divide alpha.
        .use_alpha(false);

    let mut resizer = Resizer::new();
    resizer
        .resize(&src_img, &mut dst_img, &opts)
        .expect("resize succeeds for matching F32x4 pixel types");

    // dst_img owns its buffer; reinterpret bytes back to f32.
    bytemuck::cast_slice::<u8, f32>(dst_img.buffer()).to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, c: [f32; 4]) -> Vec<f32> {
        let mut v = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..(w * h) {
            v.extend_from_slice(&c);
        }
        v
    }

    #[test]
    fn downscale_solid_preserves_color() {
        let color = [0.2, 0.4, 0.6, 0.8];
        let src = solid(64, 64, color);
        let out = resize_rgba_f32(&src, 64, 64, 8, 8, Quality::Lanczos3);
        assert_eq!(out.len(), 8 * 8 * 4);
        for px in out.chunks_exact(4) {
            for (got, want) in px.iter().zip(color.iter()) {
                assert!(
                    (got - want).abs() < 1e-4,
                    "expected {want}, got {got} (px {px:?})"
                );
            }
        }
    }

    #[test]
    fn roundtrip_flat_stays_flat() {
        let color = [0.5, 0.25, 0.75, 1.0];
        let src = solid(4, 4, color);
        let up = resize_rgba_f32(&src, 4, 4, 8, 8, Quality::Bilinear);
        assert_eq!(up.len(), 8 * 8 * 4);
        let back = resize_rgba_f32(&up, 8, 8, 4, 4, Quality::Bilinear);
        assert_eq!(back.len(), 4 * 4 * 4);
        for px in back.chunks_exact(4) {
            for (got, want) in px.iter().zip(color.iter()) {
                assert!(
                    (got - want).abs() < 1e-4,
                    "flat image drifted: expected {want}, got {got}"
                );
            }
        }
    }

    #[test]
    fn nearest_2x2_to_4x4_replicates_quadrants() {
        // Distinct color per source pixel.
        let tl = [1.0, 0.0, 0.0, 1.0];
        let tr = [0.0, 1.0, 0.0, 1.0];
        let bl = [0.0, 0.0, 1.0, 1.0];
        let br = [1.0, 1.0, 0.0, 1.0];
        let mut src = Vec::new();
        src.extend_from_slice(&tl);
        src.extend_from_slice(&tr);
        src.extend_from_slice(&bl);
        src.extend_from_slice(&br);

        let out = resize_rgba_f32(&src, 2, 2, 4, 4, Quality::Nearest);
        assert_eq!(out.len(), 4 * 4 * 4);

        let at = |x: usize, y: usize| -> [f32; 4] {
            let i = (y * 4 + x) * 4;
            [out[i], out[i + 1], out[i + 2], out[i + 3]]
        };

        // Each source pixel maps to a 2x2 quadrant.
        for (x, y, want) in [
            (0, 0, tl),
            (1, 0, tl),
            (2, 0, tr),
            (3, 0, tr),
            (0, 3, bl),
            (3, 3, br),
        ] {
            assert_eq!(at(x, y), want, "quadrant mismatch at ({x},{y})");
        }
    }
}
