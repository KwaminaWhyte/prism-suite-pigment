//! Flood fill / magic-wand selection over an RGBA `f32` buffer.
//!
//! Operates in whatever color space the caller hands us (the app passes
//! linear-light). Produces a boolean selection mask, one entry per pixel.

/// Flood fill over an RGBA buffer of `f32` channels (length == width*height*4),
/// in whatever color space the caller provides (we pass linear). Returns a
/// boolean mask, one bool per pixel (length == width*height), true where the
/// region to fill is.
///
/// - `seed_x`, `seed_y`: starting pixel (must be in bounds; if not, return all-false).
/// - `tolerance`: 0.0..=1.0. A pixel matches if the max per-channel absolute
///   difference (including alpha) from the SEED pixel's color is <= tolerance.
/// - `contiguous`: if true, 4-connected flood from the seed (BFS/DFS). If false,
///   select ALL pixels in the image matching the seed within tolerance (global).
pub fn flood_fill_mask(
    rgba: &[f32],
    width: u32,
    height: u32,
    seed_x: u32,
    seed_y: u32,
    tolerance: f32,
    contiguous: bool,
) -> Vec<bool> {
    let w = width as usize;
    let h = height as usize;
    let pixel_count = w * h;

    // Guards: malformed buffer or out-of-bounds seed -> empty selection.
    if rgba.len() != pixel_count * 4 || seed_x >= width || seed_y >= height {
        return vec![false; pixel_count];
    }

    let mut mask = vec![false; pixel_count];

    let pixel_at = |idx: usize| -> [f32; 4] {
        let base = idx * 4;
        [rgba[base], rgba[base + 1], rgba[base + 2], rgba[base + 3]]
    };

    let seed_idx = (seed_y as usize) * w + (seed_x as usize);
    let seed = pixel_at(seed_idx);

    // Max per-channel absolute difference from the seed, incl. alpha.
    let matches = |idx: usize| -> bool {
        let p = pixel_at(idx);
        let mut diff = 0.0f32;
        for c in 0..4 {
            let d = (p[c] - seed[c]).abs();
            if d > diff {
                diff = d;
            }
        }
        diff <= tolerance
    };

    if !contiguous {
        // Global: scan every pixel, no connectivity requirement.
        for idx in 0..pixel_count {
            if matches(idx) {
                mask[idx] = true;
            }
        }
        return mask;
    }

    // Contiguous: iterative 4-connected flood fill from the seed.
    // `mask` doubles as the visited set, so each pixel is processed once.
    let mut stack: Vec<usize> = Vec::with_capacity(64);
    mask[seed_idx] = true;
    stack.push(seed_idx);

    while let Some(idx) = stack.pop() {
        let x = idx % w;
        let y = idx / w;

        // Left
        if x > 0 {
            let n = idx - 1;
            if !mask[n] && matches(n) {
                mask[n] = true;
                stack.push(n);
            }
        }
        // Right
        if x + 1 < w {
            let n = idx + 1;
            if !mask[n] && matches(n) {
                mask[n] = true;
                stack.push(n);
            }
        }
        // Up
        if y > 0 {
            let n = idx - w;
            if !mask[n] && matches(n) {
                mask[n] = true;
                stack.push(n);
            }
        }
        // Down
        if y + 1 < h {
            let n = idx + w;
            if !mask[n] && matches(n) {
                mask[n] = true;
                stack.push(n);
            }
        }
    }

    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an RGBA f32 buffer from a per-pixel color closure.
    fn build(width: u32, height: u32, mut f: impl FnMut(u32, u32) -> [f32; 4]) -> Vec<f32> {
        let mut buf = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                buf.extend_from_slice(&f(x, y));
            }
        }
        buf
    }

    #[test]
    fn contiguous_selects_only_one_half() {
        // 3x3 split: left two columns red, right column blue.
        let red = [1.0, 0.0, 0.0, 1.0];
        let blue = [0.0, 0.0, 1.0, 1.0];
        let rgba = build(3, 3, |x, _| if x < 2 { red } else { blue });

        let mask = flood_fill_mask(&rgba, 3, 3, 0, 0, 0.0, true);

        // Expect the two red columns selected, blue column not.
        for y in 0..3u32 {
            for x in 0..3u32 {
                let idx = (y * 3 + x) as usize;
                let expected = x < 2;
                assert_eq!(mask[idx], expected, "pixel ({x},{y})");
            }
        }
    }

    #[test]
    fn tolerance_zero_excludes_near_but_different() {
        // 1x3 row: seed, an exact match, and a near-but-different pixel.
        let seed = [0.5, 0.5, 0.5, 1.0];
        let exact = [0.5, 0.5, 0.5, 1.0];
        let near = [0.5, 0.5, 0.51, 1.0]; // off by 0.01 in blue
        let rgba = build(3, 1, |x, _| match x {
            0 => seed,
            1 => exact,
            _ => near,
        });

        // Contiguous from x=0; exact neighbor included, near excluded -> flood stops.
        let mask = flood_fill_mask(&rgba, 3, 1, 0, 0, 0.0, true);
        assert!(mask[0]);
        assert!(mask[1]);
        assert!(!mask[2]);

        // Global, same tolerance: still excludes the near pixel.
        let mask_global = flood_fill_mask(&rgba, 3, 1, 0, 0, 0.0, false);
        assert!(mask_global[0]);
        assert!(mask_global[1]);
        assert!(!mask_global[2]);
    }

    #[test]
    fn global_selects_disconnected_matching_blocks() {
        // 5x1 row: red, gap (green), red, red, green.
        // Two separate red regions; global must pick both, contiguous only one.
        let red = [1.0, 0.0, 0.0, 1.0];
        let green = [0.0, 1.0, 0.0, 1.0];
        let rgba = build(5, 1, |x, _| match x {
            0 => red,
            1 => green,
            2 => red,
            3 => red,
            _ => green,
        });

        let global = flood_fill_mask(&rgba, 5, 1, 0, 0, 0.0, false);
        assert_eq!(global, vec![true, false, true, true, false]);

        // Contiguous from x=0 reaches only the first red, blocked by green.
        let contig = flood_fill_mask(&rgba, 5, 1, 0, 0, 0.0, true);
        assert_eq!(contig, vec![true, false, false, false, false]);
    }

    #[test]
    fn out_of_bounds_seed_returns_all_false() {
        let rgba = build(2, 2, |_, _| [1.0, 1.0, 1.0, 1.0]);

        let mask = flood_fill_mask(&rgba, 2, 2, 2, 0, 0.0, true);
        assert_eq!(mask, vec![false; 4]);

        let mask_y = flood_fill_mask(&rgba, 2, 2, 0, 5, 0.5, false);
        assert_eq!(mask_y, vec![false; 4]);
    }

    #[test]
    fn malformed_buffer_returns_all_false() {
        let rgba = vec![1.0f32; 7]; // not 2*2*4 == 16
        let mask = flood_fill_mask(&rgba, 2, 2, 0, 0, 0.0, true);
        assert_eq!(mask, vec![false; 4]);
    }
}
