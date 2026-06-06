//! Selection-mask raster algorithms.
//!
//! A *selection mask* is a `Vec<f32>` of length `width * height`, one value per
//! pixel in `0.0..=1.0` where `1.0` means fully selected. These functions build
//! and transform such masks: polygon (lasso) rasterization, feathering,
//! morphological grow/shrink, and boolean combination.

/// Rasterize a filled polygon (lasso) to a 0/1 mask. `points` are `(x, y)` in
/// pixel coordinates; the polygon is implicitly closed. Even-odd fill rule.
/// Returns a vec of length `width * height` (`1.0` inside, `0.0` outside).
pub fn polygon_mask(points: &[(f32, f32)], width: u32, height: u32) -> Vec<f32> {
    let w = width as usize;
    let h = height as usize;
    let mut mask = vec![0.0f32; w * h];
    if points.len() < 3 || w == 0 || h == 0 {
        return mask;
    }

    let n = points.len();
    let mut xs: Vec<f32> = Vec::with_capacity(n);
    for y in 0..h {
        let yc = y as f32 + 0.5;
        xs.clear();
        // Collect x crossings of edges with the horizontal scanline at `yc`.
        for i in 0..n {
            let (x0, y0) = points[i];
            let (x1, y1) = points[(i + 1) % n];
            // Edge spans the scanline if yc is within [min(y0,y1), max(y0,y1)).
            // Using a half-open interval avoids double-counting shared vertices.
            if (y0 <= yc && y1 > yc) || (y1 <= yc && y0 > yc) {
                let t = (yc - y0) / (y1 - y0);
                xs.push(x0 + t * (x1 - x0));
            }
        }
        if xs.is_empty() {
            continue;
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        // Fill spans between pairs of crossings (even-odd rule).
        let row = y * w;
        let mut i = 0;
        while i + 1 < xs.len() {
            let x_start = xs[i];
            let x_end = xs[i + 1];
            // Pixel center at px+0.5 is inside if x_start <= px+0.5 < x_end.
            let mut px_lo = (x_start - 0.5).ceil() as i64;
            let mut px_hi = (x_end - 0.5).ceil() as i64; // exclusive
            if px_lo < 0 {
                px_lo = 0;
            }
            if px_hi > w as i64 {
                px_hi = w as i64;
            }
            let mut px = px_lo;
            while px < px_hi {
                mask[row + px as usize] = 1.0;
                px += 1;
            }
            i += 2;
        }
    }
    mask
}

/// Feather (soften) a mask with a separable box blur approximating a Gaussian
/// of the given `radius` in pixels (radius 0 returns a copy). Edges clamp.
pub fn feather(mask: &[f32], width: u32, height: u32, radius: f32) -> Vec<f32> {
    let w = width as usize;
    let h = height as usize;
    if mask.len() != w * h {
        return vec![0.0f32; w * h];
    }
    if radius <= 0.0 || w == 0 || h == 0 {
        return mask.to_vec();
    }
    let r = radius.round() as i64;
    if r <= 0 {
        return mask.to_vec();
    }
    let norm = (2 * r + 1) as f32;

    // Horizontal pass.
    let mut tmp = vec![0.0f32; w * h];
    for y in 0..h {
        let row = y * w;
        for x in 0..w {
            let mut acc = 0.0f32;
            for k in -r..=r {
                let mut sx = x as i64 + k;
                if sx < 0 {
                    sx = 0;
                }
                if sx >= w as i64 {
                    sx = w as i64 - 1;
                }
                acc += mask[row + sx as usize];
            }
            tmp[row + x] = acc / norm;
        }
    }

    // Vertical pass.
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for k in -r..=r {
                let mut sy = y as i64 + k;
                if sy < 0 {
                    sy = 0;
                }
                if sy >= h as i64 {
                    sy = h as i64 - 1;
                }
                acc += tmp[sy as usize * w + x];
            }
            out[y * w + x] = acc / norm;
        }
    }
    out
}

/// Grow (dilate, `px > 0`) or shrink (erode, `px < 0`) the selected region by
/// `|px|` pixels using a 4-connected (city-block) morphological pass repeated
/// `|px|` times. `px == 0` returns a copy. Treats `value > 0.5` as selected for
/// the morphology, returns a 0/1 mask. Out-of-bounds neighbors are treated as
/// not-selected for both grow and shrink.
pub fn grow_shrink(mask: &[f32], width: u32, height: u32, px: i32) -> Vec<f32> {
    let w = width as usize;
    let h = height as usize;
    if mask.len() != w * h {
        return vec![0.0f32; w * h];
    }
    // Binarize.
    let mut cur: Vec<bool> = mask.iter().map(|&v| v > 0.5).collect();
    if px == 0 || w == 0 || h == 0 {
        return cur.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
    }

    let grow = px > 0;
    let iters = px.unsigned_abs();
    let mut next = vec![false; w * h];

    for _ in 0..iters {
        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                let here = cur[idx];
                // Neighbor selected state; OOB treated as not-selected.
                let up = if y > 0 { cur[idx - w] } else { false };
                let down = if y + 1 < h { cur[idx + w] } else { false };
                let left = if x > 0 { cur[idx - 1] } else { false };
                let right = if x + 1 < w { cur[idx + 1] } else { false };
                next[idx] = if grow {
                    here || up || down || left || right
                } else {
                    here && up && down && left && right
                };
            }
        }
        std::mem::swap(&mut cur, &mut next);
    }
    cur.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect()
}

/// Boolean combine mode for two equal-size masks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CombineMode {
    Replace,
    Add,
    Subtract,
    Intersect,
}

/// Boolean combine of two equal-size masks, element-wise.
///
/// - `Replace` = `other`
/// - `Add` = `max(base, other)`
/// - `Subtract` = `min(base, 1 - other)` i.e. `base * (1 - other)` clamped
/// - `Intersect` = `min(base, other)`
pub fn combine(base: &[f32], other: &[f32], mode: CombineMode) -> Vec<f32> {
    if base.len() != other.len() {
        // Mismatched buffers: return a sensibly-sized copy of base.
        return base.to_vec();
    }
    match mode {
        CombineMode::Replace => other.to_vec(),
        CombineMode::Add => base
            .iter()
            .zip(other)
            .map(|(&a, &b)| a.max(b))
            .collect(),
        CombineMode::Subtract => base
            .iter()
            .zip(other)
            .map(|(&a, &b)| a.min(1.0 - b))
            .collect(),
        CombineMode::Intersect => base
            .iter()
            .zip(other)
            .map(|(&a, &b)| a.min(b))
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn idx(x: usize, y: usize, w: usize) -> usize {
        y * w + x
    }

    #[test]
    fn polygon_square_fills_interior() {
        let w = 10;
        let h = 10;
        // Square covering pixel centers 2..=6 (corners at 2..7).
        let pts = [(2.0, 2.0), (7.0, 2.0), (7.0, 7.0), (2.0, 7.0)];
        let m = polygon_mask(&pts, w, h);
        // Interior pixel centers (2.5..6.5) selected.
        for y in 2..7 {
            for x in 2..7 {
                assert_eq!(m[idx(x, y, w as usize)], 1.0, "inside ({x},{y})");
            }
        }
        // Clearly-outside pixels not selected.
        assert_eq!(m[idx(0, 0, w as usize)], 0.0);
        assert_eq!(m[idx(9, 9, w as usize)], 0.0);
        assert_eq!(m[idx(8, 3, w as usize)], 0.0);
        assert_eq!(m[idx(3, 0, w as usize)], 0.0);
    }

    #[test]
    fn polygon_too_few_points_is_zero() {
        let m = polygon_mask(&[(0.0, 0.0), (5.0, 5.0)], 8, 8);
        assert!(m.iter().all(|&v| v == 0.0));
        assert_eq!(m.len(), 64);
    }

    #[test]
    fn polygon_triangle_fills() {
        let w = 10;
        let h = 10;
        // Right triangle with the right angle at top-left.
        let pts = [(1.0, 1.0), (9.0, 1.0), (1.0, 9.0)];
        let m = polygon_mask(&pts, w, h);
        // Point clearly inside near the top-left.
        assert_eq!(m[idx(2, 2, w as usize)], 1.0);
        // Bottom-right corner is outside the hypotenuse.
        assert_eq!(m[idx(8, 8, w as usize)], 0.0);
        // Outside the triangle entirely.
        assert_eq!(m[idx(0, 0, w as usize)], 0.0);
    }

    #[test]
    fn feather_radius_zero_is_copy() {
        let m = vec![0.0, 1.0, 0.0, 1.0];
        let f = feather(&m, 2, 2, 0.0);
        assert_eq!(f, m);
    }

    #[test]
    fn feather_spreads_single_pixel() {
        let w = 5;
        let h = 5;
        let mut m = vec![0.0f32; w * h];
        m[idx(2, 2, w)] = 1.0;
        let f = feather(&m, w as u32, h as u32, 1.0);
        // Center reduced below 1.
        assert!(f[idx(2, 2, w)] < 1.0, "center {}", f[idx(2, 2, w)]);
        assert!(f[idx(2, 2, w)] > 0.0);
        // 4-neighbors gained value.
        assert!(f[idx(1, 2, w)] > 0.0);
        assert!(f[idx(3, 2, w)] > 0.0);
        assert!(f[idx(2, 1, w)] > 0.0);
        assert!(f[idx(2, 3, w)] > 0.0);
        // Mass approximately conserved for an interior pixel.
        let sum: f32 = f.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "sum {sum}");
    }

    #[test]
    fn grow_one_selects_neighbors() {
        let w = 5;
        let h = 5;
        let mut m = vec![0.0f32; w * h];
        m[idx(2, 2, w)] = 1.0;
        let g = grow_shrink(&m, w as u32, h as u32, 1);
        assert_eq!(g[idx(2, 2, w)], 1.0);
        assert_eq!(g[idx(1, 2, w)], 1.0);
        assert_eq!(g[idx(3, 2, w)], 1.0);
        assert_eq!(g[idx(2, 1, w)], 1.0);
        assert_eq!(g[idx(2, 3, w)], 1.0);
        // Diagonal not selected (4-connected).
        assert_eq!(g[idx(1, 1, w)], 0.0);
    }

    #[test]
    fn shrink_one_erodes_edge() {
        let w = 5;
        let h = 5;
        let mut m = vec![0.0f32; w * h];
        // 3x3 block at rows/cols 1..=3.
        for y in 1..4 {
            for x in 1..4 {
                m[idx(x, y, w)] = 1.0;
            }
        }
        let s = grow_shrink(&m, w as u32, h as u32, -1);
        // Only the center survives.
        assert_eq!(s[idx(2, 2, w)], 1.0);
        // Edges removed.
        for &(x, y) in &[(1, 1), (2, 1), (3, 1), (1, 2), (3, 2), (1, 3), (3, 3)] {
            assert_eq!(s[idx(x, y, w)], 0.0, "edge ({x},{y})");
        }
    }

    #[test]
    fn grow_zero_is_copy() {
        let m = vec![1.0, 0.0, 0.7, 0.3];
        let g = grow_shrink(&m, 2, 2, 0);
        // Binarized copy: >0.5 -> 1, else 0.
        assert_eq!(g, vec![1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn combine_modes() {
        let w = 4;
        let h = 4;
        // base: left 2 columns selected.
        let mut base = vec![0.0f32; w * h];
        // other: top 2 rows selected.
        let mut other = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                if x < 2 {
                    base[idx(x, y, w)] = 1.0;
                }
                if y < 2 {
                    other[idx(x, y, w)] = 1.0;
                }
            }
        }

        let add = combine(&base, &other, CombineMode::Add);
        // Union: selected if in base OR other.
        assert_eq!(add[idx(0, 0, w)], 1.0); // both
        assert_eq!(add[idx(3, 0, w)], 1.0); // other only
        assert_eq!(add[idx(0, 3, w)], 1.0); // base only
        assert_eq!(add[idx(3, 3, w)], 0.0); // neither

        let inter = combine(&base, &other, CombineMode::Intersect);
        assert_eq!(inter[idx(0, 0, w)], 1.0); // overlap (top-left)
        assert_eq!(inter[idx(1, 1, w)], 1.0); // overlap
        assert_eq!(inter[idx(3, 0, w)], 0.0); // other only
        assert_eq!(inter[idx(0, 3, w)], 0.0); // base only

        let sub = combine(&base, &other, CombineMode::Subtract);
        // base minus other.
        assert_eq!(sub[idx(0, 0, w)], 0.0); // in overlap -> removed
        assert_eq!(sub[idx(0, 2, w)], 1.0); // base only -> kept
        assert_eq!(sub[idx(0, 3, w)], 1.0); // base only -> kept
        assert_eq!(sub[idx(3, 0, w)], 0.0); // not in base

        let rep = combine(&base, &other, CombineMode::Replace);
        assert_eq!(rep, other);
    }

    #[test]
    fn guards_on_mismatched_sizes() {
        // feather with wrong-length mask.
        let f = feather(&[1.0, 2.0], 4, 4, 1.0);
        assert_eq!(f.len(), 16);
        assert!(f.iter().all(|&v| v == 0.0));
        // grow with wrong-length mask.
        let g = grow_shrink(&[1.0], 4, 4, 1);
        assert_eq!(g.len(), 16);
        // combine with mismatched buffers returns base copy.
        let c = combine(&[1.0, 0.0, 1.0], &[0.0], CombineMode::Add);
        assert_eq!(c, vec![1.0, 0.0, 1.0]);
    }
}
