//! Shape + gradient rasterization.
//!
//! All public functions return an interleaved RGBA buffer of length
//! `width * height * 4`, in **linear-light premultiplied f32**. Inputs are
//! straight (non-premultiplied) linear RGBA in `0..=1`; we premultiply
//! (`rgb * a`) on the way out.
//!
//! Sampling is done at pixel centers `(x + 0.5, y + 0.5)`. Edges use ~1px
//! analytic anti-aliasing: coverage is derived from the signed distance to the
//! boundary, clamped to `0..1`.

/// Which primitive to fill.
#[derive(Clone, Copy)]
pub enum ShapeKind {
    Rectangle,
    Ellipse,
}

#[inline]
fn buf_len(width: u32, height: u32) -> usize {
    (width as usize) * (height as usize) * 4
}

/// Premultiply a straight linear RGBA color (rgb * a).
#[inline]
fn premultiply(c: [f32; 4]) -> [f32; 4] {
    let a = c[3];
    [c[0] * a, c[1] * a, c[2] * a, a]
}

/// Smooth 1px coverage from a signed distance: `d <= 0` fully inside (1.0),
/// `d >= 1` fully outside (0.0), linear ramp across the 1px boundary band.
#[inline]
fn coverage_from_distance(d: f32) -> f32 {
    (0.5 - d).clamp(0.0, 1.0)
}

/// Write `color` (already premultiplied) scaled by `coverage` into pixel `i`.
#[inline]
fn put(out: &mut [f32], i: usize, color: [f32; 4], coverage: f32) {
    let base = i * 4;
    out[base] = color[0] * coverage;
    out[base + 1] = color[1] * coverage;
    out[base + 2] = color[2] * coverage;
    out[base + 3] = color[3] * coverage;
}

/// Filled rectangle or ellipse within `rect` = `[x, y, w, h]` (pixel coords).
pub fn fill_shape(
    kind: ShapeKind,
    rect: [f32; 4],
    color: [f32; 4],
    width: u32,
    height: u32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; buf_len(width, height)];
    if width == 0 || height == 0 {
        return out;
    }

    let pm = premultiply(color);
    let [rx0, ry0, rw, rh] = rect;

    match kind {
        ShapeKind::Rectangle => {
            let x1 = rx0 + rw;
            let y1 = ry0 + rh;
            for y in 0..height {
                let py = y as f32 + 0.5;
                for x in 0..width {
                    let px = x as f32 + 0.5;
                    // Signed distance OUTSIDE the rect along each axis.
                    let dx = (rx0 - px).max(px - x1);
                    let dy = (ry0 - py).max(py - y1);
                    let d = dx.max(dy);
                    let cov = coverage_from_distance(d);
                    if cov > 0.0 {
                        put(&mut out, (y * width + x) as usize, pm, cov);
                    }
                }
            }
        }
        ShapeKind::Ellipse => {
            let cx = rx0 + rw * 0.5;
            let cy = ry0 + rh * 0.5;
            let rx = rw * 0.5;
            let ry = rh * 0.5;
            if rx <= 0.0 || ry <= 0.0 {
                return out;
            }
            for y in 0..height {
                let py = y as f32 + 0.5;
                for x in 0..width {
                    let px = x as f32 + 0.5;
                    let nx = (px - cx) / rx;
                    let ny = (py - cy) / ry;
                    // Normalized radial value; ==1 on the boundary.
                    let q = (nx * nx + ny * ny).sqrt();
                    // Convert the normalized distance to an approximate pixel
                    // distance using the gradient magnitude, for ~1px AA.
                    let grad =
                        (((px - cx) / (rx * rx)).powi(2) + ((py - cy) / (ry * ry)).powi(2)).sqrt();
                    let d = if grad > 0.0 {
                        (q - 1.0) / grad
                    } else {
                        // Center pixel of a degenerate-but-positive radius.
                        if q <= 1.0 {
                            -1.0
                        } else {
                            1.0
                        }
                    };
                    let cov = coverage_from_distance(d);
                    if cov > 0.0 {
                        put(&mut out, (y * width + x) as usize, pm, cov);
                    }
                }
            }
        }
    }

    out
}

/// A line segment from `p0` to `p1` with the given stroke `thickness` (px).
///
/// A pixel is covered when the distance from its center to the segment is
/// `<= thickness/2`, with ~1px AA on the boundary.
pub fn stroke_line(
    p0: (f32, f32),
    p1: (f32, f32),
    thickness: f32,
    color: [f32; 4],
    width: u32,
    height: u32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; buf_len(width, height)];
    if width == 0 || height == 0 || thickness <= 0.0 {
        return out;
    }

    let pm = premultiply(color);
    let half = thickness * 0.5;

    let (ax, ay) = p0;
    let (bx, by) = p1;
    let ex = bx - ax;
    let ey = by - ay;
    let len2 = ex * ex + ey * ey;

    for y in 0..height {
        let py = y as f32 + 0.5;
        for x in 0..width {
            let px = x as f32 + 0.5;
            // Distance from pixel center to the segment.
            let dist = if len2 <= 0.0 {
                // Degenerate segment: distance to the point p0.
                ((px - ax).powi(2) + (py - ay).powi(2)).sqrt()
            } else {
                let t = (((px - ax) * ex + (py - ay) * ey) / len2).clamp(0.0, 1.0);
                let projx = ax + t * ex;
                let projy = ay + t * ey;
                ((px - projx).powi(2) + (py - projy).powi(2)).sqrt()
            };
            // Signed distance to the stroke boundary.
            let d = dist - half;
            let cov = coverage_from_distance(d);
            if cov > 0.0 {
                put(&mut out, (y * width + x) as usize, pm, cov);
            }
        }
    }

    out
}

/// A linear gradient from `p0` (color `c0`) to `p1` (color `c1`), filling the
/// whole buffer. Pixels project onto the `p0 -> p1` axis; `t` is clamped to
/// `[0, 1]`; colors are interpolated straight then premultiplied.
pub fn linear_gradient(
    p0: (f32, f32),
    p1: (f32, f32),
    c0: [f32; 4],
    c1: [f32; 4],
    width: u32,
    height: u32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; buf_len(width, height)];
    if width == 0 || height == 0 {
        return out;
    }

    let (ax, ay) = p0;
    let dx = p1.0 - ax;
    let dy = p1.1 - ay;
    let len2 = dx * dx + dy * dy;

    for y in 0..height {
        let py = y as f32 + 0.5;
        for x in 0..width {
            let px = x as f32 + 0.5;
            let t = if len2 <= 0.0 {
                0.0
            } else {
                (((px - ax) * dx + (py - ay) * dy) / len2).clamp(0.0, 1.0)
            };
            let straight = [
                c0[0] + (c1[0] - c0[0]) * t,
                c0[1] + (c1[1] - c0[1]) * t,
                c0[2] + (c1[2] - c0[2]) * t,
                c0[3] + (c1[3] - c0[3]) * t,
            ];
            let pm = premultiply(straight);
            put(&mut out, (y * width + x) as usize, pm, 1.0);
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    #[inline]
    fn px(buf: &[f32], width: u32, x: u32, y: u32) -> [f32; 4] {
        let base = ((y * width + x) as usize) * 4;
        [buf[base], buf[base + 1], buf[base + 2], buf[base + 3]]
    }

    #[test]
    fn lengths_are_correct() {
        let w = 7;
        let h = 5;
        assert_eq!(
            fill_shape(ShapeKind::Rectangle, [0.0, 0.0, 7.0, 5.0], [1.0; 4], w, h).len(),
            (w * h * 4) as usize
        );
        assert_eq!(
            stroke_line((0.0, 0.0), (7.0, 5.0), 1.0, [1.0; 4], w, h).len(),
            (w * h * 4) as usize
        );
        assert_eq!(
            linear_gradient((0.0, 0.0), (7.0, 0.0), [0.0; 4], [1.0; 4], w, h).len(),
            (w * h * 4) as usize
        );
    }

    #[test]
    fn zero_dims_return_empty() {
        assert!(fill_shape(ShapeKind::Ellipse, [0.0, 0.0, 1.0, 1.0], [1.0; 4], 0, 5).is_empty());
        assert!(stroke_line((0.0, 0.0), (1.0, 1.0), 1.0, [1.0; 4], 5, 0).is_empty());
        assert!(linear_gradient((0.0, 0.0), (1.0, 0.0), [0.0; 4], [1.0; 4], 0, 0).is_empty());
    }

    #[test]
    fn rectangle_fills_whole_buffer() {
        let w = 4;
        let h = 3;
        // Straight color with alpha 0.5 -> premultiplied rgb = rgb*0.5.
        let color = [0.4, 0.6, 0.8, 0.5];
        let buf = fill_shape(
            ShapeKind::Rectangle,
            [0.0, 0.0, w as f32, h as f32],
            color,
            w,
            h,
        );
        let expected = [0.4 * 0.5, 0.6 * 0.5, 0.8 * 0.5, 0.5];
        for y in 0..h {
            for x in 0..w {
                let p = px(&buf, w, x, y);
                for c in 0..4 {
                    assert!(
                        (p[c] - expected[c]).abs() < EPS,
                        "pixel ({x},{y}) ch {c}: {} != {}",
                        p[c],
                        expected[c]
                    );
                }
            }
        }
    }

    #[test]
    fn ellipse_center_set_corners_clear() {
        let w = 21;
        let h = 21;
        let buf = fill_shape(
            ShapeKind::Ellipse,
            [0.0, 0.0, w as f32, h as f32],
            [1.0, 1.0, 1.0, 1.0],
            w,
            h,
        );
        // Center pixel fully covered.
        let center = px(&buf, w, w / 2, h / 2);
        assert!(center[3] > 0.99, "center alpha {} should be ~1", center[3]);
        // Corners clear.
        for &(x, y) in &[(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)] {
            let p = px(&buf, w, x, y);
            assert!(p[3] < EPS, "corner ({x},{y}) alpha {} should be ~0", p[3]);
        }
    }

    #[test]
    fn stroke_horizontal_marks_center_row() {
        let w = 11;
        let h = 11;
        let mid = (h / 2) as f32 + 0.5;
        let buf = stroke_line((0.0, mid), (w as f32, mid), 1.0, [1.0, 1.0, 1.0, 1.0], w, h);
        // Center row marked.
        for x in 1..w - 1 {
            let p = px(&buf, w, x, h / 2);
            assert!(p[3] > 0.5, "center-row pixel ({x}) alpha {} too low", p[3]);
        }
        // Far rows clear.
        for x in 0..w {
            let top = px(&buf, w, x, 0);
            let bot = px(&buf, w, x, h - 1);
            assert!(top[3] < EPS, "top row ({x}) alpha {} should be ~0", top[3]);
            assert!(
                bot[3] < EPS,
                "bottom row ({x}) alpha {} should be ~0",
                bot[3]
            );
        }
    }

    #[test]
    fn gradient_black_to_white_left_to_right() {
        let w = 9;
        let h = 1;
        let buf = linear_gradient(
            (0.0, 0.0),
            (w as f32, 0.0),
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            w,
            h,
        );
        // a == 1, so premultiplied == straight value.
        let left = px(&buf, w, 0, 0);
        let right = px(&buf, w, w - 1, 0);
        let midx = w / 2;
        let mid = px(&buf, w, midx, 0);

        // Leftmost pixel center is at 0.5/9 -> ~0.055, close to 0.
        assert!(left[0] < 0.1, "left {} should be ~0", left[0]);
        // Rightmost at 8.5/9 -> ~0.944, close to 1.
        assert!(right[0] > 0.9, "right {} should be ~1", right[0]);
        // Midpoint (x=4 -> center 4.5/9 = 0.5) exactly ~0.5.
        assert!((mid[0] - 0.5).abs() < 0.05, "mid {} should be ~0.5", mid[0]);
        // Alpha is opaque everywhere.
        assert!((left[3] - 1.0).abs() < EPS && (right[3] - 1.0).abs() < EPS);
    }
}
