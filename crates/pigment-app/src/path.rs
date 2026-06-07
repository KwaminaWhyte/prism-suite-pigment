//! Pen-tool **work paths**: editable cubic-Bézier vector paths drawn on the
//! canvas as an overlay (not part of the GPU composite).
//!
//! A path is a list of [`Anchor`]s. Each anchor carries an on-curve point plus
//! two off-curve control handles (`in`/`out`), stored as **absolute** document
//! pixel coordinates (not deltas), so a corner point simply has its handles
//! equal to the point. Consecutive anchors define a cubic segment
//! `P0=a.point, P1=a.out, P2=b.in, P3=b.point`; a closed path adds the final
//! segment from the last anchor back to the first.
//!
//! The geometry math here is GPU- and egui-agnostic (plain `[f32; 2]`), so it
//! unit-tests without a device. The pen *interaction* and overlay rendering live
//! in the app/view layer; this module owns only the data + flatten + fill.

pub type Pt = [f32; 2];

/// One path node: an on-curve `point` and its two Bézier control handles, all in
/// absolute document-pixel coordinates. For a sharp corner the handles coincide
/// with `point`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Anchor {
    pub point: Pt,
    /// Incoming handle (controls the segment arriving at this anchor).
    pub h_in: Pt,
    /// Outgoing handle (controls the segment leaving this anchor).
    pub h_out: Pt,
}

impl Anchor {
    /// A corner anchor (both handles collapsed onto the point).
    pub fn corner(p: Pt) -> Self {
        Self {
            point: p,
            h_in: p,
            h_out: p,
        }
    }

    /// Set a symmetric (smooth) handle pair: `h_out` is dragged to `out`, and
    /// `h_in` mirrors it through the point. Used while click-dragging a new
    /// anchor with the pen.
    pub fn set_smooth_out(&mut self, out: Pt) {
        self.h_out = out;
        self.h_in = [2.0 * self.point[0] - out[0], 2.0 * self.point[1] - out[1]];
    }

    /// Translate the whole anchor (point + both handles) by `(dx, dy)`.
    pub fn translate(&mut self, dx: f32, dy: f32) {
        for p in [&mut self.point, &mut self.h_in, &mut self.h_out] {
            p[0] += dx;
            p[1] += dy;
        }
    }
}

/// An editable work path. Anchors in draw order; `closed` joins the last back to
/// the first.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WorkPath {
    pub anchors: Vec<Anchor>,
    pub closed: bool,
}

impl WorkPath {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.anchors.is_empty()
    }

    pub fn clear(&mut self) {
        self.anchors.clear();
        self.closed = false;
    }

    /// Flatten the path to a polyline in document pixels. Each cubic segment is
    /// subdivided into `steps` line pieces (so `steps+1` points per segment,
    /// sharing endpoints between segments). An open path is *not* implicitly
    /// closed; the caller closes it for fill.
    pub fn flatten(&self, steps: u32) -> Vec<Pt> {
        let steps = steps.max(1);
        let n = self.anchors.len();
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return vec![self.anchors[0].point];
        }
        let mut out: Vec<Pt> = Vec::new();
        out.push(self.anchors[0].point);
        let seg_count = if self.closed { n } else { n - 1 };
        for i in 0..seg_count {
            let a = self.anchors[i];
            let b = self.anchors[(i + 1) % n];
            flatten_cubic(a.point, a.h_out, b.h_in, b.point, steps, &mut out);
        }
        out
    }

    /// The closed-interior polygon for fill ops: the flattened polyline with the
    /// path treated as closed. Returns `< 3` points if the path can't enclose
    /// area.
    pub fn fill_polygon(&self, steps: u32) -> Vec<Pt> {
        if self.anchors.len() < 2 {
            return Vec::new();
        }
        let steps = steps.max(1);
        let n = self.anchors.len();
        let mut out: Vec<Pt> = Vec::new();
        out.push(self.anchors[0].point);
        // Always walk every segment incl. the wrap-around closing one.
        for i in 0..n {
            let a = self.anchors[i];
            let b = self.anchors[(i + 1) % n];
            flatten_cubic(a.point, a.h_out, b.h_in, b.point, steps, &mut out);
        }
        // Drop the duplicate closing vertex (== first point) the wrap produced.
        if out.len() >= 2 && out[out.len() - 1] == out[0] {
            out.pop();
        }
        out
    }
}

/// Evaluate a cubic Bézier at parameter `t in [0,1]`.
pub fn cubic_point(p0: Pt, p1: Pt, p2: Pt, p3: Pt, t: f32) -> Pt {
    let u = 1.0 - t;
    let w0 = u * u * u;
    let w1 = 3.0 * u * u * t;
    let w2 = 3.0 * u * t * t;
    let w3 = t * t * t;
    [
        w0 * p0[0] + w1 * p1[0] + w2 * p2[0] + w3 * p3[0],
        w0 * p0[1] + w1 * p1[1] + w2 * p2[1] + w3 * p3[1],
    ]
}

/// Append `steps` flattened points of the cubic (excluding `p0`, including
/// `p3`) onto `out`. `out` is assumed to already end with `p0`.
fn flatten_cubic(p0: Pt, p1: Pt, p2: Pt, p3: Pt, steps: u32, out: &mut Vec<Pt>) {
    for s in 1..=steps {
        let t = s as f32 / steps as f32;
        out.push(cubic_point(p0, p1, p2, p3, t));
    }
}

/// Squared distance between two points.
pub fn dist2(a: Pt, b: Pt) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    dx * dx + dy * dy
}

/// Even-odd point-in-polygon test (ray-cast). `poly` is an implicitly-closed
/// ring of `(x, y)` vertices; `p` is the test point.
pub fn point_in_polygon(poly: &[Pt], p: Pt) -> bool {
    let n = poly.len();
    if n < 3 {
        return false;
    }
    let (px, py) = (p[0], p[1]);
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (poly[i][0], poly[i][1]);
        let (xj, yj) = (poly[j][0], poly[j][1]);
        // Does the upward/downward edge straddle the horizontal ray at py?
        if (yi > py) != (yj > py) {
            let x_cross = xi + (py - yi) / (yj - yi) * (xj - xi);
            if px < x_cross {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

/// Rasterize the closed interior of `poly` (a ring of `(x, y)` in pixel coords,
/// e.g. from [`WorkPath::fill_polygon`]) to a 0/1 selection mask of length
/// `width * height` — `1.0` for pixels whose center is inside. Mirrors the
/// lasso/marquee mask shape produced elsewhere, so the result drops straight
/// into the selection / layer-mask pipelines. Even-odd fill rule.
pub fn fill_mask(poly: &[Pt], width: u32, height: u32) -> Vec<f32> {
    let (w, h) = (width as usize, height as usize);
    let mut mask = vec![0.0f32; w * h];
    if poly.len() < 3 || w == 0 || h == 0 {
        return mask;
    }
    // Vertical bounds clamp the scanline range we touch.
    let (mut y0, mut y1) = (f32::INFINITY, f32::NEG_INFINITY);
    for v in poly {
        y0 = y0.min(v[1]);
        y1 = y1.max(v[1]);
    }
    let lo = (y0.floor().max(0.0)) as usize;
    let hi = ((y1.ceil()).min(h as f32)) as usize;
    for y in lo..hi {
        let yc = y as f32 + 0.5;
        let row = y * w;
        for x in 0..w {
            if point_in_polygon(poly, [x as f32 + 0.5, yc]) {
                mask[row + x] = 1.0;
            }
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn cubic_endpoints_and_midpoint() {
        // A straight diagonal: all control points on the line.
        let p0 = [0.0, 0.0];
        let p1 = [1.0, 1.0];
        let p2 = [2.0, 2.0];
        let p3 = [3.0, 3.0];
        assert_eq!(cubic_point(p0, p1, p2, p3, 0.0), p0);
        assert_eq!(cubic_point(p0, p1, p2, p3, 1.0), p3);
        let m = cubic_point(p0, p1, p2, p3, 0.5);
        assert!(close(m[0], 1.5, 1e-5) && close(m[1], 1.5, 1e-5));
    }

    #[test]
    fn flatten_points_lie_on_the_curve() {
        // A genuine curve (handles off the chord). Every flattened vertex must
        // equal the analytic Bézier evaluated at the same parameter.
        let p0 = [0.0, 0.0];
        let p1 = [0.0, 100.0];
        let p2 = [100.0, 100.0];
        let p3 = [100.0, 0.0];
        let a = Anchor {
            point: p0,
            h_in: p0,
            h_out: p1,
        };
        let b = Anchor {
            point: p3,
            h_in: p2,
            h_out: p3,
        };
        let path = WorkPath {
            anchors: vec![a, b],
            closed: false,
        };
        let steps = 16;
        let poly = path.flatten(steps);
        assert_eq!(poly.len() as u32, steps + 1);
        for (s, pt) in poly.iter().enumerate() {
            let t = s as f32 / steps as f32;
            let exp = cubic_point(p0, p1, p2, p3, t);
            assert!(
                close(pt[0], exp[0], 1e-4) && close(pt[1], exp[1], 1e-4),
                "flattened vertex {s} off curve: {pt:?} vs {exp:?}"
            );
        }
    }

    #[test]
    fn flatten_straight_segment_is_collinear() {
        // Corner handles => the cubic degenerates to the straight chord, so all
        // flattened points must be collinear with the endpoints.
        let a = Anchor::corner([0.0, 0.0]);
        let b = Anchor::corner([10.0, 5.0]);
        let path = WorkPath {
            anchors: vec![a, b],
            closed: false,
        };
        for pt in path.flatten(8) {
            // Line y = 0.5 x.
            assert!(close(pt[1], 0.5 * pt[0], 1e-4));
        }
    }

    #[test]
    fn smooth_handle_mirrors_through_point() {
        let mut a = Anchor::corner([5.0, 5.0]);
        a.set_smooth_out([8.0, 5.0]);
        assert_eq!(a.h_out, [8.0, 5.0]);
        assert_eq!(a.h_in, [2.0, 5.0]); // mirror through (5,5)
    }

    #[test]
    fn fill_polygon_of_a_square_has_no_duplicate_closing_vertex() {
        // Four corner anchors => a square; fill_polygon must not repeat the start.
        let pts = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let anchors = pts.iter().map(|p| Anchor::corner(*p)).collect();
        let path = WorkPath {
            anchors,
            closed: true,
        };
        let poly = path.fill_polygon(1);
        // 4 segments * 1 step = 4 vertices, no duplicate of the first.
        assert_eq!(poly.len(), 4);
        assert_eq!(poly[0], [0.0, 0.0]);
        assert_ne!(poly.last().copied().unwrap(), poly[0]);
    }

    #[test]
    fn point_in_polygon_on_a_square() {
        let sq = [[2.0, 2.0], [8.0, 2.0], [8.0, 8.0], [2.0, 8.0]];
        assert!(point_in_polygon(&sq, [5.0, 5.0])); // center
        assert!(!point_in_polygon(&sq, [0.0, 0.0])); // outside
        assert!(!point_in_polygon(&sq, [9.0, 5.0])); // just right of edge
        assert!(point_in_polygon(&sq, [2.5, 7.5])); // near a corner, inside
    }

    #[test]
    fn fill_mask_covers_exactly_the_square_interior() {
        // A 4x4 square inside a 10x10 canvas. Pixel centers at x.5 inside
        // [3,7) in both axes => a 4x4 = 16-pixel block.
        let sq = [[3.0, 3.0], [7.0, 3.0], [7.0, 7.0], [3.0, 7.0]];
        let (w, h) = (10u32, 10u32);
        let mask = fill_mask(&sq, w, h);
        assert_eq!(mask.len(), (w * h) as usize);
        let covered: usize = mask.iter().filter(|&&v| v > 0.5).count();
        assert_eq!(covered, 16, "square should cover a 4x4 block");
        // Spot-check: (5,5) inside, (1,1) and (8,8) outside.
        assert_eq!(mask[(5 * w + 5) as usize], 1.0);
        assert_eq!(mask[(1 * w + 1) as usize], 0.0);
        assert_eq!(mask[(8 * w + 8) as usize], 0.0);
    }

    #[test]
    fn path_to_fill_mask_matches_expected_coverage() {
        // Build a closed work path (square via corner anchors), flatten + fill,
        // and confirm the mask coverage equals the analytic interior area —
        // this is the exact `path → selection` core (sans GPU upload).
        let pts = [[3.0, 3.0], [7.0, 3.0], [7.0, 7.0], [3.0, 7.0]];
        let anchors = pts.iter().map(|p| Anchor::corner(*p)).collect();
        let path = WorkPath {
            anchors,
            closed: true,
        };
        let (w, h) = (10u32, 10u32);
        let poly = path.fill_polygon(8); // straight edges => flatten is exact
        let mask = fill_mask(&poly, w, h);
        let covered: usize = mask.iter().filter(|&&v| v > 0.5).count();
        assert_eq!(covered, 16);
    }

    #[test]
    fn fill_mask_empty_for_degenerate_path() {
        let poly = [[1.0, 1.0], [2.0, 2.0]]; // < 3 points
        let mask = fill_mask(&poly, 5, 5);
        assert!(mask.iter().all(|&v| v == 0.0));
    }
}
