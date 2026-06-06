//! Import a Contour vector document (`.contour` JSON) as a rasterized raster
//! layer — the first Prism cross-app link (Pigment <- Contour).
//!
//! This is a static "place": we read Contour's JSON, rasterize every shape in
//! paint order into a full-canvas linear-premultiplied f32 buffer, and hand the
//! buffer back for upload as a new raster layer. A live Dynamic-Link is future
//! work.
//!
//! The schema mirrors `contour-app`'s `document.rs`: a top-level object with an
//! ordered `shapes` list, where `Shape` is an externally-tagged serde enum
//! (`{"Rect": { .. }}`, `{"Ellipse": { .. }}`, `{"Line": { .. }}`,
//! `{"Path": { .. }}`). Colors are straight sRGB RGBA in `[f32; 4]`. We use
//! `#[serde(default)]` on fields and tolerate unknown fields so newer
//! `.contour` files still load.

use prism_core::color::srgb_to_linear;
use prism_core::shape::{self, ShapeKind};
use serde::Deserialize;
use std::path::Path;

/// One drawable vector primitive (matches Contour's `Shape` enum). Externally
/// tagged, like Contour's serde derive. Unknown variants would fail to parse,
/// so the set here must stay a superset of what Contour writes; per-variant
/// fields are `#[serde(default)]` to tolerate added/removed fields.
#[derive(Debug, Clone, Deserialize)]
enum Shape {
    Rect {
        #[serde(default)]
        rect: [f32; 4],
        #[serde(default)]
        fill: [f32; 4],
        #[serde(default)]
        stroke: [f32; 4],
        #[serde(default)]
        stroke_w: f32,
    },
    Ellipse {
        #[serde(default)]
        rect: [f32; 4],
        #[serde(default)]
        fill: [f32; 4],
        #[serde(default)]
        stroke: [f32; 4],
        #[serde(default)]
        stroke_w: f32,
    },
    Line {
        #[serde(default)]
        p0: (f32, f32),
        #[serde(default)]
        p1: (f32, f32),
        #[serde(default)]
        stroke: [f32; 4],
        #[serde(default)]
        stroke_w: f32,
    },
    Path {
        #[serde(default)]
        points: Vec<(f32, f32)>,
        #[serde(default)]
        closed: bool,
        #[serde(default)]
        fill: [f32; 4],
        #[serde(default)]
        stroke: [f32; 4],
        #[serde(default)]
        stroke_w: f32,
    },
}

/// The whole vector document: an ordered list of shapes (index 0 painted first,
/// last on top). Tolerates extra top-level fields (e.g. a future artboard size).
#[derive(Debug, Clone, Deserialize)]
struct ContourDoc {
    #[serde(default)]
    shapes: Vec<Shape>,
}

/// Read + parse a `.contour` file into a `ContourDoc`.
fn parse(path: &Path) -> anyhow::Result<ContourDoc> {
    let json = std::fs::read_to_string(path)?;
    let doc: ContourDoc = serde_json::from_str(&json)?;
    Ok(doc)
}

/// Straight sRGB RGBA -> straight linear RGBA (alpha unchanged). The shared
/// `shape` helpers premultiply internally, so we hand them straight-linear.
#[inline]
fn srgb_to_linear_rgba(c: [f32; 4]) -> [f32; 4] {
    [
        srgb_to_linear(c[0]),
        srgb_to_linear(c[1]),
        srgb_to_linear(c[2]),
        c[3],
    ]
}

/// Premultiplied-over composite of `src` (linear premult) onto `dst` (linear
/// premult), in place. Both buffers are `w*h*4` interleaved RGBA.
fn over(dst: &mut [f32], src: &[f32]) {
    let n = dst.len().min(src.len()) / 4;
    for i in 0..n {
        let sa = src[i * 4 + 3];
        let inv = 1.0 - sa;
        for c in 0..4 {
            dst[i * 4 + c] = src[i * 4 + c] + dst[i * 4 + c] * inv;
        }
    }
}

/// Even-odd scanline fill of a closed polygon into a linear-premultiplied f32
/// buffer. Color is straight linear; we premultiply on write. No anti-aliasing
/// (point-sampled at pixel centers) — strokes carry the AA'd edges.
fn fill_polygon(points: &[(f32, f32)], color: [f32; 4], width: u32, height: u32) -> Vec<f32> {
    let mut out = vec![0.0f32; (width as usize) * (height as usize) * 4];
    let n = points.len();
    if n < 3 || width == 0 || height == 0 || color[3] <= 0.0 {
        return out;
    }
    let a = color[3];
    let pm = [color[0] * a, color[1] * a, color[2] * a, a];
    for y in 0..height {
        let py = y as f32 + 0.5;
        // Collect x-intersections of the scanline with all edges.
        let mut xs: Vec<f32> = Vec::new();
        let mut j = n - 1;
        for i in 0..n {
            let (xi, yi) = points[i];
            let (xj, yj) = points[j];
            if (yi > py) != (yj > py) {
                let t = (py - yi) / (yj - yi);
                xs.push(xi + t * (xj - xi));
            }
            j = i;
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        // Fill between successive intersection pairs (even-odd rule).
        let mut k = 0;
        while k + 1 < xs.len() {
            let x0 = xs[k].ceil().max(0.0) as i64;
            let x1 = (xs[k + 1].floor() as i64).min(width as i64 - 1);
            let mut x = x0;
            while x <= x1 {
                let idx = ((y as usize) * (width as usize) + x as usize) * 4;
                out[idx..idx + 4].copy_from_slice(&pm);
                x += 1;
            }
            k += 2;
        }
    }
    out
}

/// Rasterize one shape into its own full-canvas linear-premultiplied buffer,
/// then composite onto `acc` (premultiplied-over). Fills paint under strokes.
fn rasterize_shape(acc: &mut [f32], s: &Shape, w: u32, h: u32) {
    match s {
        Shape::Rect {
            rect,
            fill,
            stroke,
            stroke_w,
        } => {
            if fill[3] > 0.0 {
                let buf = shape::fill_shape(
                    ShapeKind::Rectangle,
                    *rect,
                    srgb_to_linear_rgba(*fill),
                    w,
                    h,
                );
                over(acc, &buf);
            }
            if stroke[3] > 0.0 && *stroke_w > 0.0 {
                stroke_rect_outline(acc, *rect, srgb_to_linear_rgba(*stroke), *stroke_w, w, h);
            }
        }
        Shape::Ellipse {
            rect,
            fill,
            stroke,
            stroke_w,
        } => {
            if fill[3] > 0.0 {
                let buf =
                    shape::fill_shape(ShapeKind::Ellipse, *rect, srgb_to_linear_rgba(*fill), w, h);
                over(acc, &buf);
            }
            // Approximate the ellipse outline by stroking a polyline of its
            // perimeter (Contour fills+strokes; this keeps strokes visible).
            if stroke[3] > 0.0 && *stroke_w > 0.0 {
                let pts = ellipse_polyline(*rect);
                stroke_polyline(
                    acc,
                    &pts,
                    true,
                    srgb_to_linear_rgba(*stroke),
                    *stroke_w,
                    w,
                    h,
                );
            }
        }
        Shape::Line {
            p0,
            p1,
            stroke,
            stroke_w,
        } => {
            if stroke[3] > 0.0 && *stroke_w > 0.0 {
                let buf =
                    shape::stroke_line(*p0, *p1, *stroke_w, srgb_to_linear_rgba(*stroke), w, h);
                over(acc, &buf);
            }
        }
        Shape::Path {
            points,
            closed,
            fill,
            stroke,
            stroke_w,
        } => {
            // Fill first (only meaningful for closed paths), then stroke on top.
            if *closed && fill[3] > 0.0 && points.len() >= 3 {
                let buf = fill_polygon(points, srgb_to_linear_rgba(*fill), w, h);
                over(acc, &buf);
            }
            if stroke[3] > 0.0 && *stroke_w > 0.0 {
                stroke_polyline(
                    acc,
                    points,
                    *closed,
                    srgb_to_linear_rgba(*stroke),
                    *stroke_w,
                    w,
                    h,
                );
            }
        }
    }
}

/// Stroke the four edges of a rect as thick lines, compositing each over `acc`.
fn stroke_rect_outline(
    acc: &mut [f32],
    rect: [f32; 4],
    color: [f32; 4],
    thickness: f32,
    w: u32,
    h: u32,
) {
    let [x, y, rw, rh] = rect;
    let corners = [(x, y), (x + rw, y), (x + rw, y + rh), (x, y + rh)];
    stroke_polyline(acc, &corners, true, color, thickness, w, h);
}

/// Stroke a polyline (segment by segment) over `acc`. When `closed`, the last
/// point connects back to the first. Each segment is an AA'd thick line.
fn stroke_polyline(
    acc: &mut [f32],
    points: &[(f32, f32)],
    closed: bool,
    color: [f32; 4],
    thickness: f32,
    w: u32,
    h: u32,
) {
    let n = points.len();
    if n < 2 {
        return;
    }
    let last = if closed { n } else { n - 1 };
    for i in 0..last {
        let a = points[i];
        let b = points[(i + 1) % n];
        let buf = shape::stroke_line(a, b, thickness, color, w, h);
        over(acc, &buf);
    }
}

/// Sample an ellipse's perimeter into a closed polyline (for stroking).
fn ellipse_polyline(rect: [f32; 4]) -> Vec<(f32, f32)> {
    let [x, y, rw, rh] = rect;
    let (cx, cy) = (x + rw * 0.5, y + rh * 0.5);
    let (rx, ry) = (rw * 0.5, rh * 0.5);
    // Step count scales with the larger radius for smoothness.
    let steps = ((rx.abs().max(ry.abs())) as usize).clamp(24, 256);
    let mut pts = Vec::with_capacity(steps);
    for k in 0..steps {
        let t = (k as f32) / (steps as f32) * std::f32::consts::TAU;
        pts.push((cx + rx * t.cos(), cy + ry * t.sin()));
    }
    pts
}

/// Rasterize a parsed Contour document into a full-canvas linear-premultiplied
/// f32 RGBA buffer (`width*height*4`). Shapes paint in order; index 0 first.
fn rasterize_doc(doc: &ContourDoc, width: u32, height: u32) -> Vec<f32> {
    let mut acc = vec![0.0f32; (width as usize) * (height as usize) * 4];
    for s in &doc.shapes {
        rasterize_shape(&mut acc, s, width, height);
    }
    acc
}

/// Outcome of a place: the rasterized layer pixels plus a layer name.
pub struct Placed {
    /// Full-canvas linear-premultiplied f32 RGBA (`width*height*4`).
    pub pixels: Vec<f32>,
    /// Suggested layer name (derived from the file stem).
    pub name: String,
}

/// Read, parse, and rasterize a `.contour` file at the current document size.
/// Returns the placed layer pixels + name, or an error on read/parse failure.
pub fn place(path: &Path, width: u32, height: u32) -> anyhow::Result<Placed> {
    let doc = parse(path)?;
    let pixels = rasterize_doc(&doc, width, height);
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "Contour".to_string());
    Ok(Placed { pixels, name })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_external_tagged_shapes() {
        let json = r#"{"shapes":[
            {"Rect":{"rect":[0.0,0.0,4.0,4.0],"fill":[1.0,0.0,0.0,1.0],"stroke":[0.0,0.0,0.0,0.0],"stroke_w":0.0}},
            {"Line":{"p0":[0.0,0.0],"p1":[4.0,4.0],"stroke":[0.0,0.0,0.0,1.0],"stroke_w":2.0}}
        ]}"#;
        let doc: ContourDoc = serde_json::from_str(json).unwrap();
        assert_eq!(doc.shapes.len(), 2);
    }

    #[test]
    fn tolerates_unknown_top_level_fields() {
        let json = r#"{"shapes":[],"artboard":[100,100],"future":true}"#;
        let doc: ContourDoc = serde_json::from_str(json).unwrap();
        assert!(doc.shapes.is_empty());
    }

    #[test]
    fn rect_fill_lands_in_buffer() {
        let json = r#"{"shapes":[{"Rect":{"rect":[0.0,0.0,8.0,8.0],"fill":[1.0,1.0,1.0,1.0],"stroke":[0,0,0,0],"stroke_w":0.0}}]}"#;
        let doc: ContourDoc = serde_json::from_str(json).unwrap();
        let buf = rasterize_doc(&doc, 8, 8);
        // Center pixel alpha should be ~1.
        let idx = (4 * 8 + 4) * 4;
        assert!(buf[idx + 3] > 0.99, "center alpha {}", buf[idx + 3]);
    }

    #[test]
    fn closed_path_fills() {
        // A triangle covering most of an 8x8 buffer.
        let json = r#"{"shapes":[{"Path":{"points":[[0.0,0.0],[7.0,0.0],[0.0,7.0]],"closed":true,"fill":[1.0,1.0,1.0,1.0],"stroke":[0,0,0,0],"stroke_w":0.0}}]}"#;
        let doc: ContourDoc = serde_json::from_str(json).unwrap();
        let buf = rasterize_doc(&doc, 8, 8);
        // Point near the top-left corner (x=1, y=1) is inside the triangle.
        let idx = (8 + 1) * 4;
        assert!(buf[idx + 3] > 0.5, "inside-triangle alpha {}", buf[idx + 3]);
    }
}
