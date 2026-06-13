//! Pattern fill: capture the current selection's pixels (or the whole layer if
//! no selection) as a repeatable **tile**, then **fill** the active layer (or
//! selection) with that tile **tiled** across the region, with **scale** and
//! **offset** controls.
//!
//! The pixel read-back / upload follows the same destructive CPU path as the
//! gradient fill ([`super::PigmentApp::do_gradient`]): `begin_command → read
//! layer → blend the tiled pattern over the layer (selection-gated, source-over)
//! → upload`, giving region-COW undo for free. Pixels are linear **premultiplied
//! RGBA** `f32`, the compositor's working space, so blending is a plain
//! source-over.
//!
//! The tiling/coverage math — mapping a destination pixel to a pattern texel via
//! scale + offset + wrap — is the pure, testable [`pattern_texel`] below.

use super::*;

/// An in-memory pattern tile captured from a selection (or whole layer). Pixels
/// are row-major `w*h` linear **premultiplied** RGBA `f32` — the same working
/// space as the layers, so a captured tile re-fills identically.
#[derive(Clone)]
pub(crate) struct Pattern {
    pub w: u32,
    pub h: u32,
    pub px: Vec<f32>, // w*h*4, linear premultiplied RGBA
}

/// Map a destination pixel `(dx, dy)` (doc px) to the integer pattern texel it
/// samples, given a tile of size `tw × th`, a `scale` (>0; the tile is drawn
/// `scale×` its native size) and an `offset` (doc px, shifts the tile origin).
///
/// Pure and total: the result always lands inside `0..tw × 0..th` via Euclidean
/// wrap (`rem_euclid`), so the tile repeats infinitely in every direction
/// regardless of where the destination region sits. Returns `(0, 0)` for a
/// degenerate (zero-sized) tile.
pub(crate) fn pattern_texel(
    dx: u32,
    dy: u32,
    tw: u32,
    th: u32,
    scale: f32,
    offset: (f32, f32),
) -> (u32, u32) {
    if tw == 0 || th == 0 {
        return (0, 0);
    }
    let s = if scale > 1e-6 { scale } else { 1e-6 };
    // Destination → un-scaled tile space, then shift by the (un-scaled) offset.
    let fx = (dx as f32 - offset.0) / s;
    let fy = (dy as f32 - offset.1) / s;
    let tx = (fx.floor() as i64).rem_euclid(tw as i64) as u32;
    let ty = (fy.floor() as i64).rem_euclid(th as i64) as u32;
    (tx, ty)
}

/// Tile `pattern` across a `w × h` destination, blending it (source-over) over
/// `base` (linear premultiplied RGBA, mutated in place). `sel` is an optional
/// canvas-sized 0..1 coverage mask gating the write; `scale`/`offset` control
/// the tiling. Pure (no GPU) so the fill is unit-testable end to end.
pub(crate) fn tile_fill(
    base: &mut [f32],
    pattern: &Pattern,
    w: u32,
    h: u32,
    scale: f32,
    offset: (f32, f32),
    sel: Option<&[f32]>,
) {
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            let clip = sel.map(|s| s[i]).unwrap_or(1.0);
            if clip <= 0.0 {
                continue;
            }
            let (tx, ty) = pattern_texel(x, y, pattern.w, pattern.h, scale, offset);
            let pi = ((ty * pattern.w + tx) * 4) as usize;
            let sa = pattern.px[pi + 3] * clip;
            for c in 0..4 {
                base[i * 4 + c] = pattern.px[pi + c] * clip + base[i * 4 + c] * (1.0 - sa);
            }
        }
    }
}

impl PigmentApp {
    /// Capture the active layer's pixels inside the selection's bounding box as
    /// the current pattern tile; with no selection, the whole layer becomes the
    /// tile. Session-scoped (not persisted to `.pigment` — re-define after a
    /// reload). No-op if the layer can't be read or the region is empty.
    pub(crate) fn define_pattern(&mut self, frame: &mut eframe::Frame) {
        let (w, h) = (self.doc.size.width, self.doc.size.height);
        let active = self.active_id();
        let sel = if self.selection_active {
            Some(self.read_selection(frame))
        } else {
            None
        };
        let px = with_gpu(frame, |gpu, d, q| {
            gpu.read_layer(d, q, active).map(|b| f16_bytes_to_f32(&b))
        })
        .flatten();
        let Some(px) = px else { return };

        // Bounding box of the selection (whole canvas if none).
        let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0u32, 0u32);
        if let Some(s) = &sel {
            for y in 0..h {
                for x in 0..w {
                    if s[(y * w + x) as usize] > 0.0 {
                        x0 = x0.min(x);
                        y0 = y0.min(y);
                        x1 = x1.max(x + 1);
                        y1 = y1.max(y + 1);
                    }
                }
            }
            if x1 <= x0 || y1 <= y0 {
                return; // empty selection
            }
        } else {
            x0 = 0;
            y0 = 0;
            x1 = w;
            y1 = h;
        }

        let (tw, th) = (x1 - x0, y1 - y0);
        let mut tile = vec![0.0f32; (tw * th * 4) as usize];
        for ty in 0..th {
            for tx in 0..tw {
                let src = ((y0 + ty) * w + (x0 + tx)) as usize;
                // Premultiply by the selection coverage so a soft/partial
                // selection captures a correspondingly faded tile.
                let clip = sel
                    .as_ref()
                    .map(|s| s[src])
                    .unwrap_or(1.0);
                let di = ((ty * tw + tx) * 4) as usize;
                for c in 0..4 {
                    tile[di + c] = px[src * 4 + c] * clip;
                }
            }
        }
        self.pattern = Some(Pattern {
            w: tw,
            h: th,
            px: tile,
        });
    }

    /// Fill the active layer with the defined pattern, tiled across the canvas
    /// (or clipped to the active selection), with the current scale + offset.
    /// Source-over the existing pixels; region-COW undo. No-op without a pattern.
    pub(crate) fn do_pattern_fill(&mut self, frame: &mut eframe::Frame) {
        let Some(pattern) = self.pattern.clone() else {
            return;
        };
        let (w, h) = (self.doc.size.width, self.doc.size.height);
        let active = self.active_id();
        let sel = if self.selection_active {
            Some(self.read_selection(frame))
        } else {
            None
        };
        let scale = self.pattern_scale;
        let offset = self.pattern_offset;
        with_gpu(frame, |gpu, d, q| {
            gpu.begin_command_now(d, q, active, "Pattern Fill");
            let Some(b) = gpu.read_layer(d, q, active) else {
                return;
            };
            let mut base = f16_bytes_to_f32(&b);
            tile_fill(&mut base, &pattern, w, h, scale, offset, sel.as_deref());
            gpu.upload_layer(q, active, &f32_to_f16_bytes(&base));
        });
        self.force_composite = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texel_wraps_beyond_tile_size() {
        // A 4×4 tile at scale 1, no offset: dest wraps mod 4.
        assert_eq!(pattern_texel(0, 0, 4, 4, 1.0, (0.0, 0.0)), (0, 0));
        assert_eq!(pattern_texel(3, 3, 4, 4, 1.0, (0.0, 0.0)), (3, 3));
        assert_eq!(pattern_texel(4, 5, 4, 4, 1.0, (0.0, 0.0)), (0, 1));
        assert_eq!(pattern_texel(9, 4, 4, 4, 1.0, (0.0, 0.0)), (1, 0));
    }

    #[test]
    fn texel_offset_shifts_origin() {
        // Offset of (2,0) moves the tile right by 2: dest 2 → texel 0.
        assert_eq!(pattern_texel(2, 0, 4, 4, 1.0, (2.0, 0.0)), (0, 0));
        assert_eq!(pattern_texel(1, 0, 4, 4, 1.0, (2.0, 0.0)), (3, 0)); // -1 wraps to 3
    }

    #[test]
    fn texel_scale_zooms_tile() {
        // Scale 2: each tile texel covers 2 dest px, so dest 0,1 → texel 0.
        assert_eq!(pattern_texel(0, 0, 4, 4, 2.0, (0.0, 0.0)), (0, 0));
        assert_eq!(pattern_texel(1, 0, 4, 4, 2.0, (0.0, 0.0)), (0, 0));
        assert_eq!(pattern_texel(2, 0, 4, 4, 2.0, (0.0, 0.0)), (1, 0));
        assert_eq!(pattern_texel(8, 0, 4, 4, 2.0, (0.0, 0.0)), (0, 0)); // wraps
    }

    #[test]
    fn texel_degenerate_tile_is_origin() {
        assert_eq!(pattern_texel(5, 5, 0, 0, 1.0, (0.0, 0.0)), (0, 0));
    }

    /// A 2×2 opaque checker tile (white / black) tiled over a 4×4 destination
    /// reproduces the checker exactly.
    #[test]
    fn checker_tile_reproduces_checker() {
        // 2×2 premultiplied opaque checker: white where (x+y) even, else black.
        let white = [1.0, 1.0, 1.0, 1.0];
        let black = [0.0, 0.0, 0.0, 1.0];
        let mut px = vec![0.0f32; 2 * 2 * 4];
        for y in 0..2u32 {
            for x in 0..2u32 {
                let c = if (x + y) % 2 == 0 { white } else { black };
                let i = ((y * 2 + x) * 4) as usize;
                px[i..i + 4].copy_from_slice(&c);
            }
        }
        let pat = Pattern { w: 2, h: 2, px };
        let mut base = vec![0.0f32; 4 * 4 * 4];
        tile_fill(&mut base, &pat, 4, 4, 1.0, (0.0, 0.0), None);
        for y in 0..4u32 {
            for x in 0..4u32 {
                let i = ((y * 4 + x) * 4) as usize;
                let expect = if (x + y) % 2 == 0 { 1.0 } else { 0.0 };
                assert!(
                    (base[i] - expect).abs() < 1e-6,
                    "checker mismatch at ({x},{y}): got {} want {expect}",
                    base[i]
                );
                assert!((base[i + 3] - 1.0).abs() < 1e-6); // opaque everywhere
            }
        }
    }

    /// The selection mask gates the fill: pixels outside the selection are left
    /// untouched, pixels inside are overwritten by the (opaque) pattern.
    #[test]
    fn selection_gates_fill() {
        let pat = Pattern {
            w: 1,
            h: 1,
            px: vec![1.0, 0.0, 0.0, 1.0], // opaque red
        };
        let (w, h) = (3u32, 2u32);
        let mut base = vec![0.5f32; (w * h * 4) as usize]; // gray everywhere
        // Select only the left column (x==0).
        let mut sel = vec![0.0f32; (w * h) as usize];
        for y in 0..h {
            sel[(y * w) as usize] = 1.0;
        }
        tile_fill(&mut base, &pat, w, h, 1.0, (0.0, 0.0), Some(&sel));
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                if x == 0 {
                    assert!((base[i] - 1.0).abs() < 1e-6); // red R inside
                    assert!(base[i + 1].abs() < 1e-6);
                } else {
                    assert!((base[i] - 0.5).abs() < 1e-6); // untouched outside
                    assert!((base[i + 1] - 0.5).abs() < 1e-6);
                }
            }
        }
    }

    /// Partial selection coverage blends proportionally (source-over with the
    /// coverage scaling the pattern's alpha).
    #[test]
    fn partial_coverage_blends() {
        let pat = Pattern {
            w: 1,
            h: 1,
            px: vec![1.0, 1.0, 1.0, 1.0], // opaque white
        };
        let mut base = vec![0.0f32, 0.0, 0.0, 1.0]; // opaque black, 1px
        let sel = vec![0.5f32];
        tile_fill(&mut base, &pat, 1, 1, 1.0, (0.0, 0.0), Some(&sel));
        // 0.5 coverage white over black → 0.5 gray, alpha stays 1.
        assert!((base[0] - 0.5).abs() < 1e-6);
        assert!((base[3] - 1.0).abs() < 1e-6);
    }
}
