//! Camera Raw develop math — the pure, GPU-free reference for the **Camera Raw**
//! smart filter (a single re-editable filter that bundles the RAW develop
//! controls: white balance, tone, vibrance/saturation, and vignette).
//!
//! This module owns the *per-pixel* color/tone math (white balance → exposure →
//! contrast → tonal regions → vibrance/saturation) as a pure function so it can
//! be unit-tested bit-for-bit independently of wgpu. The vignette is positional
//! (it depends on the pixel's distance from the canvas center, not its colour)
//! so it is **not** part of this per-pixel reference — it is applied in the
//! shader and covered by a GPU pixel test. The shader (`filter.wgsl` kind 32)
//! mirrors the order and constants here exactly.
//!
//! ## Parameters & ranges (all default to 0 → a true no-op)
//! - `temp` `-100..=100`  white balance: + warms (boost R, cut B), − cools.
//! - `tint` `-100..=100`  white balance: + magenta (boost R&B), − green.
//! - `exposure` `-5..=5` EV: linear gain of `2^exposure`.
//! - `contrast` `-100..=100`: S-curve pivoting around mid display gray (0.5).
//! - `highlights` `-100..=100`: lift/drop the upper tones.
//! - `shadows` `-100..=100`: lift/drop the lower tones.
//! - `whites` `-100..=100`: push the extreme highlights.
//! - `blacks` `-100..=100`: push the extreme shadows.
//! - `vibrance` `-100..=100`: saturation weighted toward less-saturated pixels.
//! - `saturation` `-100..=100`: uniform saturation.
//! - `vignette` `-100..=100`: − darkens / + lightens the corners (positional —
//!   applied in the shader, see module note).
//!
//! ## Color space
//! White balance and exposure are applied in **linear light** (the pipeline's
//! working space); contrast and the tonal-region / saturation controls operate
//! in **display (sRGB) space** so the tones and the pivot land where the user
//! sees them (matching Posterize/Threshold). The reference unpremultiplies,
//! runs the pipeline on straight RGB, and the caller re-premultiplies.

/// The Camera Raw develop parameters, in UI units. `Default` is the identity
/// (every control at 0), which the pipeline guarantees is an exact no-op.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CameraRaw {
    pub temp: f32,
    pub tint: f32,
    pub exposure: f32,
    pub contrast: f32,
    pub highlights: f32,
    pub shadows: f32,
    pub whites: f32,
    pub blacks: f32,
    pub vibrance: f32,
    pub saturation: f32,
    pub vignette: f32,
}

impl Default for CameraRaw {
    fn default() -> Self {
        Self {
            temp: 0.0,
            tint: 0.0,
            exposure: 0.0,
            contrast: 0.0,
            highlights: 0.0,
            shadows: 0.0,
            whites: 0.0,
            blacks: 0.0,
            vibrance: 0.0,
            saturation: 0.0,
            vignette: 0.0,
        }
    }
}

impl CameraRaw {
    /// Pack the eleven controls into the serialized `[f32; 12]` overflow slot
    /// (last slot reserved / 0). Order matches [`Self::from_ext`].
    pub(crate) fn to_ext(self) -> [f32; 12] {
        [
            self.temp,
            self.tint,
            self.exposure,
            self.contrast,
            self.highlights,
            self.shadows,
            self.whites,
            self.blacks,
            self.vibrance,
            self.saturation,
            self.vignette,
            0.0,
        ]
    }

    /// Reconstruct from the serialized `[f32; 12]` overflow slot.
    pub(crate) fn from_ext(e: [f32; 12]) -> Self {
        Self {
            temp: e[0],
            tint: e[1],
            exposure: e[2],
            contrast: e[3],
            highlights: e[4],
            shadows: e[5],
            whites: e[6],
            blacks: e[7],
            vibrance: e[8],
            saturation: e[9],
            vignette: e[10],
        }
    }

    /// The eleven controls as the shader's `cr[12]` uniform payload (same packing
    /// as [`Self::to_ext`]). The shader reads vignette from slot 10.
    pub(crate) fn shader_params(&self) -> [f32; 12] {
        self.to_ext()
    }

    /// True if every control is at its identity (0) value.
    #[allow(dead_code)]
    pub(crate) fn is_identity(&self) -> bool {
        *self == CameraRaw::default()
    }
}

// The per-pixel develop reference and its helpers below are the pure CPU mirror
// of the `filter.wgsl` kind-32 pass, exercised by the unit tests; they have no
// non-test caller (the live pipeline runs the shader), hence the `dead_code`
// allowance.

// sRGB transfer pair, identical to `filter.wgsl` / prism-color so the GPU pass
// and this reference agree to the bit.
#[allow(dead_code)]
fn l2s1(c: f32) -> f32 {
    if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}
#[allow(dead_code)]
fn s2l1(c: f32) -> f32 {
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Rec.709 luma of a *display-space* (sRGB) RGB triple.
#[allow(dead_code)]
fn luma709(r: f32, g: f32, b: f32) -> f32 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

/// `smoothstep(0,1,t)` — the classic Hermite ramp, clamped.
#[allow(dead_code)]
fn smoothstep01(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Apply the per-pixel Camera Raw develop pipeline to a **straight (un-premult.)
/// linear** RGB pixel, returning straight linear RGB. The vignette is *not*
/// applied here (it is positional — see the module note). Identity params return
/// the input unchanged (to the bit).
///
/// Pipeline order (documented in the module header): white balance → exposure
/// (both in linear) → [to display] → contrast → highlights/shadows/whites/blacks
/// → vibrance/saturation → [back to linear].
#[allow(dead_code)]
pub(crate) fn develop_pixel(cr: &CameraRaw, rgb: [f32; 3]) -> [f32; 3] {
    let [mut r, mut g, mut b] = rgb;

    // ---- White balance (linear). temp warms R / cools B; tint trades green vs
    // magenta. Symmetric gains around 1.0 so 0 is exact identity.
    if cr.temp != 0.0 || cr.tint != 0.0 {
        let t = cr.temp / 100.0; // -1..1
        let ti = cr.tint / 100.0; // -1..1
        let gr = 1.0 + 0.30 * t + 0.15 * ti; // red
        let gg = 1.0 - 0.15 * ti; // green (tint only)
        let gb = 1.0 - 0.30 * t + 0.15 * ti; // blue
        r *= gr;
        g *= gg;
        b *= gb;
    }

    // ---- Exposure (linear): EV gain of 2^exposure.
    if cr.exposure != 0.0 {
        let gain = 2f32.powf(cr.exposure);
        r *= gain;
        g *= gain;
        b *= gain;
    }

    // Move to display (sRGB) space for the tonal + saturation controls so the
    // pivots and regions land where the user sees them. Clamp into [0,1] first
    // (exposure / WB can push beyond range).
    let mut dr = l2s1(r.clamp(0.0, 1.0));
    let mut dg = l2s1(g.clamp(0.0, 1.0));
    let mut db = l2s1(b.clamp(0.0, 1.0));

    // ---- Contrast: S-curve about mid-gray (0.5). amount -1..1.
    if cr.contrast != 0.0 {
        let c = cr.contrast / 100.0;
        // slope > 1 steepens, < 1 flattens; keep it positive for c in (-1,1).
        let slope = if c >= 0.0 { 1.0 + c } else { 1.0 + c * 0.5 };
        let f = |v: f32| (0.5 + (v - 0.5) * slope).clamp(0.0, 1.0);
        dr = f(dr);
        dg = f(dg);
        db = f(db);
    }

    // ---- Tonal regions (display space), applied on the per-pixel luma weight so
    // colour is preserved (scale the triple by a per-pixel gain). Each region has
    // a smooth weight that is 0 outside its tonal band, so 0 amount is identity.
    if cr.highlights != 0.0
        || cr.shadows != 0.0
        || cr.whites != 0.0
        || cr.blacks != 0.0
    {
        let y = luma709(dr, dg, db).clamp(0.0, 1.0);
        // Region weights: shadows/blacks weight the dark end, highlights/whites
        // the bright end; whites/blacks bite hardest at the extremes.
        let w_high = smoothstep01((y - 0.3) / 0.7); // ramps up above ~0.3
        let w_shad = smoothstep01((0.7 - y) / 0.7); // ramps up below ~0.7
        let w_white = smoothstep01((y - 0.6) / 0.4); // extreme highlights
        let w_black = smoothstep01((0.4 - y) / 0.4); // extreme shadows
        let delta = 0.5
            * (cr.highlights / 100.0 * w_high
                + cr.shadows / 100.0 * w_shad
                + cr.whites / 100.0 * w_white
                + cr.blacks / 100.0 * w_black);
        // Apply as an additive luma shift, distributed to keep hue: scale toward
        // white for + and toward black for −, proportional to the channel.
        let ny = (y + delta).clamp(0.0, 1.0);
        let gain = if y > 1e-4 { ny / y } else { 1.0 + delta };
        dr = (dr * gain).clamp(0.0, 1.0);
        dg = (dg * gain).clamp(0.0, 1.0);
        db = (db * gain).clamp(0.0, 1.0);
    }

    // ---- Vibrance + Saturation (display space). Saturation scales chroma
    // uniformly; vibrance scales it more for low-saturation pixels (protecting
    // already-saturated colours). Both 0 → identity.
    if cr.vibrance != 0.0 || cr.saturation != 0.0 {
        let y = luma709(dr, dg, db);
        let max = dr.max(dg).max(db);
        let min = dr.min(dg).min(db);
        let sat = max - min; // 0..1 rough saturation
        let sat_amt = cr.saturation / 100.0;
        // Vibrance: weight by (1 - current saturation) so muted pixels move more.
        let vib_amt = cr.vibrance / 100.0 * (1.0 - sat);
        let scale = 1.0 + sat_amt + vib_amt;
        dr = (y + (dr - y) * scale).clamp(0.0, 1.0);
        dg = (y + (dg - y) * scale).clamp(0.0, 1.0);
        db = (y + (db - y) * scale).clamp(0.0, 1.0);
    }

    [s2l1(dr), s2l1(dg), s2l1(db)]
}

/// The positional vignette gain at a point, for the GPU/CPU reference. `dist01`
/// is the normalized distance from the canvas center (0 at center, 1 at the
/// farthest corner). `vignette` is the UI control (-100..100): negative darkens
/// the corners, positive lightens them; the center is always unchanged. 0 → 1.0.
#[allow(dead_code)]
pub(crate) fn vignette_gain(vignette: f32, dist01: f32) -> f32 {
    if vignette == 0.0 {
        return 1.0;
    }
    let v = vignette / 100.0; // -1..1
    // A smooth falloff that is 1.0 at the center and reaches its full effect at
    // the corner. `1 + v * d²` keeps the center pinned (d=0 → 1) and ramps
    // quadratically outward.
    let d = dist01.clamp(0.0, 1.0);
    (1.0 + v * d * d).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MID: [f32; 3] = {
        // linear value whose sRGB is ~0.5 (mid display gray).
        [0.214_041_14, 0.214_041_14, 0.214_041_14]
    };

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    // Identity params are a true per-pixel no-op (to the bit, for all channels).
    #[test]
    fn identity_is_noop() {
        let cr = CameraRaw::default();
        assert!(cr.is_identity());
        for px in [[0.0; 3], [1.0; 3], MID, [0.1, 0.5, 0.9]] {
            let out = develop_pixel(&cr, px);
            for k in 0..3 {
                assert!(
                    approx(out[k], px[k], 1e-6),
                    "identity changed ch{k}: {} -> {}",
                    px[k],
                    out[k]
                );
            }
        }
    }

    // Positive exposure brightens; negative darkens (linear gain).
    #[test]
    fn exposure_brightens_and_darkens() {
        let up = develop_pixel(
            &CameraRaw {
                exposure: 1.0,
                ..Default::default()
            },
            MID,
        );
        let down = develop_pixel(
            &CameraRaw {
                exposure: -1.0,
                ..Default::default()
            },
            MID,
        );
        assert!(up[0] > MID[0] + 1e-3, "exposure +1 brightens: {}", up[0]);
        assert!(down[0] < MID[0] - 1e-3, "exposure -1 darkens: {}", down[0]);
        // +1 EV is ~2x the linear midpoint (clamped if it overflows, but MID*2 < 1).
        assert!(approx(up[0], MID[0] * 2.0, 1e-3));
    }

    // Contrast pivots about mid-gray: above-mid brightens, below-mid darkens,
    // and mid-gray stays put.
    #[test]
    fn contrast_pivots_about_mid() {
        let cr = CameraRaw {
            contrast: 50.0,
            ..Default::default()
        };
        // Mid-gray is the pivot — unchanged (within rounding through the transfer).
        let mid = develop_pixel(&cr, MID);
        assert!(approx(mid[0], MID[0], 2e-3), "mid pivot fixed: {}", mid[0]);
        // A bright pixel gets brighter, a dark pixel darker.
        let bright_in = [s2l1(0.75); 3];
        let dark_in = [s2l1(0.25); 3];
        let bright = develop_pixel(&cr, bright_in);
        let dark = develop_pixel(&cr, dark_in);
        assert!(bright[0] > bright_in[0], "contrast brightens highs");
        assert!(dark[0] < dark_in[0], "contrast darkens lows");
    }

    // Temperature warms (R up, B down) for +temp; the green channel is unmoved by
    // temp alone.
    #[test]
    fn temperature_warms_red_cools_blue() {
        let warm = develop_pixel(
            &CameraRaw {
                temp: 60.0,
                ..Default::default()
            },
            MID,
        );
        assert!(warm[0] > MID[0], "temp+ raises red: {}", warm[0]);
        assert!(warm[2] < MID[2], "temp+ lowers blue: {}", warm[2]);
        assert!(approx(warm[1], MID[1], 1e-4), "temp leaves green: {}", warm[1]);
    }

    // Tint trades green vs magenta: +tint raises R & B, lowers G.
    #[test]
    fn tint_shifts_green_magenta() {
        let mag = develop_pixel(
            &CameraRaw {
                tint: 60.0,
                ..Default::default()
            },
            MID,
        );
        assert!(mag[0] > MID[0] && mag[2] > MID[2], "tint+ raises R&B");
        assert!(mag[1] < MID[1], "tint+ lowers green");
    }

    // Highlights affect bright tones much more than dark ones; shadows the
    // reverse. Each lands in its own tonal region.
    #[test]
    fn highlights_and_shadows_hit_their_ranges() {
        let bright_in = [s2l1(0.85); 3];
        let dark_in = [s2l1(0.15); 3];
        // -highlights pulls the bright tone down a lot, the dark tone barely.
        let cr_h = CameraRaw {
            highlights: -80.0,
            ..Default::default()
        };
        let db = develop_pixel(&cr_h, bright_in)[0];
        let dd = develop_pixel(&cr_h, dark_in)[0];
        assert!(bright_in[0] - db > dark_in[0] - dd, "highlights bite the brights");
        assert!(db < bright_in[0], "highlights- darkens brights");
        // +shadows lifts the dark tone a lot, the bright tone barely.
        let cr_s = CameraRaw {
            shadows: 80.0,
            ..Default::default()
        };
        let sd = develop_pixel(&cr_s, dark_in)[0];
        let sb = develop_pixel(&cr_s, bright_in)[0];
        assert!(sd - dark_in[0] > sb - bright_in[0], "shadows lift the darks");
        assert!(sd > dark_in[0], "shadows+ brightens darks");
    }

    // Whites bite the extreme highlights, blacks the extreme shadows.
    #[test]
    fn whites_and_blacks_hit_extremes() {
        let near_white = [s2l1(0.95); 3];
        let near_black = [s2l1(0.05); 3];
        let mid = MID;
        let cr_w = CameraRaw {
            whites: -80.0,
            ..Default::default()
        };
        let dw = near_white[0] - develop_pixel(&cr_w, near_white)[0];
        let dm = mid[0] - develop_pixel(&cr_w, mid)[0];
        assert!(dw > dm, "whites move the extreme highlight more than mid");
        let cr_b = CameraRaw {
            blacks: 80.0,
            ..Default::default()
        };
        let dbk = develop_pixel(&cr_b, near_black)[0] - near_black[0];
        let dbm = develop_pixel(&cr_b, mid)[0] - mid[0];
        assert!(dbk > dbm, "blacks move the extreme shadow more than mid");
    }

    // Saturation pushes a colour away from / toward its gray; vibrance does too
    // but weighted toward less-saturated colours.
    #[test]
    fn saturation_and_vibrance_behave() {
        let muted = [s2l1(0.5), s2l1(0.45), s2l1(0.4)]; // low chroma
        let vivid = [s2l1(0.9), s2l1(0.2), s2l1(0.2)]; // high chroma
        // +saturation widens the channel spread of a colour.
        let spread = |p: [f32; 3]| p.iter().cloned().fold(f32::MIN, f32::max)
            - p.iter().cloned().fold(f32::MAX, f32::min);
        let sat = develop_pixel(
            &CameraRaw {
                saturation: 60.0,
                ..Default::default()
            },
            muted,
        );
        assert!(spread(sat) > spread(muted), "saturation+ widens chroma");
        // -saturation collapses toward gray.
        let desat = develop_pixel(
            &CameraRaw {
                saturation: -100.0,
                ..Default::default()
            },
            vivid,
        );
        assert!(spread(desat) < spread(vivid) - 0.1, "saturation- mutes");
        // Vibrance favours the muted colour: the *fractional* chroma boost is
        // larger for the muted pixel than the already-vivid one (absolute spread
        // change can be larger on the vivid pixel simply because its chroma is
        // bigger — the protection is in the scale factor, hence the ratio).
        let vib = CameraRaw {
            vibrance: 80.0,
            ..Default::default()
        };
        let rm = spread(develop_pixel(&vib, muted)) / spread(muted);
        let rv = spread(develop_pixel(&vib, vivid)) / spread(vivid);
        assert!(rm > rv, "vibrance favours the muted colour: {rm} vs {rv}");
    }

    // Vignette: negative darkens the corner (gain < 1) more than the center
    // (gain == 1 at d=0); positive lightens it. Zero is identity everywhere.
    #[test]
    fn vignette_darkens_corners_more_than_center() {
        // Identity.
        assert!(approx(vignette_gain(0.0, 1.0), 1.0, 1e-9));
        // Center pinned regardless of amount.
        assert!(approx(vignette_gain(-80.0, 0.0), 1.0, 1e-9));
        assert!(approx(vignette_gain(80.0, 0.0), 1.0, 1e-9));
        // Negative: corner darker than mid-radius, both < center.
        let corner = vignette_gain(-80.0, 1.0);
        let midr = vignette_gain(-80.0, 0.5);
        assert!(corner < midr, "corner darker than mid radius: {corner} {midr}");
        assert!(midr < 1.0, "mid radius darkened: {midr}");
        // Positive lightens the corner.
        assert!(vignette_gain(80.0, 1.0) > 1.0);
    }

    // ext round-trips every control losslessly.
    #[test]
    fn ext_round_trips() {
        let cr = CameraRaw {
            temp: 10.0,
            tint: -5.0,
            exposure: 0.5,
            contrast: 25.0,
            highlights: -30.0,
            shadows: 40.0,
            whites: 12.0,
            blacks: -8.0,
            vibrance: 60.0,
            saturation: -20.0,
            vignette: -45.0,
        };
        assert_eq!(CameraRaw::from_ext(cr.to_ext()), cr);
    }
}
