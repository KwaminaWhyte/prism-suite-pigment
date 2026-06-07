//! CPU reference implementations of the blur-family filters that
//! [`filter.wgsl`](../shaders/filter.wgsl) runs on the GPU.
//!
//! These are *not* used at runtime (the live path is the GPU shader pass in
//! [`compositor.rs`](super::compositor)); they exist so the averaging math
//! (motion-blur direction/length, separable box-blur normalization, radial
//! spin vs zoom) can be unit-tested deterministically without a GPU adapter,
//! mirroring the kernel the shader implements.
//!
//! Pixels are RGBA stored as `[f32; 4]` per texel in **linear premultiplied**
//! space (the working space of the compositor), so a plain component average is
//! a correct blur. Sampling uses edge-clamped bilinear, matching the WGSL
//! `textureSample` with a clamp-to-edge sampler.

/// Edge-clamped bilinear sample at floating-point pixel coords `(x, y)`.
/// `img` is row-major `w*h` RGBA. Coordinates are in pixel space where integer
/// `i` addresses the *center* of pixel `i` at `i + 0.0` (texel centers handled
/// by the caller adding 0.5 where it matches the shader's uv convention).
pub fn sample_bilinear(img: &[[f32; 4]], w: usize, h: usize, x: f32, y: f32) -> [f32; 4] {
    if w == 0 || h == 0 {
        return [0.0; 4];
    }
    let xc = x.clamp(0.0, (w - 1) as f32);
    let yc = y.clamp(0.0, (h - 1) as f32);
    let x0 = xc.floor() as usize;
    let y0 = yc.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let fx = xc - x0 as f32;
    let fy = yc - y0 as f32;
    let p00 = img[y0 * w + x0];
    let p10 = img[y0 * w + x1];
    let p01 = img[y1 * w + x0];
    let p11 = img[y1 * w + x1];
    let mut out = [0.0f32; 4];
    for c in 0..4 {
        let top = p00[c] * (1.0 - fx) + p10[c] * fx;
        let bot = p01[c] * (1.0 - fx) + p11[c] * fx;
        out[c] = top * (1.0 - fy) + bot * fy;
    }
    out
}

/// Directional (linear / "motion") blur: average `2*radius+1` taps along the
/// unit direction `(dx, dy)`, centered on each pixel. A box average along the
/// line — the classic motion-blur kernel. `radius` is the number of taps to
/// each side (so the streak length is `2*radius+1` pixels along the axis).
pub fn motion_blur(
    img: &[[f32; 4]],
    w: usize,
    h: usize,
    angle_rad: f32,
    radius: usize,
) -> Vec<[f32; 4]> {
    let (dx, dy) = (angle_rad.cos(), angle_rad.sin());
    let r = radius as i32;
    let mut out = vec![[0.0f32; 4]; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut sum = [0.0f32; 4];
            let mut n = 0.0f32;
            for i in -r..=r {
                let sx = x as f32 + dx * i as f32;
                let sy = y as f32 + dy * i as f32;
                let s = sample_bilinear(img, w, h, sx, sy);
                for c in 0..4 {
                    sum[c] += s[c];
                }
                n += 1.0;
            }
            let p = y * w + x;
            for c in 0..4 {
                out[p][c] = sum[c] / n;
            }
        }
    }
    out
}

/// One axis of a separable box blur: average `2*radius+1` taps along
/// `(ax, ay)` (use `(1,0)` for the horizontal pass, `(0,1)` for vertical).
/// Running it horizontally then vertically yields a true box blur with a flat,
/// correctly-normalized kernel.
pub fn box_blur_axis(
    img: &[[f32; 4]],
    w: usize,
    h: usize,
    ax: i32,
    ay: i32,
    radius: usize,
) -> Vec<[f32; 4]> {
    let r = radius as i32;
    let mut out = vec![[0.0f32; 4]; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut sum = [0.0f32; 4];
            let mut n = 0.0f32;
            for i in -r..=r {
                let sx = (x as i32 + ax * i).clamp(0, w as i32 - 1) as usize;
                let sy = (y as i32 + ay * i).clamp(0, h as i32 - 1) as usize;
                let s = img[sy * w + sx];
                for c in 0..4 {
                    sum[c] += s[c];
                }
                n += 1.0;
            }
            let p = y * w + x;
            for c in 0..4 {
                out[p][c] = sum[c] / n;
            }
        }
    }
    out
}

/// Full separable box blur (horizontal then vertical pass).
pub fn box_blur(img: &[[f32; 4]], w: usize, h: usize, radius: usize) -> Vec<[f32; 4]> {
    let h_pass = box_blur_axis(img, w, h, 1, 0, radius);
    box_blur_axis(&h_pass, w, h, 0, 1, radius)
}

/// Radial blur mode.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RadialMode {
    /// Rotational blur about the center (smears tangentially).
    Spin,
    /// Zoom blur toward/away from the center (smears radially).
    Zoom,
}

/// Radial blur: for each pixel, average `samples` taps taken along an arc
/// (Spin) or a ray toward the center (Zoom). `cx,cy` is the center in pixel
/// coords; `amount` is the total spin angle in radians (Spin) or the total
/// zoom fraction (Zoom, e.g. `0.2` ≈ 20%). `samples` is the number of taps
/// (>= 1); they are spread symmetrically about the source pixel so amount 0 is
/// a no-op identity.
pub fn radial_blur(
    img: &[[f32; 4]],
    w: usize,
    h: usize,
    cx: f32,
    cy: f32,
    mode: RadialMode,
    amount: f32,
    samples: usize,
) -> Vec<[f32; 4]> {
    let samples = samples.max(1);
    let mut out = vec![[0.0f32; 4]; w * h];
    for y in 0..h {
        for x in 0..w {
            let rx = x as f32 - cx;
            let ry = y as f32 - cy;
            let mut sum = [0.0f32; 4];
            let mut n = 0.0f32;
            for k in 0..samples {
                // t in [-0.5, 0.5] so the central tap is the source pixel.
                let t = if samples == 1 {
                    0.0
                } else {
                    k as f32 / (samples - 1) as f32 - 0.5
                };
                let (sx, sy) = match mode {
                    RadialMode::Spin => {
                        let a = amount * t;
                        let (ca, sa) = (a.cos(), a.sin());
                        (cx + rx * ca - ry * sa, cy + rx * sa + ry * ca)
                    }
                    RadialMode::Zoom => {
                        let scale = 1.0 + amount * t;
                        (cx + rx * scale, cy + ry * scale)
                    }
                };
                let s = sample_bilinear(img, w, h, sx, sy);
                for c in 0..4 {
                    sum[c] += s[c];
                }
                n += 1.0;
            }
            let p = y * w + x;
            for c in 0..4 {
                out[p][c] = sum[c] / n;
            }
        }
    }
    out
}

// ===========================================================================
// Distort filters (Phase 8) — per-pixel coordinate-displacement remaps. Each
// computes a *source* pixel coordinate for every output pixel and edge-clamp
// bilinear-samples there, mirroring `filter.wgsl`'s distort branches (kinds
// 8–12). All work in pixel space about a center (cx, cy).
// ===========================================================================

const TAU: f32 = std::f32::consts::TAU;

/// Twirl: rotate the source about `(cx, cy)` by an angle that falls off
/// quadratically to 0 at `radius`. Inside the radius only; outside is identity.
/// `angle` is the maximum rotation in radians at the center.
pub fn twirl(
    img: &[[f32; 4]],
    w: usize,
    h: usize,
    cx: f32,
    cy: f32,
    angle: f32,
    radius: f32,
) -> Vec<[f32; 4]> {
    let mut out = vec![[0.0f32; 4]; w * h];
    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            let (mut sx, mut sy) = (x as f32, y as f32);
            if dist < radius && radius > 0.0 {
                let falloff = 1.0 - dist / radius;
                let a = angle * falloff * falloff;
                let (ca, sa) = (a.cos(), a.sin());
                // Inverse map: sample the source rotated by -a.
                sx = cx + dx * ca + dy * sa;
                sy = cy - dx * sa + dy * ca;
            }
            out[y * w + x] = sample_bilinear(img, w, h, sx, sy);
        }
    }
    out
}

/// Pinch (`amount` > 0, pulls toward the center) / Spherize-bulge (`amount` < 0,
/// pushes outward) about `(cx, cy)` within `radius`. Radial remap with a smooth
/// falloff to the edge of the affected disc.
pub fn pinch(
    img: &[[f32; 4]],
    w: usize,
    h: usize,
    cx: f32,
    cy: f32,
    amount: f32,
    radius: f32,
) -> Vec<[f32; 4]> {
    let mut out = vec![[0.0f32; 4]; w * h];
    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            let (mut sx, mut sy) = (x as f32, y as f32);
            if dist < radius && radius > 0.0 && dist > 1e-4 {
                let nd = dist / radius; // 0 at center .. 1 at edge
                let s = nd.powf(1.0 + amount);
                let f = s / nd;
                sx = cx + dx * f;
                sy = cy + dy * f;
            }
            out[y * w + x] = sample_bilinear(img, w, h, sx, sy);
        }
    }
    out
}

/// Ripple / Wave: sinusoidal displacement — each axis is offset by a sine of the
/// *other* axis. `amplitude` is the peak displacement in pixels; `wavelength` is
/// the period in pixels.
pub fn ripple(
    img: &[[f32; 4]],
    w: usize,
    h: usize,
    amplitude: f32,
    wavelength: f32,
) -> Vec<[f32; 4]> {
    let wl = wavelength.max(1e-3);
    let k = TAU / wl;
    let mut out = vec![[0.0f32; 4]; w * h];
    for y in 0..h {
        for x in 0..w {
            let sx = x as f32 + amplitude * (y as f32 * k).sin();
            let sy = y as f32 + amplitude * (x as f32 * k).sin();
            out[y * w + x] = sample_bilinear(img, w, h, sx, sy);
        }
    }
    out
}

/// Polar Coordinates mode.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PolarMode {
    /// Rectangular → polar: output x = angle, output y = radius.
    ToPolar,
    /// Polar → rectangular: the inverse of [`PolarMode::ToPolar`].
    ToRect,
}

/// Polar Coordinates remap about `(cx, cy)`. `ToPolar` lays the image out so the
/// horizontal axis sweeps the angle (0..2π over the width, starting at the top)
/// and the vertical axis is the radius (0 at top → `maxr` at the bottom).
/// `ToRect` is the exact inverse, so `ToRect ∘ ToPolar` is (modulo resampling)
/// the identity.
pub fn polar(
    img: &[[f32; 4]],
    w: usize,
    h: usize,
    cx: f32,
    cy: f32,
    mode: PolarMode,
) -> Vec<[f32; 4]> {
    let (wf, hf) = (w as f32, h as f32);
    let maxr = wf.min(hf) * 0.5;
    let mut out = vec![[0.0f32; 4]; w * h];
    for y in 0..h {
        for x in 0..w {
            let (sx, sy) = match mode {
                PolarMode::ToPolar => {
                    let theta = (x as f32 / wf) * TAU - std::f32::consts::FRAC_PI_2;
                    let rr = (y as f32 / hf) * maxr;
                    (cx + rr * theta.cos(), cy + rr * theta.sin())
                }
                PolarMode::ToRect => {
                    let dx = x as f32 - cx;
                    let dy = y as f32 - cy;
                    let mut theta = dy.atan2(dx) + std::f32::consts::FRAC_PI_2;
                    if theta < 0.0 {
                        theta += TAU;
                    }
                    if theta >= TAU {
                        theta -= TAU;
                    }
                    let rr = (dx * dx + dy * dy).sqrt();
                    ((theta / TAU) * wf, (rr / maxr) * hf)
                }
            };
            out[y * w + x] = sample_bilinear(img, w, h, sx, sy);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    /// A 9x9 image: a single white opaque pixel at the center, else transparent.
    fn impulse(n: usize) -> Vec<[f32; 4]> {
        let mut img = vec![[0.0f32; 4]; n * n];
        img[(n / 2) * n + n / 2] = [1.0, 1.0, 1.0, 1.0];
        img
    }

    // ---- Motion blur ------------------------------------------------------

    #[test]
    fn motion_blur_smears_along_the_angle_only() {
        let n = 9;
        let img = impulse(n);
        let r = 2;
        // Horizontal streak (angle 0): the impulse spreads left/right, not up/down.
        let out = motion_blur(&img, n, n, 0.0, r);
        let c = n / 2;
        let at = |x: usize, y: usize| out[y * n + x][3];
        // Center and the two horizontal neighbours each pick up the impulse
        // (1 tap of 5) — alpha ≈ 1/5.
        assert!(approx(at(c, c), 0.2, 1e-4), "center {}", at(c, c));
        assert!(approx(at(c - 1, c), 0.2, 1e-4), "left");
        assert!(approx(at(c + 1, c), 0.2, 1e-4), "right");
        assert!(approx(at(c + 2, c), 0.2, 1e-4), "far right edge of streak");
        // Off-axis (vertical neighbours) stay empty.
        assert!(approx(at(c, c - 1), 0.0, 1e-4), "up should be untouched");
        assert!(approx(at(c, c + 1), 0.0, 1e-4), "down should be untouched");
    }

    #[test]
    fn motion_blur_conserves_energy_and_normalizes() {
        let n = 11;
        let img = impulse(n);
        let r = 3;
        let out = motion_blur(&img, n, n, 0.0, r);
        // Each tap averages 1/(2r+1); along a clear horizontal line through the
        // center, the visible taps sum back to ~1 (edge-clamp aside, center is
        // far from the border here).
        let c = n / 2;
        let line_sum: f32 = (0..n).map(|x| out[c * n + x][3]).sum();
        assert!(approx(line_sum, 1.0, 1e-3), "energy conserved: {line_sum}");
        // No single output exceeds the per-tap weight.
        let w = 1.0 / (2.0 * r as f32 + 1.0);
        assert!(out.iter().all(|p| p[3] <= w + 1e-4));
    }

    #[test]
    fn motion_blur_radius_zero_is_identity() {
        let n = 7;
        let img = impulse(n);
        let out = motion_blur(&img, n, n, 1.3, 0);
        assert_eq!(out, img, "radius 0 must be a no-op");
    }

    #[test]
    fn motion_blur_vertical_angle_smears_vertically() {
        let n = 9;
        let img = impulse(n);
        let out = motion_blur(&img, n, n, std::f32::consts::FRAC_PI_2, 2);
        let c = n / 2;
        // Now the vertical neighbours pick up the impulse, horizontals don't.
        assert!(approx(out[(c - 1) * n + c][3], 0.2, 1e-4), "up");
        assert!(approx(out[(c + 1) * n + c][3], 0.2, 1e-4), "down");
        assert!(approx(out[c * n + (c + 1)][3], 0.0, 1e-4), "right empty");
    }

    // ---- Box blur ---------------------------------------------------------

    #[test]
    fn box_blur_is_separable_and_normalized() {
        // A flat field must be preserved exactly (kernel sums to 1).
        let n = 8;
        let flat = vec![[0.3f32, 0.6, 0.9, 1.0]; n * n];
        let out = box_blur(&flat, n, n, 2);
        for p in &out {
            for c in 0..4 {
                assert!(approx(p[c], flat[0][c], 1e-5), "flat preserved: {p:?}");
            }
        }
    }

    #[test]
    fn box_blur_matches_two_axis_passes() {
        // box_blur is exactly H-pass then V-pass.
        let n = 9;
        let img = impulse(n);
        let manual = box_blur_axis(&box_blur_axis(&img, n, n, 1, 0, 2), n, n, 0, 1, 2);
        let combined = box_blur(&img, n, n, 2);
        assert_eq!(manual, combined);
    }

    #[test]
    fn box_blur_impulse_spreads_into_a_square() {
        let n = 9;
        let img = impulse(n);
        let r = 1; // 3x3 box
        let out = box_blur(&img, n, n, r);
        let c = n / 2;
        // A 3x3 box of a unit impulse: each of the 9 cells gets 1/9.
        let w = 1.0 / 9.0;
        for dy in -1..=1i32 {
            for dx in -1..=1i32 {
                let x = (c as i32 + dx) as usize;
                let y = (c as i32 + dy) as usize;
                assert!(approx(out[y * n + x][3], w, 1e-4), "cell {dx},{dy}");
            }
        }
        // The corners of a 5x5 region (2 away) stay empty.
        assert!(approx(out[(c - 2) * n + (c - 2)][3], 0.0, 1e-4));
        // Total energy conserved.
        let total: f32 = out.iter().map(|p| p[3]).sum();
        assert!(approx(total, 1.0, 1e-3), "energy: {total}");
    }

    #[test]
    fn box_blur_radius_zero_is_identity() {
        let n = 6;
        let img = impulse(n);
        assert_eq!(box_blur(&img, n, n, 0), img);
    }

    // ---- Radial blur ------------------------------------------------------

    #[test]
    fn radial_amount_zero_is_identity_for_both_modes() {
        let n = 9;
        let img = impulse(n);
        let c = (n / 2) as f32;
        let spin = radial_blur(&img, n, n, c, c, RadialMode::Spin, 0.0, 8);
        let zoom = radial_blur(&img, n, n, c, c, RadialMode::Zoom, 0.0, 8);
        for i in 0..img.len() {
            for ch in 0..4 {
                assert!(approx(spin[i][ch], img[i][ch], 1e-4), "spin identity");
                assert!(approx(zoom[i][ch], img[i][ch], 1e-4), "zoom identity");
            }
        }
    }

    #[test]
    fn radial_spin_smears_tangentially_not_radially() {
        // Bright pixel offset to the right of center: spin should smear it
        // *vertically* (tangent to the circle), zoom should smear it
        // *horizontally* (along the radius).
        let n = 21;
        let c = (n / 2) as f32;
        let mut img = vec![[0.0f32; 4]; n * n];
        let bx = n / 2 + 6;
        let by = n / 2;
        img[by * n + bx] = [1.0, 1.0, 1.0, 1.0];

        let spin = radial_blur(&img, n, n, c, c, RadialMode::Spin, 0.6, 32);
        // Tangential (vertical) neighbours of the source pick up signal.
        let tang_above = spin[(by - 2) * n + bx][3] + spin[(by - 1) * n + bx][3];
        let tang_below = spin[(by + 1) * n + bx][3] + spin[(by + 2) * n + bx][3];
        // Radial (horizontal) neighbours pick up much less.
        let radial = spin[by * n + (bx - 2)][3] + spin[by * n + (bx + 2)][3];
        assert!(
            tang_above + tang_below > radial * 2.0,
            "spin tangential ({}, {}) >> radial ({radial})",
            tang_above,
            tang_below
        );
    }

    #[test]
    fn radial_zoom_smears_radially_not_tangentially() {
        let n = 21;
        let c = (n / 2) as f32;
        let mut img = vec![[0.0f32; 4]; n * n];
        let bx = n / 2 + 6;
        let by = n / 2;
        img[by * n + bx] = [1.0, 1.0, 1.0, 1.0];

        let zoom = radial_blur(&img, n, n, c, c, RadialMode::Zoom, 0.5, 32);
        // Radial (horizontal) neighbours pick up signal.
        let radial = zoom[by * n + (bx - 2)][3]
            + zoom[by * n + (bx - 1)][3]
            + zoom[by * n + (bx + 1)][3]
            + zoom[by * n + (bx + 2)][3];
        // Tangential (vertical) neighbours pick up much less.
        let tang = zoom[(by - 2) * n + bx][3] + zoom[(by + 2) * n + bx][3];
        assert!(
            radial > tang * 2.0,
            "zoom radial ({radial}) >> tangential ({tang})"
        );
    }

    #[test]
    fn radial_blur_is_deterministic() {
        let n = 13;
        let c = (n / 2) as f32;
        let img = impulse(n);
        let a = radial_blur(&img, n, n, c, c, RadialMode::Spin, 0.7, 16);
        let b = radial_blur(&img, n, n, c, c, RadialMode::Spin, 0.7, 16);
        assert_eq!(a, b, "same inputs -> identical output");
    }

    // ---- Twirl ------------------------------------------------------------

    /// A horizontal black→white ramp; sampling its color reveals which source x
    /// a remap pulled from (color encodes the x coordinate).
    fn ramp_x(n: usize) -> Vec<[f32; 4]> {
        let mut img = vec![[0.0f32; 4]; n * n];
        for y in 0..n {
            for x in 0..n {
                let v = x as f32 / (n - 1) as f32;
                img[y * n + x] = [v, v, v, 1.0];
            }
        }
        img
    }

    #[test]
    fn twirl_angle_zero_is_identity() {
        let n = 21;
        let c = (n / 2) as f32;
        let img = ramp_x(n);
        let out = twirl(&img, n, n, c, c, 0.0, 8.0);
        for i in 0..img.len() {
            for ch in 0..4 {
                assert!(approx(out[i][ch], img[i][ch], 1e-4), "twirl 0 identity");
            }
        }
    }

    #[test]
    fn twirl_leaves_pixels_outside_radius_untouched() {
        // A point well outside the twirl disc must be exactly the source.
        let n = 31;
        let c = (n / 2) as f32;
        let img = ramp_x(n);
        let out = twirl(&img, n, n, c, c, 1.2, 5.0);
        // Corner is far outside radius 5.
        let p = n + 1;
        for ch in 0..4 {
            assert!(approx(out[p][ch], img[p][ch], 1e-4), "outside radius fixed");
        }
        // The exact center is the fixed point of the rotation.
        let cc = (n / 2) * n + n / 2;
        for ch in 0..4 {
            assert!(approx(out[cc][ch], img[cc][ch], 1e-4), "center fixed");
        }
    }

    #[test]
    fn twirl_rotates_a_point_by_the_falloff_angle() {
        // On a ramp, a CCW image rotation pulls the source from a rotated-by-(−a)
        // location. Place a probe directly above center: with a positive angle,
        // the source x shifts off the column, changing the sampled value.
        let n = 41;
        let c = (n / 2) as f32;
        let img = ramp_x(n);
        let out = twirl(&img, n, n, c, c, 1.5, 18.0);
        // A pixel above the center (dx=0): rotation moves the sample sideways, so
        // its value departs from the center column value (0.5 on a symmetric ramp).
        let px = n / 2;
        let py = n / 2 - 6;
        let v = out[py * n + px][0];
        assert!(
            (v - 0.5).abs() > 0.05,
            "twirl rotated the sample off the center column: {v}"
        );
    }

    #[test]
    fn twirl_is_deterministic() {
        let n = 25;
        let c = (n / 2) as f32;
        let img = ramp_x(n);
        let a = twirl(&img, n, n, c, c, 0.9, 10.0);
        let b = twirl(&img, n, n, c, c, 0.9, 10.0);
        assert_eq!(a, b);
    }

    // ---- Pinch / Spherize -------------------------------------------------

    #[test]
    fn pinch_amount_zero_is_identity() {
        let n = 21;
        let c = (n / 2) as f32;
        let img = ramp_x(n);
        let out = pinch(&img, n, n, c, c, 0.0, 8.0);
        for i in 0..img.len() {
            for ch in 0..4 {
                assert!(approx(out[i][ch], img[i][ch], 1e-4), "pinch 0 identity");
            }
        }
    }

    #[test]
    fn pinch_and_bulge_displace_in_opposite_directions() {
        // Probe a pixel to the right of center on a left→right ramp. Pinch
        // (amount > 0) maps the *source* radius inward (smaller src radius ⇒
        // sampled value closer to the center value 0.5, i.e. *smaller* than the
        // identity value here). Bulge (amount < 0) maps it outward (larger src
        // radius ⇒ value further from center, *larger*). So the two move the
        // sampled value to opposite sides of the identity.
        let n = 41;
        let c = (n / 2) as f32;
        let img = ramp_x(n);
        let radius = 18.0;
        let px = n / 2 + 6; // right of center
        let py = n / 2;
        let idx = py * n + px;
        let identity = img[idx][0];

        let p = pinch(&img, n, n, c, c, 0.6, radius);
        let b = pinch(&img, n, n, c, c, -0.6, radius);
        let pv = p[idx][0];
        let bv = b[idx][0];
        assert!(pv < identity - 1e-3, "pinch pulls source inward: {pv}");
        assert!(bv > identity + 1e-3, "bulge pushes source outward: {bv}");
        assert!(bv > pv, "bulge and pinch are signed-opposite");
    }

    #[test]
    fn pinch_keeps_the_center_fixed() {
        let n = 21;
        let c = (n / 2) as f32;
        let img = ramp_x(n);
        let out = pinch(&img, n, n, c, c, 0.7, 9.0);
        let cc = (n / 2) * n + n / 2;
        for ch in 0..4 {
            assert!(approx(out[cc][ch], img[cc][ch], 1e-3), "center fixed");
        }
    }

    // ---- Ripple / Wave ----------------------------------------------------

    #[test]
    fn ripple_zero_amplitude_is_identity() {
        let n = 16;
        let img = ramp_x(n);
        let out = ripple(&img, n, n, 0.0, 20.0);
        for i in 0..img.len() {
            for ch in 0..4 {
                assert!(approx(out[i][ch], img[i][ch], 1e-4), "ripple 0 identity");
            }
        }
    }

    #[test]
    fn ripple_displacement_is_periodic_in_the_wavelength() {
        // The x-axis displacement is amplitude*sin(y*k); rows one wavelength
        // apart get the same displacement, so identical output rows (on a column-
        // invariant ramp the y-offset doesn't change the value, only the x-offset
        // does — and that depends on y only). Use a ramp_x so the value depends
        // purely on the (displaced) x coordinate.
        let n = 60;
        let wl = 20.0;
        let img = ramp_x(n);
        let out = ripple(&img, n, n, 6.0, wl);
        let row = |r: usize| (0..n).map(|x| out[r * n + x][0]).collect::<Vec<_>>();
        // Rows y and y+wl share the same sin(y*k) displacement → identical rows.
        let r0 = row(10);
        let r1 = row(10 + wl as usize);
        for x in 0..n {
            assert!(
                approx(r0[x], r1[x], 1e-4),
                "rows one wavelength apart match at x={x}: {} vs {}",
                r0[x],
                r1[x]
            );
        }
    }

    #[test]
    fn ripple_actually_displaces() {
        let n = 40;
        let img = ramp_x(n);
        let out = ripple(&img, n, n, 8.0, 16.0);
        // At least some pixels must differ from the source (non-trivial warp).
        let changed = (0..img.len()).any(|i| (out[i][0] - img[i][0]).abs() > 1e-3);
        assert!(changed, "ripple must move some pixels");
    }

    // ---- Polar Coordinates ------------------------------------------------

    #[test]
    fn polar_round_trip_recovers_the_interior() {
        // ToRect ∘ ToPolar should recover the original within the inscribed disc,
        // up to resampling error. Use a smooth radial test image so bilinear
        // resampling is well-behaved, and check pixels near the center.
        let n = 64;
        let c = (n / 2) as f32;
        let mut img = vec![[0.0f32; 4]; n * n];
        for y in 0..n {
            for x in 0..n {
                let dx = x as f32 - c;
                let dy = y as f32 - c;
                let r = (dx * dx + dy * dy).sqrt() / c;
                let v = (1.0 - r).clamp(0.0, 1.0); // smooth radial bump
                img[y * n + x] = [v, v, v, 1.0];
            }
        }
        let pol = polar(&img, n, n, c, c, PolarMode::ToPolar);
        let back = polar(&pol, n, n, c, c, PolarMode::ToRect);

        // Sample a ring of points at moderate radius and compare.
        let mut max_err = 0.0f32;
        let probe_r = (n as f32) * 0.25;
        for k in 0..16 {
            let a = k as f32 / 16.0 * TAU;
            let x = (c + probe_r * a.cos()).round() as usize;
            let y = (c + probe_r * a.sin()).round() as usize;
            let err = (back[y * n + x][0] - img[y * n + x][0]).abs();
            max_err = max_err.max(err);
        }
        assert!(max_err < 0.1, "polar round-trip error too large: {max_err}");
    }

    #[test]
    fn polar_to_polar_maps_radius_to_rows() {
        // In ToPolar, the top row (y=0) maps to radius 0 → the center pixel, and
        // lower rows map to larger radii. A radial bump (bright center) should
        // therefore be bright along the top and dim toward the bottom.
        let n = 48;
        let c = (n / 2) as f32;
        let mut img = vec![[0.0f32; 4]; n * n];
        for y in 0..n {
            for x in 0..n {
                let dx = x as f32 - c;
                let dy = y as f32 - c;
                let r = (dx * dx + dy * dy).sqrt() / c;
                let v = (1.0 - r).clamp(0.0, 1.0);
                img[y * n + x] = [v, v, v, 1.0];
            }
        }
        let pol = polar(&img, n, n, c, c, PolarMode::ToPolar);
        let mid = n / 2;
        let top = pol[n + mid][0];
        let bottom = pol[(n - 2) * n + mid][0];
        assert!(
            top > bottom + 0.2,
            "top (r≈0) brighter than bottom: {top} {bottom}"
        );
    }

    #[test]
    fn polar_is_deterministic() {
        let n = 32;
        let c = (n / 2) as f32;
        let img = ramp_x(n);
        let a = polar(&img, n, n, c, c, PolarMode::ToPolar);
        let b = polar(&img, n, n, c, c, PolarMode::ToPolar);
        assert_eq!(a, b);
    }
}
