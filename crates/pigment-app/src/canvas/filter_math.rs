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

/// One axis of a separable Gaussian blur (kind 1): a normalized weighted average
/// of `2·radius+1` taps along `(ax, ay)`, weight `exp(-½·i²/σ²)` with
/// `σ = max(radius·0.5, 0.5)` — bit-for-bit the WGSL Gaussian so the CPU
/// reference matches the shader's blur. Use `(1,0)` then `(0,1)` for a full blur.
pub fn gaussian_blur_axis(
    img: &[[f32; 4]],
    w: usize,
    h: usize,
    ax: i32,
    ay: i32,
    radius: f32,
) -> Vec<[f32; 4]> {
    let r = radius.clamp(0.0, 64.0) as i32;
    let sigma = (radius * 0.5).max(0.5);
    let mut out = vec![[0.0f32; 4]; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut sum = [0.0f32; 4];
            let mut wsum = 0.0f32;
            for i in -r..=r {
                let weight = (-0.5 * (i * i) as f32 / (sigma * sigma)).exp();
                let sx = (x as i32 + ax * i).clamp(0, w as i32 - 1) as usize;
                let sy = (y as i32 + ay * i).clamp(0, h as i32 - 1) as usize;
                let s = img[sy * w + sx];
                for c in 0..4 {
                    sum[c] += s[c] * weight;
                }
                wsum += weight;
            }
            let p = y * w + x;
            for c in 0..4 {
                out[p][c] = sum[c] / wsum.max(1e-5);
            }
        }
    }
    out
}

/// Full separable Gaussian blur (horizontal then vertical pass), mirroring how
/// the compositor runs the kind-1 shader twice.
pub fn gaussian_blur(img: &[[f32; 4]], w: usize, h: usize, radius: f32) -> Vec<[f32; 4]> {
    let h_pass = gaussian_blur_axis(img, w, h, 1, 0, radius);
    gaussian_blur_axis(&h_pass, w, h, 0, 1, radius)
}

/// High Pass (kind 24): the classic Photoshop sharpen prep — subtract a
/// Gaussian-blurred copy from the original and re-centre at mid-gray, so flat
/// areas go neutral gray and only the high-frequency detail/edges survive as a
/// signed deviation about 0.5. `radius` is the Gaussian blur radius; `amount`
/// scales the detail (1 = identity high pass). The difference is taken on the
/// unpremultiplied colour (so a transparent edge doesn't bias it), clamped to
/// [0,1], then re-premultiplied keeping the source alpha — matching the shader.
pub fn high_pass(img: &[[f32; 4]], w: usize, h: usize, radius: f32, amount: f32) -> Vec<[f32; 4]> {
    let blur = gaussian_blur(img, w, h, radius);
    let mut out = vec![[0.0f32; 4]; w * h];
    for i in 0..(w * h) {
        let src = img[i];
        let blr = blur[i];
        let sa = src[3].max(1e-4);
        let ba = blr[3].max(1e-4);
        let mut col = [0.0f32; 3];
        for c in 0..3 {
            let detail = src[c] / sa - blr[c] / ba;
            col[c] = (0.5 + detail * amount).clamp(0.0, 1.0);
        }
        out[i] = [col[0] * src[3], col[1] * src[3], col[2] * src[3], src[3]];
    }
    out
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

// ===========================================================================
// Stylize filters (Phase 8) — Sobel edge / relief + a seeded diffuse scramble.
// CPU references mirroring `filter.wgsl`'s kinds 13–16. All operate in the
// compositor working space: **linear premultiplied** RGBA `[f32; 4]`.
// ===========================================================================

/// Rec.709 luma of a premultiplied RGB triple.
fn luma(p: [f32; 4]) -> f32 {
    0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]
}

/// The 3×3 Sobel gradient `(gx, gy)` of the luma field at `(x, y)`, sampling a
/// `width`-px step with edge-clamped bilinear taps (matches the WGSL Sobel).
fn sobel_grad(img: &[[f32; 4]], w: usize, h: usize, x: usize, y: usize, width: f32) -> (f32, f32) {
    let (fx, fy) = (x as f32, y as f32);
    let s = width.max(1.0);
    let l = |dx: f32, dy: f32| luma(sample_bilinear(img, w, h, fx + dx * s, fy + dy * s));
    let (tl, tc, tr) = (l(-1.0, -1.0), l(0.0, -1.0), l(1.0, -1.0));
    let (ml, mr) = (l(-1.0, 0.0), l(1.0, 0.0));
    let (bl, bc, br) = (l(-1.0, 1.0), l(0.0, 1.0), l(1.0, 1.0));
    let gx = (tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl);
    let gy = (bl + 2.0 * bc + br) - (tl + 2.0 * tc + tr);
    (gx, gy)
}

/// Find Edges: the Sobel gradient magnitude inverted to a white background with
/// dark edges (Photoshop-style). Output is a gray premultiplied RGBA keeping the
/// source alpha. `width` is the Sobel sampling step in pixels.
pub fn find_edges(img: &[[f32; 4]], w: usize, h: usize, width: f32) -> Vec<[f32; 4]> {
    let mut out = vec![[0.0f32; 4]; w * h];
    for y in 0..h {
        for x in 0..w {
            let (gx, gy) = sobel_grad(img, w, h, x, y, width);
            let mag = (gx * gx + gy * gy).sqrt();
            let a = img[y * w + x][3];
            let v = (1.0 - mag).clamp(0.0, 1.0);
            out[y * w + x] = [v * a, v * a, v * a, a];
        }
    }
    out
}

/// Emboss: a directional gray relief. The luma gradient projected onto the
/// `(dir_x, dir_y)` light direction, scaled by `amount`, biased to mid-gray.
/// Output is gray premultiplied RGBA keeping the source alpha. `width` is the
/// Sobel sampling step in pixels.
pub fn emboss(
    img: &[[f32; 4]],
    w: usize,
    h: usize,
    dir_x: f32,
    dir_y: f32,
    amount: f32,
    width: f32,
) -> Vec<[f32; 4]> {
    let mut out = vec![[0.0f32; 4]; w * h];
    for y in 0..h {
        for x in 0..w {
            let (gx, gy) = sobel_grad(img, w, h, x, y, width);
            let a = img[y * w + x][3];
            let v = (0.5 + (gx * dir_x + gy * dir_y) * amount).clamp(0.0, 1.0);
            out[y * w + x] = [v * a, v * a, v * a, a];
        }
    }
    out
}

/// Glowing Edges: bright coloured edges on black. The edge magnitude (boosted by
/// `brightness`) modulates the source colour, so flats go black and edges glow.
/// `width` is the Sobel sampling step in pixels. Output is premultiplied RGBA.
pub fn glowing_edges(
    img: &[[f32; 4]],
    w: usize,
    h: usize,
    brightness: f32,
    width: f32,
) -> Vec<[f32; 4]> {
    let mut out = vec![[0.0f32; 4]; w * h];
    for y in 0..h {
        for x in 0..w {
            let (gx, gy) = sobel_grad(img, w, h, x, y, width);
            let mag = (gx * gx + gy * gy).sqrt();
            let g = (mag * brightness).clamp(0.0, 1.0);
            let mc = img[y * w + x];
            let a = mc[3].max(1e-4);
            // Unpremultiply → scale by edge strength → re-premultiply.
            let col = [mc[0] / a * g, mc[1] / a * g, mc[2] / a * g];
            out[y * w + x] = [col[0] * mc[3], col[1] * mc[3], col[2] * mc[3], mc[3]];
        }
    }
    out
}

/// Diffuse: a seeded-deterministic anisotropic neighbour scramble — each pixel
/// is replaced by a neighbour up to `amount` px away, the offset chosen by a
/// hash of `(x, y, seed)` (matching the WGSL `fract(sin(..))` hash). Stable for
/// a given seed (no temporal noise). Output is edge-clamp-sampled.
// The hash multipliers mirror the canonical GLSL/WGSL `fract(sin(..))` hash bit
// for bit, so the CPU reference matches the shader — keep them verbatim.
#[allow(clippy::excessive_precision)]
pub fn diffuse(img: &[[f32; 4]], w: usize, h: usize, amount: f32, seed: f32) -> Vec<[f32; 4]> {
    let mut out = vec![[0.0f32; 4]; w * h];
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f32, y as f32);
            let hf = ((fx * 12.9898 + fy * 78.233 + seed).sin() * 43758.5453).fract();
            let hg = ((fx * 39.3468 + fy * 11.135 + seed).sin() * 24634.6345).fract();
            let ang = hf * TAU;
            let dist = hg * amount.max(0.0);
            let sx = fx + ang.cos() * dist;
            let sy = fy + ang.sin() * dist;
            out[y * w + x] = sample_bilinear(img, w, h, sx, sy);
        }
    }
    out
}

// ===========================================================================
// Noise filters — CPU references mirroring `filter.wgsl`'s kinds 17–19.
// All operate in the compositor working space: **linear premultiplied** RGBA.
// ===========================================================================

/// Canonical `fract(sin(..))` hash → a stable scalar in [0,1) for integer pixel
/// `(ix, iy)` and `seed`. Bit-for-bit the WGSL `hash21` (and the diffuse hash),
/// so the CPU reference reproduces the shader's noise.
// The multipliers mirror the canonical GLSL/WGSL hash; keep them verbatim.
#[allow(clippy::excessive_precision)]
fn hash21(ix: f32, iy: f32, seed: f32) -> f32 {
    ((ix * 12.9898 + iy * 78.233 + seed).sin() * 43758.5453).fract()
}

/// One ~unit-variance gaussian sample via Box–Muller from two hashes, matching
/// the shader's `gauss1`.
fn gauss1(ix: f32, iy: f32, seed: f32) -> f32 {
    let u1 = hash21(ix, iy, seed).max(1e-6);
    let u2 = hash21(ix, iy, seed + 101.0);
    (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
}

/// One zero-mean uniform noise sample in (-1, 1): the difference of two i.i.d.
/// hashes (the raw `fract(sin)` hash is biased on a regular grid), matching the
/// shader's `uniform1`.
fn uniform1(ix: f32, iy: f32, seed: f32) -> f32 {
    hash21(ix, iy, seed) - hash21(ix, iy, seed + 200.0)
}

/// Add Noise (kind 17): seeded-deterministic per-pixel noise added to the
/// (unpremultiplied) colour, then re-premultiplied. `mono` applies the same
/// noise to R/G/B; `gaussian` selects gaussian (true) vs uniform (false). The
/// noise is zero-mean so the average is preserved, and stable for a given seed.
pub fn add_noise(
    img: &[[f32; 4]],
    w: usize,
    h: usize,
    amount: f32,
    mono: bool,
    gaussian: bool,
    seed: f32,
) -> Vec<[f32; 4]> {
    let mut out = vec![[0.0f32; 4]; w * h];
    for y in 0..h {
        for x in 0..w {
            let (ix, iy) = (x as f32, y as f32);
            let mc = img[y * w + x];
            let a = mc[3].max(1e-4);
            let mut nrgb = if gaussian {
                if mono {
                    [gauss1(ix, iy, seed) * 0.4; 3]
                } else {
                    [
                        gauss1(ix, iy, seed) * 0.4,
                        gauss1(ix, iy, seed + 17.0) * 0.4,
                        gauss1(ix, iy, seed + 53.0) * 0.4,
                    ]
                }
            } else if mono {
                [uniform1(ix, iy, seed); 3]
            } else {
                [
                    uniform1(ix, iy, seed),
                    uniform1(ix, iy, seed + 17.0),
                    uniform1(ix, iy, seed + 53.0),
                ]
            };
            let mut col = [0.0f32; 3];
            for c in 0..3 {
                nrgb[c] *= amount;
                col[c] = (mc[c] / a + nrgb[c]).clamp(0.0, 1.0);
            }
            out[y * w + x] = [col[0] * mc[3], col[1] * mc[3], col[2] * mc[3], mc[3]];
        }
    }
    out
}

/// Per-channel median over a `(2·radius+1)²` window of unpremultiplied colour
/// at `(x, y)`, edge-clamped. Returns the `[r, g, b]` medians.
fn window_median(
    img: &[[f32; 4]],
    w: usize,
    h: usize,
    x: usize,
    y: usize,
    radius: i32,
) -> [f32; 3] {
    let mut med = [0.0f32; 3];
    for (ch, m) in med.iter_mut().enumerate() {
        let mut vals: Vec<f32> = Vec::new();
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let sx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                let sy = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                let p = img[sy * w + sx];
                vals.push(p[ch] / p[3].max(1e-4));
            }
        }
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        *m = vals[vals.len() / 2];
    }
    med
}

/// Median filter (kind 18): replace each pixel with the per-channel median of a
/// `(2·radius+1)²` window — despeckle / salt-pepper removal. Output is
/// premultiplied RGBA keeping the source alpha.
pub fn median(img: &[[f32; 4]], w: usize, h: usize, radius: usize) -> Vec<[f32; 4]> {
    let r = radius as i32;
    let mut out = vec![[0.0f32; 4]; w * h];
    for y in 0..h {
        for x in 0..w {
            let m = window_median(img, w, h, x, y, r);
            let a = img[y * w + x][3];
            out[y * w + x] = [m[0] * a, m[1] * a, m[2] * a, a];
        }
    }
    out
}

/// Dust & Scratches (kind 19): a thresholded median — a channel is replaced by
/// the window median only when the original differs from it by more than
/// `threshold`, preserving detail while removing specks. Output is premultiplied
/// RGBA keeping the source alpha.
pub fn dust_and_scratches(
    img: &[[f32; 4]],
    w: usize,
    h: usize,
    radius: usize,
    threshold: f32,
) -> Vec<[f32; 4]> {
    let r = radius as i32;
    let mut out = vec![[0.0f32; 4]; w * h];
    for y in 0..h {
        for x in 0..w {
            let mc = img[y * w + x];
            let a = mc[3].max(1e-4);
            let m = window_median(img, w, h, x, y, r);
            let mut col = [0.0f32; 3];
            for c in 0..3 {
                let orig = mc[c] / a;
                col[c] = if (orig - m[c]).abs() > threshold {
                    m[c]
                } else {
                    orig
                };
            }
            out[y * w + x] = [col[0] * mc[3], col[1] * mc[3], col[2] * mc[3], mc[3]];
        }
    }
    out
}

// ===========================================================================
// Pixelate filters — CPU references mirroring `filter.wgsl`'s kinds 20–23.
// All operate in the compositor working space: **linear premultiplied** RGBA.
// ===========================================================================

/// Mosaic (kind 20): average each `cell`×`cell` block to one colour (the true
/// block mean of the premultiplied RGBA — a partially transparent block is
/// alpha-weighted correctly). Every pixel in a block gets that block's average,
/// so a block is uniform. Mirrors the WGSL block walk (edge-clamped, integer
/// pixel centres).
pub fn mosaic(img: &[[f32; 4]], w: usize, h: usize, cell: usize) -> Vec<[f32; 4]> {
    let cell = cell.max(1);
    let mut out = vec![[0.0f32; 4]; w * h];
    for y in 0..h {
        for x in 0..w {
            let bx = (x / cell) * cell;
            let by = (y / cell) * cell;
            let mut sum = [0.0f32; 4];
            let mut n = 0.0f32;
            for j in 0..cell {
                for i in 0..cell {
                    let sx = (bx + i).min(w - 1);
                    let sy = (by + j).min(h - 1);
                    let p = img[sy * w + sx];
                    for c in 0..4 {
                        sum[c] += p[c];
                    }
                    n += 1.0;
                }
            }
            out[y * w + x] = [sum[0] / n, sum[1] / n, sum[2] / n, sum[3] / n];
        }
    }
    out
}

/// Crystallize (kind 21): snap each pixel to the colour at its nearest jittered
/// seed point. One seed per `cell`×`cell` block, offset within the block by a
/// hash of the block index + `seed`; the 3×3 block neighbourhood is searched so
/// an adjacent jittered seed can win, giving irregular Voronoi cells. Stable for
/// a given seed (matches the WGSL hash). Edge-clamped nearest sampling.
pub fn crystallize(img: &[[f32; 4]], w: usize, h: usize, cell: usize, seed: f32) -> Vec<[f32; 4]> {
    let cellf = cell.max(1) as f32;
    let mut out = vec![[0.0f32; 4]; w * h];
    for y in 0..h {
        for x in 0..w {
            let px = x as f32;
            let py = y as f32;
            let cx = (px / cellf).floor();
            let cy = (py / cellf).floor();
            let mut best = f32::INFINITY;
            let mut bestc = [px, py];
            for gy in -1..=1 {
                for gx in -1..=1 {
                    let bx = cx + gx as f32;
                    let by = cy + gy as f32;
                    let jx = hash21(bx, by, seed);
                    let jy = hash21(bx, by, seed + 41.0);
                    let sx = (bx + jx) * cellf;
                    let sy = (by + jy) * cellf;
                    let dx = sx - px;
                    let dy = sy - py;
                    let dist = dx * dx + dy * dy;
                    if dist < best {
                        best = dist;
                        bestc = [sx, sy];
                    }
                }
            }
            // Snap the winning seed to its integer pixel centre, edge-clamped,
            // then nearest-sample — a true snap to one source colour (no blend).
            let sx = (bestc[0].floor() as i32).clamp(0, w as i32 - 1) as usize;
            let sy = (bestc[1].floor() as i32).clamp(0, h as i32 - 1) as usize;
            out[y * w + x] = img[sy * w + sx];
        }
    }
    out
}

/// Color Halftone (kind 22): a per-channel dot screen. Tile into `cell`×`cell`
/// cells rotated by `angle_rad`; each cell's channel average sets a dot radius
/// (darker channel → bigger dot of full ink). Inside the dot the channel is 0
/// (full ink), outside 1 (paper). Output is premultiplied, keeping source alpha.
/// Mirrors the WGSL screen math (per-channel 22.5° angle offset).
pub fn color_halftone(
    img: &[[f32; 4]],
    w: usize,
    h: usize,
    cell: usize,
    angle_rad: f32,
) -> Vec<[f32; 4]> {
    let cellf = (cell.max(2)) as f32;
    let ca = angle_rad.cos();
    let sa = angle_rad.sin();
    let mut out = vec![[0.0f32; 4]; w * h];
    for y in 0..h {
        for x in 0..w {
            let pxx = x as f32;
            let pxy = y as f32;
            let mc = img[y * w + x];
            let mut col = [0.0f32; 3];
            for (ch, cv) in col.iter_mut().enumerate() {
                let off = ch as f32 * std::f32::consts::FRAC_PI_8; // 22.5°
                let cc = off.cos() * ca - off.sin() * sa;
                let ss = off.sin() * ca + off.cos() * sa;
                let rx = pxx * cc - pxy * ss;
                let ry = pxx * ss + pxy * cc;
                let cellx = (rx / cellf).floor();
                let celly = (ry / cellf).floor();
                let cellcx = (cellx + 0.5) * cellf;
                let cellcy = (celly + 0.5) * cellf;
                // Average this channel over the cell (sampling the source).
                let mut sum = 0.0f32;
                let mut n = 0.0f32;
                let mut yy = 0.0;
                while yy < cellf {
                    let mut xx = 0.0;
                    while xx < cellf {
                        let lx = cellx * cellf + xx;
                        let ly = celly * cellf + yy;
                        let ix = lx * cc + ly * ss;
                        let iy = -lx * ss + ly * cc;
                        let sxx = (ix.floor() as i32).clamp(0, w as i32 - 1) as usize;
                        let syy = (iy.floor() as i32).clamp(0, h as i32 - 1) as usize;
                        let p = img[syy * w + sxx];
                        sum += p[ch] / p[3].max(1e-4);
                        n += 1.0;
                        xx += 1.0;
                    }
                    yy += 1.0;
                }
                let avg = sum / n.max(1.0);
                let maxr = cellf * std::f32::consts::FRAC_1_SQRT_2; // half diagonal
                let dotr = (1.0 - avg) * maxr;
                let d = ((rx - cellcx).powi(2) + (ry - cellcy).powi(2)).sqrt();
                *cv = if d <= dotr { 0.0 } else { 1.0 };
            }
            out[y * w + x] = [col[0] * mc[3], col[1] * mc[3], col[2] * mc[3], mc[3]];
        }
    }
    out
}

/// Mezzotint (kind 23): a seeded threshold dither to pure black/white grain.
/// Each pixel's Rec.709 luma is compared against a per-pixel hashed threshold
/// (biased by `amount`); above → white, below → black. Stable for a given seed.
/// Output is premultiplied, keeping the source alpha.
pub fn mezzotint(img: &[[f32; 4]], w: usize, h: usize, amount: f32, seed: f32) -> Vec<[f32; 4]> {
    let lw = [0.2126f32, 0.7152, 0.0722];
    let mut out = vec![[0.0f32; 4]; w * h];
    for y in 0..h {
        for x in 0..w {
            let mc = img[y * w + x];
            let a = mc[3].max(1e-4);
            let luma = (mc[0] / a) * lw[0] + (mc[1] / a) * lw[1] + (mc[2] / a) * lw[2];
            let t = hash21(x as f32, y as f32, seed);
            let v = if luma > t + (amount - 0.5) { 1.0 } else { 0.0 };
            out[y * w + x] = [v * mc[3], v * mc[3], v * mc[3], mc[3]];
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

    /// An `n×n` opaque image split by a vertical edge at column `n/2`: black on
    /// the left, white on the right. A clean vertical gradient (gx≠0, gy≈0).
    fn vsplit(n: usize) -> Vec<[f32; 4]> {
        let mut img = vec![[0.0, 0.0, 0.0, 1.0]; n * n];
        for y in 0..n {
            for x in (n / 2)..n {
                img[y * n + x] = [1.0, 1.0, 1.0, 1.0];
            }
        }
        img
    }

    // ---- Stylize ----------------------------------------------------------

    #[test]
    fn find_edges_darkens_the_edge_and_whitens_flats() {
        let n = 9;
        let img = vsplit(n);
        let out = find_edges(&img, n, n, 1.0);
        let row = n / 2;
        // A flat interior column (far from the edge) → no gradient → white (≈1).
        let flat = out[row * n + 1][0];
        // The edge column → strong gradient → dark (≪ flat). Alpha preserved.
        let edge = out[row * n + n / 2][0];
        assert!(flat > 0.9, "flat should be near white, got {flat}");
        assert!(edge < flat - 0.3, "edge {edge} should be far darker than flat {flat}");
        assert!(approx(out[row * n + n / 2][3], 1.0, 1e-6), "alpha preserved");
    }

    #[test]
    fn emboss_is_directional() {
        let n = 9;
        let img = vsplit(n);
        let row = n / 2;
        // Light along the gradient (horizontal) shifts the edge away from mid-gray.
        let horiz = emboss(&img, n, n, 1.0, 0.0, 0.2, 1.0);
        // Light perpendicular to the gradient (vertical) leaves the edge ≈ mid-gray.
        let vert = emboss(&img, n, n, 0.0, 1.0, 0.2, 1.0);
        let edge_h = horiz[row * n + n / 2][0];
        let edge_v = vert[row * n + n / 2][0];
        assert!((edge_h - 0.5).abs() > 0.05, "horizontal light should relieve the edge");
        assert!(approx(edge_v, 0.5, 0.05), "vertical light leaves a vertical edge flat");
        // A flat interior column is mid-gray regardless of light direction.
        assert!(approx(horiz[row * n + 1][0], 0.5, 1e-3), "flat → mid-gray");
    }

    #[test]
    fn glowing_edges_lights_edges_blacks_flats() {
        let n = 9;
        let img = vsplit(n);
        let row = n / 2;
        let out = glowing_edges(&img, n, n, 2.0, 1.0);
        let flat = out[row * n + 1][0];
        let edge = out[row * n + n / 2][0];
        assert!(flat < 0.05, "flat region goes black, got {flat}");
        assert!(edge > flat, "edge glows brighter than the flat, {edge} vs {flat}");
    }

    #[test]
    fn diffuse_is_deterministic_and_amount_zero_is_identity() {
        let n = 9;
        let img = vsplit(n);
        let a = diffuse(&img, n, n, 3.0, 1.0);
        let b = diffuse(&img, n, n, 3.0, 1.0);
        assert_eq!(a, b, "same seed → identical output");
        let c = diffuse(&img, n, n, 3.0, 2.0);
        assert!(a != c, "a different seed changes the field");
        let id = diffuse(&img, n, n, 0.0, 1.0);
        assert_eq!(id, img, "amount 0 is identity");
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

    // ---- Noise ------------------------------------------------------------

    /// A flat opaque mid-gray field.
    fn flat_gray(n: usize, v: f32) -> Vec<[f32; 4]> {
        vec![[v, v, v, 1.0]; n * n]
    }

    fn channel_mean(img: &[[f32; 4]], ch: usize) -> f32 {
        img.iter().map(|p| p[ch]).sum::<f32>() / img.len() as f32
    }

    #[test]
    fn add_noise_amount_zero_is_identity() {
        let n = 16;
        let img = flat_gray(n, 0.5);
        let out = add_noise(&img, n, n, 0.0, false, true, 1.0);
        for (a, b) in out.iter().zip(img.iter()) {
            for c in 0..4 {
                assert!(approx(a[c], b[c], 1e-6), "amount 0 is identity");
            }
        }
    }

    #[test]
    fn add_noise_is_deterministic() {
        let n = 24;
        let img = flat_gray(n, 0.5);
        let a = add_noise(&img, n, n, 0.3, false, true, 7.0);
        let b = add_noise(&img, n, n, 0.3, false, true, 7.0);
        assert_eq!(a, b, "same seed → identical noise (no temporal randomness)");
        // A different seed produces a different result.
        let c = add_noise(&img, n, n, 0.3, false, true, 8.0);
        assert!(a != c, "different seed → different noise");
    }

    #[test]
    fn add_noise_monochromatic_has_equal_rgb_delta() {
        let n = 24;
        let img = flat_gray(n, 0.5);
        let out = add_noise(&img, n, n, 0.25, true, true, 3.0);
        // Monochromatic: the per-pixel R/G/B deltas from the base are identical
        // (un-clamped here because base 0.5 + small noise stays interior).
        for p in &out {
            assert!(approx(p[0], p[1], 1e-6), "mono R==G");
            assert!(approx(p[1], p[2], 1e-6), "mono G==B");
        }
        // Colour noise breaks that equality somewhere.
        let colour = add_noise(&img, n, n, 0.25, false, true, 3.0);
        let any_diff = colour
            .iter()
            .any(|p| (p[0] - p[1]).abs() > 1e-4 || (p[1] - p[2]).abs() > 1e-4);
        assert!(any_diff, "colour noise differs per channel");
    }

    #[test]
    fn add_noise_preserves_mean_and_perturbs() {
        let n = 64;
        let base = 0.5;
        let img = flat_gray(n, base);
        // Gaussian, colour.
        let g = add_noise(&img, n, n, 0.15, false, true, 11.0);
        for ch in 0..3 {
            assert!(
                (channel_mean(&g, ch) - base).abs() < 0.02,
                "zero-mean gaussian noise preserves the channel mean (ch {ch})"
            );
        }
        // Uniform, monochromatic.
        let u = add_noise(&img, n, n, 0.15, true, false, 11.0);
        for ch in 0..3 {
            assert!(
                (channel_mean(&u, ch) - base).abs() < 0.02,
                "zero-mean uniform noise preserves the channel mean (ch {ch})"
            );
        }
        // And the field is actually perturbed (not all equal to base).
        assert!(
            g.iter().any(|p| (p[0] - base).abs() > 1e-3),
            "noise actually perturbs pixels"
        );
    }

    #[test]
    fn median_removes_an_impulse_outlier() {
        // A flat gray field with a single bright (salt) outlier; a 3×3 median
        // (radius 1) replaces it with the surrounding gray.
        let n = 9;
        let mut img = flat_gray(n, 0.4);
        let cidx = (n / 2) * n + n / 2;
        img[cidx] = [1.0, 1.0, 1.0, 1.0]; // impulse
        let out = median(&img, n, n, 1);
        for c in 0..3 {
            assert!(
                approx(out[cidx][c], 0.4, 1e-5),
                "median removes the impulse (ch {c}): {}",
                out[cidx][c]
            );
        }
        // A flat neighbour is unchanged.
        assert!(approx(out[0][0], 0.4, 1e-5), "flat area preserved");
    }

    #[test]
    fn median_radius_grows_the_window() {
        // Two adjacent outliers survive a 3×3 median at one of them (a tie can
        // remain) but a 5×5 (radius 2) median over a flat field removes them.
        let n = 11;
        let mut img = flat_gray(n, 0.3);
        let c = n / 2;
        img[c * n + c] = [0.9, 0.9, 0.9, 1.0];
        img[c * n + c + 1] = [0.9, 0.9, 0.9, 1.0];
        let out = median(&img, n, n, 2);
        assert!(
            approx(out[c * n + c][0], 0.3, 1e-5),
            "radius-2 median clears clustered outliers: {}",
            out[c * n + c][0]
        );
    }

    #[test]
    fn dust_and_scratches_only_changes_above_threshold_pixels() {
        // A flat field with one strong speck and a tiny ripple. With a threshold
        // between the two deviations, only the strong speck is replaced; the
        // sub-threshold detail is preserved (unlike a plain median).
        let n = 9;
        let base = 0.5;
        let mut img = flat_gray(n, base);
        let speck = (n / 2) * n + n / 2;
        let small = 2 * n + 2;
        img[speck] = [0.95, 0.95, 0.95, 1.0]; // big deviation (0.45)
        img[small] = [base + 0.05, base + 0.05, base + 0.05, 1.0]; // tiny (0.05)
        let out = dust_and_scratches(&img, n, n, 1, 0.2);
        // The strong speck is pulled to the median (≈ base).
        assert!(
            (out[speck][0] - base).abs() < 0.02,
            "above-threshold speck replaced: {}",
            out[speck][0]
        );
        // The sub-threshold pixel is left as-is (its own value, not the median).
        assert!(
            approx(out[small][0], base + 0.05, 1e-5),
            "below-threshold pixel preserved: {}",
            out[small][0]
        );
    }

    #[test]
    fn dust_high_threshold_is_identity() {
        // With a threshold above any deviation present, nothing changes.
        let n = 9;
        let mut img = flat_gray(n, 0.5);
        img[(n / 2) * n + n / 2] = [0.9, 0.9, 0.9, 1.0];
        let out = dust_and_scratches(&img, n, n, 1, 1.0);
        for (a, b) in out.iter().zip(img.iter()) {
            for c in 0..4 {
                assert!(approx(a[c], b[c], 1e-5), "high threshold → identity");
            }
        }
    }

    // ---- Pixelate family --------------------------------------------------

    #[test]
    fn mosaic_cell_is_the_block_average() {
        // A 4×4 image, cell 2 → four 2×2 blocks. Each block becomes its own mean,
        // and every pixel of a block shares that value (the block is uniform).
        let n = 4;
        // Distinct per-pixel values so block means are non-trivial.
        let mut img = vec![[0.0f32; 4]; n * n];
        for y in 0..n {
            for x in 0..n {
                let v = (y * n + x) as f32 / 16.0;
                img[y * n + x] = [v, v * 0.5, v * 0.25, 1.0];
            }
        }
        let out = mosaic(&img, n, n, 2);
        // For each 2×2 block, compute the expected mean and check uniformity.
        for by in (0..n).step_by(2) {
            for bx in (0..n).step_by(2) {
                let mut sum = [0.0f32; 4];
                for j in 0..2 {
                    for i in 0..2 {
                        let p = img[(by + j) * n + bx + i];
                        for c in 0..4 {
                            sum[c] += p[c];
                        }
                    }
                }
                let mean = [sum[0] / 4.0, sum[1] / 4.0, sum[2] / 4.0, sum[3] / 4.0];
                for j in 0..2 {
                    for i in 0..2 {
                        let o = out[(by + j) * n + bx + i];
                        for c in 0..4 {
                            assert!(
                                approx(o[c], mean[c], 1e-6),
                                "block ({bx},{by}) pixel uniform = mean (ch {c}): {} vs {}",
                                o[c],
                                mean[c]
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn mosaic_of_a_flat_field_is_identity() {
        let n = 8;
        let img = flat_gray(n, 0.37);
        let out = mosaic(&img, n, n, 4);
        for p in &out {
            assert!(approx(p[0], 0.37, 1e-6), "flat field stays flat");
        }
    }

    #[test]
    fn crystallize_is_deterministic_and_snaps_cells() {
        // A smooth gradient field; crystallize with a fixed seed must be
        // reproducible, and pixels that resolve to the same nearest seed must
        // share an identical colour (cell-snapping).
        let n = 16;
        let mut img = vec![[0.0f32; 4]; n * n];
        for y in 0..n {
            for x in 0..n {
                let v = x as f32 / (n - 1) as f32;
                img[y * n + x] = [v, 1.0 - v, 0.5, 1.0];
            }
        }
        let a = crystallize(&img, n, n, 4, 7.0);
        let b = crystallize(&img, n, n, 4, 7.0);
        assert_eq!(a, b, "same seed → identical (no temporal randomness)");
        // The output colour set is drawn only from source pixels (snapping, not
        // blending): every output equals some input pixel.
        for o in &a {
            assert!(
                img.iter().any(|p| p == o),
                "crystallize snaps to a source colour, never blends: {o:?}"
            );
        }
        // A different seed shifts the jitter → a different tessellation.
        let c = crystallize(&img, n, n, 4, 8.0);
        assert!(a != c, "different seed → different cells");
    }

    #[test]
    fn crystallize_of_a_flat_field_is_identity() {
        let n = 12;
        let img = flat_gray(n, 0.6);
        let out = crystallize(&img, n, n, 3, 2.0);
        for p in &out {
            assert!(approx(p[0], 0.6, 1e-6), "flat field unchanged by snapping");
        }
    }

    #[test]
    fn color_halftone_dot_grows_as_the_cell_darkens() {
        // Halftone output is binary per channel (0 ink / 1 paper). A darker field
        // (lower channel average) must produce MORE ink pixels (bigger dots) than
        // a brighter one. Use angle 0 so the screen is axis-aligned.
        let n = 32;
        let ink = |img: &[[f32; 4]]| -> usize {
            color_halftone(img, n, n, 8, 0.0)
                .iter()
                .filter(|p| p[0] < 0.5) // red-channel ink (premultiplied, a=1)
                .count()
        };
        let dark = ink(&flat_gray(n, 0.2));
        let mid = ink(&flat_gray(n, 0.5));
        let bright = ink(&flat_gray(n, 0.8));
        assert!(
            dark > mid && mid > bright,
            "darker cell → larger dot → more ink: dark {dark} mid {mid} bright {bright}"
        );
        // A near-black field is (almost) solid ink; a near-white field is sparse.
        let black = ink(&flat_gray(n, 0.02));
        let white = ink(&flat_gray(n, 0.98));
        assert!(black > white, "black {black} > white {white}");
    }

    #[test]
    fn color_halftone_is_binary_per_channel() {
        let n = 16;
        let img = flat_gray(n, 0.5);
        let out = color_halftone(&img, n, n, 4, 0.0);
        for p in &out {
            for c in 0..3 {
                assert!(
                    approx(p[c], 0.0, 1e-6) || approx(p[c], 1.0, 1e-6),
                    "each channel is full ink or paper: {}",
                    p[c]
                );
            }
            assert!(approx(p[3], 1.0, 1e-6), "alpha preserved");
        }
    }

    #[test]
    fn mezzotint_is_binary_deterministic_and_tracks_brightness() {
        // Output is pure black/white, reproducible per seed, and a brighter field
        // yields more white pixels than a darker one.
        let n = 32;
        let white_count = |v: f32| -> usize {
            mezzotint(&flat_gray(n, v), n, n, 0.5, 4.0)
                .iter()
                .filter(|p| p[0] > 0.5)
                .count()
        };
        let dark = white_count(0.25);
        let bright = white_count(0.75);
        assert!(bright > dark, "brighter → more white: {bright} > {dark}");
        // Binary + deterministic.
        let a = mezzotint(&flat_gray(n, 0.5), n, n, 0.5, 9.0);
        let b = mezzotint(&flat_gray(n, 0.5), n, n, 0.5, 9.0);
        assert_eq!(a, b, "same seed → identical");
        for p in &a {
            assert!(
                approx(p[0], 0.0, 1e-6) || approx(p[0], 1.0, 1e-6),
                "binary output: {}",
                p[0]
            );
        }
    }

    // ---- Sharpen family (High Pass) --------------------------------------

    #[test]
    fn gaussian_blur_preserves_a_flat_field() {
        // A normalized Gaussian leaves a flat opaque field untouched.
        let n = 16;
        let img = flat_gray(n, 0.5);
        let out = gaussian_blur(&img, n, n, 4.0);
        for (a, b) in out.iter().zip(img.iter()) {
            for c in 0..4 {
                assert!(approx(a[c], b[c], 1e-4), "flat field preserved by blur");
            }
        }
    }

    #[test]
    fn high_pass_flattens_to_mid_gray_away_from_edges() {
        // On a vertical black/white split, columns far from the edge are locally
        // flat so the blur ≈ the original → the high pass ≈ mid-gray (0.5). The
        // edge column itself carries the detail and deviates from 0.5.
        let n = 17;
        let img = vsplit(n);
        let row = n / 2;
        let out = high_pass(&img, n, n, 2.0, 1.0);
        let flat = out[row * n + 1][0]; // interior of the black side
        let edge = out[row * n + n / 2][0]; // the boundary column
        assert!(approx(flat, 0.5, 0.02), "flat area → mid-gray, got {flat}");
        assert!(
            (edge - 0.5).abs() > 0.05,
            "edge carries detail (deviates from mid-gray): {edge}"
        );
        // Alpha is preserved everywhere.
        for p in &out {
            assert!(approx(p[3], 1.0, 1e-6), "alpha preserved");
        }
    }

    #[test]
    fn high_pass_is_signed_about_mid_gray() {
        // Either side of a step edge picks up opposite-signed detail: the lighter
        // side of the transition rises above 0.5, the darker side falls below it
        // (the blur pulls each toward the local mean).
        let n = 17;
        let img = vsplit(n);
        let row = n / 2;
        let out = high_pass(&img, n, n, 3.0, 1.0);
        let c = n / 2;
        // Just left of the edge (still black=0) → below mid-gray; just right
        // (white=1) → above mid-gray.
        let left = out[row * n + (c - 1)][0];
        let right = out[row * n + c][0];
        assert!(left < 0.5, "dark side of edge dips below mid-gray: {left}");
        assert!(right > 0.5, "light side of edge rises above mid-gray: {right}");
    }

    #[test]
    fn high_pass_amount_zero_is_flat_mid_gray() {
        // amount 0 keeps only the +0.5 re-centring → a uniform mid-gray field
        // regardless of the input detail.
        let n = 12;
        let img = vsplit(n);
        let out = high_pass(&img, n, n, 3.0, 0.0);
        for p in &out {
            assert!(approx(p[0], 0.5, 1e-5), "amount 0 → flat mid-gray: {}", p[0]);
            assert!(approx(p[3], 1.0, 1e-6), "alpha preserved");
        }
    }

    #[test]
    fn high_pass_amount_scales_the_detail() {
        // A larger amount pushes edge detail further from mid-gray (until clamp).
        let n = 17;
        let img = vsplit(n);
        let row = n / 2;
        let c = n / 2;
        let soft = high_pass(&img, n, n, 3.0, 0.5);
        let hard = high_pass(&img, n, n, 3.0, 1.0);
        let soft_dev = (soft[row * n + c][0] - 0.5).abs();
        let hard_dev = (hard[row * n + c][0] - 0.5).abs();
        assert!(
            hard_dev > soft_dev,
            "more amount → stronger detail: {hard_dev} > {soft_dev}"
        );
    }
}
