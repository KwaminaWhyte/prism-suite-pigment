// Destructive filters applied to a layer. Fullscreen pass sampling `input`.
// Separable blurs (Gaussian, Box) run this twice (horizontal then vertical).

struct FParams {
    // 1 Gaussian blur, 2 sharpen, 3 pixelate, 4 motion blur,
    // 5 box blur (separable), 6 radial spin, 7 radial zoom,
    // 8 twirl, 9 pinch/spherize, 10 ripple/wave, 11 polar (rect->polar),
    // 12 polar (polar->rect), 13 find edges, 14 emboss, 15 glowing edges,
    // 16 diffuse, 17 add noise, 18 median, 19 dust & scratches,
    // 20 mosaic, 21 crystallize, 22 color halftone, 23 mezzotint,
    // 24 high pass (orig - blur, re-centred at mid-gray),
    // 25 clouds (fBm value-noise generator), 26 difference clouds
    // (|source - clouds|), 27 oil paint (Kuwahara quadrant filter),
    // 28 posterize (quantize each channel to N display-space levels),
    // 29 threshold (luma cutoff → pure black/white).
    kind: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
    texel: vec2<f32>, // 1/size
    dir: vec2<f32>,   // blur direction * texel (Gaussian/box/motion); distort:
                      // (amplitude_px, wavelength_px) for ripple; emboss:
                      // (cos angle, sin angle) light direction; diffuse:
                      // (seed, _); add noise: (seed, monochromatic flag);
                      // color halftone: (cos angle, sin angle) screen rotation;
                      // crystallize/mezzotint: (seed, _);
                      // clouds/difference clouds: (seed, roughness)
    amount: f32,      // posterize: level count N (2..255); threshold: luma cutoff
                      // (0..1); sharpen amount; radial: spin angle (rad) / zoom fraction;
                      // twirl: max angle (rad); pinch: signed amount (-1..1);
                      // emboss: height/relief gain; glowing edges: brightness;
                      // diffuse: max neighbour displacement in pixels; add noise:
                      // noise amount (0..1); dust & scratches: threshold;
                      // mezzotint: noise/threshold style amount
    radius: f32,      // blur radius / pixelate block / motion taps / radial
                      // samples; distort: effect radius in pixels; stylize:
                      // edge sampling width in pixels; add noise: gaussian flag
                      // (1 gaussian, 0 uniform); median/dust: window radius (px);
                      // mosaic/crystallize: cell size (px); color halftone: cell
                      // (dot screen) size (px)
    center: vec2<f32>, // radial/distort center in uv (0..1)
};

@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var input: texture_2d<f32>;
@group(0) @binding(2) var<uniform> p: FParams;
// Secondary input. For every single-input kind this is bound to the *same*
// texture as `input` (so it is a harmless alias); only the High Pass combine
// (kind 24) reads it separately — `input` is the Gaussian-blurred copy and
// `orig` the untouched source — so it can subtract one from the other.
@group(0) @binding(3) var orig: texture_2d<f32>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    var v = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0),
    );
    let c = v[vi];
    var out: VsOut;
    out.pos = vec4<f32>(c, 0.0, 1.0);
    out.uv = vec2<f32>(c.x * 0.5 + 0.5, 0.5 - c.y * 0.5);
    return out;
}

// Canonical `fract(sin(..))` hash → a stable pseudo-random scalar in [0,1) for
// integer pixel `(ix, iy)` and `seed`. Matches the diffuse hash so the CPU
// reference can reproduce it bit-for-bit.
fn hash21(ix: f32, iy: f32, seed: f32) -> f32 {
    return fract(sin(dot(vec2<f32>(ix, iy), vec2<f32>(12.9898, 78.233)) + seed) * 43758.5453);
}

// sRGB transfer pair (same constants as composite.wgsl / prism-color). Posterize
// and Threshold work in *display* (sRGB) space so the levels land where the user
// sees them — quantizing in linear light would bunch the steps in the shadows.
fn l2s1(c: f32) -> f32 { if c <= 0.0031308 { return c * 12.92; } return 1.055 * pow(c, 1.0 / 2.4) - 0.055; }
fn s2l1(c: f32) -> f32 { if c <= 0.04045 { return c / 12.92; } return pow((c + 0.055) / 1.055, 2.4); }
fn l2s(c: vec3<f32>) -> vec3<f32> { return vec3<f32>(l2s1(c.x), l2s1(c.y), l2s1(c.z)); }
fn s2l(c: vec3<f32>) -> vec3<f32> { return vec3<f32>(s2l1(c.x), s2l1(c.y), s2l1(c.z)); }

// One zero-mean uniform noise sample in (-1, 1): the difference of two i.i.d.
// hashes, symmetric about 0 so the per-channel mean is preserved.
fn uniform1(ix: f32, iy: f32, seed: f32) -> f32 {
    return hash21(ix, iy, seed) - hash21(ix, iy, seed + 200.0);
}

// One ~unit-variance gaussian sample via Box–Muller from two hashes.
fn gauss1(ix: f32, iy: f32, seed: f32) -> f32 {
    let u1 = max(hash21(ix, iy, seed), 1e-6);
    let u2 = hash21(ix, iy, seed + 101.0);
    return sqrt(-2.0 * log(u1)) * cos(6.28318530718 * u2);
}

// Smooth 2D value noise in [0,1): bilinear interpolation between the four
// hashed lattice corners around `p`, smoothed with the classic Hermite curve
// (3t²−2t³). Built on `hash21` so the CPU reference reproduces it bit-for-bit.
fn value_noise(p: vec2<f32>, seed: f32) -> f32 {
    let i = floor(p);
    let f = p - i;
    let a = hash21(i.x, i.y, seed);
    let b = hash21(i.x + 1.0, i.y, seed);
    let c = hash21(i.x, i.y + 1.0, seed);
    let d = hash21(i.x + 1.0, i.y + 1.0, seed);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Fractal (fBm) sum of `octaves` value-noise layers: each octave doubles the
// frequency and halves the amplitude (`roughness` controls the falloff). The
// running amplitude sum normalises the result back into [0,1], so the cloud
// texture is a soft, seamless multi-scale field. `scale` is the base feature
// size in pixels (larger → broader puffs). Deterministic for a given seed.
fn fbm(px: vec2<f32>, seed: f32, scale: f32, roughness: f32, octaves: i32) -> f32 {
    var freq = 1.0 / max(scale, 1.0);
    var amp = 1.0;
    var sum = 0.0;
    var norm = 0.0;
    for (var o = 0; o < octaves; o = o + 1) {
        sum = sum + amp * value_noise(px * freq, seed + f32(o) * 37.0);
        norm = norm + amp;
        freq = freq * 2.0;
        amp = amp * roughness;
    }
    return sum / max(norm, 1e-5);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    if p.kind == 1u {
        // Separable Gaussian blur along p.dir.
        let r = i32(clamp(p.radius, 0.0, 64.0));
        let sigma = max(p.radius * 0.5, 0.5);
        var sum = vec4<f32>(0.0);
        var wsum = 0.0;
        for (var i = -r; i <= r; i = i + 1) {
            let w = exp(-0.5 * f32(i * i) / (sigma * sigma));
            sum = sum + textureSample(input, samp, in.uv + p.dir * f32(i)) * w;
            wsum = wsum + w;
        }
        return sum / max(wsum, 1e-5);
    } else if p.kind == 2u {
        // Unsharp: center minus a small box blur.
        let c = textureSample(input, samp, in.uv);
        var b = c;
        b = b + textureSample(input, samp, in.uv + vec2<f32>(p.texel.x, 0.0));
        b = b + textureSample(input, samp, in.uv - vec2<f32>(p.texel.x, 0.0));
        b = b + textureSample(input, samp, in.uv + vec2<f32>(0.0, p.texel.y));
        b = b + textureSample(input, samp, in.uv - vec2<f32>(0.0, p.texel.y));
        b = b / 5.0;
        return c + (c - b) * p.amount;
    } else if p.kind == 3u {
        // Pixelate: snap uv to a block grid.
        let block = max(p.radius, 1.0);
        let px = (floor(in.uv / p.texel / block) + 0.5) * block * p.texel;
        return textureSample(input, samp, px);
    } else if p.kind == 4u {
        // Motion blur: flat box average of 2*radius+1 taps along p.dir
        // (a unit direction pre-scaled by texel CPU-side).
        let r = i32(clamp(p.radius, 0.0, 256.0));
        var sum = vec4<f32>(0.0);
        var n = 0.0;
        for (var i = -r; i <= r; i = i + 1) {
            sum = sum + textureSample(input, samp, in.uv + p.dir * f32(i));
            n = n + 1.0;
        }
        return sum / max(n, 1.0);
    } else if p.kind == 5u {
        // Box blur: flat box average of 2*radius+1 taps along p.dir.
        // Run twice (H then V) for a true separable box kernel.
        let r = i32(clamp(p.radius, 0.0, 256.0));
        var sum = vec4<f32>(0.0);
        var n = 0.0;
        for (var i = -r; i <= r; i = i + 1) {
            sum = sum + textureSample(input, samp, in.uv + p.dir * f32(i));
            n = n + 1.0;
        }
        return sum / max(n, 1.0);
    } else if p.kind == 6u || p.kind == 7u {
        // Radial blur — Spin (kind 6, rotate about center) or Zoom (kind 7,
        // scale toward/from center). Average `radius` taps spread over the
        // arc/ray so amount 0 is identity.
        let samples = max(i32(p.radius), 1);
        let d = in.uv - p.center;        // offset from center in uv
        // Correct for non-square pixels so spin is circular in pixel space.
        let aspect = p.texel.y / max(p.texel.x, 1e-8); // (1/h)/(1/w) = w/h
        var sum = vec4<f32>(0.0);
        var n = 0.0;
        for (var k = 0; k < samples; k = k + 1) {
            var t = 0.0;
            if samples > 1 {
                t = f32(k) / f32(samples - 1) - 0.5;
            }
            var uv = in.uv;
            if p.kind == 6u {
                // Spin: rotate the offset by amount*t about the center.
                let a = p.amount * t;
                let ca = cos(a);
                let sa = sin(a);
                // Rotate in pixel space: scale x by aspect, rotate, unscale.
                let dx = d.x * aspect;
                let dy = d.y;
                let rx = dx * ca - dy * sa;
                let ry = dx * sa + dy * ca;
                uv = p.center + vec2<f32>(rx / aspect, ry);
            } else {
                // Zoom: scale the offset by (1 + amount*t).
                let s = 1.0 + p.amount * t;
                uv = p.center + d * s;
            }
            sum = sum + textureSample(input, samp, uv);
            n = n + 1.0;
        }
        return sum / max(n, 1.0);
    } else if p.kind >= 8u && p.kind <= 12u {
        // ---- Distort filters: per-pixel coordinate remap + edge-clamped
        // sample of the source. All work in *pixel* space about the center so
        // the warp is geometric (non-square pixels handled by mapping uv->px).
        let dim = 1.0 / p.texel;                 // (w, h) in px
        let cpx = p.center * dim;                 // center in px
        let px = in.uv * dim;                     // this pixel in px
        var spx = px;                            // source pixel to sample
        if p.kind == 8u {
            // Twirl: rotate about the center by an angle that falls off to 0 at
            // `radius`. Inside the radius only; outside is identity.
            let d = px - cpx;
            let dist = length(d);
            if dist < p.radius && p.radius > 0.0 {
                let falloff = 1.0 - dist / p.radius;
                let a = p.amount * falloff * falloff;
                let ca = cos(a);
                let sa = sin(a);
                // Inverse-map: sample the source rotated by -a so the image
                // appears rotated by +a.
                spx = cpx + vec2<f32>(d.x * ca + d.y * sa, -d.x * sa + d.y * ca);
            }
        } else if p.kind == 9u {
            // Pinch (amount > 0, pulls toward center) / Spherize-bulge
            // (amount < 0, pushes outward). Smooth radial falloff to `radius`.
            let d = px - cpx;
            let dist = length(d);
            if dist < p.radius && p.radius > 0.0 && dist > 1e-4 {
                let nd = dist / p.radius;        // 0 at center .. 1 at edge
                // Scale source radius: pinch maps the sample inward.
                let s = pow(nd, 1.0 + p.amount);
                spx = cpx + d * (s / nd);
            }
        } else if p.kind == 10u {
            // Ripple / Wave: sinusoidal displacement. dir = (amplitude_px,
            // wavelength_px); offset each axis by a sine of the *other* axis.
            let amp = p.dir.x;
            let wl = max(p.dir.y, 1e-3);
            let k = 6.28318530718 / wl;
            spx = px + vec2<f32>(amp * sin(px.y * k), amp * sin(px.x * k));
        } else if p.kind == 11u {
            // Rectangular -> Polar. Map the output's (x = angle, y = radius)
            // back to a cartesian source coordinate. x spans 0..2π over the
            // width; y spans 0..maxR over the height (0 at top = center).
            let maxr = min(dim.x, dim.y) * 0.5;
            let theta = (px.x / dim.x) * 6.28318530718 - 1.5707963268; // start at top
            let rr = (px.y / dim.y) * maxr;
            spx = cpx + vec2<f32>(rr * cos(theta), rr * sin(theta));
        } else {
            // kind 12u: Polar -> Rectangular. Map the cartesian output back to
            // the polar layout the forward pass produced (inverse of kind 11).
            let maxr = min(dim.x, dim.y) * 0.5;
            let d = px - cpx;
            var theta = atan2(d.y, d.x) + 1.5707963268;
            if theta < 0.0 { theta = theta + 6.28318530718; }
            if theta >= 6.28318530718 { theta = theta - 6.28318530718; }
            let rr = length(d);
            spx = vec2<f32>((theta / 6.28318530718) * dim.x, (rr / maxr) * dim.y);
        }
        return textureSample(input, samp, spx * p.texel);
    } else if p.kind == 17u {
        // ---- Add Noise (kind 17): seeded-deterministic per-pixel noise added
        // to the (unpremultiplied) colour, then re-premultiplied. dir.x = seed,
        // dir.y = monochromatic flag (1 = same noise on R/G/B), amount = noise
        // strength (0..1), radius = gaussian flag (1 gaussian, 0 uniform). The
        // noise is zero-mean so the average is preserved, and stable per seed
        // (matches the diffuse hash philosophy — no temporal randomness).
        let dim = 1.0 / p.texel;
        let px = in.uv * dim;
        let ix = floor(px.x);
        let iy = floor(px.y);
        let seed = p.dir.x;
        let mono = p.dir.y > 0.5;
        let gaussian = p.radius > 0.5;
        let src = textureSample(input, samp, in.uv);
        let a = max(src.a, 1e-4);
        var col = src.rgb / a;                     // unpremultiplied colour
        var nrgb = vec3<f32>(0.0);
        if gaussian {
            // Box–Muller → ~unit-variance gaussian (zero-mean), scaled.
            if mono {
                nrgb = vec3<f32>(gauss1(ix, iy, seed) * 0.4);
            } else {
                nrgb = vec3<f32>(
                    gauss1(ix, iy, seed) * 0.4,
                    gauss1(ix, iy, seed + 17.0) * 0.4,
                    gauss1(ix, iy, seed + 53.0) * 0.4,
                );
            }
        } else {
            // Uniform: a symmetric difference of two i.i.d. hashes → zero-mean
            // in [-1, 1] (the raw `fract(sin)` hash is biased on a regular grid,
            // so subtracting a second sample centres it and preserves the mean).
            if mono {
                nrgb = vec3<f32>(uniform1(ix, iy, seed));
            } else {
                nrgb = vec3<f32>(
                    uniform1(ix, iy, seed),
                    uniform1(ix, iy, seed + 17.0),
                    uniform1(ix, iy, seed + 53.0),
                );
            }
        }
        col = clamp(col + nrgb * p.amount, vec3<f32>(0.0), vec3<f32>(1.0));
        return vec4<f32>(col * src.a, src.a);
    } else if p.kind == 18u || p.kind == 19u {
        // ---- Median (kind 18) / Dust & Scratches (kind 19). Per-channel median
        // over a (2r+1)² window (despeckle / salt-pepper removal). Dust &
        // Scratches only replaces a pixel when it differs from the median by
        // more than `amount` (threshold), preserving fine detail. Work on the
        // unpremultiplied colour so a transparent edge doesn't bias the median.
        let r = i32(clamp(p.radius, 0.0, 4.0));
        let src = textureSample(input, samp, in.uv);
        let sa = max(src.a, 1e-4);
        let scol = src.rgb / sa;
        var out = scol;
        for (var ch = 0; ch < 3; ch = ch + 1) {
            var vals: array<f32, 81>;
            var n = 0;
            for (var dy = -r; dy <= r; dy = dy + 1) {
                for (var dx = -r; dx <= r; dx = dx + 1) {
                    let s = textureSample(
                        input, samp,
                        in.uv + vec2<f32>(f32(dx), f32(dy)) * p.texel,
                    );
                    let v = s.rgb / max(s.a, 1e-4);
                    vals[n] = v[ch];
                    n = n + 1;
                }
            }
            // Insertion sort the n collected values, then take the middle.
            for (var i = 1; i < n; i = i + 1) {
                let key = vals[i];
                var j = i - 1;
                loop {
                    if j < 0 || vals[j] <= key { break; }
                    vals[j + 1] = vals[j];
                    j = j - 1;
                }
                vals[j + 1] = key;
            }
            let med = vals[n / 2];
            if p.kind == 19u {
                // Dust & Scratches: keep the original unless it strays past the
                // threshold from the median.
                if abs(scol[ch] - med) > p.amount {
                    out[ch] = med;
                }
            } else {
                out[ch] = med;
            }
        }
        return vec4<f32>(out * src.a, src.a);
    } else if p.kind == 20u {
        // ---- Mosaic (kind 20): average each `cell`×`cell` block to one colour.
        // Unlike the legacy Pixelate (kind 3, which point-samples the block
        // centre), this is the true block mean. Averaging is done in
        // premultiplied space (a straight component mean of premultiplied RGBA),
        // matching the blur convention, so a partially transparent block is
        // alpha-weighted correctly.
        let dim = 1.0 / p.texel;
        let cell = max(floor(p.radius), 1.0);
        let px = floor(in.uv * dim);
        // Snap to the block's origin, then average over the (clamped) block.
        let bx = floor(px.x / cell) * cell;
        let by = floor(px.y / cell) * cell;
        var sum = vec4<f32>(0.0);
        var n = 0.0;
        var j = 0.0;
        loop {
            if j >= cell { break; }
            var i = 0.0;
            loop {
                if i >= cell { break; }
                let sx = min(bx + i, dim.x - 1.0);
                let sy = min(by + j, dim.y - 1.0);
                sum = sum + textureSample(input, samp, (vec2<f32>(sx, sy) + 0.5) * p.texel);
                n = n + 1.0;
                i = i + 1.0;
            }
            j = j + 1.0;
        }
        return sum / max(n, 1.0);
    } else if p.kind == 21u {
        // ---- Crystallize (kind 21): Voronoi-ish cells. Snap each pixel to the
        // colour sampled at its nearest seed point, where seeds are a jittered
        // grid (one per `cell`×`cell` block, offset within the block by a hash of
        // the block index + seed). The 3×3 neighbourhood of blocks is searched so
        // a jittered seed in an adjacent block can win, giving irregular polygons.
        let dim = 1.0 / p.texel;
        let cell = max(floor(p.radius), 1.0);
        let seed = p.dir.x;
        let px = in.uv * dim;
        let cx = floor(px.x / cell);
        let cy = floor(px.y / cell);
        var best = 1e30;
        var bestc = vec2<f32>(px);
        var gy = -1.0;
        loop {
            if gy > 1.0 { break; }
            var gx = -1.0;
            loop {
                if gx > 1.0 { break; }
                let bx = cx + gx;
                let by = cy + gy;
                // Jittered seed position inside this block (hash in [0,1)).
                let jx = hash21(bx, by, seed);
                let jy = hash21(bx, by, seed + 41.0);
                let spx = (vec2<f32>(bx, by) + vec2<f32>(jx, jy)) * cell;
                let d = spx - px;
                let dist = dot(d, d);
                if dist < best {
                    best = dist;
                    bestc = spx;
                }
                gx = gx + 1.0;
            }
            gy = gy + 1.0;
        }
        // Snap the winning seed to its integer pixel centre and nearest-sample
        // (edge-clamped). Sampling the exact texel centre means the bilinear
        // sampler returns that one source colour — a true snap, never a blend —
        // so every cell shares an identical input colour.
        let si = clamp(floor(bestc), vec2<f32>(0.0), dim - 1.0);
        return textureSample(input, samp, (si + 0.5) * p.texel);
    } else if p.kind == 22u {
        // ---- Color Halftone (kind 22): per-channel dot screen. Tile the image
        // into `cell`×`cell` screens (rotated by p.dir = (cos,sin)); each cell's
        // channel value sets a dot radius (a denser/darker channel → bigger dot).
        // Work on the unpremultiplied colour; coverage inside the dot = full
        // channel ink, outside = none (white-ish), so brighter cells get smaller
        // ink dots. dir = screen rotation, radius = cell size.
        let dim = 1.0 / p.texel;
        let cell = max(floor(p.radius), 2.0);
        let px = in.uv * dim;
        let ca = p.dir.x;
        let sa = p.dir.y;
        let src = textureSample(input, samp, in.uv);
        let a = max(src.a, 1e-4);
        var out = vec3<f32>(0.0);
        for (var ch = 0; ch < 3; ch = ch + 1) {
            // Rotate into the channel's screen space (a small per-channel angle
            // offset spreads the rosette, as in a CMY screen).
            let off = f32(ch) * 0.39269908; // 22.5° between channels
            let cc = cos(off) * ca - sin(off) * sa;
            let ss = sin(off) * ca + cos(off) * sa;
            let rx = px.x * cc - px.y * ss;
            let ry = px.x * ss + px.y * cc;
            // Cell index + this pixel's offset from the cell centre.
            let cellx = floor(rx / cell);
            let celly = floor(ry / cell);
            let cellcx = (cellx + 0.5) * cell;
            let cellcy = (celly + 0.5) * cell;
            // Average this channel over the cell (in source space). Walk the cell
            // in screen space, rotate back to sample the source.
            var sum = 0.0;
            var n = 0.0;
            var yy = 0.0;
            loop {
                if yy >= cell { break; }
                var xx = 0.0;
                loop {
                    if xx >= cell { break; }
                    let lx = cellx * cell + xx;
                    let ly = celly * cell + yy;
                    // Inverse-rotate the screen-space point back to image space.
                    let ix = lx * cc + ly * ss;
                    let iy = -lx * ss + ly * cc;
                    let smp = textureSample(input, samp, (vec2<f32>(ix, iy) + 0.5) * p.texel);
                    sum = sum + smp[ch] / max(smp.a, 1e-4);
                    n = n + 1.0;
                    xx = xx + 1.0;
                }
                yy = yy + 1.0;
            }
            let avg = sum / max(n, 1.0);
            // Darker channel (lower value) → bigger dot of full ink. Max dot
            // radius is half the cell diagonal so a 0-value cell is solid.
            let maxr = cell * 0.70710678; // half diagonal
            let dotr = (1.0 - avg) * maxr;
            let d = length(vec2<f32>(rx - cellcx, ry - cellcy));
            // Inside the dot → channel value 0 (full ink); outside → 1 (paper).
            if d <= dotr {
                out[ch] = 0.0;
            } else {
                out[ch] = 1.0;
            }
        }
        return vec4<f32>(out * src.a, src.a);
    } else if p.kind == 23u {
        // ---- Mezzotint (kind 23): seeded threshold dither to pure black/white
        // dots. Each pixel's luma is compared against a per-pixel hashed
        // threshold; a denser hash for darker pixels yields a stochastic
        // black/white grain. amount biases the threshold; dir.x = seed.
        let dim = 1.0 / p.texel;
        let px = in.uv * dim;
        let ix = floor(px.x);
        let iy = floor(px.y);
        let seed = p.dir.x;
        let src = textureSample(input, samp, in.uv);
        let a = max(src.a, 1e-4);
        let col = src.rgb / a;
        let lw = vec3<f32>(0.2126, 0.7152, 0.0722);
        let luma = dot(col, lw);
        let t = hash21(ix, iy, seed);
        // Dither: keep white where luma exceeds the threshold, else black.
        let v = select(0.0, 1.0, luma > t + (p.amount - 0.5));
        return vec4<f32>(vec3<f32>(v) * src.a, src.a);
    } else if p.kind == 24u {
        // ---- High Pass (kind 24): the classic Photoshop sharpen prep —
        // subtract a Gaussian-blurred copy from the original and re-centre at
        // mid-gray, so flat areas go neutral gray and only the (high-frequency)
        // detail/edges survive as a signed deviation about 0.5. The combine pass
        // reads the blurred copy from `input` and the untouched source from
        // `orig`; the Gaussian blur itself runs as two prior kind-1 passes
        // CPU-side. Work on the unpremultiplied colour (matching add-noise), then
        // re-premultiply, so a transparent edge doesn't bias the difference. The
        // source alpha is preserved. `amount` scales the detail (1 = identity
        // high pass) for a softer/stronger result.
        let src = textureSample(orig, samp, in.uv);
        let blr = textureSample(input, samp, in.uv);
        let sa = max(src.a, 1e-4);
        let ba = max(blr.a, 1e-4);
        let scol = src.rgb / sa;
        let bcol = blr.rgb / ba;
        let hp = clamp(0.5 + (scol - bcol) * p.amount, vec3<f32>(0.0), vec3<f32>(1.0));
        return vec4<f32>(hp * src.a, src.a);
    } else if p.kind == 25u || p.kind == 26u {
        // ---- Clouds (kind 25) / Difference Clouds (kind 26): a generator —
        // fill the layer with a deterministic multi-octave value-noise (fBm)
        // field. dir.x = seed, dir.y = roughness, amount = scale (base feature
        // size px), radius = octave count. The noise is seamless and stable per
        // seed (matches the diffuse/add-noise hash philosophy). Difference Clouds
        // composites it against the existing pixels via absolute difference, so
        // repeated application builds the classic veins. The result is fully
        // opaque (a generator paints the whole layer). For Clouds the source
        // pixels are ignored; for Difference Clouds the unpremultiplied source
        // colour is differenced channel-wise against the (monochrome) cloud.
        let dim = 1.0 / p.texel;
        let px = in.uv * dim;
        let seed = p.dir.x;
        let roughness = clamp(p.dir.y, 0.05, 0.95);
        let scale = max(p.amount, 1.0);
        let octaves = clamp(i32(p.radius), 1, 10);
        let n = fbm(px, seed, scale, roughness, octaves);
        if p.kind == 25u {
            return vec4<f32>(vec3<f32>(n), 1.0);
        }
        // Difference Clouds: |source − cloud| per channel (Photoshop-style).
        let src = textureSample(input, samp, in.uv);
        let a = max(src.a, 1e-4);
        let base = src.rgb / a;
        let diff = abs(base - vec3<f32>(n));
        return vec4<f32>(diff, 1.0);
    } else if p.kind == 27u {
        // ---- Oil Paint (kind 27): Kuwahara quadrant filter. Split the
        // (2r+1)² window around the pixel into four overlapping (r+1)×(r+1)
        // quadrants that share the centre; compute each quadrant's mean colour
        // and luma variance, then output the mean of the *lowest-variance*
        // quadrant. Picking the flattest quadrant smooths interiors while
        // snapping to the side of an edge, giving the characteristic painterly
        // patches with crisp boundaries. Work in premultiplied space (matching
        // the blur/mosaic convention) so partial alpha is weighted correctly.
        let r = i32(clamp(p.radius, 1.0, 8.0));
        var best_var = 1e30;
        var best_mean = textureSample(input, samp, in.uv);
        // The four quadrant offset ranges (relative to the centre pixel).
        var qx0 = array<i32, 4>(-r, 0, -r, 0);
        var qx1 = array<i32, 4>(0, r, 0, r);
        var qy0 = array<i32, 4>(-r, -r, 0, 0);
        var qy1 = array<i32, 4>(0, 0, r, r);
        for (var q = 0; q < 4; q = q + 1) {
            var sum = vec4<f32>(0.0);
            var lsum = 0.0;
            var l2sum = 0.0;
            var n = 0.0;
            for (var dy = qy0[q]; dy <= qy1[q]; dy = dy + 1) {
                for (var dx = qx0[q]; dx <= qx1[q]; dx = dx + 1) {
                    let s = textureSample(
                        input, samp,
                        in.uv + vec2<f32>(f32(dx), f32(dy)) * p.texel,
                    );
                    let lw = vec3<f32>(0.2126, 0.7152, 0.0722);
                    let lm = dot(s.rgb, lw);
                    sum = sum + s;
                    lsum = lsum + lm;
                    l2sum = l2sum + lm * lm;
                    n = n + 1.0;
                }
            }
            let mean = sum / n;
            let lmean = lsum / n;
            let variance = l2sum / n - lmean * lmean;
            if variance < best_var {
                best_var = variance;
                best_mean = mean;
            }
        }
        return best_mean;
    } else if p.kind == 28u {
        // ---- Posterize (kind 28): quantize each channel to `amount` (=N) evenly
        // spaced levels in *display* (sRGB) space, the classic Image ▸ Adjustments
        // ▸ Posterize. Unpremultiply, encode to sRGB, snap each channel to the
        // nearest of N levels — floor(c·(N−1)+0.5)/(N−1) — decode back to linear
        // and re-premultiply. Alpha is preserved. With N=2 every channel snaps to
        // its 0 or 1 extreme.
        let src = textureSample(input, samp, in.uv);
        let a = max(src.a, 1e-4);
        let levels = max(floor(p.amount + 0.5), 2.0);
        let s = l2s(src.rgb / a);
        let q = floor(s * (levels - 1.0) + 0.5) / (levels - 1.0);
        let lin = s2l(clamp(q, vec3<f32>(0.0), vec3<f32>(1.0)));
        return vec4<f32>(lin * src.a, src.a);
    } else if p.kind == 29u {
        // ---- Threshold (kind 29): convert to pure black/white by comparing each
        // pixel's display-space Rec.709 luma against the `amount` cutoff (0..1).
        // At/above the cutoff → white, below → black. Compute luma on the sRGB
        // (display) colour so the cutoff matches the histogram the user sees;
        // alpha is preserved.
        let src = textureSample(input, samp, in.uv);
        let a = max(src.a, 1e-4);
        let lw = vec3<f32>(0.2126, 0.7152, 0.0722);
        let luma = dot(l2s(src.rgb / a), lw);
        let v = select(0.0, 1.0, luma >= p.amount);
        return vec4<f32>(vec3<f32>(v) * src.a, src.a);
    } else if p.kind == 16u {
        // ---- Diffuse (kind 16): seeded-deterministic anisotropic neighbour
        // swap. Replace each pixel with one of its neighbours within `amount`
        // px, the offset chosen by a hash of (x, y, seed) — so it's a stable,
        // reproducible scramble (no temporal noise), unlike a random()-per-frame.
        let dim = 1.0 / p.texel;
        let px = in.uv * dim;
        let ix = floor(px.x);
        let iy = floor(px.y);
        // Integer hash → two angles in [0, 2π); pick a direction + distance.
        let seed = p.dir.x;
        let hf = fract(sin(dot(vec2<f32>(ix, iy), vec2<f32>(12.9898, 78.233)) + seed) * 43758.5453);
        let hg = fract(sin(dot(vec2<f32>(ix, iy), vec2<f32>(39.3468, 11.135)) + seed) * 24634.6345);
        let ang = hf * 6.28318530718;
        let dist = hg * max(p.amount, 0.0);
        let off = vec2<f32>(cos(ang), sin(ang)) * dist;
        return textureSample(input, samp, (px + off) * p.texel);
    } else {
        // ---- Stylize edge filters (kinds 13/14/15): Sobel gradient over a
        // `radius`-px sampling step. Work in the linear-premultiplied source.
        let w = max(p.radius, 1.0);
        let ox = vec2<f32>(p.texel.x * w, 0.0);
        let oy = vec2<f32>(0.0, p.texel.y * w);
        let tl = textureSample(input, samp, in.uv - ox - oy);
        let tc = textureSample(input, samp, in.uv - oy);
        let tr = textureSample(input, samp, in.uv + ox - oy);
        let ml = textureSample(input, samp, in.uv - ox);
        let mc = textureSample(input, samp, in.uv);
        let mr = textureSample(input, samp, in.uv + ox);
        let bl = textureSample(input, samp, in.uv - ox + oy);
        let bc = textureSample(input, samp, in.uv + oy);
        let br = textureSample(input, samp, in.uv + ox + oy);
        // Rec.709 luma of each (premultiplied) tap.
        let lw = vec3<f32>(0.2126, 0.7152, 0.0722);
        let l_tl = dot(tl.rgb, lw); let l_tc = dot(tc.rgb, lw); let l_tr = dot(tr.rgb, lw);
        let l_ml = dot(ml.rgb, lw);                              let l_mr = dot(mr.rgb, lw);
        let l_bl = dot(bl.rgb, lw); let l_bc = dot(bc.rgb, lw); let l_br = dot(br.rgb, lw);
        // Sobel kernels.
        let gx = (l_tr + 2.0 * l_mr + l_br) - (l_tl + 2.0 * l_ml + l_bl);
        let gy = (l_bl + 2.0 * l_bc + l_br) - (l_tl + 2.0 * l_tc + l_tr);
        let mag = sqrt(gx * gx + gy * gy);
        if p.kind == 13u {
            // Find Edges: white background, dark edges (PS-style). Invert the
            // gradient magnitude → high edges go dark. Keep the source alpha.
            let v = clamp(1.0 - mag, 0.0, 1.0);
            return vec4<f32>(v * mc.a, v * mc.a, v * mc.a, mc.a);
        } else if p.kind == 14u {
            // Emboss: directional gray relief. Project the gradient onto the
            // light direction (p.dir = unit (cos,sin)); mid-gray + signed slope.
            let g = vec2<f32>(gx, gy);
            let v = clamp(0.5 + dot(g, p.dir) * p.amount, 0.0, 1.0);
            return vec4<f32>(v * mc.a, v * mc.a, v * mc.a, mc.a);
        } else {
            // Glowing Edges (kind 15): bright coloured edges on black. Scale the
            // center colour by the (boosted) edge magnitude → edges glow, flats
            // go black. amount = brightness gain.
            let g = clamp(mag * p.amount, 0.0, 1.0);
            // Recover an unpremultiplied edge colour, then re-premultiply by the
            // edge strength so the result is a glowing, alpha-consistent edge.
            let base = mc.rgb / max(mc.a, 1e-4);
            let col = base * g;
            return vec4<f32>(col * mc.a, mc.a);
        }
    }
}
