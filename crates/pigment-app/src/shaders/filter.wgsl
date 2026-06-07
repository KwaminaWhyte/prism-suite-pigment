// Destructive filters applied to a layer. Fullscreen pass sampling `input`.
// Separable blurs (Gaussian, Box) run this twice (horizontal then vertical).

struct FParams {
    // 1 Gaussian blur, 2 sharpen, 3 pixelate, 4 motion blur,
    // 5 box blur (separable), 6 radial spin, 7 radial zoom,
    // 8 twirl, 9 pinch/spherize, 10 ripple/wave, 11 polar (rect->polar),
    // 12 polar (polar->rect).
    kind: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
    texel: vec2<f32>, // 1/size
    dir: vec2<f32>,   // blur direction * texel (Gaussian/box/motion); distort:
                      // (amplitude_px, wavelength_px) for ripple
    amount: f32,      // sharpen amount; radial: spin angle (rad) / zoom fraction;
                      // twirl: max angle (rad); pinch: signed amount (-1..1)
    radius: f32,      // blur radius / pixelate block / motion taps / radial
                      // samples; distort: effect radius in pixels
    center: vec2<f32>, // radial/distort center in uv (0..1)
};

@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var input: texture_2d<f32>;
@group(0) @binding(2) var<uniform> p: FParams;

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
    } else {
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
    }
}
