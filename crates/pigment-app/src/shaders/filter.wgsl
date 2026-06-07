// Destructive filters applied to a layer. Fullscreen pass sampling `input`.
// Separable blurs (Gaussian, Box) run this twice (horizontal then vertical).

struct FParams {
    // 1 Gaussian blur, 2 sharpen, 3 pixelate, 4 motion blur,
    // 5 box blur (separable), 6 radial spin, 7 radial zoom.
    kind: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
    texel: vec2<f32>, // 1/size
    dir: vec2<f32>,   // blur direction * texel (Gaussian/box/motion)
    amount: f32,      // sharpen amount; radial: spin angle (rad) / zoom fraction
    radius: f32,      // blur radius / pixelate block / motion taps / radial samples
    center: vec2<f32>, // radial center in uv (0..1)
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
    } else {
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
    }
}
