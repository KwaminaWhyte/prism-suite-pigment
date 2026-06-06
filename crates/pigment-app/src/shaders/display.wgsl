// Display pass, run inside egui's render pass (non-sRGB Bgra8Unorm target).
// Samples the final composite (linear premultiplied), composites it over a
// transparency checkerboard in linear light, then sRGB-encodes for display.

struct Display {
    clip_min: vec2<f32>,
    clip_max: vec2<f32>,
    checker_px: f32,
    has_selection: f32,
    time: f32,
    canvas_w: f32,
    canvas_h: f32,
    _p0: f32,
    _p1: f32,
    _p2: f32,
};

@group(0) @binding(0) var<uniform> d: Display;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var composite: texture_2d<f32>;
@group(0) @binding(3) var selection: texture_2d<f32>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    var uvs = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
    );
    let uv = uvs[vi];
    let x = mix(d.clip_min.x, d.clip_max.x, uv.x);
    let y = mix(d.clip_min.y, d.clip_max.y, uv.y);
    var out: VsOut;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = uv;
    return out;
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let cl = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    let lo = cl * 12.92;
    let hi = 1.055 * pow(cl, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, cl <= vec3<f32>(0.0031308));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let c = textureSample(composite, samp, in.uv); // premultiplied linear

    // Checkerboard in linear light (≈ sRGB 0.5 / 0.66 grays).
    let sz = max(d.checker_px, 1.0);
    let cell = floor(in.pos.xy / sz);
    let parity = (cell.x + cell.y) - 2.0 * floor((cell.x + cell.y) * 0.5);
    let g = select(0.21, 0.4, parity < 0.5);
    let bg = vec3<f32>(g, g, g);

    let over = c.rgb + bg * (1.0 - c.a); // premultiplied over opaque bg
    var rgb = linear_to_srgb(over);

    // Marching ants: detect selection-mask boundary, draw animated dashes.
    if d.has_selection > 0.5 {
        let tx = vec2<f32>(1.0 / max(d.canvas_w, 1.0), 1.0 / max(d.canvas_h, 1.0));
        let m = textureSample(selection, samp, in.uv).r;
        let mx = textureSample(selection, samp, in.uv + vec2<f32>(tx.x, 0.0)).r;
        let my = textureSample(selection, samp, in.uv + vec2<f32>(0.0, tx.y)).r;
        let edge = max(abs(m - mx), abs(m - my));
        if edge > 0.3 {
            // Dash pattern marching along the boundary in screen space.
            let phase = fract((in.pos.x + in.pos.y) * 0.12 - d.time * 4.0);
            let ant = select(0.0, 1.0, phase < 0.5);
            rgb = vec3<f32>(ant, ant, ant);
        }
    }
    return vec4<f32>(rgb, 1.0);
}
