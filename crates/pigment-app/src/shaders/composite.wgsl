// Layer compositing pass (ping-pong). Reads the running backdrop + one layer,
// both Rgba16Float linear premultiplied, and writes the blended result.
// Blend math per RESEARCH.md §2 (W3C compositing, premultiplied).

struct Params {
    opacity: f32,
    blend_mode: u32,
    _p0: u32,
    _p1: u32,
};

@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var backdrop: texture_2d<f32>;
@group(0) @binding(2) var layer_tex: texture_2d<f32>;
@group(0) @binding(3) var<uniform> params: Params;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    // Fullscreen triangle.
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0),
    );
    let c = p[vi];
    var out: VsOut;
    out.pos = vec4<f32>(c, 0.0, 1.0);
    out.uv = vec2<f32>(c.x * 0.5 + 0.5, 0.5 - c.y * 0.5); // top-left origin
    return out;
}

fn blend_fn(mode: u32, b: vec3<f32>, s: vec3<f32>) -> vec3<f32> {
    switch mode {
        case 1u: { return b * s; }                       // Multiply
        case 2u: { return b + s - b * s; }               // Screen
        case 3u: {                                        // Overlay
            let lo = 2.0 * b * s;
            let hi = 1.0 - 2.0 * (1.0 - b) * (1.0 - s);
            return select(hi, lo, b <= vec3<f32>(0.5));
        }
        case 4u: { return min(b, s); }                   // Darken
        case 5u: { return max(b, s); }                   // Lighten
        case 12u: { return min(vec3<f32>(1.0), b + s); } // LinearDodge (Add)
        default: { return s; }                           // Normal
    }
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let b = textureSample(backdrop, samp, in.uv);          // premultiplied
    var s = textureSample(layer_tex, samp, in.uv) * params.opacity;

    let sa = max(s.a, 1e-5);
    let ba = max(b.a, 1e-5);
    let sc = s.rgb / sa;                                    // straight source
    let bc = b.rgb / ba;                                    // straight backdrop

    let blended = blend_fn(params.blend_mode, bc, sc);
    let mixed = (1.0 - b.a) * sc + b.a * blended;           // W3C blended color
    let out_rgb = s.a * mixed + b.rgb * (1.0 - s.a);        // premultiplied over
    let out_a = s.a + b.a * (1.0 - s.a);
    return vec4<f32>(out_rgb, out_a);
}
