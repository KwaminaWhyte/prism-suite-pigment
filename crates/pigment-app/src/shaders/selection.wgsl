// Selection mask passes. The mask is an R16Float texture (1 = selected). One
// pipeline rasterizes a rectangle/ellipse marquee; another inverts the mask.

struct Shape {
    rect: vec4<f32>, // x, y, w, h in document pixels
    size: vec2<f32>, // canvas size in pixels
    kind: u32,       // 0 = rectangle, 1 = ellipse
    _p: u32,
};

@group(0) @binding(0) var<uniform> shape: Shape;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0),
    );
    let c = p[vi];
    var out: VsOut;
    out.pos = vec4<f32>(c, 0.0, 1.0);
    out.uv = vec2<f32>(c.x * 0.5 + 0.5, 0.5 - c.y * 0.5);
    return out;
}

@fragment
fn fs_shape(in: VsOut) -> @location(0) vec4<f32> {
    let p = in.uv * shape.size;
    var inside = 0.0;
    if shape.kind == 0u {
        if p.x >= shape.rect.x && p.x <= shape.rect.x + shape.rect.z
            && p.y >= shape.rect.y && p.y <= shape.rect.y + shape.rect.w {
            inside = 1.0;
        }
    } else {
        let center = shape.rect.xy + shape.rect.zw * 0.5;
        let half = max(shape.rect.zw * 0.5, vec2<f32>(1e-3));
        let d = (p - center) / half;
        if dot(d, d) <= 1.0 {
            inside = 1.0;
        }
    }
    return vec4<f32>(inside, 0.0, 0.0, 1.0);
}

// --- Invert pass (distinct bindings so both pipelines share one module) ---
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var mask: texture_2d<f32>;

@fragment
fn fs_invert(in: VsOut) -> @location(0) vec4<f32> {
    let m = textureSample(mask, samp, in.uv).r;
    return vec4<f32>(1.0 - m, 0.0, 0.0, 1.0);
}
