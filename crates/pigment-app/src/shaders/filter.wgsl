// Destructive filters applied to a layer. Fullscreen pass sampling `input`.
// Separable Gaussian blur runs this twice (horizontal then vertical).

struct FParams {
    kind: u32,     // 1 blur, 2 sharpen, 3 pixelate
    _p0: u32,
    _p1: u32,
    _p2: u32,
    texel: vec2<f32>, // 1/size
    dir: vec2<f32>,   // blur direction * texel
    amount: f32,      // sharpen amount
    radius: f32,      // blur radius / pixelate block size
    _q: vec2<f32>,
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
    } else {
        // Pixelate: snap uv to a block grid.
        let block = max(p.radius, 1.0);
        let px = (floor(in.uv / p.texel / block) + 0.5) * block * p.texel;
        return textureSample(input, samp, px);
    }
}
