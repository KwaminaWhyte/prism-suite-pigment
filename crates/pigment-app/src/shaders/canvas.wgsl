// Phase 0 canvas shader.
//
// Draws the document as a textured quad positioned in clip space by the CPU
// (pan/zoom already folded into clip_min/clip_max), plus a screen-fixed
// transparency checkerboard behind it. Two fragment entry points share one
// vertex shader; the app binds two pipelines over them.

struct Canvas {
    clip_min: vec2<f32>,   // top-left of the document quad, clip space
    clip_max: vec2<f32>,   // bottom-right of the document quad, clip space
    checker_px: f32,       // checker square size in physical pixels
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> canvas: Canvas;
@group(0) @binding(1) var img_tex: texture_2d<f32>;
@group(0) @binding(2) var img_smp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    // Two-triangle quad. uv (0,0)=top-left .. (1,1)=bottom-right.
    var uvs = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
    );
    let uv = uvs[vi];
    let x = mix(canvas.clip_min.x, canvas.clip_max.x, uv.x);
    let y = mix(canvas.clip_min.y, canvas.clip_max.y, uv.y);
    var out: VsOut;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = uv;
    return out;
}

@fragment
fn fs_checker(in: VsOut) -> @location(0) vec4<f32> {
    let sz = max(canvas.checker_px, 1.0);
    let cell = floor(in.pos.xy / sz);
    let parity = (cell.x + cell.y) - 2.0 * floor((cell.x + cell.y) * 0.5);
    let shade = select(0.5, 0.7, parity < 0.5); // two mid grays
    return vec4<f32>(shade, shade, shade, 1.0);
}

@fragment
fn fs_image(in: VsOut) -> @location(0) vec4<f32> {
    // img_tex is Rgba8Unorm holding sRGB-encoded bytes; the egui target is
    // also non-sRGB, so pass the sampled value straight through.
    return textureSample(img_tex, img_smp, in.uv);
}
