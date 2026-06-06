// Clone-stamp dab pass. Same instanced soft-dab geometry as dab.wgsl, but each
// fragment samples a frozen source snapshot at (destDoc - offset) and composites
// it premultiplied-over, shaped by the brush falloff. This is the Clone Stamp
// (Phase 6 retouch): paint pixels copied from elsewhere in the image.

struct LayerInfo {
    size: vec2<f32>,
    has_selection: f32,
    _pad: f32,
};

// offset = destAnchor - sourceAnchor, in document pixels (aligned clone).
struct CloneParams {
    offset: vec2<f32>,
    _pad: vec2<f32>,
};

@group(0) @binding(0) var<uniform> layer: LayerInfo;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var sel_mask: texture_2d<f32>;
@group(0) @binding(3) var src_tex: texture_2d<f32>; // frozen pre-stroke source (premultiplied)
@group(0) @binding(4) var<uniform> clone: CloneParams;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) local: vec2<f32>,   // -1..1 within the dab
    @location(1) flow: f32,          // dab opacity/flow (color.a)
    @location(2) hardness: f32,
    @location(3) uv: vec2<f32>,       // document uv of the destination
};

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    @location(0) center: vec2<f32>,
    @location(1) radius: f32,
    @location(2) hardness: f32,
    @location(3) color: vec4<f32>,
) -> VsOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    );
    let local = corners[vi];
    let doc = center + local * radius;
    let clip = vec2<f32>(
        doc.x / layer.size.x * 2.0 - 1.0,
        1.0 - doc.y / layer.size.y * 2.0,
    );
    var out: VsOut;
    out.pos = vec4<f32>(clip, 0.0, 1.0);
    out.local = local;
    out.flow = color.a;
    out.hardness = hardness;
    out.uv = doc / layer.size;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let d = length(in.local);
    var a = in.flow * (1.0 - smoothstep(in.hardness, 1.0, d));
    if layer.has_selection > 0.5 {
        a = a * textureSampleLevel(sel_mask, samp, in.uv, 0.0).r;
    }
    let dst_doc = in.uv * layer.size;
    let src_uv = (dst_doc - clone.offset) / layer.size;
    // Outside the source image: nothing to stamp.
    if src_uv.x < 0.0 || src_uv.x > 1.0 || src_uv.y < 0.0 || src_uv.y > 1.0 {
        a = 0.0;
    }
    let src = textureSampleLevel(
        src_tex, samp, clamp(src_uv, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0
    ); // premultiplied
    let sa = max(src.a, 1e-5);
    let sc = src.rgb / sa;          // straight source color
    let out_a = a * src.a;          // coverage scaled by source alpha
    return vec4<f32>(sc * out_a, out_a); // premultiplied (composited over by the pipeline blend)
}
