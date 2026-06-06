// Brush dab pass. Renders instanced soft circles into a layer's Rgba16Float
// texture with premultiplied-over blending, building up a stroke. Dab data is
// in document pixel space; the layer size uniform maps it to clip space.

struct LayerInfo {
    size: vec2<f32>, // layer width/height in pixels
    _pad: vec2<f32>,
};

@group(0) @binding(0) var<uniform> layer: LayerInfo;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) local: vec2<f32>,   // -1..1 within the dab
    @location(1) color: vec4<f32>,   // straight linear rgba
    @location(2) hardness: f32,
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
    out.color = color;
    out.hardness = hardness;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let d = length(in.local);
    // 1 inside `hardness`, ramping to 0 at the edge.
    let a = in.color.a * (1.0 - smoothstep(in.hardness, 1.0, d));
    return vec4<f32>(in.color.rgb * a, a); // premultiplied
}
