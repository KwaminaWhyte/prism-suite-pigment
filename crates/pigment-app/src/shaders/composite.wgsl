// Layer compositing pass (ping-pong). Reads the running backdrop + one layer,
// both Rgba16Float linear premultiplied, and writes the blended result.
// Blend math per RESEARCH.md §2 (W3C compositing, premultiplied).

struct Params {
    opacity: f32,
    blend_mode: u32,
    has_xform: u32,
    adjust_kind: u32, // 0 = raster layer; else an adjustment of the backdrop
    m: vec4<f32>,    // 2x2 layer-from-canvas matrix (a,b,c,d), uv space
    off: vec2<f32>,  // uv-space offset
    _p1: vec2<f32>,
    adjust: vec4<f32>, // adjustment params
    has_blend_if: u32,
    _p2a: u32, // scalar pads (NOT vec3 — vec3 would force 16-byte align, mismatching [u32;3])
    _p2b: u32,
    _p2c: u32,
    blend_if: vec4<f32>, // this_black, this_white, under_black, under_white
};

@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var backdrop: texture_2d<f32>;
@group(0) @binding(2) var layer_tex: texture_2d<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var mask_tex: texture_2d<f32>; // R = layer mask (1x1 white if none)
@group(0) @binding(5) var lut_tex: texture_2d<f32>;  // Curves LUT 256x1: rgba = (rCurve, gCurve, bCurve, masterCurve)

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

fn lum(c: vec3<f32>) -> f32 { return dot(c, vec3<f32>(0.3, 0.59, 0.11)); }

// Blend-If keep factor: 1 when luma x is inside [lo, hi], ramping to 0 just
// outside (soft 0.05 feather), à la Photoshop's this/underlying sliders.
fn blend_if_factor(x: f32, lo: f32, hi: f32) -> f32 {
    let f = 0.05;
    let lower = smoothstep(lo, lo + f, x);
    let upper = 1.0 - smoothstep(hi - f, hi, x);
    return clamp(lower * upper, 0.0, 1.0);
}

fn clip_color(c: vec3<f32>) -> vec3<f32> {
    let l = lum(c);
    let n = min(c.r, min(c.g, c.b));
    let x = max(c.r, max(c.g, c.b));
    var col = c;
    if n < 0.0 { col = l + (col - l) * (l / (l - n + 1e-6)); }
    if x > 1.0 { col = l + (col - l) * ((1.0 - l) / (x - l + 1e-6)); }
    return col;
}
fn set_lum(c: vec3<f32>, l: f32) -> vec3<f32> { return clip_color(c + (l - lum(c))); }
fn sat(c: vec3<f32>) -> f32 { return max(c.r, max(c.g, c.b)) - min(c.r, min(c.g, c.b)); }
fn set_sat(c: vec3<f32>, s: f32) -> vec3<f32> {
    let mn = min(c.r, min(c.g, c.b));
    let mx = max(c.r, max(c.g, c.b));
    if mx > mn { return (c - mn) * (s / (mx - mn)); }
    return vec3<f32>(0.0);
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
        case 6u: {                                        // ColorDodge
            return select(min(vec3<f32>(1.0), b / max(1.0 - s, vec3<f32>(1e-6))),
                          vec3<f32>(1.0), s >= vec3<f32>(1.0));
        }
        case 7u: {                                        // ColorBurn
            return select(1.0 - min(vec3<f32>(1.0), (1.0 - b) / max(s, vec3<f32>(1e-6))),
                          vec3<f32>(0.0), s <= vec3<f32>(0.0));
        }
        case 8u: {                                        // HardLight
            let lo = 2.0 * b * s;
            let hi = 1.0 - 2.0 * (1.0 - b) * (1.0 - s);
            return select(hi, lo, s <= vec3<f32>(0.5));
        }
        case 9u: {                                        // SoftLight
            let d = select(sqrt(b), ((16.0 * b - 12.0) * b + 4.0) * b, b <= vec3<f32>(0.25));
            let lo = b - (1.0 - 2.0 * s) * b * (1.0 - b);
            let hi = b + (2.0 * s - 1.0) * (d - b);
            return select(hi, lo, s <= vec3<f32>(0.5));
        }
        case 10u: { return abs(b - s); }                 // Difference
        case 11u: { return b + s - 2.0 * b * s; }        // Exclusion
        case 12u: { return min(vec3<f32>(1.0), b + s); } // LinearDodge (Add)
        case 13u: { return max(vec3<f32>(0.0), b + s - 1.0); } // LinearBurn
        case 20u: { return set_lum(set_sat(s, sat(b)), lum(b)); } // Hue
        case 21u: { return set_lum(set_sat(b, sat(s)), lum(b)); } // Saturation
        case 22u: { return set_lum(s, lum(b)); }         // Color
        case 23u: { return set_lum(b, lum(s)); }         // Luminosity
        default: { return s; }                           // Normal
    }
}

fn l2s1(c: f32) -> f32 { if c <= 0.0031308 { return c * 12.92; } return 1.055 * pow(c, 1.0 / 2.4) - 0.055; }
fn s2l1(c: f32) -> f32 { if c <= 0.04045 { return c / 12.92; } return pow((c + 0.055) / 1.055, 2.4); }
fn l2s(c: vec3<f32>) -> vec3<f32> { return vec3<f32>(l2s1(c.x), l2s1(c.y), l2s1(c.z)); }
fn s2l(c: vec3<f32>) -> vec3<f32> { return vec3<f32>(s2l1(c.x), s2l1(c.y), s2l1(c.z)); }

fn rgb2hsl(c: vec3<f32>) -> vec3<f32> {
    let mx = max(c.r, max(c.g, c.b));
    let mn = min(c.r, min(c.g, c.b));
    let l = (mx + mn) * 0.5;
    var h = 0.0;
    var s = 0.0;
    let d = mx - mn;
    if d > 1e-6 {
        s = d / (1.0 - abs(2.0 * l - 1.0) + 1e-6);
        if mx == c.r { h = (c.g - c.b) / d + select(0.0, 6.0, c.g < c.b); }
        else if mx == c.g { h = (c.b - c.r) / d + 2.0; }
        else { h = (c.r - c.g) / d + 4.0; }
        h = h / 6.0;
    }
    return vec3<f32>(h, s, l);
}
fn hue2rgb(p: f32, q: f32, t_in: f32) -> f32 {
    var t = fract(t_in);
    if t < 1.0 / 6.0 { return p + (q - p) * 6.0 * t; }
    if t < 0.5 { return q; }
    if t < 2.0 / 3.0 { return p + (q - p) * (2.0 / 3.0 - t) * 6.0; }
    return p;
}
fn hsl2rgb(h: f32, s: f32, l: f32) -> vec3<f32> {
    if s <= 1e-6 { return vec3<f32>(l); }
    let q = select(l + s - l * s, l * (1.0 + s), l < 0.5);
    let p = 2.0 * l - q;
    return vec3<f32>(hue2rgb(p, q, h + 1.0 / 3.0), hue2rgb(p, q, h), hue2rgb(p, q, h - 1.0 / 3.0));
}

// Curves: master (composite) curve first — stored in the LUT's .a — then the
// per-channel curves (.rgb). textureSampleLevel (explicit LOD) so it's legal in
// the conditional adjustment branch. Identity LUT => no-op.
fn apply_curves(c: vec3<f32>) -> vec3<f32> {
    let mr = textureSampleLevel(lut_tex, samp, vec2<f32>(c.r, 0.5), 0.0).a;
    let mg = textureSampleLevel(lut_tex, samp, vec2<f32>(c.g, 0.5), 0.0).a;
    let mb = textureSampleLevel(lut_tex, samp, vec2<f32>(c.b, 0.5), 0.0).a;
    let cr = textureSampleLevel(lut_tex, samp, vec2<f32>(mr, 0.5), 0.0).r;
    let cg = textureSampleLevel(lut_tex, samp, vec2<f32>(mg, 0.5), 0.0).g;
    let cb = textureSampleLevel(lut_tex, samp, vec2<f32>(mb, 0.5), 0.0).b;
    return vec3<f32>(cr, cg, cb);
}

fn apply_adjust(kind: u32, p: vec4<f32>, c_lin: vec3<f32>) -> vec3<f32> {
    if kind == 5u {
        return c_lin * exp2(p.x); // exposure: linear-light multiply
    }
    var c = clamp(l2s(c_lin), vec3<f32>(0.0), vec3<f32>(1.0)); // perceptual ops in sRGB
    switch kind {
        case 1u: { c = (c - 0.5) * (1.0 + p.y) + 0.5 + p.x; }
        case 2u: {
            let denom = max(p.y - p.x, 1e-4);
            c = clamp((c - p.x) / denom, vec3<f32>(0.0), vec3<f32>(1.0));
            c = pow(c, vec3<f32>(1.0 / max(p.z, 1e-3)));
        }
        case 3u: {
            var hsl = rgb2hsl(c);
            hsl.x = fract(hsl.x + p.x / 360.0);
            hsl.y = clamp(hsl.y * (1.0 + p.y), 0.0, 1.0);
            hsl.z = clamp(hsl.z + p.z, 0.0, 1.0);
            c = hsl2rgb(hsl.x, hsl.y, hsl.z);
        }
        case 4u: { c = 1.0 - c; }
        case 6u: { let y = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722)); c = vec3<f32>(select(0.0, 1.0, y >= p.x)); }
        case 7u: { c = vec3<f32>(dot(c, vec3<f32>(0.2126, 0.7152, 0.0722))); }
        case 8u: { c = clamp(apply_curves(c), vec3<f32>(0.0), vec3<f32>(1.0)); }
        case 9u: { // Vibrance: boost more where saturation is low
            let mx = max(c.r, max(c.g, c.b));
            let mn = min(c.r, min(c.g, c.b));
            let sat = mx - mn;
            let lum = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
            let boost = p.x * (1.0 - sat);
            c = lum + (c - lum) * (1.0 + boost);
        }
        case 10u: { // Photo Filter: luminosity-preserving color tint
            let lum0 = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
            let tinted = c * p.xyz;
            let lum1 = max(dot(tinted, vec3<f32>(0.2126, 0.7152, 0.0722)), 1e-4);
            let preserved = tinted * (lum0 / lum1);
            c = mix(c, preserved, p.w);
        }
        case 11u: { // Posterize: quantize to p.x levels
            let n = max(p.x, 2.0) - 1.0;
            c = floor(c * n + 0.5) / n;
        }
        default: {}
    }
    return s2l(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let b = textureSample(backdrop, samp, in.uv);          // premultiplied

    // Adjustment layer: transform the backdrop, keep its alpha, mix by opacity.
    if params.adjust_kind != 0u {
        let ba = max(b.a, 1e-5);
        let bc = b.rgb / ba;
        let adj = apply_adjust(params.adjust_kind, params.adjust, bc);
        let mixed = mix(bc, adj, params.opacity);
        return vec4<f32>(mixed * b.a, b.a);
    }
    // Optional per-layer affine (move/transform preview): sample the layer at the
    // transformed uv, masking out anything that falls outside the layer.
    var luv = in.uv;
    if params.has_xform != 0u {
        let mm = mat2x2<f32>(params.m.x, params.m.y, params.m.z, params.m.w);
        luv = mm * in.uv + params.off;
    }
    let in_bounds = luv.x >= 0.0 && luv.x <= 1.0 && luv.y >= 0.0 && luv.y <= 1.0;
    let mask = textureSample(mask_tex, samp, in.uv).r; // mask is in canvas space
    var s = textureSample(layer_tex, samp, clamp(luv, vec2<f32>(0.0), vec2<f32>(1.0)))
        * (params.opacity * mask);
    if !in_bounds {
        s = vec4<f32>(0.0);
    }

    // Blend-If: gate the source by its own luma + the backdrop's luma.
    if params.has_blend_if != 0u {
        let s_lum = lum(s.rgb / max(s.a, 1e-5));
        let b_lum = lum(b.rgb / max(b.a, 1e-5));
        let f = blend_if_factor(s_lum, params.blend_if.x, params.blend_if.y)
              * blend_if_factor(b_lum, params.blend_if.z, params.blend_if.w);
        s = s * f;
    }

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
