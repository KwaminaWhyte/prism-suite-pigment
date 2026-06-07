//! GPU canvas: per-layer Rgba16Float linear-premultiplied textures, a ping-pong
//! compositor (blend modes in-shader), instanced brush-dab painting, and a
//! display pass that composites over a checkerboard and sRGB-encodes for egui.
//! Phase 1 (PLAN.md §4). Tiling/atlas is the next refinement — layers are
//! currently one canvas-sized texture each (a degenerate single tile).

use std::collections::HashMap;

use eframe::egui_wgpu::{CallbackResources, CallbackTrait, ScreenDescriptor};
use eframe::wgpu;
use prism_core::{LayerId, Size};

/// One instanced brush dab, in document pixel space; color is straight linear.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Dab {
    pub center: [f32; 2],
    pub radius: f32,
    pub hardness: f32,
    pub color: [f32; 4],
}

/// Per-frame layer compositing parameters supplied by the app.
#[derive(Clone, Copy)]
pub struct LayerDraw {
    pub id: LayerId,
    pub opacity: f32,
    pub blend: u32,
    pub visible: bool,
    /// 0 = raster layer; else an adjustment kind applied to the backdrop.
    pub adjust_kind: u32,
    pub adjust: [f32; 4],
    /// Channel-Mixer matrix (adjust_kind 14): per-output [from_r,from_g,from_b,const].
    pub mix_r: [f32; 4],
    pub mix_g: [f32; 4],
    pub mix_b: [f32; 4],
    /// Blend-If: gate the layer by its own + the backdrop's luma.
    pub has_blend_if: bool,
    pub blend_if: [f32; 4], // [this_black, this_white, under_black, under_white]
    /// Clip to the layer directly below (its alpha gates this layer).
    pub clipped: bool,
    /// Outer-stroke layer style.
    pub has_stroke: bool,
    pub stroke_color: [f32; 4],
    pub stroke_width: f32, // uv units
    /// Drop-shadow layer style.
    pub has_shadow: bool,
    pub shadow_color: [f32; 4],
    pub shadow_offset: [f32; 2], // uv units
    pub shadow_blur: f32,        // uv units
    /// Color-overlay layer style (a = strength).
    pub has_overlay: bool,
    pub overlay_color: [f32; 4],
    /// Inner-shadow layer style.
    pub has_inner_shadow: bool,
    pub inner_shadow_color: [f32; 4],
    pub inner_shadow_offset: [f32; 2], // uv units
    pub inner_shadow_blur: f32,        // uv units
    /// Outer-glow layer style.
    pub has_outer_glow: bool,
    pub outer_glow_color: [f32; 4],
    pub outer_glow_size: f32, // uv units
    /// Inner-glow layer style.
    pub has_inner_glow: bool,
    pub inner_glow_color: [f32; 4],
    pub inner_glow_size: f32, // uv units
    /// Gradient-overlay layer style.
    pub has_grad_overlay: bool,
    pub grad_color0: [f32; 4],
    pub grad_color1: [f32; 4],
    pub grad_angle: f32,   // radians
    pub grad_opacity: f32, // 0..1
    /// Bevel-&-emboss layer style (Inner Bevel).
    pub has_bevel: bool,
    pub bevel_highlight: [f32; 4], // straight rgba (a = opacity)
    pub bevel_shadow: [f32; 4],    // straight rgba (a = opacity)
    pub bevel_size: f32,           // edge width, uv units
    pub bevel_soften: f32,         // extra blur of the normal field, uv units
    pub bevel_angle: f32,          // light azimuth, radians
    pub bevel_altitude: f32,       // light altitude, radians
}

/// A selection operation requested by the app for this frame.
#[derive(Clone, Copy)]
pub enum SelectionOp {
    /// Replace the selection with a rectangle/ellipse marquee (doc px).
    #[allow(dead_code)]
    Marquee {
        rect: [f32; 4],
        ellipse: bool,
    },
    All,
    None,
    Invert,
}

/// Pan/zoom state for the viewport.
#[derive(Clone, Copy, Debug)]
pub struct ViewTransform {
    pub pan: egui::Vec2,
    pub zoom: f32,
}

impl Default for ViewTransform {
    fn default() -> Self {
        Self {
            pan: egui::Vec2::ZERO,
            zoom: 1.0,
        }
    }
}

impl ViewTransform {
    pub fn zoom_to(&mut self, factor: f32, anchor_from_center: egui::Vec2) {
        let new_zoom = (self.zoom * factor).clamp(0.02, 64.0);
        let real = new_zoom / self.zoom;
        self.pan = (self.pan - anchor_from_center) * real + anchor_from_center;
        self.zoom = new_zoom;
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DisplayUniform {
    clip_min: [f32; 2],
    clip_max: [f32; 2],
    checker_px: f32,
    has_selection: f32,
    time: f32,
    canvas_w: f32,
    canvas_h: f32,
    _pad: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FilterParams {
    kind: u32,
    _p: [u32; 3],
    texel: [f32; 2],
    dir: [f32; 2],
    amount: f32,
    radius: f32,
    center: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ShapeUniform {
    rect: [f32; 4],
    size: [f32; 2],
    kind: u32,
    _p: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CompositeParams {
    opacity: f32,
    blend_mode: u32,
    has_xform: u32,
    adjust_kind: u32,
    m: [f32; 4],
    off: [f32; 2],
    _p1: [f32; 2],
    adjust: [f32; 4],
    has_blend_if: u32,
    has_clip: u32,   // clip this layer to the layer below's alpha
    has_stroke: u32, // outer-stroke layer style
    stroke_w: f32,   // stroke half-width in uv units
    /// Blend-If luma ranges: [this_black, this_white, under_black, under_white].
    blend_if: [f32; 4],
    /// Stroke color (straight, premultiplied at use): [r, g, b, a].
    stroke_color: [f32; 4],
    has_shadow: u32,         // drop-shadow layer style
    shadow_blur: f32,        // shadow softness radius, uv units
    shadow_off: [f32; 2],    // shadow offset, uv units
    shadow_color: [f32; 4],  // straight rgba
    has_overlay: u32,             // color-overlay layer style
    has_inner_shadow: u32,        // inner-shadow layer style
    has_outer_glow: u32,          // outer-glow layer style
    has_inner_glow: u32,          // inner-glow layer style
    overlay_color: [f32; 4],      // straight rgba (a = strength)
    inner_shadow_color: [f32; 4], // straight rgba
    inner_shadow_off: [f32; 2],   // uv units
    inner_shadow_blur: f32,       // uv units
    outer_glow_size: f32,         // uv units
    outer_glow_color: [f32; 4],   // straight rgba
    inner_glow_color: [f32; 4],   // straight rgba
    inner_glow_size: f32,         // uv units
    has_grad_overlay: u32,        // gradient-overlay layer style
    grad_angle: f32,              // radians
    grad_opacity: f32,            // 0..1
    grad_color0: [f32; 4],        // straight rgb (a unused)
    grad_color1: [f32; 4],        // straight rgb (a unused)
    has_bevel: u32,               // bevel-&-emboss layer style (Inner Bevel)
    bevel_size: f32,              // edge width, uv units
    bevel_soften: f32,            // extra normal-field blur, uv units
    _pb: f32,
    bevel_light: [f32; 4],        // light direction (xyz unit vector; w unused)
    bevel_highlight: [f32; 4],    // straight rgba (a = opacity)
    bevel_shadow: [f32; 4],       // straight rgba (a = opacity)
    // Channel-Mixer matrix (adjust_kind 14): per-output [from_r, from_g, from_b, const].
    mix_r: [f32; 4],
    mix_g: [f32; 4],
    mix_b: [f32; 4],
}

impl CompositeParams {
    fn plain(opacity: f32, blend_mode: u32) -> Self {
        Self {
            opacity,
            blend_mode,
            has_xform: 0,
            adjust_kind: 0,
            m: [1.0, 0.0, 0.0, 1.0],
            off: [0.0; 2],
            _p1: [0.0; 2],
            adjust: [0.0; 4],
            has_blend_if: 0,
            has_clip: 0,
            has_stroke: 0,
            stroke_w: 0.0,
            blend_if: [0.0, 1.0, 0.0, 1.0],
            stroke_color: [0.0; 4],
            has_shadow: 0,
            shadow_blur: 0.0,
            shadow_off: [0.0; 2],
            shadow_color: [0.0; 4],
            has_overlay: 0,
            has_inner_shadow: 0,
            has_outer_glow: 0,
            has_inner_glow: 0,
            overlay_color: [0.0; 4],
            inner_shadow_color: [0.0; 4],
            inner_shadow_off: [0.0; 2],
            inner_shadow_blur: 0.0,
            outer_glow_size: 0.0,
            outer_glow_color: [0.0; 4],
            inner_glow_color: [0.0; 4],
            inner_glow_size: 0.0,
            has_grad_overlay: 0,
            grad_angle: 0.0,
            grad_opacity: 0.0,
            grad_color0: [0.0; 4],
            grad_color1: [0.0; 4],
            has_bevel: 0,
            bevel_size: 0.0,
            bevel_soften: 0.0,
            _pb: 0.0,
            bevel_light: [0.0; 4],
            bevel_highlight: [0.0; 4],
            bevel_shadow: [0.0; 4],
            mix_r: [1.0, 0.0, 0.0, 0.0],
            mix_g: [0.0, 1.0, 0.0, 0.0],
            mix_b: [0.0, 0.0, 1.0, 0.0],
        }
    }
}

// The compositor binds CompositeParams via dynamic-offset slots spaced by
// PARAMS_STRIDE, so the struct must fit within one slot. It must also be a
// multiple of 16 bytes to match the WGSL std140 uniform layout exactly. Both
// are compile-time invariants.
const _: () = {
    assert!(std::mem::size_of::<CompositeParams>() as u64 <= PARAMS_STRIDE);
    assert!(std::mem::size_of::<CompositeParams>().is_multiple_of(16));
};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LayerInfo {
    size: [f32; 2],
    has_selection: f32,
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CloneParams {
    /// destAnchor − sourceAnchor in document px (aligned Clone Stamp).
    offset: [f32; 2],
    _pad: [f32; 2],
}

struct GpuLayer {
    tex: wgpu::Texture,
    view: wgpu::TextureView,
}

/// A region pixel snapshot held GPU-side for one undo step. `rect` is the
/// `[x, y, w, h]` sub-region of the layer the snapshot covers (region-COW: a
/// stroke only snapshots the tiles/area it touched, not the whole layer).
struct Snapshot {
    id: LayerId,
    tex: wgpu::Texture,
    rect: [u32; 4],
    label: String,
}

const FMT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const SEL_FMT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;
/// Width of a Curves tone-curve LUT (256 input levels, 1px tall, Rgba16Float).
const LUT_W: u32 = 256;
const MAX_LAYERS: u64 = 64;
// Dynamic-offset alignment must be a multiple of 256; CompositeParams grew past
// 256 bytes once the full layer-style set landed (currently 352, after Bevel &
// Emboss), so the slot stride is 512. A compile-time assert guards this.
const PARAMS_STRIDE: u64 = 512;
const UNDO_MAX: usize = 32;
/// Params slot reserved for the wet stroke (last slot in the buffer).
const WET_PARAMS_OFFSET: u32 = ((MAX_LAYERS - 1) * PARAMS_STRIDE) as u32;

/// Erase blend: dst *= (1 - dab_alpha). Removes coverage.
const ERASE_BLEND: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::Zero,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::Zero,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    },
};

fn make_target(device: &wgpu::Device, size: Size, label: &str) -> GpuLayer {
    make_target_fmt(device, size, label, FMT)
}

/// A 256×1 Rgba16Float tone-curve LUT target.
fn make_lut_target(device: &wgpu::Device, label: &str) -> GpuLayer {
    make_target_fmt(device, Size::new(LUT_W, 1), label, FMT)
}

/// Upload `LUT_W*4` f16 texels (interleaved rgba) into a LUT texture.
fn write_lut(queue: &wgpu::Queue, tex: &wgpu::Texture, texels: &[half::f16]) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(texels),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(LUT_W * 4 * 2), // rgba * 2 bytes (f16)
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: LUT_W,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
}

fn make_target_fmt(
    device: &wgpu::Device,
    size: Size,
    label: &str,
    format: wgpu::TextureFormat,
) -> GpuLayer {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size.width.max(1),
            height: size.height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    GpuLayer { tex, view }
}

/// All long-lived canvas GPU state, stored as a single egui callback resource.
pub struct CanvasGpu {
    sampler: wgpu::Sampler,

    // Compositor.
    composite_pipeline: wgpu::RenderPipeline,
    composite_bgl: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,

    // Brush dabs.
    dab_pipeline: wgpu::RenderPipeline,
    dab_erase_pipeline: wgpu::RenderPipeline,
    dab_bgl: wgpu::BindGroupLayout,
    dab_info_buf: wgpu::Buffer,
    dab_instances: wgpu::Buffer,
    dab_capacity: u64,

    // Clone Stamp: a clone dab pass samples a frozen source snapshot.
    clone_pipeline: wgpu::RenderPipeline,
    clone_bgl: wgpu::BindGroupLayout,
    clone_params_buf: wgpu::Buffer,
    /// Pre-stroke source snapshot for the active clone stroke (sampleable).
    clone_src: Option<GpuLayer>,

    // Undo/redo: region snapshots. A stroke copies its layer to `stroke_pre` at
    // start, then on commit extracts only the dirty sub-rect into the stack.
    undo_stack: Vec<Snapshot>,
    redo_stack: Vec<Snapshot>,
    stroke_pre: Option<wgpu::Texture>,
    stroke_owner: Option<LayerId>,
    stroke_label: String,

    // Display.
    display_pipeline: wgpu::RenderPipeline,
    display_bgl: wgpu::BindGroupLayout,
    display_uniform: wgpu::Buffer,
    display_bind_group: Option<wgpu::BindGroup>,

    // Document state mirror.
    layers: HashMap<LayerId, GpuLayer>,
    /// Per-Curves-layer 256×1 LUT textures (rgba = r/g/b/master tone curves).
    curve_luts: HashMap<LayerId, GpuLayer>,
    /// Identity LUT bound to composite passes that have no Curves layer (the
    /// shader only samples it when `adjust_kind == 8`, so it must still be valid).
    identity_lut: Option<GpuLayer>,
    canvas_size: Size,
    ping: Option<GpuLayer>,
    pong: Option<GpuLayer>,

    // In-progress brush stroke ("wet ink") composited over its owner layer and
    // flattened on pen-up — gives correct per-stroke opacity (RESEARCH.md §4).
    wet: Option<GpuLayer>,
    wet_owner: Option<LayerId>,
    wet_opacity: f32,

    // Frame-level dirty tracking: skip recompositing when the document is
    // unchanged (pan/zoom only touch the display pass). Per-tile region
    // invalidation needs the tile model (deferred).
    last_final_is_ping: bool,
    composite_valid: bool,

    // Selection (R16Float mask; 1 = selected). Marquee + invert pipelines.
    selection: Option<GpuLayer>,
    selection_tmp: Option<GpuLayer>,
    has_selection: bool,
    /// Saved selections (alpha channels): name → R16Float mask, canvas-sized.
    channels: Vec<(String, GpuLayer)>,
    shape_pipeline: wgpu::RenderPipeline,
    shape_bgl: wgpu::BindGroupLayout,
    shape_uniform: wgpu::Buffer,
    invert_pipeline: wgpu::RenderPipeline,
    invert_bgl: wgpu::BindGroupLayout,

    // Live move/transform of the active layer (uv-space layer-from-canvas affine).
    xform_layer: Option<LayerId>,
    xform_m: [f32; 4],
    xform_off: [f32; 2],

    // Per-layer masks (Rgba16Float; .r multiplies layer alpha). 1x1 white fallback.
    masks: HashMap<LayerId, GpuLayer>,
    white_mask: GpuLayer,
    white_written: bool,

    // Destructive filters.
    filter_pipeline: wgpu::RenderPipeline,
    filter_bgl: wgpu::BindGroupLayout,
    filter_uniform: wgpu::Buffer,
}

impl CanvasGpu {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let composite_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("composite.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/composite.wgsl").into()),
        });
        let dab_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dab.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/dab.wgsl").into()),
        });
        let display_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("display.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/display.wgsl").into()),
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("canvas.sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        // --- Compositor pipeline ---
        let composite_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("composite.bgl"),
            entries: &[
                sampler_entry(0),
                tex_entry(1),
                tex_entry(2),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<CompositeParams>() as u64,
                        ),
                    },
                    count: None,
                },
                tex_entry(4),
                tex_entry(5),
                tex_entry(6),
            ],
        });
        let composite_pipeline = make_render_pipeline(
            device,
            "composite",
            &composite_mod,
            "fs_main",
            &[&composite_bgl],
            &[],
            FMT,
            None, // we overwrite every pixel (read backdrop via sampler)
        );
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("composite.params"),
            size: PARAMS_STRIDE * MAX_LAYERS,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Dab pipeline ---
        let dab_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dab.bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                sampler_entry(1),
                tex_entry(2),
            ],
        });
        const DAB_ATTRS: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
            0 => Float32x2, 1 => Float32, 2 => Float32, 3 => Float32x4
        ];
        let dab_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Dab>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &DAB_ATTRS,
        };
        let dab_pipeline = make_render_pipeline(
            device,
            "dab",
            &dab_mod,
            "fs_main",
            &[&dab_bgl],
            std::slice::from_ref(&dab_layout),
            FMT,
            Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
        );
        let dab_erase_pipeline = make_render_pipeline(
            device,
            "dab.erase",
            &dab_mod,
            "fs_main",
            &[&dab_bgl],
            std::slice::from_ref(&dab_layout),
            FMT,
            Some(ERASE_BLEND),
        );
        let dab_info_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dab.info"),
            size: std::mem::size_of::<LayerInfo>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dab_capacity = 1024;
        let dab_instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dab.instances"),
            size: dab_capacity * std::mem::size_of::<Dab>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Clone-stamp pipeline (samples a frozen source + offset) ---
        let clone_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("clone.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/clone.wgsl").into()),
        });
        let clone_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("clone.bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                sampler_entry(1),
                tex_entry(2),
                tex_entry(3),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let clone_pipeline = make_render_pipeline(
            device,
            "clone",
            &clone_mod,
            "fs_main",
            &[&clone_bgl],
            std::slice::from_ref(&dab_layout),
            FMT,
            Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
        );
        let clone_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("clone.params"),
            size: std::mem::size_of::<CloneParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Display pipeline ---
        let display_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("display.bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                sampler_entry(1),
                tex_entry(2),
                tex_entry(3),
            ],
        });
        let display_pipeline = make_render_pipeline(
            device,
            "display",
            &display_mod,
            "fs_main",
            &[&display_bgl],
            &[],
            target_format,
            None,
        );

        // --- Selection pipelines ---
        let selection_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("selection.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/selection.wgsl").into()),
        });
        let shape_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sel.shape.bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let shape_pipeline = make_render_pipeline(
            device,
            "sel.shape",
            &selection_mod,
            "fs_shape",
            &[&shape_bgl],
            &[],
            SEL_FMT,
            None,
        );
        let invert_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sel.invert.bgl"),
            entries: &[sampler_entry(1), tex_entry(2)],
        });
        let invert_pipeline = make_render_pipeline(
            device,
            "sel.invert",
            &selection_mod,
            "fs_invert",
            &[&invert_bgl],
            &[],
            SEL_FMT,
            None,
        );
        let shape_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sel.shape.uniform"),
            size: std::mem::size_of::<ShapeUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Filter pipeline ---
        let filter_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("filter.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/filter.wgsl").into()),
        });
        let filter_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("filter.bgl"),
            entries: &[
                sampler_entry(0),
                tex_entry(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let filter_pipeline = make_render_pipeline(
            device,
            "filter",
            &filter_mod,
            "fs_main",
            &[&filter_bgl],
            &[],
            FMT,
            None,
        );
        let filter_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("filter.uniform"),
            size: std::mem::size_of::<FilterParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let display_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("display.uniform"),
            size: std::mem::size_of::<DisplayUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            sampler,
            composite_pipeline,
            composite_bgl,
            params_buf,
            dab_pipeline,
            dab_erase_pipeline,
            dab_bgl,
            dab_info_buf,
            dab_instances,
            clone_pipeline,
            clone_bgl,
            clone_params_buf,
            clone_src: None,
            dab_capacity,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            stroke_pre: None,
            stroke_owner: None,
            stroke_label: String::new(),
            display_pipeline,
            display_bgl,
            display_uniform,
            display_bind_group: None,
            layers: HashMap::new(),
            curve_luts: HashMap::new(),
            identity_lut: None,
            canvas_size: Size::new(1, 1),
            ping: None,
            pong: None,
            wet: None,
            wet_owner: None,
            wet_opacity: 1.0,
            last_final_is_ping: true,
            composite_valid: false,
            selection: None,
            selection_tmp: None,
            has_selection: false,
            channels: Vec::new(),
            shape_pipeline,
            shape_bgl,
            shape_uniform,
            invert_pipeline,
            invert_bgl,
            xform_layer: None,
            xform_m: [1.0, 0.0, 0.0, 1.0],
            xform_off: [0.0; 2],
            masks: HashMap::new(),
            white_mask: make_target_fmt(device, Size::new(1, 1), "white.mask", FMT),
            white_written: false,
            filter_pipeline,
            filter_bgl,
            filter_uniform,
        }
    }
}

mod brush;
mod channels;
mod clone;
mod command;
mod compositor;
mod layers;
mod paint;
mod selection;

pub use paint::CanvasPaint;

// CPU reference math for the blur-family filters — compiled only for tests
// (the live path is the GPU shader pass in `compositor`).
#[cfg(test)]
mod filter_math;

#[cfg(test)]
mod gpu_tests;

fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

fn tex_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

/// GPU-side full-texture copy (same size/format).
fn copy_tex(
    encoder: &mut wgpu::CommandEncoder,
    src: &wgpu::Texture,
    dst: &wgpu::Texture,
    size: Size,
) {
    encoder.copy_texture_to_texture(
        wgpu::TexelCopyTextureInfo {
            texture: src,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyTextureInfo {
            texture: dst,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::Extent3d {
            width: size.width.max(1),
            height: size.height.max(1),
            depth_or_array_layers: 1,
        },
    );
}

/// GPU-side copy of a `[w,h]` region from `src@src_xy` to `dst@dst_xy`.
fn copy_region(
    encoder: &mut wgpu::CommandEncoder,
    src: &wgpu::Texture,
    src_xy: [u32; 2],
    dst: &wgpu::Texture,
    dst_xy: [u32; 2],
    wh: [u32; 2],
) {
    encoder.copy_texture_to_texture(
        wgpu::TexelCopyTextureInfo {
            texture: src,
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: src_xy[0],
                y: src_xy[1],
                z: 0,
            },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyTextureInfo {
            texture: dst,
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: dst_xy[0],
                y: dst_xy[1],
                z: 0,
            },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::Extent3d {
            width: wh[0].max(1),
            height: wh[1].max(1),
            depth_or_array_layers: 1,
        },
    );
}

/// A clear-to-transparent color attachment for `view` (helper for inline descriptors).
fn clear_attachment(view: &wgpu::TextureView) -> wgpu::RenderPassColorAttachment<'_> {
    wgpu::RenderPassColorAttachment {
        view,
        depth_slice: None,
        resolve_target: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            store: wgpu::StoreOp::Store,
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn make_render_pipeline(
    device: &wgpu::Device,
    label: &str,
    module: &wgpu::ShaderModule,
    fs_entry: &str,
    bind_group_layouts: &[&wgpu::BindGroupLayout],
    vertex_buffers: &[wgpu::VertexBufferLayout],
    format: wgpu::TextureFormat,
    blend: Option<wgpu::BlendState>,
) -> wgpu::RenderPipeline {
    let bgls: Vec<Option<&wgpu::BindGroupLayout>> =
        bind_group_layouts.iter().map(|b| Some(*b)).collect();
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &bgls,
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some("vs_main"),
            buffers: vertex_buffers,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(fs_entry),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}
