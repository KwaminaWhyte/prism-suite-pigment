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
    /// Blend-If: gate the layer by its own + the backdrop's luma.
    pub has_blend_if: bool,
    pub blend_if: [f32; 4], // [this_black, this_white, under_black, under_white]
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
    _q: [f32; 2],
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
    _p2: [u32; 3],
    /// Blend-If luma ranges: [this_black, this_white, under_black, under_white].
    blend_if: [f32; 4],
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
            _p2: [0; 3],
            blend_if: [0.0, 1.0, 0.0, 1.0],
        }
    }
}

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
const PARAMS_STRIDE: u64 = 256; // min uniform dynamic-offset alignment
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
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/composite.wgsl").into()),
        });
        let dab_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dab.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/dab.wgsl").into()),
        });
        let display_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("display.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/display.wgsl").into()),
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
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/clone.wgsl").into()),
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
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/selection.wgsl").into()),
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
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/filter.wgsl").into()),
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

    #[allow(clippy::too_many_arguments)]
    fn filter_pass(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &wgpu::TextureView,
        output: &wgpu::TextureView,
        kind: u32,
        dir: [f32; 2],
        amount: f32,
        radius: f32,
    ) {
        let (w, h) = (
            self.canvas_size.width as f32,
            self.canvas_size.height as f32,
        );
        queue.write_buffer(
            &self.filter_uniform,
            0,
            bytemuck::bytes_of(&FilterParams {
                kind,
                _p: [0; 3],
                texel: [1.0 / w, 1.0 / h],
                dir,
                amount,
                radius,
                _q: [0.0; 2],
            }),
        );
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("filter.bg"),
            layout: &self.filter_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(input),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.filter_uniform.as_entire_binding(),
                },
            ],
        });
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("filter.pass"),
                color_attachments: &[Some(clear_attachment(output))],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.filter_pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit([enc.finish()]);
    }

    /// Apply a destructive filter to a layer (kind: 1 blur, 2 sharpen, 3 pixelate).
    pub fn apply_filter(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: LayerId,
        kind: u32,
        radius: f32,
        amount: f32,
    ) {
        self.begin_command_now(device, queue, id, "Filter");
        let (Some(layer), Some(pong)) = (self.layers.get(&id), self.pong.as_ref()) else {
            return;
        };
        if kind == 1 {
            let tx = [1.0 / self.canvas_size.width as f32, 0.0];
            let ty = [0.0, 1.0 / self.canvas_size.height as f32];
            self.filter_pass(device, queue, &layer.view, &pong.view, 1, tx, 0.0, radius);
            self.filter_pass(device, queue, &pong.view, &layer.view, 1, ty, 0.0, radius);
        } else {
            self.filter_pass(
                device,
                queue,
                &layer.view,
                &pong.view,
                kind,
                [0.0; 2],
                amount,
                radius,
            );
            let mut enc = device.create_command_encoder(&Default::default());
            copy_tex(&mut enc, &pong.tex, &layer.tex, self.canvas_size);
            queue.submit([enc.finish()]);
        }
    }

    fn mask_view(&self, id: LayerId) -> &wgpu::TextureView {
        self.masks
            .get(&id)
            .map(|m| &m.view)
            .unwrap_or(&self.white_mask.view)
    }

    /// Initialize the 1x1 white mask fallback (used by maskless layers).
    fn ensure_white(&mut self, queue: &wgpu::Queue) {
        if self.white_written {
            return;
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.white_mask.tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[half::f16::from_f32(1.0).to_le_bytes(); 4].concat(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(8),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        self.white_written = true;
    }

    #[allow(dead_code)]
    pub fn has_mask(&self, id: LayerId) -> bool {
        self.masks.contains_key(&id)
    }

    /// Add/replace a layer mask from per-pixel values (1 = reveal). `None` => white.
    pub fn set_mask(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: LayerId,
        values: Option<&[f32]>,
    ) {
        let (w, h) = (self.canvas_size.width, self.canvas_size.height);
        let m = make_target_fmt(device, self.canvas_size, "layer.mask", FMT);
        let n = (w * h) as usize;
        let mut bytes = Vec::with_capacity(n * 8);
        for i in 0..n {
            let v = values.map_or(1.0, |s| s.get(i).copied().unwrap_or(0.0));
            for _ in 0..4 {
                bytes.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
            }
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &m.tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 8),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.masks.insert(id, m);
    }

    pub fn delete_mask(&mut self, id: LayerId) {
        self.masks.remove(&id);
    }

    /// Set the live affine on a layer (uv-space layer-from-canvas matrix + offset).
    pub fn set_layer_transform(&mut self, layer: Option<LayerId>, m: [f32; 4], off: [f32; 2]) {
        self.xform_layer = layer;
        self.xform_m = m;
        self.xform_off = off;
    }

    /// Bake the active transform into its layer's pixels, then clear it.
    pub fn bake_transform(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        self.ensure_white(queue);
        self.ensure_identity_lut(device, queue);
        let Some(id) = self.xform_layer else { return };
        let (Some(layer), Some(ping), Some(pong)) =
            (self.layers.get(&id), self.ping.as_ref(), self.pong.as_ref())
        else {
            return;
        };
        // Bake params live in the wet slot to avoid clashing with composite's slots.
        let p = CompositeParams {
            opacity: 1.0,
            blend_mode: 0,
            has_xform: 1,
            adjust_kind: 0,
            m: self.xform_m,
            off: self.xform_off,
            _p1: [0.0; 2],
            adjust: [0.0; 4],
            has_blend_if: 0,
            _p2: [0; 3],
            blend_if: [0.0, 1.0, 0.0, 1.0],
        };
        queue.write_buffer(
            &self.params_buf,
            WET_PARAMS_OFFSET as u64,
            bytemuck::bytes_of(&p),
        );
        // Clear ping (transparent backdrop).
        {
            let _c = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("xform.clear"),
                color_attachments: &[Some(clear_attachment(&ping.view))],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("xform.bake.bg"),
            layout: &self.composite_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&ping.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&layer.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.params_buf,
                        offset: 0,
                        size: wgpu::BufferSize::new(std::mem::size_of::<CompositeParams>() as u64),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.white_mask.view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(
                        &self.identity_lut.as_ref().unwrap().view,
                    ),
                },
            ],
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("xform.bake"),
                color_attachments: &[Some(clear_attachment(&pong.view))],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.composite_pipeline);
            pass.set_bind_group(0, &bind, &[WET_PARAMS_OFFSET]);
            pass.draw(0..3, 0..1);
        }
        copy_tex(encoder, &pong.tex, &layer.tex, self.canvas_size);
        self.xform_layer = None;
        self.xform_m = [1.0, 0.0, 0.0, 1.0];
        self.xform_off = [0.0; 2];
    }

    /// (Re)create the canvas targets when the document size changes. Clears
    /// layers + history (new document).
    pub fn ensure_canvas(&mut self, device: &wgpu::Device, size: Size) {
        if self.canvas_size == size && self.ping.is_some() {
            return;
        }
        self.canvas_size = size;
        self.ping = Some(make_target(device, size, "composite.ping"));
        self.pong = Some(make_target(device, size, "composite.pong"));
        self.wet = Some(make_target(device, size, "composite.wet"));
        self.wet_owner = None;
        self.selection = Some(make_target_fmt(device, size, "selection", SEL_FMT));
        self.selection_tmp = Some(make_target_fmt(device, size, "selection.tmp", SEL_FMT));
        self.has_selection = false;
        self.xform_layer = None;
        self.composite_valid = false;
        self.stroke_owner = None;
        self.stroke_pre = None;
        self.layers.clear(); // new document
        self.curve_luts.clear();
        self.clone_src = None;
        self.masks.clear();
        self.undo_stack.clear();
        self.redo_stack.clear();
    }

    pub fn ensure_layer(&mut self, device: &wgpu::Device, id: LayerId) {
        let size = self.canvas_size;
        self.layers
            .entry(id)
            .or_insert_with(|| make_target(device, size, "layer"));
    }

    pub fn drop_layer(&mut self, id: LayerId) {
        self.layers.remove(&id);
        self.curve_luts.remove(&id);
    }

    /// Lazily build the identity LUT (ramp in every channel). Needed before any
    /// composite-layout bind group is created (the binding must be valid even
    /// when unused).
    fn ensure_identity_lut(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.identity_lut.is_some() {
            return;
        }
        let t = make_lut_target(device, "lut.identity");
        let n = LUT_W as usize;
        let mut texels = vec![half::f16::from_f32(0.0); n * 4];
        for (i, px) in texels.chunks_mut(4).enumerate() {
            let v = half::f16::from_f32(i as f32 / (n - 1) as f32);
            px[0] = v;
            px[1] = v;
            px[2] = v;
            px[3] = v;
        }
        write_lut(queue, &t.tex, &texels);
        self.identity_lut = Some(t);
    }

    /// Build + upload a Curves layer's tone-curve LUT from per-channel control
    /// points. `rgb` is the composite (master) curve; `r`/`g`/`b` are per-channel.
    #[allow(clippy::too_many_arguments)]
    pub fn set_curve_lut(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: LayerId,
        rgb: &[(f32, f32)],
        r: &[(f32, f32)],
        g: &[(f32, f32)],
        b: &[(f32, f32)],
    ) {
        let n = LUT_W as usize;
        let lm = prism_core::curve::build_lut(rgb, n);
        let lr = prism_core::curve::build_lut(r, n);
        let lg = prism_core::curve::build_lut(g, n);
        let lb = prism_core::curve::build_lut(b, n);
        let mut texels = vec![half::f16::from_f32(0.0); n * 4];
        for (i, px) in texels.chunks_mut(4).enumerate() {
            px[0] = half::f16::from_f32(lr[i]);
            px[1] = half::f16::from_f32(lg[i]);
            px[2] = half::f16::from_f32(lb[i]);
            px[3] = half::f16::from_f32(lm[i]);
        }
        let t = self
            .curve_luts
            .entry(id)
            .or_insert_with(|| make_lut_target(device, "lut.curve"));
        write_lut(queue, &t.tex, &texels);
    }

    /// Upload raw RGBA16F (linear premultiplied) bytes into a layer.
    pub fn upload_layer(&mut self, queue: &wgpu::Queue, id: LayerId, bytes: &[u8]) {
        let Some(layer) = self.layers.get(&id) else {
            return;
        };
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &layer.tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.canvas_size.width * 4 * 2),
                rows_per_image: Some(self.canvas_size.height),
            },
            wgpu::Extent3d {
                width: self.canvas_size.width,
                height: self.canvas_size.height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Read a whole texture back to CPU as tightly-packed RGBA16F bytes.
    fn readback_texture(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tex: &wgpu::Texture,
    ) -> Option<Vec<u8>> {
        let (w, h) = (self.canvas_size.width, self.canvas_size.height);
        let unpadded = w * 8; // 4ch * f16
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = unpadded.div_ceil(align) * align;
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (padded * h) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([enc.finish()]);
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .ok()?;
        let data = slice.get_mapped_range();
        let mut out = Vec::with_capacity((unpadded * h) as usize);
        for row in 0..h {
            let s = (row * padded) as usize;
            out.extend_from_slice(&data[s..s + unpadded as usize]);
        }
        drop(data);
        buf.unmap();
        Some(out)
    }

    fn readback_pixel(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tex: &wgpu::Texture,
        x: u32,
        y: u32,
    ) -> Option<[f32; 4]> {
        if x >= self.canvas_size.width || y >= self.canvas_size.height {
            return None;
        }
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readpixel"),
            size: 256,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([enc.finish()]);
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .ok()?;
        let data = slice.get_mapped_range();
        let px = [
            half::f16::from_le_bytes([data[0], data[1]]).to_f32(),
            half::f16::from_le_bytes([data[2], data[3]]).to_f32(),
            half::f16::from_le_bytes([data[4], data[5]]).to_f32(),
            half::f16::from_le_bytes([data[6], data[7]]).to_f32(),
        ];
        drop(data);
        buf.unmap();
        Some(px)
    }

    /// Read a whole layer back to CPU as tightly-packed RGBA16F bytes.
    pub fn read_layer(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: LayerId,
    ) -> Option<Vec<u8>> {
        let tex = &self.layers.get(&id)?.tex;
        self.readback_texture(device, queue, tex)
    }

    /// Read a single layer pixel (linear premultiplied RGBA) — for the eyedropper.
    pub fn read_pixel(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: LayerId,
        x: u32,
        y: u32,
    ) -> Option<[f32; 4]> {
        let tex = &self.layers.get(&id)?.tex;
        self.readback_pixel(device, queue, tex, x, y)
    }

    /// Composite all layers now (own encoder) and return whether the result is in `ping`.
    pub fn composite_now(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        order: &[LayerDraw],
    ) -> bool {
        let mut enc = device.create_command_encoder(&Default::default());
        let r = self.composite(device, queue, &mut enc, order);
        queue.submit([enc.finish()]);
        r
    }

    fn target_tex(&self, is_ping: bool) -> Option<&wgpu::Texture> {
        if is_ping {
            self.ping.as_ref()
        } else {
            self.pong.as_ref()
        }
        .map(|t| &t.tex)
    }

    /// Read the full composite (call right after `composite_now`).
    pub fn read_composite(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        is_ping: bool,
    ) -> Option<Vec<u8>> {
        self.readback_texture(device, queue, self.target_tex(is_ping)?)
    }

    /// Read one composite pixel (linear premultiplied).
    pub fn read_composite_pixel(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        is_ping: bool,
        x: u32,
        y: u32,
    ) -> Option<[f32; 4]> {
        self.readback_pixel(device, queue, self.target_tex(is_ping)?, x, y)
    }

    /// Read the selection mask to CPU as one f32 per pixel (len = w*h).
    pub fn read_selection(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Vec<f32>> {
        let sel = self.selection.as_ref()?;
        let (w, h) = (self.canvas_size.width, self.canvas_size.height);
        let unpadded = w * 2; // R16
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = unpadded.div_ceil(align) * align;
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sel.readback"),
            size: (padded * h) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &sel.tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([enc.finish()]);
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .ok()?;
        let data = slice.get_mapped_range();
        let mut out = Vec::with_capacity((w * h) as usize);
        for row in 0..h {
            let s = (row * padded) as usize;
            for px in data[s..s + unpadded as usize].chunks_exact(2) {
                out.push(half::f16::from_le_bytes([px[0], px[1]]).to_f32());
            }
        }
        drop(data);
        buf.unmap();
        Some(out)
    }

    /// Upload a CPU selection mask (one f32 per pixel) and mark a selection active.
    pub fn upload_selection(&mut self, queue: &wgpu::Queue, mask: &[f32]) {
        let Some(sel) = self.selection.as_ref() else {
            return;
        };
        let (w, h) = (self.canvas_size.width, self.canvas_size.height);
        let mut bytes = Vec::with_capacity(mask.len() * 2);
        for &m in mask {
            bytes.extend_from_slice(&half::f16::from_f32(m).to_le_bytes());
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &sel.tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 2),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.has_selection = true;
    }

    fn new_region_tex(&self, device: &wgpu::Device, w: u32, h: u32) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("undo.region"),
            size: wgpu::Extent3d {
                width: w.max(1),
                height: h.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FMT,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    }

    /// Clamp a requested `[x, y, w, h]` to the canvas bounds.
    fn clamp_rect(&self, rect: [u32; 4]) -> [u32; 4] {
        let (cw, ch) = (self.canvas_size.width, self.canvas_size.height);
        let x = rect[0].min(cw.saturating_sub(1));
        let y = rect[1].min(ch.saturating_sub(1));
        let w = rect[2].clamp(1, cw - x);
        let h = rect[3].clamp(1, ch - y);
        [x, y, w, h]
    }

    fn push_undo(&mut self, snap: Snapshot) {
        self.undo_stack.push(snap);
        if self.undo_stack.len() > UNDO_MAX {
            self.undo_stack.remove(0);
        }
        self.redo_stack.clear();
    }

    /// Begin a command: copy the layer's current pixels into the transient
    /// pre-stroke buffer. The undo entry (just the dirty region) is taken at
    /// [`commit_command`].
    pub fn begin_command(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        id: LayerId,
        label: &str,
    ) {
        if !self.layers.contains_key(&id) {
            return;
        }
        let (w, h) = (self.canvas_size.width, self.canvas_size.height);
        if self.stroke_pre.is_none() {
            self.stroke_pre = Some(self.new_region_tex(device, w, h));
        }
        let pre = self.stroke_pre.as_ref().unwrap();
        let layer = self.layers.get(&id).unwrap();
        copy_tex(encoder, &layer.tex, pre, self.canvas_size);
        self.stroke_owner = Some(id);
        self.stroke_label = label.to_string();
    }

    /// Commit the open command, snapshotting only `rect` (the touched region)
    /// from the pre-stroke buffer onto the undo stack.
    pub fn commit_command(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        rect: [u32; 4],
    ) {
        let Some(id) = self.stroke_owner.take() else {
            return;
        };
        let Some(pre) = self.stroke_pre.as_ref() else {
            return;
        };
        let [x, y, w, h] = self.clamp_rect(rect);
        let region = self.new_region_tex(device, w, h);
        copy_region(encoder, pre, [x, y], &region, [0, 0], [w, h]);
        let label = std::mem::take(&mut self.stroke_label);
        self.push_undo(Snapshot {
            id,
            tex: region,
            rect: [x, y, w, h],
            label,
        });
    }

    /// Snapshot a whole-layer command immediately (own encoder), for callers
    /// outside the frame callback such as bucket fill.
    pub fn begin_command_now(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: LayerId,
        label: &str,
    ) {
        let Some(layer) = self.layers.get(&id) else {
            return;
        };
        let (w, h) = (self.canvas_size.width, self.canvas_size.height);
        let region = self.new_region_tex(device, w, h);
        let mut enc = device.create_command_encoder(&Default::default());
        copy_region(&mut enc, &layer.tex, [0, 0], &region, [0, 0], [w, h]);
        queue.submit([enc.finish()]);
        self.push_undo(Snapshot {
            id,
            tex: region,
            rect: [0, 0, w, h],
            label: label.to_string(),
        });
    }

    /// Labels of pending undo steps (oldest→newest) and redo steps (next→furthest).
    pub fn history_labels(&self) -> (Vec<String>, Vec<String>) {
        let undo = self.undo_stack.iter().map(|s| s.label.clone()).collect();
        let redo = self
            .redo_stack
            .iter()
            .rev()
            .map(|s| s.label.clone())
            .collect();
        (undo, redo)
    }

    fn restore(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        from_undo: bool,
    ) {
        let snap = if from_undo {
            self.undo_stack.pop()
        } else {
            self.redo_stack.pop()
        };
        let Some(snap) = snap else { return };
        let Some(layer) = self.layers.get(&snap.id) else {
            return;
        };
        let [x, y, w, h] = snap.rect;
        // Save the layer's current region to the opposite stack, then restore.
        let cur = self.new_region_tex(device, w, h);
        copy_region(encoder, &layer.tex, [x, y], &cur, [0, 0], [w, h]);
        copy_region(encoder, &snap.tex, [0, 0], &layer.tex, [x, y], [w, h]);
        let saved = Snapshot {
            id: snap.id,
            tex: cur,
            rect: snap.rect,
            label: snap.label,
        };
        if from_undo {
            self.redo_stack.push(saved);
        } else {
            self.undo_stack.push(saved);
        }
    }

    pub fn undo(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        self.restore(device, encoder, true);
    }

    pub fn redo(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        self.restore(device, encoder, false);
    }

    pub fn has_selection(&self) -> bool {
        self.has_selection
    }

    /// Apply a selection operation (records into `encoder`).
    pub fn apply_selection(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        op: &SelectionOp,
    ) {
        match *op {
            SelectionOp::All => self.has_selection = false,
            SelectionOp::None => {
                // Clear the selection texture so it's in a clean state for
                // the next selection op, then remove the mask constraint so
                // painting is unrestricted (matches Photoshop "Select > None").
                if let Some(sel) = self.selection.as_ref() {
                    let _clear = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("sel.none"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &sel.view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                }
                self.has_selection = false; // was true — bug: empty mask blocked all painting
            }
            SelectionOp::Marquee { rect, ellipse } => {
                let Some(sel) = self.selection.as_ref() else {
                    return;
                };
                let uni = ShapeUniform {
                    rect,
                    size: [
                        self.canvas_size.width as f32,
                        self.canvas_size.height as f32,
                    ],
                    kind: if ellipse { 1 } else { 0 },
                    _p: 0,
                };
                queue.write_buffer(&self.shape_uniform, 0, bytemuck::bytes_of(&uni));
                let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("sel.shape.bg"),
                    layout: &self.shape_bgl,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.shape_uniform.as_entire_binding(),
                    }],
                });
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("sel.shape.pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &sel.view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&self.shape_pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.draw(0..3, 0..1);
                drop(pass);
                self.has_selection = true;
            }
            SelectionOp::Invert => {
                let (Some(sel), Some(tmp)) = (self.selection.as_ref(), self.selection_tmp.as_ref())
                else {
                    return;
                };
                let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("sel.invert.bg"),
                    layout: &self.invert_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&sel.view),
                        },
                    ],
                });
                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("sel.invert.pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &tmp.view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    pass.set_pipeline(&self.invert_pipeline);
                    pass.set_bind_group(0, &bind, &[]);
                    pass.draw(0..3, 0..1);
                }
                copy_tex(encoder, &tmp.tex, &sel.tex, self.canvas_size);
                self.has_selection = true;
            }
        }
    }

    /// Begin a wet brush stroke over `owner`: clear the wet buffer.
    pub fn wet_begin(&mut self, encoder: &mut wgpu::CommandEncoder, owner: LayerId, opacity: f32) {
        self.wet_owner = Some(owner);
        self.wet_opacity = opacity;
        if let Some(wet) = self.wet.as_ref() {
            let _clear = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("wet.clear"),
                color_attachments: &[Some(clear_attachment(&wet.view))],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
    }

    /// Flatten the wet stroke into its owner layer (pen-up) and clear it.
    pub fn wet_end(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let Some(owner) = self.wet_owner.take() else {
            return;
        };
        self.ensure_identity_lut(device, queue);
        let (Some(layer), Some(wet), Some(pong)) = (
            self.layers.get(&owner),
            self.wet.as_ref(),
            self.pong.as_ref(),
        ) else {
            return;
        };
        queue.write_buffer(
            &self.params_buf,
            WET_PARAMS_OFFSET as u64,
            bytemuck::bytes_of(&CompositeParams::plain(self.wet_opacity, 0)),
        );
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wet.flatten.bg"),
            layout: &self.composite_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&layer.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&wet.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.params_buf,
                        offset: 0,
                        size: wgpu::BufferSize::new(std::mem::size_of::<CompositeParams>() as u64),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.white_mask.view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(
                        &self.identity_lut.as_ref().unwrap().view,
                    ),
                },
            ],
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("wet.flatten"),
                color_attachments: &[Some(clear_attachment(&pong.view))],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.composite_pipeline);
            pass.set_bind_group(0, &bind, &[WET_PARAMS_OFFSET]);
            pass.draw(0..3, 0..1);
        }
        copy_tex(encoder, &pong.tex, &layer.tex, self.canvas_size);
        let _clear = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("wet.clear.end"),
            color_attachments: &[Some(clear_attachment(&wet.view))],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn paint_dabs(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        id: LayerId,
        dabs: &[Dab],
        erase: bool,
        into_wet: bool,
        into_mask: bool,
    ) {
        if dabs.is_empty() {
            return;
        }
        let target = if into_mask {
            self.masks.get(&id)
        } else if into_wet {
            self.wet.as_ref()
        } else {
            self.layers.get(&id)
        };
        let Some(target) = target else { return };

        if dabs.len() as u64 > self.dab_capacity {
            self.dab_capacity = (dabs.len() as u64).next_power_of_two();
            self.dab_instances = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("dab.instances"),
                size: self.dab_capacity * std::mem::size_of::<Dab>() as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        let Some(sel) = self.selection.as_ref() else {
            return;
        };
        queue.write_buffer(&self.dab_instances, 0, bytemuck::cast_slice(dabs));
        queue.write_buffer(
            &self.dab_info_buf,
            0,
            bytemuck::bytes_of(&LayerInfo {
                size: [
                    self.canvas_size.width as f32,
                    self.canvas_size.height as f32,
                ],
                has_selection: if self.has_selection { 1.0 } else { 0.0 },
                _pad: 0.0,
            }),
        );
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dab.bg"),
            layout: &self.dab_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.dab_info_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&sel.view),
                },
            ],
        });

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("dab.pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target.view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        let pipeline = if erase {
            &self.dab_erase_pipeline
        } else {
            &self.dab_pipeline
        };
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.set_vertex_buffer(0, self.dab_instances.slice(..));
        pass.draw(0..6, 0..dabs.len() as u32);
    }

    /// Freeze the active layer into the clone source so a clone stroke samples a
    /// stable snapshot (no feedback). Called at clone-stroke begin.
    pub fn snapshot_clone_source(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        id: LayerId,
    ) {
        let src = make_target(device, self.canvas_size, "clone.src");
        {
            let Some(layer) = self.layers.get(&id) else {
                return;
            };
            copy_tex(encoder, &layer.tex, &src.tex, self.canvas_size);
        }
        self.clone_src = Some(src);
    }

    /// Clone-stamp dabs: copy pixels from the frozen source at `offset`
    /// (destAnchor − sourceAnchor) into the active layer, shaped by brush falloff.
    fn paint_clone_dabs(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        id: LayerId,
        dabs: &[Dab],
        offset: [f32; 2],
    ) {
        if dabs.is_empty() {
            return;
        }
        if !self.layers.contains_key(&id) || self.clone_src.is_none() || self.selection.is_none() {
            return;
        }
        if dabs.len() as u64 > self.dab_capacity {
            self.dab_capacity = (dabs.len() as u64).next_power_of_two();
            self.dab_instances = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("dab.instances"),
                size: self.dab_capacity * std::mem::size_of::<Dab>() as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        queue.write_buffer(&self.dab_instances, 0, bytemuck::cast_slice(dabs));
        queue.write_buffer(
            &self.dab_info_buf,
            0,
            bytemuck::bytes_of(&LayerInfo {
                size: [
                    self.canvas_size.width as f32,
                    self.canvas_size.height as f32,
                ],
                has_selection: if self.has_selection { 1.0 } else { 0.0 },
                _pad: 0.0,
            }),
        );
        queue.write_buffer(
            &self.clone_params_buf,
            0,
            bytemuck::bytes_of(&CloneParams {
                offset,
                _pad: [0.0; 2],
            }),
        );
        let target = self.layers.get(&id).unwrap();
        let src = self.clone_src.as_ref().unwrap();
        let sel = self.selection.as_ref().unwrap();
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("clone.bg"),
            layout: &self.clone_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.dab_info_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&sel.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&src.view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.clone_params_buf.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("clone.pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target.view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.clone_pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.set_vertex_buffer(0, self.dab_instances.slice(..));
        pass.draw(0..6, 0..dabs.len() as u32);
    }

    /// Ping-pong composite all visible layers; returns the final view index.
    fn composite(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        order: &[LayerDraw],
    ) -> bool {
        self.ensure_white(queue);
        self.ensure_identity_lut(device, queue);

        // Write all layer params up front (one buffer, dynamic offsets).
        let visible: Vec<&LayerDraw> = order.iter().filter(|l| l.visible).collect();
        for (i, l) in visible.iter().enumerate() {
            let mut p = CompositeParams::plain(l.opacity, l.blend);
            p.adjust_kind = l.adjust_kind;
            p.adjust = l.adjust;
            if l.has_blend_if {
                p.has_blend_if = 1;
                p.blend_if = l.blend_if;
            }
            if self.xform_layer == Some(l.id) {
                p.has_xform = 1;
                p.m = self.xform_m;
                p.off = self.xform_off;
            }
            queue.write_buffer(
                &self.params_buf,
                i as u64 * PARAMS_STRIDE,
                bytemuck::bytes_of(&p),
            );
        }
        if self.wet_owner.is_some() {
            queue.write_buffer(
                &self.params_buf,
                WET_PARAMS_OFFSET as u64,
                bytemuck::bytes_of(&CompositeParams::plain(self.wet_opacity, 0)),
            );
        }

        let ping = self.ping.as_ref().unwrap();
        let pong = self.pong.as_ref().unwrap();
        let identity_lut = &self.identity_lut.as_ref().unwrap().view;

        // Ordered passes: each visible layer (+ its Curves LUT, or the identity
        // LUT), plus the wet stroke just above its owner so the in-progress
        // stroke previews at the right depth.
        let mut passes: Vec<(
            &wgpu::TextureView,
            u32,
            &wgpu::TextureView,
            &wgpu::TextureView,
        )> = Vec::new();
        for (i, l) in visible.iter().enumerate() {
            if let Some(layer) = self.layers.get(&l.id) {
                let lut = self
                    .curve_luts
                    .get(&l.id)
                    .map(|g| &g.view)
                    .unwrap_or(identity_lut);
                passes.push((
                    &layer.view,
                    (i as u64 * PARAMS_STRIDE) as u32,
                    self.mask_view(l.id),
                    lut,
                ));
                if self.wet_owner == Some(l.id) {
                    if let Some(wet) = self.wet.as_ref() {
                        passes.push((
                            &wet.view,
                            WET_PARAMS_OFFSET,
                            &self.white_mask.view,
                            identity_lut,
                        ));
                    }
                }
            }
        }

        // Clear ping to transparent — the initial backdrop (LoadOp::Clear, no draw).
        {
            let _clear = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("composite.clear"),
                color_attachments: &[Some(clear_attachment(&ping.view))],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }

        let mut src_is_ping = true;
        for (layer_view, offset, mask_view, lut_view) in passes {
            let (src, dst) = if src_is_ping {
                (ping, pong)
            } else {
                (pong, ping)
            };
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("composite.bg"),
                layout: &self.composite_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&src.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(layer_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.params_buf,
                            offset: 0,
                            size: wgpu::BufferSize::new(
                                std::mem::size_of::<CompositeParams>() as u64
                            ),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(mask_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(lut_view),
                    },
                ],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("composite.layer"),
                color_attachments: &[Some(clear_attachment(&dst.view))],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.composite_pipeline);
            pass.set_bind_group(0, &bind, &[offset]);
            pass.draw(0..3, 0..1);
            drop(pass);
            src_is_ping = !src_is_ping;
        }
        // Final result lives in `src` after the last swap.
        src_is_ping
    }

    fn build_display_bind_group(&mut self, device: &wgpu::Device, final_is_ping: bool) {
        let final_view = if final_is_ping {
            &self.ping.as_ref().unwrap().view
        } else {
            &self.pong.as_ref().unwrap().view
        };
        let sel_view = &self.selection.as_ref().unwrap().view;
        self.display_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("display.bg"),
            layout: &self.display_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.display_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(final_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(sel_view),
                },
            ],
        }));
    }
}

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

/// Per-frame data the app hands to the canvas.
pub struct CanvasPaint {
    pub doc_rect: egui::Rect,
    pub checker_pts: f32,
    pub canvas_size: Size,
    pub layers: Vec<LayerDraw>,
    pub active_id: LayerId,
    pub dabs: Vec<Dab>,
    pub erase: bool,
    /// Set on the first frame of a stroke — copies the layer to the pre-stroke buffer.
    pub begin_command: bool,
    pub command_label: String,
    /// Set on the last frame of a stroke — snapshots `dirty_rect` for undo.
    pub commit_command: bool,
    pub dirty_rect: [u32; 4],
    pub undo: u32,
    pub redo: u32,
    // Wet brush stroke lifecycle.
    pub wet_begin: bool,
    pub wet_end: bool,
    pub wet_opacity: f32,
    pub paint_into_wet: bool,
    /// Route brush dabs to the active layer's mask instead of its pixels.
    pub paint_mask: bool,
    /// Clone Stamp: dabs copy from the frozen source at `clone_offset` instead
    /// of painting a flat color.
    pub clone: bool,
    /// destAnchor − sourceAnchor in document px (aligned clone).
    pub clone_offset: [f32; 2],
    /// Whether the document changed this frame (gates recompositing).
    pub dirty: bool,
    /// Selection operation to apply this frame, if any.
    pub selection_op: Option<SelectionOp>,
    /// Seconds, for marching-ants animation.
    pub time: f32,
    /// Live affine on the active layer (uv-space matrix, offset), if transforming.
    pub xform: Option<([f32; 4], [f32; 2])>,
    /// Bake the active transform into the layer this frame.
    pub bake: bool,
}

impl CallbackTrait for CanvasPaint {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen_descriptor: &ScreenDescriptor,
        encoder: &mut wgpu::CommandEncoder,
        resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let gpu: &mut CanvasGpu = resources.get_mut().unwrap();

        gpu.ensure_canvas(device, self.canvas_size);
        for l in &self.layers {
            gpu.ensure_layer(device, l.id);
        }
        match self.xform {
            Some((m, off)) => gpu.set_layer_transform(Some(self.active_id), m, off),
            None => gpu.set_layer_transform(None, [1.0, 0.0, 0.0, 1.0], [0.0; 2]),
        }

        for _ in 0..self.undo {
            gpu.undo(device, encoder);
        }
        for _ in 0..self.redo {
            gpu.redo(device, encoder);
        }
        if let Some(op) = &self.selection_op {
            gpu.apply_selection(device, queue, encoder, op);
        }
        if self.begin_command {
            gpu.begin_command(device, encoder, self.active_id, &self.command_label);
            if self.clone {
                gpu.snapshot_clone_source(device, encoder, self.active_id);
            }
        }
        if self.wet_begin {
            gpu.wet_begin(encoder, self.active_id, self.wet_opacity);
        }
        if self.clone {
            gpu.paint_clone_dabs(
                device,
                queue,
                encoder,
                self.active_id,
                &self.dabs,
                self.clone_offset,
            );
        } else {
            gpu.paint_dabs(
                device,
                queue,
                encoder,
                self.active_id,
                &self.dabs,
                self.erase,
                self.paint_into_wet,
                self.paint_mask,
            );
        }
        if self.wet_end {
            gpu.wet_end(device, queue, encoder);
        }
        if self.bake {
            gpu.bake_transform(device, queue, encoder);
        }
        if self.commit_command {
            gpu.commit_command(device, encoder, self.dirty_rect);
        }
        // Recomposite only when the document changed; pan/zoom alone reuse the
        // last composite (only the display pass re-runs each frame).
        let final_is_ping = if self.dirty || !gpu.composite_valid {
            let f = gpu.composite(device, queue, encoder, &self.layers);
            gpu.last_final_is_ping = f;
            gpu.composite_valid = true;
            f
        } else {
            gpu.last_final_is_ping
        };
        gpu.build_display_bind_group(device, final_is_ping);

        let [sw, sh] = screen_descriptor.size_in_pixels;
        let ppp = screen_descriptor.pixels_per_point;
        let to_clip = |p: egui::Pos2| -> [f32; 2] {
            [
                p.x * ppp / sw as f32 * 2.0 - 1.0,
                1.0 - p.y * ppp / sh as f32 * 2.0,
            ]
        };
        let uni = DisplayUniform {
            clip_min: to_clip(self.doc_rect.min),
            clip_max: to_clip(self.doc_rect.max),
            checker_px: self.checker_pts * ppp,
            has_selection: if gpu.has_selection() { 1.0 } else { 0.0 },
            time: self.time,
            canvas_w: self.canvas_size.width as f32,
            canvas_h: self.canvas_size.height as f32,
            _pad: [0.0; 3],
        };
        queue.write_buffer(&gpu.display_uniform, 0, bytemuck::bytes_of(&uni));
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &CallbackResources,
    ) {
        let gpu: &CanvasGpu = resources.get().unwrap();
        let Some(bg) = &gpu.display_bind_group else {
            return;
        };
        render_pass.set_pipeline(&gpu.display_pipeline);
        render_pass.set_bind_group(0, bg, &[]);
        render_pass.draw(0..6, 0..1);
    }
}

#[cfg(test)]
mod gpu_tests {
    use super::*;

    fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
    }

    /// A solid `n`x`n` buffer of linear-premultiplied RGBA16F bytes.
    fn solid(n: u32, r: f32, g: f32, b: f32, a: f32) -> Vec<u8> {
        let px = [r * a, g * a, b * a, a];
        let mut o = Vec::new();
        for _ in 0..(n * n) {
            for &c in &px {
                o.extend_from_slice(&half::f16::from_f32(c).to_le_bytes());
            }
        }
        o
    }

    // Drives the real GPU compositor: upload -> composite -> brush(wet)+flatten
    // -> undo, asserting pixels via readback. Skips if no GPU adapter.
    #[test]
    fn compositor_brush_wet_undo() {
        let Some((device, queue)) = device() else {
            eprintln!("no GPU adapter; skipping compositor_brush_wet_undo");
            return;
        };
        let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
        let size = Size::new(8, 8);
        gpu.ensure_canvas(&device, size);
        let l0 = LayerId(0);
        gpu.ensure_layer(&device, l0);
        gpu.upload_layer(&queue, l0, &solid(8, 1.0, 0.0, 0.0, 1.0)); // red

        let order = vec![LayerDraw {
            id: l0,
            opacity: 1.0,
            blend: 0,
            visible: true,
            adjust_kind: 0,
            adjust: [0.0; 4],
            has_blend_if: false,
            blend_if: [0.0, 1.0, 0.0, 1.0],
        }];
        let p = gpu.composite_now(&device, &queue, &order);
        let px = gpu.read_composite_pixel(&device, &queue, p, 4, 4).unwrap();
        assert!(
            px[0] > 0.9 && px[1] < 0.1 && px[3] > 0.9,
            "composite red: {px:?}"
        );

        // Blue wet brush dab over the center, flattened.
        let mut enc = device.create_command_encoder(&Default::default());
        gpu.begin_command(&device, &mut enc, l0, "test");
        gpu.wet_begin(&mut enc, l0, 1.0);
        let dab = Dab {
            center: [2.0, 2.0],
            radius: 2.5,
            hardness: 0.95,
            color: [0.0, 0.0, 1.0, 1.0],
        };
        gpu.paint_dabs(&device, &queue, &mut enc, l0, &[dab], false, true, false);
        gpu.wet_end(&device, &queue, &mut enc);
        // Region-COW: only snapshot the touched corner.
        gpu.commit_command(&device, &mut enc, [0, 0, 5, 5]);
        queue.submit([enc.finish()]);

        let p = gpu.composite_now(&device, &queue, &order);
        let near = gpu.read_composite_pixel(&device, &queue, p, 2, 2).unwrap();
        let far = gpu.read_composite_pixel(&device, &queue, p, 7, 7).unwrap();
        assert!(
            near[2] > 0.9 && near[0] < 0.1,
            "baked blue at stroke: {near:?}"
        );
        assert!(
            far[0] > 0.9 && far[2] < 0.1,
            "far pixel untouched red: {far:?}"
        );

        // Undo restores the region to red.
        let mut enc = device.create_command_encoder(&Default::default());
        gpu.undo(&device, &mut enc);
        queue.submit([enc.finish()]);
        let p = gpu.composite_now(&device, &queue, &order);
        let near = gpu.read_composite_pixel(&device, &queue, p, 2, 2).unwrap();
        assert!(near[0] > 0.9 && near[2] < 0.1, "undo back to red: {near:?}");
    }

    // A rectangle selection must clip painting to the selected region.
    #[test]
    fn selection_clips_paint() {
        let Some((device, queue)) = device() else {
            eprintln!("no GPU adapter; skipping selection_clips_paint");
            return;
        };
        let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
        let size = Size::new(8, 8);
        gpu.ensure_canvas(&device, size);
        let l0 = LayerId(0);
        gpu.ensure_layer(&device, l0);
        gpu.upload_layer(&queue, l0, &solid(8, 1.0, 0.0, 0.0, 1.0)); // red
        let order = vec![LayerDraw {
            id: l0,
            opacity: 1.0,
            blend: 0,
            visible: true,
            adjust_kind: 0,
            adjust: [0.0; 4],
            has_blend_if: false,
            blend_if: [0.0, 1.0, 0.0, 1.0],
        }];

        // Select the left half, then paint blue over the whole canvas.
        let mut enc = device.create_command_encoder(&Default::default());
        gpu.apply_selection(
            &device,
            &queue,
            &mut enc,
            &SelectionOp::Marquee {
                rect: [0.0, 0.0, 4.0, 8.0],
                ellipse: false,
            },
        );
        let dab = Dab {
            center: [4.0, 4.0],
            radius: 16.0,
            hardness: 0.99,
            color: [0.0, 0.0, 1.0, 1.0],
        };
        gpu.paint_dabs(&device, &queue, &mut enc, l0, &[dab], false, false, false);
        queue.submit([enc.finish()]);
        assert!(gpu.has_selection());

        let p = gpu.composite_now(&device, &queue, &order);
        let inside = gpu.read_composite_pixel(&device, &queue, p, 1, 4).unwrap();
        let outside = gpu.read_composite_pixel(&device, &queue, p, 6, 4).unwrap();
        assert!(inside[2] > 0.5, "selected area painted blue: {inside:?}");
        assert!(
            outside[0] > 0.5 && outside[2] < 0.2,
            "unselected area untouched red: {outside:?}"
        );
    }

    // Translating a layer by +half-width then baking should clear the left half
    // and keep the right half.
    #[test]
    fn transform_bake_translates() {
        let Some((device, queue)) = device() else {
            eprintln!("no GPU adapter; skipping transform_bake_translates");
            return;
        };
        let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
        let size = Size::new(8, 8);
        gpu.ensure_canvas(&device, size);
        let l0 = LayerId(0);
        gpu.ensure_layer(&device, l0);
        gpu.upload_layer(&queue, l0, &solid(8, 1.0, 0.0, 0.0, 1.0)); // red everywhere

        // Move right by half width: layer-from-canvas off.x = -0.5.
        gpu.set_layer_transform(Some(l0), [1.0, 0.0, 0.0, 1.0], [-0.5, 0.0]);
        let mut enc = device.create_command_encoder(&Default::default());
        gpu.bake_transform(&device, &queue, &mut enc);
        queue.submit([enc.finish()]);

        let left = gpu.read_pixel(&device, &queue, l0, 1, 4).unwrap();
        let right = gpu.read_pixel(&device, &queue, l0, 6, 4).unwrap();
        assert!(left[3] < 0.1, "left half cleared after move: {left:?}");
        assert!(
            right[0] > 0.9 && right[3] > 0.9,
            "right half keeps red: {right:?}"
        );
    }

    // An Invert adjustment layer over a red layer yields cyan in the composite.
    #[test]
    fn adjustment_invert() {
        let Some((device, queue)) = device() else {
            eprintln!("no GPU adapter; skipping adjustment_invert");
            return;
        };
        let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
        let size = Size::new(8, 8);
        gpu.ensure_canvas(&device, size);
        let l0 = LayerId(0);
        let l1 = LayerId(1);
        gpu.ensure_layer(&device, l0);
        gpu.ensure_layer(&device, l1); // adjustment layer (no pixels needed)
        gpu.upload_layer(&queue, l0, &solid(8, 1.0, 0.0, 0.0, 1.0)); // red

        let (k, p) = prism_core::adjust::Adjustment::Invert.encode();
        let order = vec![
            LayerDraw {
                id: l0,
                opacity: 1.0,
                blend: 0,
                visible: true,
                adjust_kind: 0,
                adjust: [0.0; 4],
                has_blend_if: false,
                blend_if: [0.0, 1.0, 0.0, 1.0],
            },
            LayerDraw {
                id: l1,
                opacity: 1.0,
                blend: 0,
                visible: true,
                adjust_kind: k,
                adjust: p,
                has_blend_if: false,
                blend_if: [0.0, 1.0, 0.0, 1.0],
            },
        ];
        let pp = gpu.composite_now(&device, &queue, &order);
        let px = gpu.read_composite_pixel(&device, &queue, pp, 4, 4).unwrap();
        // Inverted red (premultiplied, alpha 1): low red, high green+blue.
        assert!(
            px[0] < 0.2 && px[1] > 0.8 && px[2] > 0.8,
            "invert red -> cyan: {px:?}"
        );
    }

    // A Curves adjustment layer with an inverting master curve turns red -> cyan,
    // exercising the full LUT build + upload + shader-sample path.
    #[test]
    fn curves_invert_master() {
        let Some((device, queue)) = device() else {
            eprintln!("no GPU adapter; skipping curves_invert_master");
            return;
        };
        let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
        let size = Size::new(8, 8);
        gpu.ensure_canvas(&device, size);
        let l0 = LayerId(0);
        let l1 = LayerId(1);
        gpu.ensure_layer(&device, l0);
        gpu.ensure_layer(&device, l1); // adjustment layer (no pixels needed)
        gpu.upload_layer(&queue, l0, &solid(8, 1.0, 0.0, 0.0, 1.0)); // red

        // Master curve inverts (0->1, 1->0); per-channel curves stay identity.
        let inv = [(0.0, 1.0), (1.0, 0.0)];
        let idc = [(0.0, 0.0), (1.0, 1.0)];
        gpu.set_curve_lut(&device, &queue, l1, &inv, &idc, &idc, &idc);

        let order = vec![
            LayerDraw {
                id: l0,
                opacity: 1.0,
                blend: 0,
                visible: true,
                adjust_kind: 0,
                adjust: [0.0; 4],
                has_blend_if: false,
                blend_if: [0.0, 1.0, 0.0, 1.0],
            },
            LayerDraw {
                id: l1,
                opacity: 1.0,
                blend: 0,
                visible: true,
                adjust_kind: 8, // Curves
                adjust: [0.0; 4],
                has_blend_if: false,
                blend_if: [0.0, 1.0, 0.0, 1.0],
            },
        ];
        let pp = gpu.composite_now(&device, &queue, &order);
        let px = gpu.read_composite_pixel(&device, &queue, pp, 4, 4).unwrap();
        assert!(
            px[0] < 0.2 && px[1] > 0.8 && px[2] > 0.8,
            "curves invert red -> cyan: {px:?}"
        );
    }

    // Clone stamp: source = left-half green / right-half red. Stamping the right
    // half with offset +4px samples the green left half, so the dab paints green.
    #[test]
    fn clone_stamp_copies_source() {
        let Some((device, queue)) = device() else {
            eprintln!("no GPU adapter; skipping clone_stamp_copies_source");
            return;
        };
        let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
        let size = Size::new(8, 8);
        gpu.ensure_canvas(&device, size); // selection exists, has_selection = false
        let l0 = LayerId(0);
        gpu.ensure_layer(&device, l0);

        // Left half (x<4) green, right half red — premultiplied f16.
        let mut buf = Vec::new();
        for _y in 0..8 {
            for x in 0..8 {
                let px = if x < 4 {
                    [0.0, 1.0, 0.0, 1.0]
                } else {
                    [1.0, 0.0, 0.0, 1.0]
                };
                for &c in &px {
                    buf.extend_from_slice(&half::f16::from_f32(c).to_le_bytes());
                }
            }
        }
        gpu.upload_layer(&queue, l0, &buf);

        // Stamp the right half sampling 4px to the left (green source).
        let dab = Dab {
            center: [6.0, 4.0],
            radius: 2.0,
            hardness: 0.99,
            color: [0.0, 0.0, 0.0, 1.0],
        };
        let mut enc = device.create_command_encoder(&Default::default());
        gpu.snapshot_clone_source(&device, &mut enc, l0);
        gpu.paint_clone_dabs(&device, &queue, &mut enc, l0, &[dab], [4.0, 0.0]);
        queue.submit([enc.finish()]);

        let px = gpu.read_pixel(&device, &queue, l0, 6, 4).unwrap();
        assert!(
            px[1] > 0.8 && px[0] < 0.2,
            "clone stamped green over red: {px:?}"
        );
    }

    // A Posterize(2) adjustment over a mid-gray layer snaps it to white (the
    // sRGB-space value rounds up at 2 levels) — exercises a new adjustment kind.
    #[test]
    fn posterize_adjustment() {
        let Some((device, queue)) = device() else {
            eprintln!("no GPU adapter; skipping posterize_adjustment");
            return;
        };
        let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
        let size = Size::new(8, 8);
        gpu.ensure_canvas(&device, size);
        let l0 = LayerId(0);
        let l1 = LayerId(1);
        gpu.ensure_layer(&device, l0);
        gpu.ensure_layer(&device, l1);
        gpu.upload_layer(&queue, l0, &solid(8, 0.6, 0.6, 0.6, 1.0)); // mid gray

        let (k, p) = prism_core::adjust::Adjustment::Posterize { levels: 2 }.encode();
        let order = vec![
            LayerDraw {
                id: l0,
                opacity: 1.0,
                blend: 0,
                visible: true,
                adjust_kind: 0,
                adjust: [0.0; 4],
                has_blend_if: false,
                blend_if: [0.0, 1.0, 0.0, 1.0],
            },
            LayerDraw {
                id: l1,
                opacity: 1.0,
                blend: 0,
                visible: true,
                adjust_kind: k,
                adjust: p,
                has_blend_if: false,
                blend_if: [0.0, 1.0, 0.0, 1.0],
            },
        ];
        let pp = gpu.composite_now(&device, &queue, &order);
        let px = gpu.read_composite_pixel(&device, &queue, pp, 4, 4).unwrap();
        assert!(
            px[0] > 0.9,
            "posterize(2) snaps mid gray up to white: {px:?}"
        );
    }

    // Blend-If: a gray top layer with "this layer white" pulled below its luma is
    // hidden, revealing the white layer beneath.
    #[test]
    fn blend_if_hides_bright_source() {
        let Some((device, queue)) = device() else {
            eprintln!("no GPU adapter; skipping blend_if_hides_bright_source");
            return;
        };
        let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
        let size = Size::new(8, 8);
        gpu.ensure_canvas(&device, size);
        let l0 = LayerId(0);
        let l1 = LayerId(1);
        gpu.ensure_layer(&device, l0);
        gpu.ensure_layer(&device, l1);
        gpu.upload_layer(&queue, l0, &solid(8, 1.0, 1.0, 1.0, 1.0)); // white base
        gpu.upload_layer(&queue, l1, &solid(8, 0.5, 0.5, 0.5, 1.0)); // gray top

        // this_white = 0.45 < gray luma 0.5 -> top fully gated out.
        let order = vec![
            LayerDraw {
                id: l0,
                opacity: 1.0,
                blend: 0,
                visible: true,
                adjust_kind: 0,
                adjust: [0.0; 4],
                has_blend_if: false,
                blend_if: [0.0, 1.0, 0.0, 1.0],
            },
            LayerDraw {
                id: l1,
                opacity: 1.0,
                blend: 0,
                visible: true,
                adjust_kind: 0,
                adjust: [0.0; 4],
                has_blend_if: true,
                blend_if: [0.0, 0.45, 0.0, 1.0],
            },
        ];
        let pp = gpu.composite_now(&device, &queue, &order);
        let px = gpu.read_composite_pixel(&device, &queue, pp, 4, 4).unwrap();
        assert!(
            px[0] > 0.9,
            "blend-if hides gray top -> white shows through: {px:?}"
        );
    }

    // A layer mask (0 on the left half) hides those pixels in the composite.
    #[test]
    fn layer_mask_hides() {
        let Some((device, queue)) = device() else {
            eprintln!("no GPU adapter; skipping layer_mask_hides");
            return;
        };
        let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
        let size = Size::new(8, 8);
        gpu.ensure_canvas(&device, size);
        let l0 = LayerId(0);
        gpu.ensure_layer(&device, l0);
        gpu.upload_layer(&queue, l0, &solid(8, 1.0, 0.0, 0.0, 1.0)); // red
        let mut mvals = vec![0.0f32; 64];
        for y in 0..8 {
            for x in 4..8 {
                mvals[y * 8 + x] = 1.0; // reveal right half
            }
        }
        gpu.set_mask(&device, &queue, l0, Some(&mvals));
        let order = vec![LayerDraw {
            id: l0,
            opacity: 1.0,
            blend: 0,
            visible: true,
            adjust_kind: 0,
            adjust: [0.0; 4],
            has_blend_if: false,
            blend_if: [0.0, 1.0, 0.0, 1.0],
        }];
        let p = gpu.composite_now(&device, &queue, &order);
        let left = gpu.read_composite_pixel(&device, &queue, p, 1, 4).unwrap();
        let right = gpu.read_composite_pixel(&device, &queue, p, 6, 4).unwrap();
        assert!(left[3] < 0.1, "masked-out left is transparent: {left:?}");
        assert!(
            right[0] > 0.9 && right[3] > 0.9,
            "revealed right is red: {right:?}"
        );
    }
}
