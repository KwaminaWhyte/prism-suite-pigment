//! GPU canvas: per-layer Rgba16Float linear-premultiplied textures, a ping-pong
//! compositor (blend modes in-shader), instanced brush-dab painting, and a
//! display pass that composites over a checkerboard and sRGB-encodes for egui.
//! Phase 1 (PLAN.md §4). Tiling/atlas is the next refinement — layers are
//! currently one canvas-sized texture each (a degenerate single tile).

use std::collections::HashMap;

use eframe::egui_wgpu::{CallbackResources, CallbackTrait, ScreenDescriptor};
use eframe::wgpu;
use pigment_core::{LayerId, Size};

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
}

/// Pan/zoom state for the viewport.
#[derive(Clone, Copy, Debug)]
pub struct ViewTransform {
    pub pan: egui::Vec2,
    pub zoom: f32,
}

impl Default for ViewTransform {
    fn default() -> Self {
        Self { pan: egui::Vec2::ZERO, zoom: 1.0 }
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
    _pad: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CompositeParams {
    opacity: f32,
    blend_mode: u32,
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LayerInfo {
    size: [f32; 2],
    _pad: [f32; 2],
}

struct GpuLayer {
    tex: wgpu::Texture,
    view: wgpu::TextureView,
}

/// A full-layer pixel snapshot held GPU-side for one undo step.
struct Snapshot {
    id: LayerId,
    tex: wgpu::Texture,
    label: String,
}

const FMT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
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
        format: FMT,
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

    // Undo/redo: GPU-side full-layer snapshots (tile-COW diffs come with tiling).
    undo_stack: Vec<Snapshot>,
    redo_stack: Vec<Snapshot>,

    // Display.
    display_pipeline: wgpu::RenderPipeline,
    display_bgl: wgpu::BindGroupLayout,
    display_uniform: wgpu::Buffer,
    display_bind_group: Option<wgpu::BindGroup>,

    // Document state mirror.
    layers: HashMap<LayerId, GpuLayer>,
    canvas_size: Size,
    ping: Option<GpuLayer>,
    pong: Option<GpuLayer>,

    // In-progress brush stroke ("wet ink") composited over its owner layer and
    // flattened on pen-up — gives correct per-stroke opacity (RESEARCH.md §4).
    wet: Option<GpuLayer>,
    wet_owner: Option<LayerId>,
    wet_opacity: f32,
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
            ],
        });
        let composite_pipeline = make_render_pipeline(
            device,
            "composite",
            &composite_mod,
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
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
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
            &[&dab_bgl],
            std::slice::from_ref(&dab_layout),
            FMT,
            Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
        );
        let dab_erase_pipeline = make_render_pipeline(
            device,
            "dab.erase",
            &dab_mod,
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
            ],
        });
        let display_pipeline = make_render_pipeline(
            device,
            "display",
            &display_mod,
            &[&display_bgl],
            &[],
            target_format,
            None,
        );
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
            dab_capacity,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            display_pipeline,
            display_bgl,
            display_uniform,
            display_bind_group: None,
            layers: HashMap::new(),
            canvas_size: Size::new(1, 1),
            ping: None,
            pong: None,
            wet: None,
            wet_owner: None,
            wet_opacity: 1.0,
        }
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
        self.layers.clear(); // new document
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
    }

    /// Upload raw RGBA16F (linear premultiplied) bytes into a layer.
    pub fn upload_layer(&mut self, queue: &wgpu::Queue, id: LayerId, bytes: &[u8]) {
        let Some(layer) = self.layers.get(&id) else { return };
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
    fn readback_texture(&self, device: &wgpu::Device, queue: &wgpu::Queue, tex: &wgpu::Texture) -> Option<Vec<u8>> {
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
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        queue.submit([enc.finish()]);
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok()?;
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

    fn readback_pixel(&self, device: &wgpu::Device, queue: &wgpu::Queue, tex: &wgpu::Texture, x: u32, y: u32) -> Option<[f32; 4]> {
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
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        queue.submit([enc.finish()]);
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok()?;
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
    pub fn read_layer(&self, device: &wgpu::Device, queue: &wgpu::Queue, id: LayerId) -> Option<Vec<u8>> {
        let tex = &self.layers.get(&id)?.tex;
        self.readback_texture(device, queue, tex)
    }

    /// Read a single layer pixel (linear premultiplied RGBA) — for the eyedropper.
    pub fn read_pixel(&self, device: &wgpu::Device, queue: &wgpu::Queue, id: LayerId, x: u32, y: u32) -> Option<[f32; 4]> {
        let tex = &self.layers.get(&id)?.tex;
        self.readback_pixel(device, queue, tex, x, y)
    }

    /// Composite all layers now (own encoder) and return whether the result is in `ping`.
    pub fn composite_now(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, order: &[LayerDraw]) -> bool {
        let mut enc = device.create_command_encoder(&Default::default());
        let r = self.composite(device, queue, &mut enc, order);
        queue.submit([enc.finish()]);
        r
    }

    fn target_tex(&self, is_ping: bool) -> Option<&wgpu::Texture> {
        if is_ping { self.ping.as_ref() } else { self.pong.as_ref() }.map(|t| &t.tex)
    }

    /// Read the full composite (call right after `composite_now`).
    pub fn read_composite(&self, device: &wgpu::Device, queue: &wgpu::Queue, is_ping: bool) -> Option<Vec<u8>> {
        self.readback_texture(device, queue, self.target_tex(is_ping)?)
    }

    /// Read one composite pixel (linear premultiplied).
    pub fn read_composite_pixel(&self, device: &wgpu::Device, queue: &wgpu::Queue, is_ping: bool, x: u32, y: u32) -> Option<[f32; 4]> {
        self.readback_pixel(device, queue, self.target_tex(is_ping)?, x, y)
    }

    fn new_snapshot_tex(&self, device: &wgpu::Device) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("undo.snapshot"),
            size: wgpu::Extent3d {
                width: self.canvas_size.width.max(1),
                height: self.canvas_size.height.max(1),
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

    /// Copy a layer's pixels into a fresh GPU texture (one undo unit).
    fn snapshot_of(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, id: LayerId, label: &str) -> Option<Snapshot> {
        let layer = self.layers.get(&id)?;
        let tex = self.new_snapshot_tex(device);
        copy_tex(encoder, &layer.tex, &tex, self.canvas_size);
        Some(Snapshot { id, tex, label: label.to_string() })
    }

    fn push_undo(&mut self, snap: Snapshot) {
        self.undo_stack.push(snap);
        if self.undo_stack.len() > UNDO_MAX {
            self.undo_stack.remove(0);
        }
        self.redo_stack.clear();
    }

    /// Push the active layer's current pixels onto the undo stack (stroke start).
    pub fn begin_command(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, id: LayerId, label: &str) {
        if let Some(snap) = self.snapshot_of(device, encoder, id, label) {
            self.push_undo(snap);
        }
    }

    /// Snapshot using a self-contained encoder (for callers outside the frame
    /// callback, e.g. bucket fill).
    pub fn begin_command_now(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, id: LayerId, label: &str) {
        let mut enc = device.create_command_encoder(&Default::default());
        if let Some(snap) = self.snapshot_of(device, &mut enc, id, label) {
            queue.submit([enc.finish()]);
            self.push_undo(snap);
        }
    }

    /// Labels of pending undo steps (oldest→newest) and redo steps (next→furthest).
    pub fn history_labels(&self) -> (Vec<String>, Vec<String>) {
        let undo = self.undo_stack.iter().map(|s| s.label.clone()).collect();
        let redo = self.redo_stack.iter().rev().map(|s| s.label.clone()).collect();
        (undo, redo)
    }

    fn restore(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, from_undo: bool) {
        let snap = if from_undo { self.undo_stack.pop() } else { self.redo_stack.pop() };
        let Some(snap) = snap else { return };
        if let Some(layer) = self.layers.get(&snap.id) {
            let cur = self.new_snapshot_tex(device);
            copy_tex(encoder, &layer.tex, &cur, self.canvas_size);
            let saved = Snapshot { id: snap.id, tex: cur, label: snap.label.clone() };
            if from_undo {
                self.redo_stack.push(saved);
            } else {
                self.undo_stack.push(saved);
            }
            copy_tex(encoder, &snap.tex, &layer.tex, self.canvas_size);
        }
    }

    pub fn undo(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        self.restore(device, encoder, true);
    }

    pub fn redo(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        self.restore(device, encoder, false);
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
    pub fn wet_end(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder) {
        let Some(owner) = self.wet_owner.take() else { return };
        let (Some(layer), Some(wet), Some(pong)) =
            (self.layers.get(&owner), self.wet.as_ref(), self.pong.as_ref())
        else {
            return;
        };
        queue.write_buffer(
            &self.params_buf,
            WET_PARAMS_OFFSET as u64,
            bytemuck::bytes_of(&CompositeParams { opacity: self.wet_opacity, blend_mode: 0, _pad: [0; 2] }),
        );
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wet.flatten.bg"),
            layout: &self.composite_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&layer.view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&wet.view) },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.params_buf,
                        offset: 0,
                        size: wgpu::BufferSize::new(std::mem::size_of::<CompositeParams>() as u64),
                    }),
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

    fn paint_dabs(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        id: LayerId,
        dabs: &[Dab],
        erase: bool,
        into_wet: bool,
    ) {
        if dabs.is_empty() {
            return;
        }
        let target = if into_wet { self.wet.as_ref() } else { self.layers.get(&id) };
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
        queue.write_buffer(&self.dab_instances, 0, bytemuck::cast_slice(dabs));
        queue.write_buffer(
            &self.dab_info_buf,
            0,
            bytemuck::bytes_of(&LayerInfo {
                size: [self.canvas_size.width as f32, self.canvas_size.height as f32],
                _pad: [0.0; 2],
            }),
        );
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dab.bg"),
            layout: &self.dab_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.dab_info_buf.as_entire_binding(),
            }],
        });

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("dab.pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target.view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        let pipeline = if erase { &self.dab_erase_pipeline } else { &self.dab_pipeline };
        pass.set_pipeline(pipeline);
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
        // Write all layer params up front (one buffer, dynamic offsets).
        let visible: Vec<&LayerDraw> = order.iter().filter(|l| l.visible).collect();
        for (i, l) in visible.iter().enumerate() {
            let p = CompositeParams { opacity: l.opacity, blend_mode: l.blend, _pad: [0; 2] };
            queue.write_buffer(&self.params_buf, i as u64 * PARAMS_STRIDE, bytemuck::bytes_of(&p));
        }
        if self.wet_owner.is_some() {
            let p = CompositeParams { opacity: self.wet_opacity, blend_mode: 0, _pad: [0; 2] };
            queue.write_buffer(&self.params_buf, WET_PARAMS_OFFSET as u64, bytemuck::bytes_of(&p));
        }

        let ping = self.ping.as_ref().unwrap();
        let pong = self.pong.as_ref().unwrap();

        // Ordered passes: each visible layer, plus the wet stroke just above its
        // owner so the in-progress stroke previews at the right depth.
        let mut passes: Vec<(&wgpu::TextureView, u32)> = Vec::new();
        for (i, l) in visible.iter().enumerate() {
            if let Some(layer) = self.layers.get(&l.id) {
                passes.push((&layer.view, (i as u64 * PARAMS_STRIDE) as u32));
                if self.wet_owner == Some(l.id) {
                    if let Some(wet) = self.wet.as_ref() {
                        passes.push((&wet.view, WET_PARAMS_OFFSET));
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
        for (layer_view, offset) in passes {
            let (src, dst) = if src_is_ping { (ping, pong) } else { (pong, ping) };
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("composite.bg"),
                layout: &self.composite_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&src.view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(layer_view) },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.params_buf,
                            offset: 0,
                            size: wgpu::BufferSize::new(std::mem::size_of::<CompositeParams>() as u64),
                        }),
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
        self.display_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("display.bg"),
            layout: &self.display_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.display_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(final_view) },
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
fn copy_tex(encoder: &mut wgpu::CommandEncoder, src: &wgpu::Texture, dst: &wgpu::Texture, size: Size) {
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
    bind_group_layouts: &[&wgpu::BindGroupLayout],
    vertex_buffers: &[wgpu::VertexBufferLayout],
    format: wgpu::TextureFormat,
    blend: Option<wgpu::BlendState>,
) -> wgpu::RenderPipeline {
    let bgls: Vec<Option<&wgpu::BindGroupLayout>> = bind_group_layouts.iter().map(|b| Some(*b)).collect();
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
            entry_point: Some("fs_main"),
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
    /// Set on the first frame of a stroke — snapshots the layer for undo.
    pub begin_command: bool,
    pub command_label: String,
    pub undo: u32,
    pub redo: u32,
    // Wet brush stroke lifecycle.
    pub wet_begin: bool,
    pub wet_end: bool,
    pub wet_opacity: f32,
    pub paint_into_wet: bool,
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

        for _ in 0..self.undo {
            gpu.undo(device, encoder);
        }
        for _ in 0..self.redo {
            gpu.redo(device, encoder);
        }
        if self.begin_command {
            gpu.begin_command(device, encoder, self.active_id, &self.command_label);
        }
        if self.wet_begin {
            gpu.wet_begin(encoder, self.active_id, self.wet_opacity);
        }
        gpu.paint_dabs(device, queue, encoder, self.active_id, &self.dabs, self.erase, self.paint_into_wet);
        if self.wet_end {
            gpu.wet_end(device, queue, encoder);
        }
        let final_is_ping = gpu.composite(device, queue, encoder, &self.layers);
        gpu.build_display_bind_group(device, final_is_ping);

        let [sw, sh] = screen_descriptor.size_in_pixels;
        let ppp = screen_descriptor.pixels_per_point;
        let to_clip = |p: egui::Pos2| -> [f32; 2] {
            [p.x * ppp / sw as f32 * 2.0 - 1.0, 1.0 - p.y * ppp / sh as f32 * 2.0]
        };
        let uni = DisplayUniform {
            clip_min: to_clip(self.doc_rect.min),
            clip_max: to_clip(self.doc_rect.max),
            checker_px: self.checker_pts * ppp,
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
        let Some(bg) = &gpu.display_bind_group else { return };
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
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
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

        let order = vec![LayerDraw { id: l0, opacity: 1.0, blend: 0, visible: true }];
        let p = gpu.composite_now(&device, &queue, &order);
        let px = gpu.read_composite_pixel(&device, &queue, p, 4, 4).unwrap();
        assert!(px[0] > 0.9 && px[1] < 0.1 && px[3] > 0.9, "composite red: {px:?}");

        // Blue wet brush dab over the center, flattened.
        let mut enc = device.create_command_encoder(&Default::default());
        gpu.begin_command(&device, &mut enc, l0, "test");
        gpu.wet_begin(&mut enc, l0, 1.0);
        let dab = Dab { center: [4.0, 4.0], radius: 8.0, hardness: 0.95, color: [0.0, 0.0, 1.0, 1.0] };
        gpu.paint_dabs(&device, &queue, &mut enc, l0, &[dab], false, true);
        gpu.wet_end(&device, &queue, &mut enc);
        queue.submit([enc.finish()]);

        let p = gpu.composite_now(&device, &queue, &order);
        let px = gpu.read_composite_pixel(&device, &queue, p, 4, 4).unwrap();
        assert!(px[2] > 0.9 && px[0] < 0.1, "baked blue at center: {px:?}");

        // Undo restores red.
        let mut enc = device.create_command_encoder(&Default::default());
        gpu.undo(&device, &mut enc);
        queue.submit([enc.finish()]);
        let p = gpu.composite_now(&device, &queue, &order);
        let px = gpu.read_composite_pixel(&device, &queue, p, 4, 4).unwrap();
        assert!(px[0] > 0.9 && px[2] < 0.1, "undo back to red: {px:?}");
    }
}
