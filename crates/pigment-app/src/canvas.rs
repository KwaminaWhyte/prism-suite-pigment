//! The GPU canvas viewport: an `egui_wgpu` paint callback that draws the
//! document (textured quad) with a transparency checkerboard behind it,
//! positioned by a CPU-side pan/zoom transform. Phase 0 (PLAN.md §4).

use std::sync::Arc;

use eframe::egui_wgpu::{CallbackResources, CallbackTrait, ScreenDescriptor};
use eframe::wgpu;
use pigment_io::LoadedImage;

/// Pan/zoom state for the viewport. Pan is in egui points; zoom is a scalar
/// (1.0 = one document pixel per point).
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
    /// Zoom toward a cursor anchor (in points, relative to viewport center) so
    /// the point under the cursor stays put.
    pub fn zoom_to(&mut self, factor: f32, anchor_from_center: egui::Vec2) {
        let new_zoom = (self.zoom * factor).clamp(0.02, 64.0);
        let real = new_zoom / self.zoom;
        // Keep the anchor stationary: pan must scale about the anchor.
        self.pan = (self.pan - anchor_from_center) * real + anchor_from_center;
        self.zoom = new_zoom;
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CanvasUniform {
    clip_min: [f32; 2],
    clip_max: [f32; 2],
    checker_px: f32,
    _pad: [f32; 3],
}

/// Long-lived GPU resources, stored in egui's `callback_resources`.
pub struct CanvasRenderer {
    uniform: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler_nearest: wgpu::Sampler,
    pipeline_checker: wgpu::RenderPipeline,
    pipeline_image: wgpu::RenderPipeline,
    // Rebuilt when the loaded image changes.
    bind_group: Option<wgpu::BindGroup>,
    image_gen: Option<u64>,
}

impl CanvasRenderer {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("canvas.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/canvas.wgsl").into()),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("canvas.bgl"),
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("canvas.pl"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let make_pipeline = |fs_entry: &str, label: &str, blend: Option<wgpu::BlendState>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(fs_entry),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target_format,
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
        };

        let pipeline_checker = make_pipeline("fs_checker", "canvas.checker", None);
        let pipeline_image = make_pipeline(
            "fs_image",
            "canvas.image",
            Some(wgpu::BlendState::ALPHA_BLENDING),
        );

        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("canvas.uniform"),
            size: std::mem::size_of::<CanvasUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sampler_nearest = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("canvas.sampler.nearest"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            uniform,
            bind_group_layout,
            sampler_nearest,
            pipeline_checker,
            pipeline_image,
            bind_group: None,
            image_gen: None,
        }
    }

    /// Upload a newly loaded image and (re)build the bind group.
    fn ensure_image(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, gen: u64, img: &LoadedImage) {
        if self.image_gen == Some(gen) {
            return;
        }
        let size = wgpu::Extent3d {
            width: img.size.width,
            height: img.size.height,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("canvas.image.tex"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // egui's target is non-sRGB (Bgra8Unorm), i.e. it holds
            // display-referred sRGB-encoded bytes. Sample the raw PNG bytes as
            // Unorm and pass them straight through. The real compositor (Phase 1)
            // renders to an f16 linear offscreen target and sRGB-encodes at blit.
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &img.rgba8,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * img.size.width),
                rows_per_image: Some(img.size.height),
            },
            size,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("canvas.bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.uniform.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler_nearest),
                },
            ],
        });
        self.bind_group = Some(bind_group);
        self.image_gen = Some(gen);
    }
}

/// Per-frame paint callback carrying the document layout (cheap to clone:
/// the image is shared via `Arc`).
pub struct CanvasPaint {
    /// Document quad in egui screen points (pan/zoom already applied).
    pub doc_rect: egui::Rect,
    pub checker_pts: f32,
    pub image: Option<(u64, Arc<LoadedImage>)>,
}

impl CallbackTrait for CanvasPaint {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen_descriptor: &ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let renderer: &mut CanvasRenderer = resources.get_mut().unwrap();

        if let Some((gen, img)) = &self.image {
            renderer.ensure_image(device, queue, *gen, img);
        }

        let [sw, sh] = screen_descriptor.size_in_pixels;
        let ppp = screen_descriptor.pixels_per_point;
        let to_clip = |p: egui::Pos2| -> [f32; 2] {
            let px = p.x * ppp;
            let py = p.y * ppp;
            [px / sw as f32 * 2.0 - 1.0, 1.0 - py / sh as f32 * 2.0]
        };
        let uni = CanvasUniform {
            clip_min: to_clip(self.doc_rect.min),
            clip_max: to_clip(self.doc_rect.max),
            checker_px: self.checker_pts * ppp,
            _pad: [0.0; 3],
        };
        queue.write_buffer(&renderer.uniform, 0, bytemuck::bytes_of(&uni));
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &CallbackResources,
    ) {
        let renderer: &CanvasRenderer = resources.get().unwrap();
        let Some(bind_group) = &renderer.bind_group else {
            return;
        };
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.set_pipeline(&renderer.pipeline_checker);
        render_pass.draw(0..6, 0..1);
        render_pass.set_pipeline(&renderer.pipeline_image);
        render_pass.draw(0..6, 0..1);
    }
}
