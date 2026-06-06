use super::*;

impl CanvasGpu {
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
}
