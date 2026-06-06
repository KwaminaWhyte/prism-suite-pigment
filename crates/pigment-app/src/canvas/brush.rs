use super::*;

impl CanvasGpu {
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
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&self.white_mask.view),
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
    pub(crate) fn paint_dabs(
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
}
