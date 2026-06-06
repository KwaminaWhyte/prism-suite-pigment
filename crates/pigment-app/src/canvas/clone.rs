use super::*;

impl CanvasGpu {
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
    pub(crate) fn paint_clone_dabs(
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
}
