use super::*;

impl CanvasGpu {
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

    /// Ping-pong composite all visible layers; returns the final view index.
    pub(crate) fn composite(
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
            // Clip to the layer below (needs a layer beneath it in the stack).
            if l.clipped && i > 0 {
                p.has_clip = 1;
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
            &wgpu::TextureView, // clipping-mask base (layer below) or white
        )> = Vec::new();
        for (i, l) in visible.iter().enumerate() {
            if let Some(layer) = self.layers.get(&l.id) {
                let lut = self
                    .curve_luts
                    .get(&l.id)
                    .map(|g| &g.view)
                    .unwrap_or(identity_lut);
                // Clip base = the layer directly below (white = no clip).
                let clip_base = if l.clipped && i > 0 {
                    self.layers
                        .get(&visible[i - 1].id)
                        .map(|b| &b.view)
                        .unwrap_or(&self.white_mask.view)
                } else {
                    &self.white_mask.view
                };
                passes.push((
                    &layer.view,
                    (i as u64 * PARAMS_STRIDE) as u32,
                    self.mask_view(l.id),
                    lut,
                    clip_base,
                ));
                if self.wet_owner == Some(l.id) {
                    if let Some(wet) = self.wet.as_ref() {
                        passes.push((
                            &wet.view,
                            WET_PARAMS_OFFSET,
                            &self.white_mask.view,
                            identity_lut,
                            &self.white_mask.view,
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
        for (layer_view, offset, mask_view, lut_view, clip_base_view) in passes {
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
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(clip_base_view),
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

    pub(crate) fn build_display_bind_group(&mut self, device: &wgpu::Device, final_is_ping: bool) {
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
