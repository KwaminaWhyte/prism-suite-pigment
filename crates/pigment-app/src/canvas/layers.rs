use super::*;

impl CanvasGpu {
    pub(crate) fn mask_view(&self, id: LayerId) -> &wgpu::TextureView {
        self.masks
            .get(&id)
            .map(|m| &m.view)
            .unwrap_or(&self.white_mask.view)
    }

    /// Initialize the 1x1 white mask fallback (used by maskless layers).
    pub(crate) fn ensure_white(&mut self, queue: &wgpu::Queue) {
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
        // Start from a plain (no-style) param block and turn on the affine only.
        let p = CompositeParams {
            has_xform: 1,
            m: self.xform_m,
            off: self.xform_off,
            ..CompositeParams::plain(1.0, 0)
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
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&self.white_mask.view),
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
        self.channels.clear();
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
    pub(crate) fn ensure_identity_lut(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
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

    /// Build + upload a Gradient-Map LUT (luma → lerp(low, high)) for layer `id`,
    /// stored in the same per-layer LUT slot the compositor samples.
    pub fn set_gradient_lut(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: LayerId,
        low: [f32; 3],
        high: [f32; 3],
    ) {
        let n = LUT_W as usize;
        let mut texels = vec![half::f16::from_f32(0.0); n * 4];
        for (i, px) in texels.chunks_mut(4).enumerate() {
            let t = i as f32 / (n - 1) as f32;
            px[0] = half::f16::from_f32(low[0] + (high[0] - low[0]) * t);
            px[1] = half::f16::from_f32(low[1] + (high[1] - low[1]) * t);
            px[2] = half::f16::from_f32(low[2] + (high[2] - low[2]) * t);
            px[3] = half::f16::from_f32(1.0);
        }
        let tex = self
            .curve_luts
            .entry(id)
            .or_insert_with(|| make_lut_target(device, "lut.gradmap"));
        write_lut(queue, &tex.tex, &texels);
    }

    /// Build + upload a Color-Balance layer's per-channel transfer LUT (shadow/
    /// midtone/highlight RGB shifts → a per-channel curve in the LUT's .rgb),
    /// stored in the same per-layer LUT slot the compositor samples (kind 13).
    pub fn set_color_balance_lut(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: LayerId,
        shadows: [f32; 3],
        midtones: [f32; 3],
        highlights: [f32; 3],
    ) {
        let n = LUT_W as usize;
        let cb = prism_core::adjust::ColorBalanceLuts::build(shadows, midtones, highlights, n);
        let mut texels = vec![half::f16::from_f32(0.0); n * 4];
        for (i, px) in texels.chunks_mut(4).enumerate() {
            px[0] = half::f16::from_f32(cb.r[i]);
            px[1] = half::f16::from_f32(cb.g[i]);
            px[2] = half::f16::from_f32(cb.b[i]);
            px[3] = half::f16::from_f32(1.0);
        }
        let t = self
            .curve_luts
            .entry(id)
            .or_insert_with(|| make_lut_target(device, "lut.colorbalance"));
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
    pub(crate) fn readback_texture(
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

    pub(crate) fn readback_pixel(
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
}
