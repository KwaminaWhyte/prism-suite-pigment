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
        self.filter_pass_c(
            device, queue, input, output, kind, dir, amount, radius, [0.0; 2],
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn filter_pass_c(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &wgpu::TextureView,
        output: &wgpu::TextureView,
        kind: u32,
        dir: [f32; 2],
        amount: f32,
        radius: f32,
        center: [f32; 2],
    ) {
        // Single-input pass: alias the secondary texture to `input` (only the
        // High Pass combine, kind 24, needs a distinct secondary — see
        // `filter_pass_2`).
        self.filter_pass_2(
            device, queue, input, input, output, kind, dir, amount, radius, center,
        )
    }

    /// Like `filter_pass_c` but with a distinct secondary input texture bound at
    /// binding 3 (`orig` in the shader). Used by High Pass (kind 24) to subtract
    /// a Gaussian-blurred copy (`input`) from the untouched source (`orig`).
    #[allow(clippy::too_many_arguments)]
    fn filter_pass_2(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &wgpu::TextureView,
        orig: &wgpu::TextureView,
        output: &wgpu::TextureView,
        kind: u32,
        dir: [f32; 2],
        amount: f32,
        radius: f32,
        center: [f32; 2],
    ) {
        self.filter_pass_cr(
            device,
            queue,
            input,
            orig,
            output,
            kind,
            dir,
            amount,
            radius,
            center,
            [[0.0; 4]; 3],
        )
    }

    /// Like `filter_pass_2` but also writes the Camera Raw `cr` overflow payload
    /// (shader kind 32). Every other path passes an all-zero `cr` (a no-op).
    #[allow(clippy::too_many_arguments)]
    fn filter_pass_cr(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &wgpu::TextureView,
        orig: &wgpu::TextureView,
        output: &wgpu::TextureView,
        kind: u32,
        dir: [f32; 2],
        amount: f32,
        radius: f32,
        center: [f32; 2],
        cr: [[f32; 4]; 3],
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
                center,
                cr,
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
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(orig),
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

    /// Apply a destructive filter to a layer.
    /// kind: 1 Gaussian blur, 2 sharpen, 3 pixelate, 5 box blur. Gaussian and
    /// box blur are separable and run two passes (H then V); the rest run once.
    /// (Motion blur and radial blur take extra geometry — see their methods.)
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
        // Separable blurs (Gaussian=1, Box=5): horizontal then vertical pass.
        if kind == 1 || kind == 5 {
            let tx = [1.0 / self.canvas_size.width as f32, 0.0];
            let ty = [0.0, 1.0 / self.canvas_size.height as f32];
            self.filter_pass(
                device,
                queue,
                &layer.view,
                &pong.view,
                kind,
                tx,
                0.0,
                radius,
            );
            self.filter_pass(
                device,
                queue,
                &pong.view,
                &layer.view,
                kind,
                ty,
                0.0,
                radius,
            );
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

    /// Capture the layer's current pixels as its **smart-filter source** (the
    /// un-filtered baseline the stack re-applies from). Idempotent: if a source
    /// already exists for `id` it is left untouched, so re-applying an edited
    /// stack never re-snapshots an already-filtered result. Call this once, when
    /// the layer's first smart filter is added.
    pub fn ensure_smart_source(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, id: LayerId) {
        if self.smart_sources.contains_key(&id) {
            return;
        }
        let Some(layer) = self.layers.get(&id) else {
            return;
        };
        let src = make_target(device, self.canvas_size, "smart.source");
        let mut enc = device.create_command_encoder(&Default::default());
        copy_tex(&mut enc, &layer.tex, &src.tex, self.canvas_size);
        queue.submit([enc.finish()]);
        self.smart_sources.insert(id, src);
    }

    /// Whether layer `id` currently holds a smart-filter source snapshot.
    pub fn has_smart_source(&self, id: LayerId) -> bool {
        self.smart_sources.contains_key(&id)
    }

    /// Restore the layer's pixels from its smart-filter source and drop the
    /// source. Used when the last smart filter is removed (the layer goes back to
    /// being plain, editable pixels — the source becomes the live layer again).
    pub fn clear_smart_source(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: LayerId,
    ) {
        let (Some(src), Some(layer)) = (self.smart_sources.get(&id), self.layers.get(&id)) else {
            self.smart_sources.remove(&id);
            return;
        };
        let mut enc = device.create_command_encoder(&Default::default());
        copy_tex(&mut enc, &src.tex, &layer.tex, self.canvas_size);
        queue.submit([enc.finish()]);
        self.smart_sources.remove(&id);
    }

    /// Re-apply a layer's **non-destructive smart-filter stack**: reset the layer
    /// to its source pixels, then run each `(kind, radius, amount)` pass over it
    /// in order. The source is snapshotted on the first call (see
    /// [`Self::ensure_smart_source`]) and never overwritten, so this is fully
    /// reversible — passing an empty `passes` (all filters disabled) leaves the
    /// layer equal to the source. Reuses the same GPU filter passes the
    /// destructive Filter menu uses (separable blur runs H then V); no undo
    /// snapshot is taken (the stack itself, held app-side, is the edit history).
    pub fn reapply_smart_filters(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: LayerId,
        passes: &[crate::app::smart_filter::SmartPass],
    ) {
        self.ensure_smart_source(device, queue, id);
        let (Some(src), Some(layer), Some(pong)) = (
            self.smart_sources.get(&id),
            self.layers.get(&id),
            self.pong.as_ref(),
        ) else {
            return;
        };
        // Start from the un-filtered source pixels.
        let mut enc = device.create_command_encoder(&Default::default());
        copy_tex(&mut enc, &src.tex, &layer.tex, self.canvas_size);
        queue.submit([enc.finish()]);
        // Apply each enabled filter in order, in place on the layer texture.
        let (txw, txh) = (
            1.0 / self.canvas_size.width as f32,
            1.0 / self.canvas_size.height as f32,
        );
        for pass in passes {
            let (kind, radius, amount) = (pass.kind, pass.radius, pass.amount);
            if kind == 1 || kind == 5 {
                // Separable blur: layer -> pong (H), pong -> layer (V).
                self.filter_pass(device, queue, &layer.view, &pong.view, kind, [txw, 0.0], 0.0, radius);
                self.filter_pass(device, queue, &pong.view, &layer.view, kind, [0.0, txh], 0.0, radius);
            } else if kind == 32 {
                // Camera Raw: single pass carrying the develop controls in `cr`.
                let cr = [
                    [pass.cr[0], pass.cr[1], pass.cr[2], pass.cr[3]],
                    [pass.cr[4], pass.cr[5], pass.cr[6], pass.cr[7]],
                    [pass.cr[8], pass.cr[9], pass.cr[10], pass.cr[11]],
                ];
                self.filter_pass_cr(
                    device, queue, &layer.view, &layer.view, &pong.view, kind, [0.0; 2], amount,
                    radius, [0.0; 2], cr,
                );
                let mut enc = device.create_command_encoder(&Default::default());
                copy_tex(&mut enc, &pong.tex, &layer.tex, self.canvas_size);
                queue.submit([enc.finish()]);
            } else {
                // Single pass: layer -> pong, then copy pong back into layer.
                self.filter_pass(device, queue, &layer.view, &pong.view, kind, [0.0; 2], amount, radius);
                let mut enc = device.create_command_encoder(&Default::default());
                copy_tex(&mut enc, &pong.tex, &layer.tex, self.canvas_size);
                queue.submit([enc.finish()]);
            }
        }
    }

    /// Motion blur: a flat box average of `2*radius+1` taps along `angle_rad`
    /// (a directional/linear blur). Single pass; destructive + undoable.
    pub fn apply_motion_blur(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: LayerId,
        angle_rad: f32,
        radius: f32,
    ) {
        self.begin_command_now(device, queue, id, "Motion Blur");
        let (Some(layer), Some(pong)) = (self.layers.get(&id), self.pong.as_ref()) else {
            return;
        };
        // Unit direction scaled into uv (texel) space, matching the CPU ref
        // which steps one pixel per tap along (cos, sin).
        let dir = [
            angle_rad.cos() / self.canvas_size.width as f32,
            angle_rad.sin() / self.canvas_size.height as f32,
        ];
        self.filter_pass(device, queue, &layer.view, &pong.view, 4, dir, 0.0, radius);
        let mut enc = device.create_command_encoder(&Default::default());
        copy_tex(&mut enc, &pong.tex, &layer.tex, self.canvas_size);
        queue.submit([enc.finish()]);
    }

    /// Radial blur about `(cx, cy)` (pixel coords). `spin` selects rotational
    /// (true) vs zoom (false). `amount` is the spin angle in radians or the
    /// zoom fraction; `samples` is the tap count. Single pass; destructive.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_radial_blur(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: LayerId,
        cx: f32,
        cy: f32,
        spin: bool,
        amount: f32,
        samples: u32,
    ) {
        self.begin_command_now(device, queue, id, "Radial Blur");
        let (Some(layer), Some(pong)) = (self.layers.get(&id), self.pong.as_ref()) else {
            return;
        };
        let kind = if spin { 6 } else { 7 };
        let center = [
            cx / self.canvas_size.width as f32,
            cy / self.canvas_size.height as f32,
        ];
        self.filter_pass_c(
            device,
            queue,
            &layer.view,
            &pong.view,
            kind,
            [0.0; 2],
            amount,
            samples.max(1) as f32,
            center,
        );
        let mut enc = device.create_command_encoder(&Default::default());
        copy_tex(&mut enc, &pong.tex, &layer.tex, self.canvas_size);
        queue.submit([enc.finish()]);
    }

    /// Apply a coordinate-displacement Distort filter about `(cx, cy)` (pixel
    /// coords). Each Distort kind remaps the sampled source coordinate per pixel
    /// and edge-clamps: Twirl (kind 8, `amount` = max angle rad, `radius` px),
    /// Pinch/Spherize (kind 9, signed `amount`, `radius` px), Ripple/Wave (kind
    /// 10, `dir` = `[amplitude_px, wavelength_px]`), Polar rect→polar (kind 11)
    /// and polar→rect (kind 12). Single pass; destructive + undoable.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_distort(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: LayerId,
        kind: u32,
        cx: f32,
        cy: f32,
        amount: f32,
        radius: f32,
        dir: [f32; 2],
    ) {
        self.begin_command_now(device, queue, id, "Distort");
        let (Some(layer), Some(pong)) = (self.layers.get(&id), self.pong.as_ref()) else {
            return;
        };
        let center = [
            cx / self.canvas_size.width as f32,
            cy / self.canvas_size.height as f32,
        ];
        self.filter_pass_c(
            device,
            queue,
            &layer.view,
            &pong.view,
            kind,
            dir,
            amount,
            radius,
            center,
        );
        let mut enc = device.create_command_encoder(&Default::default());
        copy_tex(&mut enc, &pong.tex, &layer.tex, self.canvas_size);
        queue.submit([enc.finish()]);
    }

    /// Apply a Stylize edge/relief filter to the active layer. Each is a single
    /// neighbour-sampling pass over a `width`-px Sobel step (`radius` in the
    /// shader): Find Edges (kind 13), Emboss (kind 14, `dir` = unit light dir,
    /// `amount` = relief gain), Glowing Edges (kind 15, `amount` = brightness),
    /// Diffuse (kind 16, `dir.x` = seed, `amount` = max neighbour displacement
    /// px, `width` ignored). Single pass; destructive + undoable (region-COW).
    #[allow(clippy::too_many_arguments)]
    pub fn apply_stylize(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: LayerId,
        kind: u32,
        amount: f32,
        width: f32,
        dir: [f32; 2],
    ) {
        self.begin_command_now(device, queue, id, "Stylize");
        let (Some(layer), Some(pong)) = (self.layers.get(&id), self.pong.as_ref()) else {
            return;
        };
        self.filter_pass_c(
            device,
            queue,
            &layer.view,
            &pong.view,
            kind,
            dir,
            amount,
            width,
            [0.0; 2],
        );
        let mut enc = device.create_command_encoder(&Default::default());
        copy_tex(&mut enc, &pong.tex, &layer.tex, self.canvas_size);
        queue.submit([enc.finish()]);
    }

    /// Oil Paint (Kuwahara quadrant filter, kind 27) on the active layer:
    /// replace each pixel with the mean colour of the lowest-luma-variance
    /// quadrant of its `(2·radius+1)²` window — painterly patches with crisp
    /// edges. `radius` is the quadrant half-size in px (clamped 1..8 in-shader).
    /// Single pass; destructive + undoable (region-COW).
    pub fn apply_oil_paint(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: LayerId,
        radius: f32,
    ) {
        self.begin_command_now(device, queue, id, "Oil Paint");
        let (Some(layer), Some(pong)) = (self.layers.get(&id), self.pong.as_ref()) else {
            return;
        };
        self.filter_pass_c(
            device,
            queue,
            &layer.view,
            &pong.view,
            27,
            [0.0; 2],
            0.0,
            radius,
            [0.0; 2],
        );
        let mut enc = device.create_command_encoder(&Default::default());
        copy_tex(&mut enc, &pong.tex, &layer.tex, self.canvas_size);
        queue.submit([enc.finish()]);
    }

    /// Add seeded-deterministic noise to the active layer (kind 17). `amount` is
    /// the noise strength (0..1); `mono` applies the same noise to R/G/B;
    /// `gaussian` selects gaussian (true) vs uniform (false) noise; `seed` makes
    /// it reproducible. Single pass; destructive + undoable (region-COW).
    pub fn apply_noise(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: LayerId,
        amount: f32,
        mono: bool,
        gaussian: bool,
        seed: f32,
    ) {
        self.begin_command_now(device, queue, id, "Add Noise");
        let (Some(layer), Some(pong)) = (self.layers.get(&id), self.pong.as_ref()) else {
            return;
        };
        let dir = [seed, if mono { 1.0 } else { 0.0 }];
        let gflag = if gaussian { 1.0 } else { 0.0 };
        self.filter_pass_c(
            device,
            queue,
            &layer.view,
            &pong.view,
            17,
            dir,
            amount,
            gflag,
            [0.0; 2],
        );
        let mut enc = device.create_command_encoder(&Default::default());
        copy_tex(&mut enc, &pong.tex, &layer.tex, self.canvas_size);
        queue.submit([enc.finish()]);
    }

    /// Per-channel median despeckle on the active layer: Median (kind 18, full
    /// replacement) or Dust & Scratches (kind 19, replace only when the pixel
    /// differs from the window median by more than `threshold`). `radius` is the
    /// window radius in px (window = `2·radius+1`). Single pass; destructive +
    /// undoable (region-COW).
    pub fn apply_median(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: LayerId,
        radius: f32,
        threshold: Option<f32>,
    ) {
        self.begin_command_now(device, queue, id, "Median");
        let (Some(layer), Some(pong)) = (self.layers.get(&id), self.pong.as_ref()) else {
            return;
        };
        let (kind, amount) = match threshold {
            Some(t) => (19, t),
            None => (18, 0.0),
        };
        self.filter_pass_c(
            device,
            queue,
            &layer.view,
            &pong.view,
            kind,
            [0.0; 2],
            amount,
            radius,
            [0.0; 2],
        );
        let mut enc = device.create_command_encoder(&Default::default());
        copy_tex(&mut enc, &pong.tex, &layer.tex, self.canvas_size);
        queue.submit([enc.finish()]);
    }

    /// Mosaic on the active layer (kind 20): average each `cell`×`cell` block to
    /// one colour (the true block mean, vs the legacy point-sampling Pixelate).
    /// `cell` is the cell size in px. Single pass; destructive + undoable.
    pub fn apply_mosaic(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: LayerId,
        cell: f32,
    ) {
        self.begin_command_now(device, queue, id, "Mosaic");
        let (Some(layer), Some(pong)) = (self.layers.get(&id), self.pong.as_ref()) else {
            return;
        };
        self.filter_pass_c(
            device,
            queue,
            &layer.view,
            &pong.view,
            20,
            [0.0; 2],
            0.0,
            cell,
            [0.0; 2],
        );
        let mut enc = device.create_command_encoder(&Default::default());
        copy_tex(&mut enc, &pong.tex, &layer.tex, self.canvas_size);
        queue.submit([enc.finish()]);
    }

    /// Crystallize on the active layer (kind 21): snap each pixel to the colour
    /// of its nearest jittered seed (one per `cell`×`cell` block, jittered by a
    /// hash of the block index + `seed`), giving irregular Voronoi cells. Single
    /// pass; destructive + undoable.
    pub fn apply_crystallize(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: LayerId,
        cell: f32,
        seed: f32,
    ) {
        self.begin_command_now(device, queue, id, "Crystallize");
        let (Some(layer), Some(pong)) = (self.layers.get(&id), self.pong.as_ref()) else {
            return;
        };
        self.filter_pass_c(
            device,
            queue,
            &layer.view,
            &pong.view,
            21,
            [seed, 0.0],
            0.0,
            cell,
            [0.0; 2],
        );
        let mut enc = device.create_command_encoder(&Default::default());
        copy_tex(&mut enc, &pong.tex, &layer.tex, self.canvas_size);
        queue.submit([enc.finish()]);
    }

    /// Color Halftone on the active layer (kind 22): a per-channel dot screen of
    /// `cell`-px cells rotated by `angle_rad`, each cell's channel average setting
    /// a dot radius (denser ink for darker channels). Single pass; destructive +
    /// undoable.
    pub fn apply_color_halftone(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: LayerId,
        cell: f32,
        angle_rad: f32,
    ) {
        self.begin_command_now(device, queue, id, "Color Halftone");
        let (Some(layer), Some(pong)) = (self.layers.get(&id), self.pong.as_ref()) else {
            return;
        };
        let dir = [angle_rad.cos(), angle_rad.sin()];
        self.filter_pass_c(
            device,
            queue,
            &layer.view,
            &pong.view,
            22,
            dir,
            0.0,
            cell,
            [0.0; 2],
        );
        let mut enc = device.create_command_encoder(&Default::default());
        copy_tex(&mut enc, &pong.tex, &layer.tex, self.canvas_size);
        queue.submit([enc.finish()]);
    }

    /// Mezzotint on the active layer (kind 23): seeded threshold dither to pure
    /// black/white grain. `amount` biases the threshold; `seed` makes it
    /// reproducible. Single pass; destructive + undoable.
    pub fn apply_mezzotint(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: LayerId,
        amount: f32,
        seed: f32,
    ) {
        self.begin_command_now(device, queue, id, "Mezzotint");
        let (Some(layer), Some(pong)) = (self.layers.get(&id), self.pong.as_ref()) else {
            return;
        };
        self.filter_pass_c(
            device,
            queue,
            &layer.view,
            &pong.view,
            23,
            [seed, 0.0],
            amount,
            0.0,
            [0.0; 2],
        );
        let mut enc = device.create_command_encoder(&Default::default());
        copy_tex(&mut enc, &pong.tex, &layer.tex, self.canvas_size);
        queue.submit([enc.finish()]);
    }

    /// High Pass on the active layer (kind 24): the classic Photoshop sharpen
    /// prep — subtract a Gaussian-blurred copy from the original and re-centre at
    /// mid-gray, leaving only the high-frequency detail/edges as a signed
    /// deviation about 0.5. `radius` is the Gaussian blur radius (larger →
    /// coarser detail kept); `amount` scales the detail (1 = identity high pass).
    /// Runs the separable Gaussian (kind 1, H then V) into the layer, then a
    /// two-input combine pass that reads the blurred layer + the saved original.
    /// Destructive + undoable (region-COW).
    pub fn apply_high_pass(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: LayerId,
        radius: f32,
        amount: f32,
    ) {
        self.begin_command_now(device, queue, id, "High Pass");
        let (Some(layer), Some(ping), Some(pong)) =
            (self.layers.get(&id), self.ping.as_ref(), self.pong.as_ref())
        else {
            return;
        };
        // Stash the untouched source in `ping` so the combine pass can read it
        // after the blur has overwritten the layer.
        let mut enc = device.create_command_encoder(&Default::default());
        copy_tex(&mut enc, &layer.tex, &ping.tex, self.canvas_size);
        queue.submit([enc.finish()]);
        // Separable Gaussian blur in place (kind 1): H into pong, V back into the
        // layer — mirrors `apply_filter`'s blur path exactly.
        let tx = [1.0 / self.canvas_size.width as f32, 0.0];
        let ty = [0.0, 1.0 / self.canvas_size.height as f32];
        self.filter_pass(device, queue, &layer.view, &pong.view, 1, tx, 0.0, radius);
        self.filter_pass(device, queue, &pong.view, &layer.view, 1, ty, 0.0, radius);
        // Combine: input = blurred layer, orig = saved original (ping) → pong.
        self.filter_pass_2(
            device,
            queue,
            &layer.view,
            &ping.view,
            &pong.view,
            24,
            [0.0; 2],
            amount,
            radius,
            [0.0; 2],
        );
        let mut enc = device.create_command_encoder(&Default::default());
        copy_tex(&mut enc, &pong.tex, &layer.tex, self.canvas_size);
        queue.submit([enc.finish()]);
    }

    /// Render Clouds (kind 25) / Difference Clouds (kind 26) into the active
    /// layer — a generator that fills it with a deterministic multi-octave
    /// value-noise (fBm) field. `seed` makes it reproducible; `scale` is the base
    /// feature size (px), `roughness` the per-octave amplitude falloff, `octaves`
    /// the layer count. Clouds ignores the source; Difference Clouds composites
    /// the field against the existing pixels via per-channel absolute difference
    /// (so repeated application builds veins). Single pass; destructive + undoable
    /// (region-COW).
    #[allow(clippy::too_many_arguments)]
    pub fn apply_clouds(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: LayerId,
        difference: bool,
        seed: f32,
        scale: f32,
        roughness: f32,
        octaves: u32,
    ) {
        let label = if difference {
            "Difference Clouds"
        } else {
            "Clouds"
        };
        self.begin_command_now(device, queue, id, label);
        let (Some(layer), Some(pong)) = (self.layers.get(&id), self.pong.as_ref()) else {
            return;
        };
        let kind = if difference { 26 } else { 25 };
        self.filter_pass_c(
            device,
            queue,
            &layer.view,
            &pong.view,
            kind,
            [seed, roughness],
            scale,
            octaves.max(1) as f32,
            [0.0; 2],
        );
        let mut enc = device.create_command_encoder(&Default::default());
        copy_tex(&mut enc, &pong.tex, &layer.tex, self.canvas_size);
        queue.submit([enc.finish()]);
    }

    /// Posterize the active layer (kind 28): quantize each colour channel to
    /// `levels` (2..=255) evenly spaced steps in display (sRGB) space, the classic
    /// destructive Image ▸ Adjustments ▸ Posterize. Alpha is preserved. Single
    /// pass; destructive + undoable (region-COW).
    pub fn apply_posterize(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: LayerId,
        levels: u32,
    ) {
        self.begin_command_now(device, queue, id, "Posterize");
        let (Some(layer), Some(pong)) = (self.layers.get(&id), self.pong.as_ref()) else {
            return;
        };
        self.filter_pass_c(
            device,
            queue,
            &layer.view,
            &pong.view,
            28,
            [0.0; 2],
            levels.clamp(2, 255) as f32,
            0.0,
            [0.0; 2],
        );
        let mut enc = device.create_command_encoder(&Default::default());
        copy_tex(&mut enc, &pong.tex, &layer.tex, self.canvas_size);
        queue.submit([enc.finish()]);
    }

    /// Threshold the active layer (kind 29): convert to pure black/white at a
    /// display-space Rec.709 luma cutoff `level` (0..1) — at/above → white, below
    /// → black — the destructive Image ▸ Adjustments ▸ Threshold. Alpha is
    /// preserved. Single pass; destructive + undoable (region-COW).
    pub fn apply_threshold(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: LayerId,
        level: f32,
    ) {
        self.begin_command_now(device, queue, id, "Threshold");
        let (Some(layer), Some(pong)) = (self.layers.get(&id), self.pong.as_ref()) else {
            return;
        };
        self.filter_pass_c(
            device,
            queue,
            &layer.view,
            &pong.view,
            29,
            [0.0; 2],
            level.clamp(0.0, 1.0),
            0.0,
            [0.0; 2],
        );
        let mut enc = device.create_command_encoder(&Default::default());
        copy_tex(&mut enc, &pong.tex, &layer.tex, self.canvas_size);
        queue.submit([enc.finish()]);
    }

    /// Tilt-Shift (Blur Gallery, kind 30) on the active layer: a graduated focus
    /// blur. The image stays sharp inside a horizontal focus band centred on
    /// `center_y` (uv 0..1) of `half_band` px half-width, and blurs progressively
    /// up to `max_radius` px once past `half_band + feather` (px). `angle_rad`
    /// tilts the band (0 = horizontal band, normal pointing down). Single pass;
    /// destructive + undoable (region-COW).
    #[allow(clippy::too_many_arguments)]
    pub fn apply_tilt_shift(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: LayerId,
        center_y: f32,
        half_band: f32,
        feather: f32,
        max_radius: f32,
        angle_rad: f32,
    ) {
        self.begin_command_now(device, queue, id, "Tilt-Shift");
        let (Some(layer), Some(pong)) = (self.layers.get(&id), self.pong.as_ref()) else {
            return;
        };
        // Band normal: angle 0 → (0, 1) (horizontal band). `center.x` carries the
        // feather (px), `center.y` the focus line (uv).
        let nrm = [-angle_rad.sin(), angle_rad.cos()];
        let center = [feather, center_y.clamp(0.0, 1.0)];
        self.filter_pass_c(
            device,
            queue,
            &layer.view,
            &pong.view,
            30,
            nrm,
            half_band.max(0.0),
            max_radius.max(0.0),
            center,
        );
        let mut enc = device.create_command_encoder(&Default::default());
        copy_tex(&mut enc, &pong.tex, &layer.tex, self.canvas_size);
        queue.submit([enc.finish()]);
    }

    /// Iris Blur (Blur Gallery, kind 31) on the active layer: the radial sibling
    /// of Tilt-Shift. The image stays sharp inside an elliptical region centred at
    /// `(cx, cy)` (pixel coords) with pixel radii `(rx, ry)`, and blurs
    /// progressively up to `max_radius` px outside it. `feather` is a normalized
    /// fraction of the ellipse radius (how far past the boundary the blur ramps to
    /// full). Single pass; destructive + undoable (region-COW).
    #[allow(clippy::too_many_arguments)]
    pub fn apply_iris_blur(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: LayerId,
        cx: f32,
        cy: f32,
        rx: f32,
        ry: f32,
        feather: f32,
        max_radius: f32,
    ) {
        self.begin_command_now(device, queue, id, "Iris Blur");
        let (Some(layer), Some(pong)) = (self.layers.get(&id), self.pong.as_ref()) else {
            return;
        };
        // `dir` carries the ellipse radii (px), `center` the center (uv),
        // `amount` the feather (normalized), `radius` the max blur (px).
        let center = [
            cx / self.canvas_size.width as f32,
            cy / self.canvas_size.height as f32,
        ];
        self.filter_pass_c(
            device,
            queue,
            &layer.view,
            &pong.view,
            31,
            [rx.max(0.0), ry.max(0.0)],
            feather.max(0.0),
            max_radius.max(0.0),
            center,
        );
        let mut enc = device.create_command_encoder(&Default::default());
        copy_tex(&mut enc, &pong.tex, &layer.tex, self.canvas_size);
        queue.submit([enc.finish()]);
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
            if l.adjust_kind == 14 {
                p.mix_r = l.mix_r;
                p.mix_g = l.mix_g;
                p.mix_b = l.mix_b;
            }
            if l.has_blend_if {
                p.has_blend_if = 1;
                p.blend_if = l.blend_if;
            }
            // Clip to the layer below (needs a layer beneath it in the stack).
            if l.clipped && i > 0 {
                p.has_clip = 1;
            }
            if l.has_stroke {
                p.has_stroke = 1;
                p.stroke_color = l.stroke_color;
                p.stroke_w = l.stroke_width;
            }
            if l.has_shadow {
                p.has_shadow = 1;
                p.shadow_color = l.shadow_color;
                p.shadow_off = l.shadow_offset;
                p.shadow_blur = l.shadow_blur;
            }
            if l.has_overlay {
                p.has_overlay = 1;
                p.overlay_color = l.overlay_color;
            }
            if l.has_grad_overlay {
                p.has_grad_overlay = 1;
                p.grad_color0 = l.grad_color0;
                p.grad_color1 = l.grad_color1;
                p.grad_angle = l.grad_angle;
                p.grad_opacity = l.grad_opacity;
            }
            if l.has_inner_shadow {
                p.has_inner_shadow = 1;
                p.inner_shadow_color = l.inner_shadow_color;
                p.inner_shadow_off = l.inner_shadow_offset;
                p.inner_shadow_blur = l.inner_shadow_blur;
            }
            if l.has_outer_glow {
                p.has_outer_glow = 1;
                p.outer_glow_color = l.outer_glow_color;
                p.outer_glow_size = l.outer_glow_size;
            }
            if l.has_inner_glow {
                p.has_inner_glow = 1;
                p.inner_glow_color = l.inner_glow_color;
                p.inner_glow_size = l.inner_glow_size;
            }
            if l.has_bevel {
                p.has_bevel = 1;
                p.bevel_highlight = l.bevel_highlight;
                p.bevel_shadow = l.bevel_shadow;
                p.bevel_size = l.bevel_size;
                p.bevel_soften = l.bevel_soften;
                // Light direction from azimuth (angle) + altitude. uv y runs
                // top-down, so flip y to put the light where the angle points.
                let (ca, sa) = (l.bevel_angle.cos(), l.bevel_angle.sin());
                let cz = l.bevel_altitude.cos();
                p.bevel_light = [cz * ca, -cz * sa, l.bevel_altitude.sin(), 0.0];
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
