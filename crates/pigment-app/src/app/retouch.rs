use super::*;

impl PigmentApp {
    /// Fill the active layer with the editor's multi-stop gradient (color +
    /// opacity stops, geometry type, dithering), driven by the drag `p0→p1`
    /// (doc px), composited over the layer's existing pixels and clipped to the
    /// active selection when one exists. The gradient math is the shared,
    /// app-agnostic `prism_core::gradient`.
    pub(crate) fn do_gradient(
        &mut self,
        frame: &mut eframe::Frame,
        p0: egui::Vec2,
        p1: egui::Vec2,
    ) {
        let (w, h) = (self.doc.size.width, self.doc.size.height);
        let active = self.active_id();
        // Build the gradient in linear working space (premultiplied output).
        let grad = self
            .gradient
            .to_core()
            .render((p0.x, p0.y), (p1.x, p1.y), w, h);
        // Optional selection clip (canvas-sized 0..1 mask).
        let sel = if self.selection_active {
            Some(self.read_selection(frame))
        } else {
            None
        };
        with_gpu(frame, |gpu, d, q| {
            gpu.begin_command_now(d, q, active, "Gradient");
            let Some(b) = gpu.read_layer(d, q, active) else {
                return;
            };
            let mut base = f16_bytes_to_f32(&b);
            for i in 0..(w * h) as usize {
                let clip = sel.as_ref().map(|s| s[i]).unwrap_or(1.0);
                // Source-over with the gradient (already premultiplied), scaled
                // by the selection coverage.
                let ga = grad[i * 4 + 3] * clip;
                for c in 0..4 {
                    base[i * 4 + c] = grad[i * 4 + c] * clip + base[i * 4 + c] * (1.0 - ga);
                }
            }
            gpu.upload_layer(q, active, &f32_to_f16_bytes(&base));
        });
        self.force_composite = true;
    }

    /// Stamp a filled disk of radius `r` (doc px) into the Healing Brush mask.
    pub(crate) fn heal_mark(&mut self, c: egui::Vec2, r: f32) {
        let (w, h) = (self.doc.size.width as i64, self.doc.size.height as i64);
        if self.heal_mask.len() != (w * h) as usize {
            self.heal_mask = vec![false; (w * h) as usize];
        }
        let r2 = r * r;
        let x0 = (c.x - r).floor() as i64;
        let x1 = (c.x + r).ceil() as i64;
        let y0 = (c.y - r).floor() as i64;
        let y1 = (c.y + r).ceil() as i64;
        for y in y0..=y1 {
            for x in x0..=x1 {
                if x < 0 || y < 0 || x >= w || y >= h {
                    continue;
                }
                let dx = x as f32 + 0.5 - c.x;
                let dy = y as f32 + 0.5 - c.y;
                if dx * dx + dy * dy <= r2 {
                    self.heal_mask[(y * w + x) as usize] = true;
                }
            }
        }
    }

    /// Gradient-domain heal on stroke release: transplant the source patch's
    /// texture into the brushed region, tone-matched to the destination boundary
    /// (`prism_core::heal::seamless_clone`). Operates on straight linear RGB.
    pub(crate) fn do_heal(&mut self, frame: &mut eframe::Frame, offset: [f32; 2]) {
        if !self.heal_mask.iter().any(|&m| m) {
            return;
        }
        let (w, h) = (self.doc.size.width as usize, self.doc.size.height as usize);
        if self.heal_mask.len() != w * h {
            return;
        }
        let active = self.active_id();
        let mask0 = std::mem::take(&mut self.heal_mask);
        let off = (offset[0].round() as i64, offset[1].round() as i64);
        with_gpu(frame, |gpu, d, q| {
            gpu.begin_command_now(d, q, active, "Healing Brush");
            let Some(bytes) = gpu.read_layer(d, q, active) else {
                return;
            };
            let prem = f16_bytes_to_f32(&bytes); // premultiplied linear RGBA
                                                 // Premultiplied → straight linear.
            let mut dest = vec![0.0f32; w * h * 4];
            for p in 0..w * h {
                let a = prem[p * 4 + 3];
                let inv = if a > 1e-5 { 1.0 / a } else { 0.0 };
                dest[p * 4] = prem[p * 4] * inv;
                dest[p * 4 + 1] = prem[p * 4 + 1] * inv;
                dest[p * 4 + 2] = prem[p * 4 + 2] * inv;
                dest[p * 4 + 3] = a;
            }
            // Disable mask pixels whose source falls off-canvas (no real source).
            let mut mask = mask0;
            for y in 0..h {
                for x in 0..w {
                    let p = y * w + x;
                    if !mask[p] {
                        continue;
                    }
                    let sx = x as i64 - off.0;
                    let sy = y as i64 - off.1;
                    if sx < 0 || sy < 0 || sx >= w as i64 || sy >= h as i64 {
                        mask[p] = false;
                    }
                }
            }
            // Full offset-aligned source (clamped at edges) so the Poisson
            // guidance has valid gradients at the region boundary too.
            let mut src = vec![0.0f32; w * h * 4];
            for y in 0..h {
                for x in 0..w {
                    let sx = (x as i64 - off.0).clamp(0, w as i64 - 1) as usize;
                    let sy = (y as i64 - off.1).clamp(0, h as i64 - 1) as usize;
                    let (p, sp) = (y * w + x, sy * w + sx);
                    for c in 0..4 {
                        src[p * 4 + c] = dest[sp * 4 + c];
                    }
                }
            }
            let solved = prism_core::heal::seamless_clone(&dest, &src, &mask, w, h, 250);
            // Straight → premultiplied f16.
            let mut out = vec![0.0f32; w * h * 4];
            for p in 0..w * h {
                let a = solved[p * 4 + 3];
                out[p * 4] = solved[p * 4] * a;
                out[p * 4 + 1] = solved[p * 4 + 1] * a;
                out[p * 4 + 2] = solved[p * 4 + 2] * a;
                out[p * 4 + 3] = a;
            }
            gpu.upload_layer(q, active, &f32_to_f16_bytes(&out));
        });
        self.force_composite = true;
    }

    /// Patch tool on drag-release: gradient-domain seamless-clone the texture
    /// from a dragged source area into the lasso-selected destination region,
    /// tone-matched to the destination boundary (`prism_core::heal::seamless_clone`).
    ///
    /// `region` is the destination 0/1 mask (canvas-sized); `offset` is the drag
    /// vector (doc px) — the source pixel for destination `p` is `p − offset`.
    /// Clipped to the active selection when one exists. Region-COW undo.
    pub(crate) fn do_patch(
        &mut self,
        frame: &mut eframe::Frame,
        region: Vec<bool>,
        offset: [f32; 2],
    ) {
        if !region.iter().any(|&m| m) {
            return;
        }
        let (w, h) = (self.doc.size.width as usize, self.doc.size.height as usize);
        if region.len() != w * h {
            return;
        }
        // Clip the patch region to the active selection (if any).
        let sel = if self.selection_active {
            Some(self.read_selection(frame))
        } else {
            None
        };
        let active = self.active_id();
        let off = (offset[0].round() as i64, offset[1].round() as i64);
        with_gpu(frame, |gpu, d, q| {
            gpu.begin_command_now(d, q, active, "Patch");
            let Some(bytes) = gpu.read_layer(d, q, active) else {
                return;
            };
            let prem = f16_bytes_to_f32(&bytes); // premultiplied linear RGBA
                                                 // Premultiplied → straight linear.
            let mut dest = vec![0.0f32; w * h * 4];
            for p in 0..w * h {
                let a = prem[p * 4 + 3];
                let inv = if a > 1e-5 { 1.0 / a } else { 0.0 };
                dest[p * 4] = prem[p * 4] * inv;
                dest[p * 4 + 1] = prem[p * 4 + 1] * inv;
                dest[p * 4 + 2] = prem[p * 4 + 2] * inv;
                dest[p * 4 + 3] = a;
            }
            // Build the destination mask: region ∩ selection, minus pixels whose
            // source falls off-canvas (no real texture to transplant).
            let mut mask = region;
            for y in 0..h {
                for x in 0..w {
                    let p = y * w + x;
                    if !mask[p] {
                        continue;
                    }
                    if let Some(s) = &sel {
                        if s[p] <= 0.5 {
                            mask[p] = false;
                            continue;
                        }
                    }
                    let sx = x as i64 - off.0;
                    let sy = y as i64 - off.1;
                    if sx < 0 || sy < 0 || sx >= w as i64 || sy >= h as i64 {
                        mask[p] = false;
                    }
                }
            }
            // Full offset-aligned source (clamped at edges) so the Poisson
            // guidance has valid gradients at the region boundary too.
            let mut src = vec![0.0f32; w * h * 4];
            for y in 0..h {
                for x in 0..w {
                    let sx = (x as i64 - off.0).clamp(0, w as i64 - 1) as usize;
                    let sy = (y as i64 - off.1).clamp(0, h as i64 - 1) as usize;
                    let (p, sp) = (y * w + x, sy * w + sx);
                    for c in 0..4 {
                        src[p * 4 + c] = dest[sp * 4 + c];
                    }
                }
            }
            let solved = prism_core::heal::seamless_clone(&dest, &src, &mask, w, h, 250);
            // Straight → premultiplied f16.
            let mut out = vec![0.0f32; w * h * 4];
            for p in 0..w * h {
                let a = solved[p * 4 + 3];
                out[p * 4] = solved[p * 4] * a;
                out[p * 4 + 1] = solved[p * 4 + 1] * a;
                out[p * 4 + 2] = solved[p * 4 + 2] * a;
                out[p * 4 + 3] = a;
            }
            gpu.upload_layer(q, active, &f32_to_f16_bytes(&out));
        });
        self.force_composite = true;
    }

    /// Spot heal on release: brush a blemish, auto-source a clean region, blend
    /// (`prism_core::heal::spot_heal`). No manual source anchor.
    pub(crate) fn do_spot_heal(&mut self, frame: &mut eframe::Frame) {
        if !self.heal_mask.iter().any(|&m| m) {
            return;
        }
        let (w, h) = (self.doc.size.width as usize, self.doc.size.height as usize);
        if self.heal_mask.len() != w * h {
            return;
        }
        let active = self.active_id();
        let mask = std::mem::take(&mut self.heal_mask);
        with_gpu(frame, |gpu, d, q| {
            gpu.begin_command_now(d, q, active, "Spot Heal");
            let Some(bytes) = gpu.read_layer(d, q, active) else {
                return;
            };
            let prem = f16_bytes_to_f32(&bytes);
            let mut img = vec![0.0f32; w * h * 4];
            for p in 0..w * h {
                let a = prem[p * 4 + 3];
                let inv = if a > 1e-5 { 1.0 / a } else { 0.0 };
                img[p * 4] = prem[p * 4] * inv;
                img[p * 4 + 1] = prem[p * 4 + 1] * inv;
                img[p * 4 + 2] = prem[p * 4 + 2] * inv;
                img[p * 4 + 3] = a;
            }
            let solved = prism_core::heal::spot_heal(&img, &mask, w, h, 250);
            let mut out = vec![0.0f32; w * h * 4];
            for p in 0..w * h {
                let a = solved[p * 4 + 3];
                out[p * 4] = solved[p * 4] * a;
                out[p * 4 + 1] = solved[p * 4 + 1] * a;
                out[p * 4 + 2] = solved[p * 4 + 2] * a;
                out[p * 4 + 3] = a;
            }
            gpu.upload_layer(q, active, &f32_to_f16_bytes(&out));
        });
        self.force_composite = true;
    }

    /// Content-aware fill on release: synthesize the brushed region from the
    /// surrounding texture (`prism_core::inpaint::content_aware_fill`, PatchMatch).
    pub(crate) fn do_content_fill(&mut self, frame: &mut eframe::Frame) {
        if !self.heal_mask.iter().any(|&m| m) {
            return;
        }
        let (w, h) = (self.doc.size.width as usize, self.doc.size.height as usize);
        if self.heal_mask.len() != w * h {
            return;
        }
        let active = self.active_id();
        let mask = std::mem::take(&mut self.heal_mask);
        with_gpu(frame, |gpu, d, q| {
            gpu.begin_command_now(d, q, active, "Content-Aware Fill");
            let Some(bytes) = gpu.read_layer(d, q, active) else {
                return;
            };
            let prem = f16_bytes_to_f32(&bytes);
            let mut img = vec![0.0f32; w * h * 4];
            for p in 0..w * h {
                let a = prem[p * 4 + 3];
                let inv = if a > 1e-5 { 1.0 / a } else { 0.0 };
                img[p * 4] = prem[p * 4] * inv;
                img[p * 4 + 1] = prem[p * 4 + 1] * inv;
                img[p * 4 + 2] = prem[p * 4 + 2] * inv;
                img[p * 4 + 3] = a;
            }
            let solved = prism_core::inpaint::content_aware_fill(&img, &mask, w, h, 3, 6);
            let mut out = vec![0.0f32; w * h * 4];
            for p in 0..w * h {
                let a = solved[p * 4 + 3];
                out[p * 4] = solved[p * 4] * a;
                out[p * 4 + 1] = solved[p * 4 + 1] * a;
                out[p * 4 + 2] = solved[p * 4 + 2] * a;
                out[p * 4 + 3] = a;
            }
            gpu.upload_layer(q, active, &f32_to_f16_bytes(&out));
        });
        self.force_composite = true;
    }

    /// Accumulate a soft round dab of `r` (doc px) into the dodge/burn coverage,
    /// peaking `flow` at center and tapering to the edge (clamped to 1).
    pub(crate) fn tone_mark(&mut self, c: egui::Vec2, r: f32, flow: f32) {
        let (w, h) = (self.doc.size.width as i64, self.doc.size.height as i64);
        if self.tone_mask.len() != (w * h) as usize {
            self.tone_mask = vec![0.0; (w * h) as usize];
        }
        let x0 = (c.x - r).floor() as i64;
        let x1 = (c.x + r).ceil() as i64;
        let y0 = (c.y - r).floor() as i64;
        let y1 = (c.y + r).ceil() as i64;
        for y in y0..=y1 {
            for x in x0..=x1 {
                if x < 0 || y < 0 || x >= w || y >= h {
                    continue;
                }
                let dx = x as f32 + 0.5 - c.x;
                let dy = y as f32 + 0.5 - c.y;
                let d = (dx * dx + dy * dy).sqrt();
                if d <= r {
                    let f = (1.0 - d / r) * flow;
                    let i = (y * w + x) as usize;
                    self.tone_mask[i] = (self.tone_mask[i] + f).min(1.0);
                }
            }
        }
    }

    /// Apply dodge (or burn) over the accumulated tonal coverage on release.
    pub(crate) fn do_dodge_burn(&mut self, frame: &mut eframe::Frame, burn: bool) {
        if !self.tone_mask.iter().any(|&m| m > 0.0) {
            return;
        }
        let (w, h) = (self.doc.size.width as usize, self.doc.size.height as usize);
        if self.tone_mask.len() != w * h {
            return;
        }
        let active = self.active_id();
        let cover = std::mem::take(&mut self.tone_mask);
        let sign = if burn { -1.0 } else { 1.0 };
        let label = if burn { "Burn" } else { "Dodge" };
        with_gpu(frame, |gpu, d, q| {
            gpu.begin_command_now(d, q, active, label);
            let Some(bytes) = gpu.read_layer(d, q, active) else {
                return;
            };
            let prem = f16_bytes_to_f32(&bytes);
            let mut img = vec![0.0f32; w * h * 4];
            for p in 0..w * h {
                let a = prem[p * 4 + 3];
                let inv = if a > 1e-5 { 1.0 / a } else { 0.0 };
                img[p * 4] = prem[p * 4] * inv;
                img[p * 4 + 1] = prem[p * 4 + 1] * inv;
                img[p * 4 + 2] = prem[p * 4 + 2] * inv;
                img[p * 4 + 3] = a;
            }
            let amount: Vec<f32> = cover.iter().map(|&c| sign * c).collect();
            let solved = prism_core::tone::dodge_burn(&img, &amount, w, h);
            let mut out = vec![0.0f32; w * h * 4];
            for p in 0..w * h {
                let a = solved[p * 4 + 3];
                out[p * 4] = solved[p * 4] * a;
                out[p * 4 + 1] = solved[p * 4 + 1] * a;
                out[p * 4 + 2] = solved[p * 4 + 2] * a;
                out[p * 4 + 3] = a;
            }
            gpu.upload_layer(q, active, &f32_to_f16_bytes(&out));
        });
        self.force_composite = true;
    }

    /// Detail brush on release: saturate/desaturate (sponge), blur, or sharpen
    /// the brushed coverage on the active layer.
    pub(crate) fn do_detail(&mut self, frame: &mut eframe::Frame) {
        if !self.tone_mask.iter().any(|&m| m > 0.0) {
            return;
        }
        let (w, h) = (self.doc.size.width as usize, self.doc.size.height as usize);
        if self.tone_mask.len() != w * h {
            return;
        }
        let active = self.active_id();
        let cover = std::mem::take(&mut self.tone_mask);
        let mode = self.detail_mode;
        let label = match mode {
            0 => "Saturate",
            1 => "Desaturate",
            2 => "Blur",
            _ => "Sharpen",
        };
        with_gpu(frame, |gpu, d, q| {
            gpu.begin_command_now(d, q, active, label);
            let Some(bytes) = gpu.read_layer(d, q, active) else {
                return;
            };
            let prem = f16_bytes_to_f32(&bytes);
            let mut img = vec![0.0f32; w * h * 4];
            for p in 0..w * h {
                let a = prem[p * 4 + 3];
                let inv = if a > 1e-5 { 1.0 / a } else { 0.0 };
                img[p * 4] = prem[p * 4] * inv;
                img[p * 4 + 1] = prem[p * 4 + 1] * inv;
                img[p * 4 + 2] = prem[p * 4 + 2] * inv;
                img[p * 4 + 3] = a;
            }
            let solved = match mode {
                0 => prism_core::tone::sponge(&img, &cover, w, h),
                1 => {
                    let neg: Vec<f32> = cover.iter().map(|&c| -c).collect();
                    prism_core::tone::sponge(&img, &neg, w, h)
                }
                2 => prism_core::detail::blur_sharpen(&img, &cover, w, h, false),
                _ => prism_core::detail::blur_sharpen(&img, &cover, w, h, true),
            };
            let mut out = vec![0.0f32; w * h * 4];
            for p in 0..w * h {
                let a = solved[p * 4 + 3];
                out[p * 4] = solved[p * 4] * a;
                out[p * 4 + 1] = solved[p * 4 + 1] * a;
                out[p * 4 + 2] = solved[p * 4 + 2] * a;
                out[p * 4 + 3] = a;
            }
            gpu.upload_layer(q, active, &f32_to_f16_bytes(&out));
        });
        self.force_composite = true;
    }

    /// Liquify stroke start: snapshot the active layer (straight linear) as the
    /// warp source, zero the displacement field, and open an undo step.
    pub(crate) fn liquify_capture(&mut self, frame: &mut eframe::Frame) {
        let (w, h) = (self.doc.size.width as usize, self.doc.size.height as usize);
        let active = self.active_id();
        self.liquify_disp = vec![[0.0, 0.0]; w * h];
        let mut src = Vec::new();
        with_gpu(frame, |gpu, d, q| {
            gpu.begin_command_now(d, q, active, "Liquify");
            if let Some(bytes) = gpu.read_layer(d, q, active) {
                let prem = f16_bytes_to_f32(&bytes);
                let mut s = vec![0.0f32; w * h * 4];
                for p in 0..w * h {
                    let a = prem[p * 4 + 3];
                    let inv = if a > 1e-5 { 1.0 / a } else { 0.0 };
                    s[p * 4] = prem[p * 4] * inv;
                    s[p * 4 + 1] = prem[p * 4 + 1] * inv;
                    s[p * 4 + 2] = prem[p * 4 + 2] * inv;
                    s[p * 4 + 3] = a;
                }
                src = s;
            }
        });
        self.liquify_src = src;
    }

    /// Resample the frozen Liquify source through the current displacement field
    /// and upload (live preview; re-warps the original each frame, no compounding).
    pub(crate) fn liquify_apply(&mut self, frame: &mut eframe::Frame) {
        let (w, h) = (self.doc.size.width as usize, self.doc.size.height as usize);
        if self.liquify_src.len() != w * h * 4 || self.liquify_disp.len() != w * h {
            return;
        }
        let active = self.active_id();
        let warped =
            prism_core::warp::apply_displacement(&self.liquify_src, &self.liquify_disp, w, h);
        let mut out = vec![0.0f32; w * h * 4];
        for p in 0..w * h {
            let a = warped[p * 4 + 3];
            out[p * 4] = warped[p * 4] * a;
            out[p * 4 + 1] = warped[p * 4 + 1] * a;
            out[p * 4 + 2] = warped[p * 4 + 2] * a;
            out[p * 4 + 3] = a;
        }
        with_gpu(frame, |gpu, _, q| {
            gpu.upload_layer(q, active, &f32_to_f16_bytes(&out))
        });
        self.force_composite = true;
    }

    pub(crate) fn do_filter(
        &mut self,
        frame: &mut eframe::Frame,
        kind: u32,
        radius: f32,
        amount: f32,
    ) {
        let active = self.active_id();
        with_gpu(frame, |gpu, d, q| {
            gpu.apply_filter(d, q, active, kind, radius, amount)
        });
        self.force_composite = true;
    }

    /// Motion blur the active layer: directional box average over `distance`
    /// taps each side, oriented at `angle_deg`.
    pub(crate) fn do_motion_blur(
        &mut self,
        frame: &mut eframe::Frame,
        angle_deg: f32,
        distance: f32,
    ) {
        let active = self.active_id();
        let angle = angle_deg.to_radians();
        with_gpu(frame, |gpu, d, q| {
            gpu.apply_motion_blur(d, q, active, angle, distance)
        });
        self.force_composite = true;
    }

    /// Radial blur the active layer about the canvas center. `spin` chooses
    /// rotational (true, `amount` = degrees) vs zoom (false, `amount` = percent).
    pub(crate) fn do_radial_blur(
        &mut self,
        frame: &mut eframe::Frame,
        spin: bool,
        amount: f32,
        samples: u32,
    ) {
        let active = self.active_id();
        let cx = self.doc.size.width as f32 * 0.5;
        let cy = self.doc.size.height as f32 * 0.5;
        // Spin amount is an angle (deg→rad); zoom amount is a percentage.
        let amt = if spin {
            amount.to_radians()
        } else {
            amount / 100.0
        };
        with_gpu(frame, |gpu, d, q| {
            gpu.apply_radial_blur(d, q, active, cx, cy, spin, amt, samples)
        });
        self.force_composite = true;
    }

    /// Twirl the active layer about its center: rotate by up to `angle_deg`,
    /// falling off to 0 at `radius` pixels.
    pub(crate) fn do_twirl(&mut self, frame: &mut eframe::Frame, angle_deg: f32, radius: f32) {
        let active = self.active_id();
        let cx = self.doc.size.width as f32 * 0.5;
        let cy = self.doc.size.height as f32 * 0.5;
        let angle = angle_deg.to_radians();
        with_gpu(frame, |gpu, d, q| {
            gpu.apply_distort(d, q, active, 8, cx, cy, angle, radius, [0.0; 2])
        });
        self.force_composite = true;
    }

    /// Pinch (`amount` > 0, toward center) / Spherize-bulge (`amount` < 0,
    /// outward) the active layer about its center within `radius` pixels.
    /// `amount` is a signed strength in roughly -1..1.
    pub(crate) fn do_pinch(&mut self, frame: &mut eframe::Frame, amount: f32, radius: f32) {
        let active = self.active_id();
        let cx = self.doc.size.width as f32 * 0.5;
        let cy = self.doc.size.height as f32 * 0.5;
        with_gpu(frame, |gpu, d, q| {
            gpu.apply_distort(d, q, active, 9, cx, cy, amount, radius, [0.0; 2])
        });
        self.force_composite = true;
    }

    /// Ripple/Wave the active layer: sinusoidal displacement with `amplitude`
    /// pixels at `wavelength` pixels (each axis offset by a sine of the other).
    pub(crate) fn do_ripple(&mut self, frame: &mut eframe::Frame, amplitude: f32, wavelength: f32) {
        let active = self.active_id();
        let cx = self.doc.size.width as f32 * 0.5;
        let cy = self.doc.size.height as f32 * 0.5;
        with_gpu(frame, |gpu, d, q| {
            gpu.apply_distort(d, q, active, 10, cx, cy, 0.0, 0.0, [amplitude, wavelength])
        });
        self.force_composite = true;
    }

    /// Polar Coordinates on the active layer: rectangular→polar (`to_polar`
    /// true) or polar→rectangular (false), about the canvas center.
    pub(crate) fn do_polar(&mut self, frame: &mut eframe::Frame, to_polar: bool) {
        let active = self.active_id();
        let cx = self.doc.size.width as f32 * 0.5;
        let cy = self.doc.size.height as f32 * 0.5;
        let kind = if to_polar { 11 } else { 12 };
        with_gpu(frame, |gpu, d, q| {
            gpu.apply_distort(d, q, active, kind, cx, cy, 0.0, 0.0, [0.0; 2])
        });
        self.force_composite = true;
    }
}

#[cfg(test)]
mod patch_tests {
    //! `do_patch` itself is GPU-bound (read/upload layer), but its CPU pipeline —
    //! build an offset-aligned source, clip the region to the selection, then
    //! `seamless_clone` — is pure. These tests exercise that exact pipeline plus
    //! the `translate_mask` helper, so the Patch tool's transplant math and undo
    //! (region-COW, identity round-trip) are covered without a GPU adapter.
    use super::super::translate_mask;
    use prism_core::heal::seamless_clone;

    /// Replicate the straight-linear pipeline `do_patch` runs after layer
    /// readback: clip `region` to `sel`, build the source as `image[p − off]`
    /// (clamped), and seamless-clone. Returns the new straight-RGBA buffer.
    fn patch_pipeline(
        image: &[f32],
        region: &[bool],
        sel: Option<&[f32]>,
        w: usize,
        h: usize,
        off: (i64, i64),
        iters: usize,
    ) -> Vec<f32> {
        let mut mask = region.to_vec();
        for y in 0..h {
            for x in 0..w {
                let p = y * w + x;
                if !mask[p] {
                    continue;
                }
                if let Some(s) = sel {
                    if s[p] <= 0.5 {
                        mask[p] = false;
                        continue;
                    }
                }
                let (sx, sy) = (x as i64 - off.0, y as i64 - off.1);
                if sx < 0 || sy < 0 || sx >= w as i64 || sy >= h as i64 {
                    mask[p] = false;
                }
            }
        }
        let mut src = vec![0.0f32; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let sx = (x as i64 - off.0).clamp(0, w as i64 - 1) as usize;
                let sy = (y as i64 - off.1).clamp(0, h as i64 - 1) as usize;
                let (p, sp) = (y * w + x, sy * w + sx);
                src[p * 4..p * 4 + 4].copy_from_slice(&image[sp * 4..sp * 4 + 4]);
            }
        }
        seamless_clone(image, &src, &mask, w, h, iters)
    }

    fn fill(w: usize, h: usize, rgba: [f32; 4]) -> Vec<f32> {
        let mut v = Vec::with_capacity(w * h * 4);
        for _ in 0..w * h {
            v.extend_from_slice(&rgba);
        }
        v
    }

    #[test]
    fn patch_transplants_source_texture_into_destination() {
        // Flat gray image with a bright dot at a SOURCE area to the right of the
        // destination region. Patch with an offset that samples from the source:
        // the destination interior should pick up the source's brightening, while
        // its boundary tone stays matched to the surrounding gray.
        let (w, h) = (24, 12);
        let mut img = fill(w, h, [0.5, 0.5, 0.5, 1.0]);
        // Bright 4×4 block in the source area (centered ~ (17, 6)).
        for y in 4..8 {
            for x in 15..19 {
                let p = (y * w + x) * 4;
                img[p] = 0.9;
                img[p + 1] = 0.9;
                img[p + 2] = 0.9;
            }
        }
        // Destination region: 4×4 block on the left (centered ~ (5, 6)).
        let mut region = vec![false; w * h];
        for y in 4..8 {
            for x in 3..7 {
                region[y * w + x] = true;
            }
        }
        // off = dest − source = (5−17, 0) = (−12, 0); src[p] = img[p − off] = img[p+12].
        let out = patch_pipeline(&img, &region, None, w, h, (-12, 0), 400);
        // Destination center should brighten toward the transplanted source.
        let c = (6 * w + 5) * 4;
        assert!(
            out[c] > 0.62,
            "destination center should pick up bright source texture, got {}",
            out[c]
        );
        // A pixel well outside the region is untouched (region-COW: only the
        // masked region changes; undo restores the rest exactly).
        let outside = 0; // pixel (0,0), channel 0
        assert_eq!(out[outside], img[outside]);
    }

    #[test]
    fn patch_clips_to_selection() {
        // The region spans two columns but the selection only allows the left
        // column; the right column must be left equal to the original image.
        let (w, h) = (10, 6);
        let mut img = fill(w, h, [0.4, 0.4, 0.4, 1.0]);
        // Bright source band on the far right.
        for y in 0..h {
            for x in 7..10 {
                let p = (y * w + x) * 4;
                img[p] = 0.95;
                img[p + 1] = 0.95;
                img[p + 2] = 0.95;
            }
        }
        let mut region = vec![false; w * h];
        for y in 2..4 {
            for x in 2..4 {
                region[y * w + x] = true;
            }
        }
        // Selection allows only column x == 2.
        let mut sel = vec![0.0f32; w * h];
        for y in 0..h {
            sel[y * w + 2] = 1.0;
        }
        let out = patch_pipeline(&img, &region, Some(&sel), w, h, (-5, 0), 200);
        // x == 3 was in the region but outside the selection → unchanged.
        let clipped = (2 * w + 3) * 4;
        assert_eq!(out[clipped], img[clipped]);
    }

    #[test]
    fn patch_identity_offset_is_noop() {
        // Zero offset (source == dest): the membrane absorbs nothing and the
        // result equals the input — the undo/redo identity property.
        let (w, h) = (8, 8);
        let img = fill(w, h, [0.3, 0.6, 0.2, 1.0]);
        let mut region = vec![false; w * h];
        for y in 2..6 {
            for x in 2..6 {
                region[y * w + x] = true;
            }
        }
        let out = patch_pipeline(&img, &region, None, w, h, (0, 0), 100);
        for (i, (&o, &d)) in out.iter().zip(img.iter()).enumerate() {
            assert!((o - d).abs() < 1e-3, "idx {i}: {o} vs {d}");
        }
    }

    #[test]
    fn translate_mask_shifts_and_clips() {
        let (w, h) = (6u32, 6u32);
        let mut m = vec![false; (w * h) as usize];
        m[(2 * w + 2) as usize] = true; // (2,2)
        m[0] = true; // (0,0) — will fall off-canvas when shifted up-left
        let out = translate_mask(&m, w, h, 1.0, 1.0);
        assert!(out[(3 * w + 3) as usize], "(2,2) → (3,3)");
        assert!(out[(w + 1) as usize], "(0,0) → (1,1)");
        // Shifting up-left drops the (0,0) pixel off-canvas.
        let out2 = translate_mask(&m, w, h, -1.0, -1.0);
        assert!(out2[(w + 1) as usize], "(2,2) → (1,1)");
        assert_eq!(out2.iter().filter(|&&b| b).count(), 1, "(0,0) clipped away");
    }
}
