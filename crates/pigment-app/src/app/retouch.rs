use super::*;

impl PigmentApp {
    /// Fill the active layer with a foreground→transparent linear gradient,
    /// composited over its existing pixels.
    pub(crate) fn do_gradient(
        &mut self,
        frame: &mut eframe::Frame,
        p0: egui::Vec2,
        p1: egui::Vec2,
    ) {
        let (w, h) = (self.doc.size.width, self.doc.size.height);
        let active = self.active_id();
        let c = self.brush_color;
        let c0 = [
            srgb_to_linear(c.r() as f32 / 255.0),
            srgb_to_linear(c.g() as f32 / 255.0),
            srgb_to_linear(c.b() as f32 / 255.0),
            1.0,
        ];
        let grad = shape::linear_gradient((p0.x, p0.y), (p1.x, p1.y), c0, [0.0; 4], w, h);
        with_gpu(frame, |gpu, d, q| {
            gpu.begin_command_now(d, q, active, "Gradient");
            let Some(b) = gpu.read_layer(d, q, active) else {
                return;
            };
            let mut base = f16_bytes_to_f32(&b);
            for i in 0..(w * h) as usize {
                let ga = grad[i * 4 + 3];
                for c in 0..4 {
                    base[i * 4 + c] = grad[i * 4 + c] + base[i * 4 + c] * (1.0 - ga);
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
}
