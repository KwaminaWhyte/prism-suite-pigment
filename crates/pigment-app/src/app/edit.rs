use super::*;

impl PigmentApp {
    /// Read every layer to CPU f32, map each through `map`, recreate the canvas
    /// at `new_size`, and re-upload. Used by resize/canvas/crop.
    fn rebuild_document(
        &mut self,
        frame: &mut eframe::Frame,
        new_size: Size,
        map: impl Fn(&[f32]) -> Vec<f32>,
    ) {
        let ids: Vec<LayerId> = self.doc.layers.layers.iter().map(|l| l.id).collect();
        let layers_px = with_gpu(frame, |gpu, d, q| {
            ids.iter()
                .filter_map(|id| {
                    gpu.read_layer(d, q, *id)
                        .map(|b| (*id, f16_bytes_to_f32(&b)))
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
        let mapped: Vec<(LayerId, Vec<u8>)> = layers_px
            .iter()
            .map(|(id, px)| (*id, f32_to_f16_bytes(&map(px))))
            .collect();
        self.doc.size = new_size;
        with_gpu(frame, |gpu, d, q| {
            gpu.ensure_canvas(d, new_size);
            for (id, bytes) in &mapped {
                gpu.ensure_layer(d, *id);
                gpu.upload_layer(q, *id, bytes);
            }
        });
        self.selection_active = false;
        self.force_composite = true;
        self.needs_fit = true;
    }

    pub(crate) fn resize_image(&mut self, frame: &mut eframe::Frame, nw: u32, nh: u32) {
        let old = self.doc.size;
        let q = if nw < old.width {
            Quality::Lanczos3
        } else {
            Quality::Bicubic
        };
        self.rebuild_document(frame, Size::new(nw, nh), move |px| {
            resize_rgba_f32(px, old.width, old.height, nw, nh, q)
        });
    }

    pub(crate) fn resize_canvas(&mut self, frame: &mut eframe::Frame, nw: u32, nh: u32) {
        let old = self.doc.size;
        self.rebuild_document(frame, Size::new(nw, nh), move |px| {
            reposition(px, old.width, old.height, nw, nh, 0, 0)
        });
    }

    pub(crate) fn crop_to_selection(&mut self, frame: &mut eframe::Frame) {
        if !self.selection_active {
            return;
        }
        let (w, h) = (self.doc.size.width, self.doc.size.height);
        let sel = self.read_selection(frame);
        let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0u32, 0u32);
        for y in 0..h {
            for x in 0..w {
                if sel[(y * w + x) as usize] > 0.5 {
                    x0 = x0.min(x);
                    y0 = y0.min(y);
                    x1 = x1.max(x + 1);
                    y1 = y1.max(y + 1);
                }
            }
        }
        if x1 <= x0 || y1 <= y0 {
            return;
        }
        let (rw, rh, ox, oy) = (x1 - x0, y1 - y0, x0 as i32, y0 as i32);
        self.rebuild_document(frame, Size::new(rw, rh), move |px| {
            reposition(px, w, h, rw, rh, ox, oy)
        });
    }

    pub(crate) fn flip_active(&mut self, frame: &mut eframe::Frame, horizontal: bool) {
        let active = self.active_id();
        let (w, h) = (self.doc.size.width, self.doc.size.height);
        with_gpu(frame, |gpu, d, q| {
            gpu.begin_command_now(d, q, active, "Flip");
            if let Some(b) = gpu.read_layer(d, q, active) {
                let f = flip(&f16_bytes_to_f32(&b), w, h, horizontal);
                gpu.upload_layer(q, active, &f32_to_f16_bytes(&f));
            }
        });
        self.force_composite = true;
    }

    /// Copy the active layer (masked by the selection) into the clipboard.
    pub(crate) fn copy_selection(&mut self, frame: &mut eframe::Frame) {
        let active = self.active_id();
        let n = (self.doc.size.width * self.doc.size.height) as usize;
        let has_sel = self.selection_active;
        let res = with_gpu(frame, |gpu, d, q| {
            let px = gpu.read_layer(d, q, active)?;
            let sel = if has_sel {
                gpu.read_selection(d, q)
            } else {
                None
            };
            Some((f16_bytes_to_f32(&px), sel))
        })
        .flatten();
        let Some((mut px, sel)) = res else { return };
        if let Some(s) = sel {
            for i in 0..n {
                let a = s[i];
                for c in 0..4 {
                    px[i * 4 + c] *= a;
                }
            }
        }
        self.clipboard = Some(px);
        self.clipboard_size = self.doc.size;
    }

    pub(crate) fn cut_selection(&mut self, frame: &mut eframe::Frame) {
        self.copy_selection(frame);
        if !self.selection_active {
            return;
        }
        let active = self.active_id();
        let n = (self.doc.size.width * self.doc.size.height) as usize;
        with_gpu(frame, |gpu, d, q| {
            gpu.begin_command_now(d, q, active, "Cut");
            let (Some(b), Some(sel)) = (gpu.read_layer(d, q, active), gpu.read_selection(d, q))
            else {
                return;
            };
            let mut px = f16_bytes_to_f32(&b);
            for i in 0..n {
                if sel[i] > 0.5 {
                    for c in 0..4 {
                        px[i * 4 + c] = 0.0;
                    }
                }
            }
            gpu.upload_layer(q, active, &f32_to_f16_bytes(&px));
        });
        self.force_composite = true;
    }

    pub(crate) fn paste(&mut self, frame: &mut eframe::Frame) {
        let Some(mut cb) = self.clipboard.clone() else {
            return;
        };
        if self.clipboard_size != self.doc.size {
            cb = reposition(
                &cb,
                self.clipboard_size.width,
                self.clipboard_size.height,
                self.doc.size.width,
                self.doc.size.height,
                0,
                0,
            );
        }
        let id = self.doc.layers.add_raster("Pasted");
        self.doc.active_layer = Some(id);
        with_gpu(frame, |gpu, d, q| {
            gpu.ensure_layer(d, id);
            gpu.upload_layer(q, id, &f32_to_f16_bytes(&cb));
        });
        self.force_composite = true;
    }

    pub(crate) fn layer_from_selection(&mut self, frame: &mut eframe::Frame) {
        self.copy_selection(frame);
        self.paste(frame);
    }

    /// Read the current GPU selection mask (zeros if none active).
    pub(crate) fn read_selection(&mut self, frame: &mut eframe::Frame) -> Vec<f32> {
        let n = (self.doc.size.width * self.doc.size.height) as usize;
        if !self.selection_active {
            return vec![0.0; n];
        }
        with_gpu(frame, |gpu, device, queue| {
            gpu.read_selection(device, queue)
        })
        .flatten()
        .unwrap_or_else(|| vec![0.0; n])
    }

    /// Upload a CPU selection mask and mark a selection active.
    fn set_selection(&mut self, frame: &mut eframe::Frame, mask: &[f32]) {
        with_gpu(frame, |gpu, _, queue| gpu.upload_selection(queue, mask));
        self.selection_active = true;
    }

    /// Read → transform → upload the selection (feather / grow / shrink).
    pub(crate) fn map_selection(
        &mut self,
        frame: &mut eframe::Frame,
        f: impl Fn(&[f32], u32, u32) -> Vec<f32>,
    ) {
        if !self.selection_active {
            return;
        }
        let (w, h) = (self.doc.size.width, self.doc.size.height);
        let cur = self.read_selection(frame);
        let next = f(&cur, w, h);
        self.set_selection(frame, &next);
    }

    /// Combine a freshly-computed mask with the op's base per the active mode.
    pub(crate) fn commit_selection(&mut self, frame: &mut eframe::Frame, new_mask: Vec<f32>) {
        let combined = raster::combine(&self.sel_base, &new_mask, self.sel_mode);
        self.set_selection(frame, &combined);
    }

    pub(crate) fn add_mask(&mut self, frame: &mut eframe::Frame, from_selection: bool) {
        let active = self.active_id();
        let values = if from_selection && self.selection_active {
            Some(self.read_selection(frame))
        } else {
            None
        };
        with_gpu(frame, |gpu, d, q| {
            gpu.set_mask(d, q, active, values.as_deref())
        });
        self.masked_layers.insert(active);
        self.force_composite = true;
    }

    pub(crate) fn delete_mask(&mut self, frame: &mut eframe::Frame) {
        let active = self.active_id();
        with_gpu(frame, |gpu, _, _| gpu.delete_mask(active));
        self.masked_layers.remove(&active);
        self.edit_mask = false;
        self.force_composite = true;
    }
}
