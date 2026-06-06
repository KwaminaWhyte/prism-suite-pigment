use super::*;

impl CanvasGpu {
    fn new_region_tex(&self, device: &wgpu::Device, w: u32, h: u32) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("undo.region"),
            size: wgpu::Extent3d {
                width: w.max(1),
                height: h.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FMT,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    }

    /// Clamp a requested `[x, y, w, h]` to the canvas bounds.
    fn clamp_rect(&self, rect: [u32; 4]) -> [u32; 4] {
        let (cw, ch) = (self.canvas_size.width, self.canvas_size.height);
        let x = rect[0].min(cw.saturating_sub(1));
        let y = rect[1].min(ch.saturating_sub(1));
        let w = rect[2].clamp(1, cw - x);
        let h = rect[3].clamp(1, ch - y);
        [x, y, w, h]
    }

    fn push_undo(&mut self, snap: Snapshot) {
        self.undo_stack.push(snap);
        if self.undo_stack.len() > UNDO_MAX {
            self.undo_stack.remove(0);
        }
        self.redo_stack.clear();
    }

    /// Begin a command: copy the layer's current pixels into the transient
    /// pre-stroke buffer. The undo entry (just the dirty region) is taken at
    /// [`commit_command`].
    pub fn begin_command(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        id: LayerId,
        label: &str,
    ) {
        if !self.layers.contains_key(&id) {
            return;
        }
        let (w, h) = (self.canvas_size.width, self.canvas_size.height);
        if self.stroke_pre.is_none() {
            self.stroke_pre = Some(self.new_region_tex(device, w, h));
        }
        let pre = self.stroke_pre.as_ref().unwrap();
        let layer = self.layers.get(&id).unwrap();
        copy_tex(encoder, &layer.tex, pre, self.canvas_size);
        self.stroke_owner = Some(id);
        self.stroke_label = label.to_string();
    }

    /// Commit the open command, snapshotting only `rect` (the touched region)
    /// from the pre-stroke buffer onto the undo stack.
    pub fn commit_command(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        rect: [u32; 4],
    ) {
        let Some(id) = self.stroke_owner.take() else {
            return;
        };
        let Some(pre) = self.stroke_pre.as_ref() else {
            return;
        };
        let [x, y, w, h] = self.clamp_rect(rect);
        let region = self.new_region_tex(device, w, h);
        copy_region(encoder, pre, [x, y], &region, [0, 0], [w, h]);
        let label = std::mem::take(&mut self.stroke_label);
        self.push_undo(Snapshot {
            id,
            tex: region,
            rect: [x, y, w, h],
            label,
        });
    }

    /// Snapshot a whole-layer command immediately (own encoder), for callers
    /// outside the frame callback such as bucket fill.
    pub fn begin_command_now(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: LayerId,
        label: &str,
    ) {
        let Some(layer) = self.layers.get(&id) else {
            return;
        };
        let (w, h) = (self.canvas_size.width, self.canvas_size.height);
        let region = self.new_region_tex(device, w, h);
        let mut enc = device.create_command_encoder(&Default::default());
        copy_region(&mut enc, &layer.tex, [0, 0], &region, [0, 0], [w, h]);
        queue.submit([enc.finish()]);
        self.push_undo(Snapshot {
            id,
            tex: region,
            rect: [0, 0, w, h],
            label: label.to_string(),
        });
    }

    /// Labels of pending undo steps (oldest→newest) and redo steps (next→furthest).
    pub fn history_labels(&self) -> (Vec<String>, Vec<String>) {
        let undo = self.undo_stack.iter().map(|s| s.label.clone()).collect();
        let redo = self
            .redo_stack
            .iter()
            .rev()
            .map(|s| s.label.clone())
            .collect();
        (undo, redo)
    }

    fn restore(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        from_undo: bool,
    ) {
        let snap = if from_undo {
            self.undo_stack.pop()
        } else {
            self.redo_stack.pop()
        };
        let Some(snap) = snap else { return };
        let Some(layer) = self.layers.get(&snap.id) else {
            return;
        };
        let [x, y, w, h] = snap.rect;
        // Save the layer's current region to the opposite stack, then restore.
        let cur = self.new_region_tex(device, w, h);
        copy_region(encoder, &layer.tex, [x, y], &cur, [0, 0], [w, h]);
        copy_region(encoder, &snap.tex, [0, 0], &layer.tex, [x, y], [w, h]);
        let saved = Snapshot {
            id: snap.id,
            tex: cur,
            rect: snap.rect,
            label: snap.label,
        };
        if from_undo {
            self.redo_stack.push(saved);
        } else {
            self.undo_stack.push(saved);
        }
    }

    pub fn undo(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        self.restore(device, encoder, true);
    }

    pub fn redo(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        self.restore(device, encoder, false);
    }
}
