use super::*;

/// Per-frame data the app hands to the canvas.
pub struct CanvasPaint {
    pub doc_rect: egui::Rect,
    pub checker_pts: f32,
    pub canvas_size: Size,
    pub layers: Vec<LayerDraw>,
    pub active_id: LayerId,
    pub dabs: Vec<Dab>,
    pub erase: bool,
    /// Set on the first frame of a stroke — copies the layer to the pre-stroke buffer.
    pub begin_command: bool,
    pub command_label: String,
    /// Set on the last frame of a stroke — snapshots `dirty_rect` for undo.
    pub commit_command: bool,
    pub dirty_rect: [u32; 4],
    pub undo: u32,
    pub redo: u32,
    // Wet brush stroke lifecycle.
    pub wet_begin: bool,
    pub wet_end: bool,
    pub wet_opacity: f32,
    pub paint_into_wet: bool,
    /// Route brush dabs to the active layer's mask instead of its pixels.
    pub paint_mask: bool,
    /// Clone Stamp: dabs copy from the frozen source at `clone_offset` instead
    /// of painting a flat color.
    pub clone: bool,
    /// destAnchor − sourceAnchor in document px (aligned clone).
    pub clone_offset: [f32; 2],
    /// Whether the document changed this frame (gates recompositing).
    pub dirty: bool,
    /// Selection operation to apply this frame, if any.
    pub selection_op: Option<SelectionOp>,
    /// Seconds, for marching-ants animation.
    pub time: f32,
    /// Live affine on the active layer (uv-space matrix, offset), if transforming.
    pub xform: Option<([f32; 4], [f32; 2])>,
    /// Bake the active transform into the layer this frame.
    pub bake: bool,
}

impl CallbackTrait for CanvasPaint {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen_descriptor: &ScreenDescriptor,
        encoder: &mut wgpu::CommandEncoder,
        resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let gpu: &mut CanvasGpu = resources.get_mut().unwrap();

        gpu.ensure_canvas(device, self.canvas_size);
        for l in &self.layers {
            gpu.ensure_layer(device, l.id);
        }
        match self.xform {
            Some((m, off)) => gpu.set_layer_transform(Some(self.active_id), m, off),
            None => gpu.set_layer_transform(None, [1.0, 0.0, 0.0, 1.0], [0.0; 2]),
        }

        for _ in 0..self.undo {
            gpu.undo(device, encoder);
        }
        for _ in 0..self.redo {
            gpu.redo(device, encoder);
        }
        if let Some(op) = &self.selection_op {
            gpu.apply_selection(device, queue, encoder, op);
        }
        if self.begin_command {
            gpu.begin_command(device, encoder, self.active_id, &self.command_label);
            if self.clone {
                gpu.snapshot_clone_source(device, encoder, self.active_id);
            }
        }
        if self.wet_begin {
            gpu.wet_begin(encoder, self.active_id, self.wet_opacity);
        }
        if self.clone {
            gpu.paint_clone_dabs(
                device,
                queue,
                encoder,
                self.active_id,
                &self.dabs,
                self.clone_offset,
            );
        } else {
            gpu.paint_dabs(
                device,
                queue,
                encoder,
                self.active_id,
                &self.dabs,
                self.erase,
                self.paint_into_wet,
                self.paint_mask,
            );
        }
        if self.wet_end {
            gpu.wet_end(device, queue, encoder);
        }
        if self.bake {
            gpu.bake_transform(device, queue, encoder);
        }
        if self.commit_command {
            gpu.commit_command(device, encoder, self.dirty_rect);
        }
        // Recomposite only when the document changed; pan/zoom alone reuse the
        // last composite (only the display pass re-runs each frame).
        let final_is_ping = if self.dirty || !gpu.composite_valid {
            let f = gpu.composite(device, queue, encoder, &self.layers);
            gpu.last_final_is_ping = f;
            gpu.composite_valid = true;
            f
        } else {
            gpu.last_final_is_ping
        };
        gpu.build_display_bind_group(device, final_is_ping);

        let [sw, sh] = screen_descriptor.size_in_pixels;
        let ppp = screen_descriptor.pixels_per_point;
        let to_clip = |p: egui::Pos2| -> [f32; 2] {
            [
                p.x * ppp / sw as f32 * 2.0 - 1.0,
                1.0 - p.y * ppp / sh as f32 * 2.0,
            ]
        };
        let uni = DisplayUniform {
            clip_min: to_clip(self.doc_rect.min),
            clip_max: to_clip(self.doc_rect.max),
            checker_px: self.checker_pts * ppp,
            has_selection: if gpu.has_selection() { 1.0 } else { 0.0 },
            time: self.time,
            canvas_w: self.canvas_size.width as f32,
            canvas_h: self.canvas_size.height as f32,
            _pad: [0.0; 3],
        };
        queue.write_buffer(&gpu.display_uniform, 0, bytemuck::bytes_of(&uni));
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &CallbackResources,
    ) {
        let gpu: &CanvasGpu = resources.get().unwrap();
        let Some(bg) = &gpu.display_bind_group else {
            return;
        };
        render_pass.set_pipeline(&gpu.display_pipeline);
        render_pass.set_bind_group(0, bg, &[]);
        render_pass.draw(0..6, 0..1);
    }
}
