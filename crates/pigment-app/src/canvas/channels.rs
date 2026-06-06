//! Saved selections as named alpha channels (the Channels panel). A channel is a
//! stored copy of the selection mask (R16Float); loading copies it back into the
//! live selection. Phase 7 (PLAN.md §7 "Channels panel").

use super::*;

impl CanvasGpu {
    /// Save the current selection mask under `name`, replacing any existing one.
    pub fn save_selection_as_channel(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        name: String,
    ) {
        let Some(sel) = self.selection.as_ref() else {
            return;
        };
        let tex = make_target_fmt(device, self.canvas_size, "channel", SEL_FMT);
        let mut enc = device.create_command_encoder(&Default::default());
        copy_tex(&mut enc, &sel.tex, &tex.tex, self.canvas_size);
        queue.submit([enc.finish()]);
        self.channels.retain(|(n, _)| n != &name);
        self.channels.push((name, tex));
    }

    /// Load channel `name` into the live selection (overwrites it).
    pub fn load_channel(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, name: &str) {
        let (Some(src), Some(dst)) = (
            self.channels
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, t)| t),
            self.selection.as_ref(),
        ) else {
            return;
        };
        let mut enc = device.create_command_encoder(&Default::default());
        copy_tex(&mut enc, &src.tex, &dst.tex, self.canvas_size);
        queue.submit([enc.finish()]);
        self.has_selection = true;
        self.composite_valid = false;
    }

    pub fn delete_channel(&mut self, name: &str) {
        self.channels.retain(|(n, _)| n != name);
    }

    /// Names of the saved channels, in save order.
    pub fn channel_names(&self) -> Vec<String> {
        self.channels.iter().map(|(n, _)| n.clone()).collect()
    }
}
