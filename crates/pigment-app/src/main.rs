//! Pigment — open source Photoshop alternative. Phase 0 entry point.

// egui 0.34 deprecates several panel/menu aliases mid-cycle; the replacements
// are still settling. Tracked as a cleanup task — silence the churn for now.
#![allow(deprecated)]

mod app;
mod canvas;

use app::PigmentApp;

fn main() -> eframe::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_title("Pigment"),
        // wgpu backend; AutoVsync falls back gracefully (RESEARCH.md §1).
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };

    eframe::run_native(
        "Pigment",
        options,
        Box::new(|cc| Ok(Box::new(PigmentApp::new(cc)))),
    )
}
