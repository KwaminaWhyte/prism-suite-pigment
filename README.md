<h1 align="center">Pigment</h1>

<p align="center">
  <b>An open source, GPU-accelerated, non-destructive raster image editor.</b><br>
  A professional-grade alternative to Adobe Photoshop — built in Rust.
</p>

---

Pigment is app #1 of the planned **[Prism Suite](./SUITE.md)** — four
interoperating creative apps (raster, vector, video, motion) that work together
the way Adobe's Creative Cloud does. It is early and under active construction.

## Status

**Phase 0 complete** — the app builds, launches, and renders a document on the
GPU with pan/zoom and image loading. See [PLAN.md](./PLAN.md) for the full
roadmap and [ARCHITECTURE.md](./ARCHITECTURE.md) for current internals.

| Milestone | State |
|-----------|-------|
| Phase 0 — GPU canvas, pan/zoom, open image | ✅ done |
| Phase 1 — layers, blend modes, brush/eraser/fill, wet-layer, undo+history, save | ✅ done |
| Phase 2 — selection & transform | 🚧 next |
| Phase 3 — adjustments, masks, filters | ⏳ planned |
| Phase 4 — text, vector, smart objects | ⏳ planned |
| Phase 5 — color mgmt, PSD, AI, plugins | ⏳ planned |

## Design principles

- **GPU-resident** — wgpu (Vulkan/Metal/DX12/WebGPU) compositor from day one.
- **Non-destructive** — layer tree / node graph; pixels re-derived and cached.
- **Linear-light, premultiplied** — correct color math everywhere; ICC/OCIO.
- **Sparse tiles** — paint huge documents; only touched tiles allocate.
- **Polish over feature count** — the engine is free; the product is the feel.

## Tech stack

`wgpu` 29 · `eframe`/`egui` 0.34 · `winit` 0.30 · `image` 0.25 · `lcms2` ·
`cosmic-text` · `kurbo`/`lyon` · `ort` (ONNX) · `fast_image_resize`.
Full matrix + rationale in [RESEARCH.md](./RESEARCH.md).

## Workspace layout

```
crates/
  pigment-core/   document model: layers, tiles, blend modes, color
  pigment-io/     image load/save (PNG/JPEG/WebP/TIFF/…)
  pigment-app/    eframe desktop app + wgpu canvas
assets/shaders/   WGSL
```

## Build & run

Requires a recent Rust toolchain (1.88+).

```bash
cargo run -p pigment-app        # launch the editor
cargo build                     # build everything
cargo test                      # run tests
```

File → Open loads an image; drag to pan, scroll to zoom, View → Fit to screen.

## Roadmap

- [PLAN.md](./PLAN.md) — phased, actionable task backlog for Pigment.
- [SUITE.md](./SUITE.md) — the four-app Prism Suite vision and interop plan.
- [RESEARCH.md](./RESEARCH.md) — cited research backing the technical choices.

## License

Dual-licensed under MIT or Apache-2.0.
