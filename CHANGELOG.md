# Changelog

All notable changes to **Pigment** (the Prism suite's raster editor) are documented
here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project is pre-1.0, so versions are `0.0.x` milestones.

## [Unreleased]

### Added
- **Content-Aware Fill** (Phase 6 retouch). Brush a region; on release
  `prism_core::inpaint::content_aware_fill` synthesizes it from the surrounding
  texture via PatchMatch (approximate-NNF propagation + random search + patch
  voting, deterministic). Better than translate-and-blend for textured fills /
  larger removals. Unit-tested (uniform fill, content-awareness, determinism).
- **Spot Healing** (Phase 6 retouch). Brush over a blemish — **no manual source**;
  on release `prism_core::heal::spot_heal` auto-finds a clean nearby source by
  scoring boundary-ring match across candidate translations, then gradient-domain
  blends it in. Unit-tested (blemish removal, empty-mask no-op).
- **Healing Brush** (Phase 6 retouch). Alt-click sets a source; brush over the area
  to repair; on release a gradient-domain Poisson solve transplants the source's
  *texture* while matching the destination's tone/color at the region boundary —
  seamless repair, not a hard-edged copy. Solver lives in the shared core
  (`prism_core::heal::seamless_clone`, Gauss–Seidel membrane), unit-tested for
  tone-matching and texture transfer.

### Fixed
- Healing Brush: the Poisson guidance read source gradients at the region
  boundary from an unfilled (zero) source buffer, causing tone overshoot. The
  source is now built over the full image (offset-shifted, edge-clamped).

### Added
- **Clone Stamp tool** (Phase 6 retouch). Alt-click sets a source anchor; dragging
  stamps pixels copied from a frozen pre-stroke snapshot at a locked (aligned)
  offset, through a dedicated GPU clone-dab pass (`clone.wgsl`). Soft brush +
  opacity, selection-clipped, region-COW undo, on-canvas source crosshair.
  Pixel-verified by a headless-GPU test.
- **Curves adjustment** (completes Phase 4). Draggable monotone-cubic tone-curve
  editor with a composite (master) curve **plus** per-channel R/G/B; built to a
  256×1 LUT uploaded as a texture and sampled in the compositor (master then
  per-channel). Add/move/delete knots, pinned endpoints. Pixel-verified.

## [0.0.1] - 2026-06-06

First end-to-end raster editor on a GPU, linear-light, non-destructive engine.
Phases 0–5 of [PLAN.md](./PLAN.md), plus the suite's shared crates and first
cross-app interop.

### Added
- **Phase 0 — GPU canvas.** eframe + wgpu 29 shell; document textured-quad render
  with cursor-anchored pan/zoom, HiDPI; checkerboard transparency; open
  PNG/JPEG/etc; fit/100%.
- **Phase 1 — tiles, layers, paint.** Ping-pong compositor (Rgba16Float
  linear-premultiplied), blend modes; layers panel (add/delete/reorder/rename/
  visibility/opacity/blend/opacity); brush engine with arc-length dab walker,
  **wet-layer** stroke separation, velocity→size dynamics; eraser; bucket fill +
  eyedropper (sample-all-layers); undo/redo with a History panel; **region-COW
  undo**; frame-level dirty compositing; `.pigment` save/load (lz4 + JSON).
- **Phase 2 — selection & transform.** Marquee/ellipse/lasso/magic-wand with
  marching ants; feather/grow/shrink/invert + add/subtract/intersect; move +
  transform (translate/scale) with bake; crop, canvas/image resize (Lanczos),
  flip; copy/cut/paste + layer-from-selection; phosphor-icon dark-theme UI.
- **Phase 3 — adjustments, masks, filters.** Non-destructive adjustment layers
  (Brightness/Contrast, Levels, Hue/Saturation, Exposure, Invert, Threshold,
  Black&White); layer masks (paint reveal/hide); Gaussian blur / sharpen /
  pixelate; all 18 blend modes incl. HSL non-separable; histogram panel.
- **Phase 4 — text, vector, gradient.** Editable text layers (cosmic-text);
  rectangle/ellipse vector shape layers; linear gradient tool; generated layers
  stay editable.
- **Phase 5 — interchange.** PSD import (layers/opacity/blend/visibility);
  EXR/HDR open; export to PNG/JPEG/WebP/TIFF/BMP.
- **Suite interop.** Place a Contour `.contour` artboard as a rasterized layer;
  **live Dynamic-Link** — linked `.contour` layers re-render when the source
  file changes.
- **Shared engine.** Depend on suite-level `prism-core` / `prism-color` /
  `prism-io` crates (was app-local `pigment-core`/`-io`).

### Testing / CI
- Core unit tests (color/blend/tile/fill/raster/curve/histogram/shape), IO
  round-trips, and **headless-GPU pixel assertions** (compositor, wet brush,
  region undo, selection clip, transform bake, adjustment, layer mask).
- CI: `fmt --check` + `clippy -D warnings` + `test` on Linux/macOS/Windows.
