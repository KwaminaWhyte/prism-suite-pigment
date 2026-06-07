# Changelog

All notable changes to **Pigment** (the Prism suite's raster editor) are documented
here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project is pre-1.0, so versions are `0.0.x` milestones.

## [Unreleased]

### Added
- **Layer styles persist to `.pigment`** (Phase 7). Closes a known data-loss
  gap: the 8 non-destructive layer styles (Stroke, Drop Shadow, Color Overlay,
  Inner Shadow, Outer Glow, Inner Glow, Gradient Overlay, Bevel & Emboss) were
  runtime-only — saving and reopening a document silently dropped them. They now
  round-trip. The `.pigment` doc model (`prism-io::document_file`) gains an
  optional, per-layer `styles` payload (`Option<LayerStyles>` with one optional
  struct per style, units documented: colors as straight RGBA/RGB, pixel
  offsets/sizes/blur in document px, angles in degrees), serialized with serde
  `default` + `skip_serializing_if` so **old documents (no `styles` key) still
  load** and **new documents with no styles stay byte-compact** (no empty keys).
  On save Pigment maps each layer's runtime style HashMaps into `LayerMeta.styles`;
  on open it re-installs them under the freshly-allocated layer ids and forces a
  recomposite so restored styles render immediately. Pure mapping functions
  (`runtime_styles_to_meta` / `meta_styles_to_runtime`) are unit-tested for a
  full 8-style lossless round-trip (GPU upload untested per convention); prism-io
  adds tests for a full-payload serde round-trip and old-doc back-compat.
- **Pen tool + work paths, path → selection, vector mask** (Phase 4 completion).
  A cubic-Bézier **pen**: click to drop corner anchors, click-drag to pull
  symmetric Bézier handles, and click the first anchor (within 8 screen px,
  ≥ 3 anchors) to **close** the path. A companion **Direct Select** tool grabs
  on-curve points (moves the whole anchor) or individual handles to reshape the
  curve after the fact. The work path renders as a vector **overlay** via the
  egui painter (flattened curve + anchor dots + handle lines/rings) — it is *not*
  part of the GPU composite. From the tool-options bar: **Path → selection**
  flattens the closed interior and fills it into the selection mask (reusing the
  selection pipeline, replace mode), and **Apply as vector mask** rasterizes the
  same interior into the active layer's mask via the existing layer-mask pipeline
  (`set_mask`). Bézier evaluation, polyline flattening, even-odd point-in-polygon
  and interior fill all live in-app (`path.rs`, no new shared-crate dep); nine
  unit tests cover curve evaluation, flatten-on-curve, smooth-handle mirroring,
  point-in-polygon, and exact fill-mask coverage on a known square (= the
  path→selection core, GPU-free). *Deferred:* shape layers from paths, boolean
  path ops, stroking a path with a brush, and persisting paths to the `.pigment`
  doc (doc-format change out of scope this pass).
- **Layer style: Bevel & Emboss (Inner Bevel)** (Phase 7). The last and hardest
  of the common Photoshop layer styles, evaluated live in the compositor with no
  separate height pass. A screen-space surface normal is derived from a central
  difference of the layer's alpha (height) field; a directional light (azimuth
  *angle* + *altitude*) shades it, painting a **highlight** where the surface
  faces the light and a **shadow** where it faces away, concentrated within *size*
  pixels of the edge (with optional *soften*). Per-layer highlight/shadow color +
  opacity, size, soften, angle, altitude controls; one GPU pixel test (highlight
  brighter on the light-facing edge, shadow darker on the opposite edge, nothing
  outside the shape). This rounds out the common PS layer-style set — only
  **Satin** and **Pattern Overlay** remain. (`CompositeParams` is now 352 bytes,
  still within the 512-byte `PARAMS_STRIDE` slot.)
- **Layer styles: Inner Shadow, Outer Glow, Inner Glow, Gradient Overlay**
  (Phase 7). Four more non-destructive layer FX evaluated live in the compositor,
  reusing the Drop-Shadow / Stroke alpha-neighborhood machinery. **Inner Shadow**
  casts a blurred, offset copy of the layer's *inverse* alpha clipped to its own
  coverage (dark band inside the edge). **Outer Glow** halos a centered, soft
  colored copy of the alpha outward; **Inner Glow** tints inward from the edge.
  **Gradient Overlay** recolors the layer's fill with an angled two-color linear
  gradient at adjustable opacity. Per-layer controls (color / offset / blur / size
  / angle / opacity as each needs); four GPU pixel tests. With Stroke, Drop
  Shadow, and Color Overlay this brings the common PS layer-style set near
  completion (Bevel & Emboss landed next). (`CompositeParams` outgrew the
  256-byte uniform slot at 288 bytes, so `PARAMS_STRIDE` is now 512, guarded by a
  compile-time size/alignment assert.)
- **Patch tool** (Phase 6 retouch). Lasso/freehand-select a region and drag it to
  a source area; on release a gradient-domain Poisson solve
  (`prism_core::heal::seamless_clone`) transplants the source's *texture* into the
  destination while tone-matching the region boundary — seamless, not a hard copy.
  PS-style **Source / Destination** mode toggle in the tool-options bar;
  selection-clipped, region-COW undo. Four unit tests (texture transplant,
  selection clipping, identity-offset no-op, mask translate/clip).
- **Layer style: Color Overlay** (Phase 7). Recolors a layer's covered pixels
  toward a chosen color by strength, evaluated live in the compositor. Per-layer
  color picker; GPU pixel-tested. Completes the Stroke/Drop-Shadow/Overlay trio.
- **Gradient Map adjustment** (Phase 7). Non-destructive — maps the backdrop's
  luminance through a two-color gradient (shadows→highlights) built into the
  per-layer LUT texture and sampled in the compositor (shader kind 12). Two color
  pickers; GPU pixel-tested.
- **Layer style: Drop Shadow** (Phase 7). Non-destructive — a blurred, offset,
  tinted copy of the layer's alpha drawn behind it, evaluated live in the
  compositor (16-tap disk blur). Per-layer color (premultiplied) + offset + blur;
  GPU pixel-tested. Reuses the stroke FX alpha-neighborhood machinery.
- **Layer style: Stroke** (Phase 7). Non-destructive outer stroke — an alpha-edge
  ring sampled live in the compositor shader, tinted and drawn behind the layer.
  Per-layer color + width sliders; GPU pixel-tested. (First layer FX; the
  alpha-neighborhood machinery generalizes to drop shadow / glow.)
- **Channels panel** (Phase 7). Save the current selection as a named alpha channel,
  load a channel back into the selection, or delete it (Channels panel section).
  GPU round-trip-tested.
- **Clipping masks** (Phase 7). A layer can clip to the layer directly below — its
  alpha gates where the layer shows, evaluated in the compositor via a clip-base
  texture binding. Per-layer "Clip to layer below" toggle; GPU pixel-tested.
- **Blend-If** (Phase 7). Per-layer "this layer" + "underlying" luma-range sliders
  (soft-feathered) that gate where the layer shows, evaluated in the compositor
  shader against the source and backdrop luma. GPU pixel-tested.
- **Adjustment expansion** (Phase 7). Three new non-destructive adjustment layers —
  **Vibrance** (saturation weighted to low-sat pixels), **Photo Filter**
  (luminosity-preserving warm/cool tint), **Posterize** (level quantize) — as
  compositor shader kinds 9/10/11, with sliders + a GPU pixel test.
- **Detail brush** (Phase 6). One brush, four modes — **Saturate / Desaturate**
  (sponge, `prism_core::tone::sponge`) and **Blur / Sharpen**
  (`prism_core::detail::blur_sharpen`) — applied over a soft brushed coverage mask
  on release. Unit-tested core math.
- **Liquify** (Phase 6). Mesh warp via a per-pixel displacement field with
  **Push / Twirl / Pucker / Bloat** modes (panel selector). Live preview
  re-warps a frozen snapshot each frame (no compounding blur);
  `prism_core::warp` provides the bilinear resample + brush stamps (unit-tested).
- **Dodge & Burn** (Phase 6 retouch). Brush to lighten (dodge), or hold Alt to
  darken (burn); a soft coverage mask accumulates over the stroke and is applied
  in linear light on release (`prism_core::tone::dodge_burn`, unit-tested).
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
