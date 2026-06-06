# Pigment — Open Source Photoshop Alternative

Professional-grade, non-destructive, GPU-accelerated raster image editor in Rust.
Goal: match ~80% of Photoshop core in staged milestones with a modern GPU pipeline and clean UX.

> Companion docs: [RESEARCH.md](./RESEARCH.md) (cited findings + crate matrix), [ARCHITECTURE.md](./ARCHITECTURE.md) (module/data-flow detail as it lands).

---

## 0. Why this can work

- Algorithms are solved and free (blend math, resampling, color science, segmentation).
- Mature Rust crates cover GPU, image IO, text, vector, color, AI.
- The real moat is **polish, performance, and a unified non-destructive engine** — that's where we focus.

**Non-negotiable principle:** non-destructive, GPU-resident, **linear-light premultiplied** pixel pipeline from day one. Retrofitting GPU/linear later = rewrite.

---

## 1. Validated Tech Stack (June 2026 versions — verify at build with `cargo add`)

| Concern | Crate | Version | Notes |
|---|---|---|---|
| GPU | `wgpu` | 29 | Vulkan/Metal/DX12/WebGPU. Pin major. |
| Windowing | `winit` | 0.30 | `ApplicationHandler` pattern; unified Pointer events (pen pressure/tilt). |
| App shell + UI | `eframe` / `egui` / `egui-wgpu` / `egui-winit` | 0.34 | Lockstep versions. wgpu backend default. Shares GPU device with our canvas. |
| Pen/tablet axes | `octotablet` | 0.x | Pressure/tilt cross-platform; Wintab fallback (`wintab_lite`) for pro Wacom on Win. |
| Image IO | `image` | 0.25 | PNG/JPEG/WebP/TIFF/etc, 16-bit. Pixel IO only — not the doc model. |
| HDR / scene-linear | `exr` | 1.74 | Pure-safe-Rust OpenEXR (f16/f32). |
| Color mgmt | `lcms2` | 6.1 | Little CMS 2.17; ICC v2/v4, CMYK, soft-proof. `qcms` for wasm/RGB path. |
| Resize | `fast_image_resize` | 6.0 | SIMD Lanczos3/Bicubic, premultiplied-correct. |
| Text | `cosmic-text` + `glyphon` | 0.18 / latest | Editable buffers (HarfRust+swash); glyphon atlases to wgpu. |
| Vector math | `kurbo` | 0.11 | Bezier/path math, hit-test. |
| Vector tessellation | `lyon` | 1.0 | Path → GPU triangle mesh. |
| Boolean path ops | `i_overlay` | 1.x | Union/intersect/diff (kurbo native booleans still WIP). |
| PSD import | `psd` | 0.3.5 | Read-only; we build blend/mask compositing on top. No PSD write (custom serializer or `.ora`). |
| AI runtime | `ort` | 2.0-rc | ONNX Runtime; `candle` for pure-Rust/wasm. |
| Undo | `undo`/custom | — | Command stack + tile-COW for pixels. |
| Encode extras | `mozjpeg`, `webp`, `ravif`, `oxipng` | — | Better-than-baseline JPEG/WebP/AVIF/PNG. |
| Serde/util | `serde`, `glam`, `bytemuck`, `thiserror`, `anyhow`, `rayon`, `arc-swap`, `lz4_flex` | — | Doc serialization, math, GPU casts, tile compression. |

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────┐
│  pigment-app  (eframe + egui)                            │
│  panels: tools · layers · history · color · properties   │
│  CentralPanel → CanvasWidget (egui_wgpu paint callback)  │
├──────────────────────────────────────────────────────────┤
│  pigment-core  (engine, GPU-agnostic state)              │
│  Document · LayerTree · Selection · CommandStack(undo)   │
│  Tile model (sparse, COW) · color types · doc IO         │
├──────────────────────────────────────────────────────────┤
│  pigment-gpu  (wgpu compositor + tools)                  │
│  TileCache(atlas+page table) · Compositor(render graph)  │
│  blend/adjustment/filter shaders · brush(wet layer)      │
├──────────────────────────────────────────────────────────┤
│  pigment-io  · pigment-ai  (load/save, ONNX features)    │
├──────────────────────────────────────────────────────────┤
│  wgpu 29   ·   lcms2 color mgmt                          │
└──────────────────────────────────────────────────────────┘
```

### Core data model
- **Document** = canvas size, color profile, `LayerTree`, active selection.
- **LayerTree** = recursive: `Layer { Raster | Group | Adjustment | Text | Vector | SmartObject }`, each with blend mode, opacity, mask, visibility.
- **Tile** = 256×256 RGBA16F **linear premultiplied**. Sparse `HashMap<(layer_id, tx, ty), Arc<Tile>>`. COW: edits clone only touched tiles → cheap undo + layer clones.
- **CommandStack** = every edit is a reversible `Command`. Pixel ops store pre-edit dirty-tile copies (Arc-shared). Structural/param edits store small graph deltas.

### Render graph (compositor)
- Nodes: raster source, blend, group boundary, adjustment, mask, filter. Each = a GPU pass (compute preferred) writing intermediate tiles.
- **Unit of work = (node, dirty tile).** Recomposite only dirty tiles; propagate dirtiness upward. Cache per-node per-tile output. This keeps 100-layer docs interactive.
- All math in **linear light**, premultiplied alpha; sRGB encode only at final blit to the egui/swapchain `*Srgb` surface.

### GPU specifics (from research)
- Working/intermediate/composite tiles: `Rgba16Float` (filterable, blendable, storage-capable).
- Imported 8-bit sources: `Rgba8UnormSrgb` (auto-decode, half VRAM); promote to f16 only during compositing.
- Surface can't be f16 → final tonemap/encode pass to `*Srgb`.
- Simple modes (Normal/Multiply/Add) → fixed-function `BlendState`. Complex modes (Overlay/SoftLight/ColorDodge/all HSL) → shader sampling both backdrop+source, `blend_mode` uniform + switch, per dirty tile.
- Canvas viewport rendered via `egui_wgpu::CallbackTrait`: `prepare()` runs our offscreen compositor pass; `paint()` blits result into the egui-allocated rect. Store pipelines/bind groups in `renderer.callback_resources`. `response` from `allocate_exact_size` feeds pan/zoom/brush input.
- Present mode `AutoVsync` (Fifo); `desired_maximum_frame_latency = 1` for brush snappiness. Allocate canvas texture in **physical** px (× `pixels_per_point`) for crisp HiDPI.

### Brush engine (low-latency)
- Input: winit Pointer events + `octotablet` for pressure/tilt → drain **all** queued samples per frame into a lock-free queue (tablets 100–240Hz > frame rate; never sample only latest).
- Stamp/dab-based. Arc-length walker emits a dab every `spacing × radius`; Catmull-Rom interp through points; lerp pressure/tilt; carry leftover distance across segments.
- **Wet layer**: in-progress stroke renders to its own GPU texture composited over committed pixels; flatten wet→dry on pen-up. Append only new dabs each frame (flat per-frame cost).
- Smoothing: EMA/weighted stabilizer + 1–2 dab prediction to hide latency.
- Smudge: read canvas color under dab → blend into brush state → write (RMW per dab).
- Consider FFI to **libmypaint** for mature artist-tuned dynamics; implement `MyPaintSurface::draw_dab` backed by our wgpu wet layer.

---

## 3. Workspace Layout

```
pigment/
├── Cargo.toml                # workspace
├── crates/
│   ├── pigment-core/         # doc model, layers, tiles, commands, color types
│   ├── pigment-gpu/          # wgpu compositor, tile cache, shaders, brush
│   ├── pigment-io/           # image/psd/exr load+save, ICC
│   ├── pigment-ai/           # ort ONNX features (bg-remove, inpaint, upscale)
│   └── pigment-app/          # eframe binary: panels + canvas widget
├── assets/shaders/           # WGSL: blit, blend, brush, adjustments
├── PLAN.md  RESEARCH.md  ARCHITECTURE.md
```

---

## 4. End-to-End Task Backlog (actionable, checkbox-tracked)

### Phase 0 — Skeleton & GPU canvas  *(DONE — runs, shows image, pan/zoom)*
- [x] `cargo` workspace + crate stubs (`pigment-core`/`-io`/`-app`)
- [x] `pigment-core`: `Document`, `Layer`, `LayerTree`, `Tile`, `BlendMode`, geometry/color types
- [x] `pigment-app`: eframe app, panel layout (tools/layers/menu), CentralPanel
- [x] `CanvasWidget`: `egui_wgpu::CallbackTrait` (`CanvasPaint`); allocate rect + `Sense::click_and_drag`
- [x] wgpu: vertex/frag pipeline drawing the document textured quad with CPU-folded view transform
- [x] Pan (drag) + cursor-anchored zoom (scroll) via `ViewTransform`
- [x] Checkerboard transparency backing shader (`fs_checker`)
- [x] Load an image (`pigment-io` via `image`) into a GPU texture; File→Open dialog (`rfd`); placeholder on launch
- [x] Nearest-mag / linear-min sampling; Fit-to-screen + 100%
- [x] **DoD met:** builds, launches (Metal/wgpu 29), opens PNG/JPEG/etc, pan/zoom works, HiDPI-aware
- [ ] *Follow-up:* migrate off egui 0.34 deprecated panel/menu aliases; screenshot/CI smoke test

### Phase 1 — Tiles, layers, paint  *(COMPLETE)*
- [x] Compositor v1: GPU ping-pong, layers as `Rgba16Float` linear-premul textures, display pass (checker + sRGB encode)
- [x] Blend modes as shaders: Normal, Multiply, Screen, Overlay, Darken, Lighten, Add (rest of the separable + HSL set: Phase 3)
- [x] Layers panel: add, delete, reorder (▲▼), inline rename, visibility, opacity, blend-mode dropdown, active select
- [x] Brush engine v1: arc-length dab walker, instanced soft dabs, size/hardness/opacity, color picker
- [x] **Wet-layer separation:** brush strokes render to a wet buffer, composited over the owner layer, flattened on pen-up (correct per-stroke opacity)
- [x] **Brush dynamics:** velocity → size taper; per-dab modulation plumbed for a future pressure source
- [x] Eraser (destination-out dab pipeline)
- [x] Bucket fill (CPU flood fill, `pigment_core::fill`) + eyedropper (1px GPU readback); both with "sample all layers"
- [x] CommandStack: undo/redo (Cmd+Z / Cmd+Shift+Z / menu) + **History panel** (labeled steps, click to jump)
- [x] **Region-COW undo:** snapshots only the stroke's dirty rect, not the whole layer
- [x] **Dirty compositing:** recomposite only when the document changes; pan/zoom reuse the last composite
- [x] `.pigment` doc format: lz4 RGBA16F layer blobs + JSON metadata (`pigment_io::document_file`); save via GPU readback, open via staged upload
- [x] Image loads into the background layer (linear-premul f16 conversion)
- [x] Tests: core unit (color/blend/tile/fill) + io round-trip + **headless GPU test** (upload→composite→wet-brush→region-undo, pixel-asserted)
- [x] **DoD met:** multi-layer painting + blend + wet strokes + undo + save/reopen

**Deferred to later phases (rationale):**
- *GPU sparse-virtual-texture streaming* (atlas + page table + RAM/disk spill for docs > VRAM) → **Phase 5** "out-of-core huge docs". Layers are currently one full-canvas `Rgba16Float` texture each (degenerate single tile); dirty tracking is frame-level + region-COW. Streaming is a large self-contained perf subsystem that belongs with its Phase 5 sibling.
- *Hardware stylus pressure/tilt* → platform-blocked: eframe/egui doesn't surface pen pressure and `octotablet` has no macOS backend. Velocity dynamics stand in; per-dab modulation is ready for a raw-winit/`octotablet` source.

### Phase 2 — Selection & transform  *(COMPLETE)*
- [x] Selection mask as `R16F` GPU texture; animated marching-ants overlay (display shader)
- [x] Tools: rectangle, ellipse, lasso (freehand polygon), magic wand (flood by tolerance)
- [x] Feather, grow/shrink, invert, select-all/none, add/subtract/intersect modifiers (`pigment_core::raster`)
- [x] Selection-aware brush/eraser/fill (dabs + fill clip to the mask)
- [x] Move tool (translate) + Transform (translate + Shift-drag scale) via composite-time uv affine, baked on release
- [x] Crop to selection; canvas size (no resample); image size (resample via `fast_image_resize`); flip layer H/V
- [x] Copy/cut/paste (Cmd+C/X/V) + layer-from-selection; selection-masked clipboard
- [x] UI polish: phosphor icon toolbar + modern dark theme
- [x] Tests: core `raster` (10) + `resize` (4) + GPU selection-clip & transform-bake
- [x] **DoD met:** select region, transform, crop, resize with quality resampling

**Deferred (polish):** free-transform *rotation/skew* + interactive corner/rotate handles (current transform is translate + uniform scale via Shift-drag; the composite affine already supports rotation — only the handle UI + aspect-correct rotation math remain). Modifier preview during marquee uses replace-then-combine.

### Phase 3 — Adjustments, masks, filters  *(COMPLETE)*
- [x] Adjustment layers (read backdrop, transform): Brightness/Contrast, Levels, Hue/Saturation, Exposure, Invert, Threshold, Black&White — non-destructive, live param sliders (`pigment_core::adjust` + composite-shader branch)
- [x] Layer masks (`α *= mask`): add white / from selection / delete; paint reveal (brush) / hide (eraser); composite multiplies layer alpha by mask
- [x] Filters as GPU passes: Gaussian blur (separable), Sharpen (unsharp), Pixelate — destructive on the active layer, undoable
- [x] HSL non-separable blend modes (Hue/Saturation/Color/Luminosity) + the previously-missing separable modes — all 18 now correct
- [x] Histogram panel (Rec.709 luma + per-channel, `pigment_core::histogram`)
- [x] Tests: core `adjust`/`curve`/`histogram` + GPU adjustment-invert & layer-mask
- [x] **DoD met:** non-destructive adjustment stack + masks + core filters

**Deferred (polish):** Curves spline-editor UI (the monotone-cubic LUT builder exists in `pigment_core::curve`, only the draggable widget + a Curves adjustment kind remain); Color Balance / per-channel curves; clipping & group masks; layer styles (stroke/shadow/glow); motion blur / noise.

### Phase 4 — Text, vector, smart objects  *(MOSTLY COMPLETE)*
- [x] Text layers (`cosmic-text` rasterizer → layer texture): editable text/size/color/align, re-rasterized on edit
- [x] Vector shape layers: rectangle + ellipse (drag-create), editable fill color, re-rasterized (`pigment_core::shape`, AA)
- [x] Gradient tool: drag a foreground→transparent linear gradient, composited over the active layer
- [x] Generated layers stay editable after creation (`sync_generated_layers` re-rasterizes on def change)
- [x] Tests: `pigment_io::text` + `pigment_core::shape` rasterizers
- [x] **DoD (partial):** text + vector editable after creation ✓; smart objects deferred
- [ ] **Deferred:** pen tool (`kurbo` bezier + handles), polygon/line shapes, boolean shape ops (`i_overlay`), vector masks, smart objects, gradient editor (multi-stop) + pattern fill. Foundations present: `lyon`/`kurbo`/`i_overlay` in the stack; `curve` LUT for gradient stops.

### Phase 5 — Pro features
- [ ] Color management: ICC load/embed (`lcms2`), working-space conversion, soft-proofing, CMYK, 16/32-bit per channel, HDR/EXR
- [ ] PSD import (layers/masks/blend via `psd` + our compositor); PSD export (custom serializer) or `.ora`
- [ ] AI tools (`ort`): background removal (BiRefNet MIT / U²-Net fast preview), inpaint/heal (LaMa), super-res (Real-ESRGAN), select-subject
- [ ] Heal/clone stamp, content-aware fill
- [ ] Plugin API (WASM-sandboxed effects + OFX bridge consideration)
- [ ] Performance: out-of-core huge docs, GPU profiling, multi-thread tile ops (`rayon`)
- [ ] Web build: WASM + WebGPU target (`qcms` color path)
- [ ] **DoD:** color-accurate, PSD-compatible, AI-assisted, extensible

### Cross-cutting (every phase)
- [ ] Tests: core model unit tests, blend-math golden-image tests, doc round-trip
- [ ] CI: fmt + clippy + test + cross-platform build matrix
- [ ] Input mapping/shortcuts config; preferences
- [ ] Crash recovery / autosave; error surfacing
- [ ] Benchmarks: brush latency, composite time per layer count, large-image open

---

## 5. Milestones (definition of "usable")

| Milestone | Phase done | Capability | Solo | Team 3–4 |
|---|---|---|---|---|
| **Spike** | 0 | Open/view/pan/zoom on GPU | 2–4 wk | 1–2 wk |
| **MVP** | 2 | Paint, layers, blend, select, transform, undo, save | 6–9 mo | 3–4 mo |
| **Beta** | 3 | + adjustments, masks, filters → real photo editing | 12–18 mo | 6–9 mo |
| **1.0** | 4 | + text, vector, smart objects → general-purpose | 2–3 yr | 12–18 mo |
| **Pro** | 5 | + color mgmt, PSD, AI, plugins | 3+ yr | 2+ yr |

Realistic solo target: **MVP → Beta**, depth over breadth. Polish > feature count.

---

## 6. Hard Problems (mitigations baked in)

1. **Brush latency** → wet-layer + drain-all-samples + prediction + `frame_latency=1`.
2. **Large docs > VRAM** → sparse tiles + atlas/page-table LRU + RAM/disk spill.
3. **Color correctness** → linear-light premultiplied everywhere; lcms2 ICC; f16 working buffers (8-bit linear bands).
4. **PSD compat** → import via `psd` + our compositor; export is a custom serializer (budget real effort) — ship `.ora` interchange first.
5. **Undo memory** → tile-COW diffs, not full snapshots; compress cold history (lz4).
6. **Non-destructive + undoable** → document is a node-graph/command log; pixels re-derived & cached.

---

## 7. Immediate Next Steps (this draft)

1. [x] Scaffold cargo workspace + crates.
2. [ ] Build Phase 0 skeleton: eframe app + wgpu textured-quad canvas + pan/zoom + image load.
3. [ ] `cargo run` → window with a loaded image, pan/zoom working.
4. [ ] Then Phase 1: tiles + compositor + brush.

*Foundations are free. The product is the polish.*
