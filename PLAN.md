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

### Phase 1 — Tiles, layers, paint  *(MVP painting — IN PROGRESS)*
- [x] Compositor v1: GPU ping-pong, layers as `Rgba16Float` linear-premul textures, display pass (checker + sRGB encode)
- [x] Blend modes as shaders: Normal, Multiply, Screen, Overlay, Darken, Lighten, Add (rest of the separable + HSL set: TODO)
- [x] Layers panel: add, visibility toggle, opacity slider, blend-mode dropdown, active-layer select
- [x] Brush engine v1: arc-length dab walker, instanced soft dabs, size/hardness/opacity, color picker
- [x] Image loads into the background layer (linear-premul f16 conversion)
- [x] Core unit tests (color roundtrip, blend ids, tile coords)
- [ ] **Tile model:** sparse map + `Rgba16Float` tiles + COW (`Arc`) — currently full-canvas textures (degenerate single tile)
- [ ] `TileCache`: GPU atlas + indirection/page table; LRU residency; RAM spill
- [x] Eraser (destination-out dab pipeline)
- [x] CommandStack: undo/redo (Cmd+Z / Cmd+Shift+Z + Edit menu) via GPU-side `copy_texture_to_texture` layer snapshots, depth-capped
- [ ] Wet-layer separation (in-progress stroke buffer) + pressure (tablet via `octotablet`)
- [ ] Dirty-tile invalidation (currently recomposite every frame)
- [ ] Bucket fill, eyedropper; layer delete/reorder/rename
- [ ] History panel; tile-COW snapshots (replace full-layer; readback pattern ready)
- [ ] `.pigment` doc format: serialize layer tree + tiles (lz4), load back
- [ ] **DoD:** multi-layer painting with blend modes, undo, save/reopen

### Phase 2 — Selection & transform
- [ ] Selection mask as `R16F` GPU texture; marching-ants overlay
- [ ] Tools: rectangle, ellipse, lasso, polygon lasso, magic wand (flood by tolerance)
- [ ] Feather, grow/shrink, invert, select-all/none, add/subtract/intersect modifiers
- [ ] Move tool; free transform (scale/rotate/skew) with GPU resampling
- [ ] Crop, canvas size, image size (resize via `fast_image_resize` Lanczos3/Bicubic)
- [ ] Copy/cut/paste, layer-from-selection, clipboard interop
- [ ] **DoD:** select region, transform, crop, resize with quality resampling

### Phase 3 — Adjustments, masks, filters  *(real photo editing)*
- [ ] Adjustment layers (read backdrop, transform): Levels, Curves, Brightness/Contrast, Hue/Sat, Color Balance, Exposure, Black&White, Invert, Threshold
- [ ] Curves UI widget (spline editor) + per-channel
- [ ] Layer masks (paint to reveal/hide; `α *= mask`), clipping masks, group masks
- [ ] Filters as GPU passes: Gaussian blur, Box/Motion blur, Sharpen/Unsharp, Noise, Pixelate
- [ ] Layer styles: stroke, drop shadow, inner shadow, outer/inner glow, color/gradient overlay
- [ ] HSL non-separable blend modes (Hue/Saturation/Color/Luminosity)
- [ ] Histogram panel
- [ ] **DoD:** non-destructive adjustment stack + masks + core filters

### Phase 4 — Text, vector, smart objects  *(general-purpose 1.0)*
- [ ] Text layers: `cosmic-text` editable buffer → `glyphon` atlas → layer tile cache; font/size/color/align/spacing
- [ ] Vector shapes: rectangle, ellipse, polygon, line; `lyon` tessellation
- [ ] Pen tool: `kurbo` bezier path model, anchor/handle editing, path panel
- [ ] Boolean shape ops (`i_overlay`); vector masks
- [ ] Smart objects: embedded non-destructive content re-rasterized on transform
- [ ] Gradient tool + gradient editor; pattern fill
- [ ] **DoD:** text + vector + smart objects, editable after creation

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
