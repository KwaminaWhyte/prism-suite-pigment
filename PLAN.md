# Pigment — Open Source Photoshop Alternative

Professional-grade, non-destructive, GPU-accelerated raster image editor in Rust.
**Goal: reach ≥85% of Photoshop's real-world capability** — features, reliability, and
ease-of-use — in staged milestones on a modern GPU pipeline, while leaving the suite's
shared engine clean for Contour (vector) and Pulse (motion) to reuse.

Phases 0–5 stood up the engine and the photo-editing core; Phases 6–12 below close the
gap to parity (retouching, layer power, the filter galleries, pro color/IO, AI, automation,
and the reliability/UX polish that makes a tool *feel* finished). The §"Parity coverage
matrix" tracks where we stand against the full Photoshop surface.

> Companion docs: [RESEARCH.md](./RESEARCH.md) (cited findings + crate matrix), [ARCHITECTURE.md](./ARCHITECTURE.md) (module/data-flow detail as it lands), [../SUITE.md](../SUITE.md) (four-app vision + interop).

---

## 0a. Suite boundaries — what belongs in Pigment vs Contour / Pulse

Pigment shares the `prism-core` / `prism-color` / `prism-io` crates with **Contour** (vector,
the Illustrator analog) and the future **Pulse** (motion/VFX, the After Effects analog). To
avoid duplicating or overwriting their work, every new feature is filed against one of three rules:

- **Pigment-owned (raster):** painting, retouch/heal, raster filters, adjustments, masks,
  channels, raster layer styles, image/color IO, raster-side AI. Lives in `pigment-app` or
  raster-only modules of the shared crates. This is the bulk of Phases 6–12.
- **Shared-crate, app-agnostic:** anything promoted into `prism-core`/`prism-color`/`prism-io`
  (blend math, tile model, color transforms, file containers, curve/gradient/LUT, geometry,
  path math) **must not** assume Pigment. Contour already depends on these — additions are
  additive and feature-gated, never raster-coupled. Touch these crates only to *add* generic
  primitives, never to bend them toward Pigment's UI.
- **Out of scope — belongs to a sibling app (do not build here):**
  - *Deep vector authoring* (full pen/node editing, boolean shape trees, stroke profiles,
    multi-artboard vector layout) → **Contour**. Pigment keeps only Photoshop-grade vector:
    shape layers, simple pen paths, vector masks — re-rasterized into the raster doc.
  - *Timeline, keyframes, motion graphics, node compositing, video frames* → **Pulse / Reel**.
    Pigment ships only Photoshop's *frame/video timeline* scope (frame animation, GIF/APNG/
    short-clip export); anything beyond per-frame raster is a Pulse comp placed via Dynamic Link.
  - *Cross-app interop glue* (Dynamic Link host, the `prism-doc` interchange container, shared
    clipboard, shared asset library) is **suite-level** — Pigment consumes it (smart objects can
    reference a `.contour` or Pulse comp) but does not define it unilaterally.

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
| AI runtime | `ort` | 2.0-rc | ONNX Runtime; CoreML/DirectML/CUDA EPs. `candle`/`tract` for pure-Rust/wasm fallback. |
| Undo | `undo`/custom | — | Command stack + tile-COW for pixels. |
| Encode extras | `mozjpeg`, `webp`, `ravif`, `oxipng` | — | Better-than-baseline JPEG/WebP/AVIF/PNG. |
| Serde/util | `serde`, `glam`, `bytemuck`, `thiserror`, `anyhow`, `rayon`, `arc-swap`, `lz4_flex` | — | Doc serialization, math, GPU casts, tile compression. |
| **RAW / Camera Raw** | `rawler` (+ `rawloader`) | latest | Decode CR2/CR3/NEF/ARW/RAF/DNG; demosaic (Malvar-He-Cutler, X-Trans). `zenraw` = safe-Rust, scene-linear f32 alt. |
| **Lens correction** | `lensfun` (pure-Rust port) | 0.x pre-alpha | Distortion / TCA / vignetting from the Lensfun DB; bit-exact vs C++ ref. Polynomial fallback if pre-alpha. |
| **Inpainting / content-aware** | custom PatchMatch + `ort` (LaMa) | — | Classic PatchMatch on GPU/CPU (no model, high-res); optional LaMa ONNX for structure, hybrid guided fill. |
| **Segmentation / select-subject** | `ort` (BiRefNet, RMBG-2.0, SAM2/3) | — | BiRefNet_dynamic for matting; SAM2/SAM3 for prompted object selection. |
| **Super-resolution** | `ort` (Real-ESRGAN / SwinIR) | — | 2–4× upscale; tile-and-blend for big images. |
| **Scripting / automation** | `rhai` (embedded) + `serde_json` actions | latest | Sandboxed action/script engine; JSON-recorded actions; batch runner. Optional `boa`/`deno_core` for JS parity. |
| **Seamless cloning / heal** | custom (Poisson solve) + `nalgebra`/`faer` | — | Gradient-domain heal (Poisson blend) for healing brush / patch. |
| **Plugin / effects** | OpenFX-style `prism-fx` host (suite) | — | Shared with the suite; raster effects authored once, run in any compositing app. |

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

**Deferred (polish):** ~~Curves spline-editor UI~~ (**done** — see Phase 4); Color Balance; clipping & group masks; layer styles (stroke/shadow/glow); motion blur / noise.

### Phase 4 — Text, vector, smart objects  *(MOSTLY COMPLETE)*
- [x] Text layers (`cosmic-text` rasterizer → layer texture): editable text/size/color/align, re-rasterized on edit
- [x] Vector shape layers: rectangle + ellipse (drag-create), editable fill color, re-rasterized (`pigment_core::shape`, AA)
- [x] Gradient tool: drag a foreground→transparent linear gradient, composited over the active layer
- [x] Generated layers stay editable after creation (`sync_generated_layers` re-rasterizes on def change)
- [x] Tests: `pigment_io::text` + `pigment_core::shape` rasterizers
- [x] **DoD (partial):** text + vector editable after creation ✓; smart objects deferred

**Phase 4 completion tasks (promote from "deferred" — needed for parity):**
- [ ] Pen tool: `kurbo` `BezPath` with draggable anchor/handle UI; add/delete/convert points; rubber-band preview
- [ ] More shapes: polygon, line, rounded-rect (live corner radius), custom-shape from path; shape stroke + fill + dashes
- [ ] Boolean shape ops (`i_overlay`): unite / subtract / intersect / exclude on selected shapes
- [ ] Vector masks (path clips a layer; composite multiplies α by rasterized path coverage) + clipping masks (layer clipped to one below) + group masks
- [x] Curves adjustment **UI** (draggable monotone-cubic editor; composite **+ per-channel R/G/B** curves; LUT uploaded as a 256×1 texture, sampled in the compositor; GPU-pixel-tested)
- [ ] Gradient editor: multi-stop, opacity stops, linear/radial/angle/reflected/diamond types, dithering; saved presets
- [ ] Pattern fill / pattern stamp + define-pattern from selection
- [ ] **Type richness:** character + paragraph panels (kerning/tracking/leading, OpenType features, justification), text-on-path, warp text, type masks
- [ ] **Smart objects** (big — see Phase 7): embedded + linked; non-destructive transform/filter; "edit contents" reopens source; place `.pigment`/`.contour`/image as smart object

### Phase 5 — Pro features  *(IN PROGRESS — interop landed)*
- [x] PSD import (`psd` 0.3.5): layers, opacity, blend, visibility → our model + compositor
- [x] HDR/EXR open (`exr` 1.74, linear RGBA f32) + HDR via `image`
- [x] Image export (`image`): PNG/JPEG/WebP/TIFF/BMP by extension (composite → straight sRGB8)
- [x] **DoD (partial):** PSD-compatible ✓, standard-format export ✓, HDR in ✓

The remaining Phase-5 "pro" bullets are large self-contained subsystems; they are
**re-scoped as dedicated Phases 8–12 below** rather than a single deferred list, because
each needs its own task breakdown to reach parity:
- Color management (ICC/CMYK/soft-proof) → **Phase 9**
- AI tools (select-subject, remove, inpaint, super-res, colorize, generative) → **Phase 10**
- PSD export, RAW/Camera Raw, export-as/save-for-web, more containers → **Phase 9**
- Heal / clone / content-aware fill → **Phase 6**
- Plugin API + actions/scripting/batch → **Phase 11**
- Out-of-core huge docs + sparse-virtual-texture streaming + web build → **Phase 12**

---

## 4b. Parity expansion — Phases 6–12 (the road to ≥85%)

Each phase is a coherent, shippable slice. Effort tags: **S** ≤1wk-equiv, **M** 1–3wk, **L** >3wk
(solo-equivalent, GPU/algorithm work). "shared?" = touches a `prism-*` crate, so keep it app-agnostic.

### Phase 6 — Retouching, healing & Liquify  *(the photo-repair core)*
The single biggest "feels like Photoshop" gap. All operate on the active raster layer (or a
sampled-merged source), undoable via region-COW, selection-clipped.
- [x] **Clone Stamp** (Alt-click source → aligned offset; frozen pre-stroke snapshot sampled in a GPU clone-dab pass; soft brush + opacity; selection-clipped; on-canvas source crosshair; GPU-pixel-tested). *Still: non-aligned mode, sample all-layers/below, flow vs opacity, rotation.*
- [x] **Healing Brush** (Alt-click source → brush a region → on release a gradient-domain Poisson solve transplants the source *texture* with the destination tone matched at the region boundary). Solver is `prism_core::heal::seamless_clone` (Gauss–Seidel membrane, shared core, unit-tested incl. tone-match + texture-transfer). *Still: continuous (per-dab) heal, spot/auto-source, content-aware fallback.*
- [ ] **Spot Healing** (M): auto-source from surrounding ring (proximity match) → Poisson blend; content-aware mode falls through to PatchMatch fill
- [ ] **Patch tool** (M): lasso a region, drag onto a source area, seamless-blend (Poisson) the result
- [ ] **Content-Aware Fill** (L): PatchMatch synthesis over the selection from sampled regions; sampling-area mask + scale/mirror/rotation adaptation; optional LaMa-ONNX guidance for structure (Phase 10 dep, degrades gracefully without)
- [ ] **Remove tool** (M, AI-assist): brush over an object → content-aware/LaMa removal in one stroke
- [ ] **Dodge / Burn / Sponge** (S): exposure/saturation brushes with shadows/mids/highlights range + protect-tones
- [ ] **Blur / Sharpen / Smudge tools** (S): localized GPU brush variants (smudge RMW already prototyped in brush research)
- [ ] **Red-eye** (S): detect + desaturate/darken pupil
- [ ] **Liquify** (L): forward-mapping mesh warp — push/pull, twirl, pucker, bloat, freeze/thaw mask, reconstruct; GPU displacement texture; face-aware later (AI, Phase 10)
- [ ] Tests: gradient-domain heal seam continuity; PatchMatch determinism (seeded); liquify mesh round-trip

### Phase 7 — Layer power: styles, smart objects, channels  *(non-destructive depth)*
- [ ] **Layer styles / FX** (L): Stroke, Drop Shadow, Inner Shadow, Outer/Inner Glow, Bevel & Emboss, Color/Gradient/Pattern Overlay, Satin — live, re-evaluated as GPU passes around the layer; per-effect blend mode/opacity; copy/paste/scale styles; FX as collapsible layer children
- [ ] **Smart Objects** (L): embedded + linked; wrap any layer/selection; transforms & filters re-applied non-destructively to the source render; "edit contents" → child document; replace-contents; place external (`.pigment`/`.contour`/image/PDF/Pulse-comp via Dynamic Link)
- [ ] **Smart Filters** (M): any Phase-6/8 filter applied to a smart object stays editable, re-orderable, masked, toggle-able
- [ ] **Clipping masks** (S) + **vector masks** (M, from Phase 4 pen) + **group/nested masks** + mask density/feather + mask panel
- [ ] **Blend-If / advanced blending** (M): this-layer / underlying-layer gray + per-channel split sliders; fill-vs-opacity; knockout; blend interior effects
- [ ] **Channels panel** (M, shared types): view/edit per-channel; alpha channels (save/load selections as masks); spot channels; channel from selection; split/merge channels
- [ ] **Layer comps** (S): named snapshots of visibility/position/appearance
- [ ] **Adjustment expansion** (M): Curves (UI from Ph4), Color Balance, Vibrance, Photo Filter, Channel Mixer, Selective Color, Gradient Map, Color Lookup (`.cube`/`.3dl` LUT, shared with suite), Posterize, Shadows/Highlights, HDR Toning, Equalize, Replace/Match Color
- [ ] Tests: layer-style pass pixel asserts; smart-object re-render on source edit; blend-if math; alpha-channel round-trip

### Phase 8 — Filters & distort galleries  *(creative + corrective filters)*
Implement as a unified `prism-fx` GPU pass registry (OpenFX-style; shared with the suite) so each
filter is authored once. Destructive on a layer **or** non-destructive as a smart filter (Phase 7).
- [ ] **Blur Gallery** (M): Field, Iris, Tilt-Shift, Path, Spin blur; bokeh shape; Lens Blur (depth/alpha-aware)
- [ ] **Motion blur, Box, Surface, Radial, Smart Blur** (S/M)
- [ ] **Sharpen family** (S): Smart Sharpen (radius/amount/noise-reduce/lens-vs-motion), Unsharp Mask (have), High Pass
- [ ] **Noise** (S): Add Noise (gaussian/uniform, monochromatic), Reduce Noise, Median, Dust & Scratches, Despeckle
- [ ] **Distort** (M): Warp (mesh + presets), Pinch, Spherize, Polar Coords, Ripple, Wave, Twirl, Shear, Displace (displacement map), Lens Correction (`lensfun`: distortion/CA/vignette), Adaptive Wide Angle
- [ ] **Render** (M): Clouds/Difference Clouds (Perlin/Simplex), Fibers, Lens Flare, Lighting Effects (normal-from-bump), Picture-frame/Tree/Flame-style generators (lower priority)
- [ ] **Stylize / Artistic — Filter Gallery** (M): Emboss, Find Edges, Glowing Edges, Oil Paint, Posterize Edges, Wind, Diffuse, plus the classic artistic set (Poster, Cutout, Dry Brush, Watercolor…)
- [ ] **Pixelate** (have) + Mosaic, Crystallize, Mezzotint, Color Halftone
- [ ] **Camera Raw filter** (M): apply the Phase-9 RAW develop controls (white balance, tone, HSL, sharpening, grain, vignette) as a non-destructive filter on any layer
- [ ] **Vanishing Point / Perspective Warp** (L, optional): plane-defined clone & paste
- [ ] Tests: golden-image per filter; separable-blur correctness; displacement-map sampling

### Phase 9 — Pro color, RAW & interchange  *(color-accurate + opens/saves everything)*
- [ ] **Color management** (L, shared `prism-color`): `lcms2` ICC v2/v4 load/embed; assign/convert profile; working spaces (sRGB/AdobeRGB/Display-P3/ProPhoto/linear); rendering intents; **soft-proofing** + gamut warning; **CMYK** mode + separations; Lab mode; `qcms` fast path for wasm/RGB
- [ ] **Bit depth & color modes** (M): 8/16/32-bit UI + conversions; Grayscale, Duotone, Indexed (palette + dither), Bitmap, Lab modes
- [ ] **RAW / Camera Raw develop** (L): `rawler`/`zenraw` decode + demosaic → scene-linear; white balance, exposure/contrast/highlights/shadows/whites/blacks, tone curve, texture/clarity/dehaze, HSL/color mixer, split toning, detail (sharpen + NR), lens corrections (`lensfun`), crop/straighten, profiles; open as smart object for re-edit
- [ ] **PSD export** (L): hand-written serializer (documented binary format) — layers, groups, masks, blend modes, opacity, text (as layers), basic layer styles; round-trip-tested against import
- [ ] **More containers** (M): `.ora` (zip+PNG, easy interchange first), layered TIFF, PDF (single + multipage placement), PSB (>30k px / >2GB)
- [ ] **Export As / Save for Web** (M): per-format quality/size preview, multiple scales (@1x/@2x/@3x), metadata strip, color-profile embed; PNG/JPEG/WebP/AVIF/GIF via the better-than-baseline encoders already in stack
- [ ] **Asset/Generator export** (S): layer/group → file by naming convention; SVG export for shape/vector layers (bridge to Contour)
- [ ] **Metadata** (S): EXIF/IPTC/XMP read + preserve on export; copyright/watermark template
- [ ] Tests: ICC round-trip ΔE bound; PSD export→reimport layer fidelity; RAW decode vs reference thumbnail

### Phase 10 — AI / neural tools  *(`ort`, feature-gated, models fetched on first use)*
Runtime is `ort` (ONNX) with CoreML/DirectML/CUDA EPs; `candle`/`tract` pure-Rust fallback.
Models are **not bundled** — downloaded to a cache on first use behind a feature flag, with
clear license surfacing; every tool degrades gracefully when models/GPU are absent.
- [ ] **Select Subject / Object Select** (M): SAM2/SAM3 prompted (click/box) + BiRefNet saliency → editable selection/mask
- [ ] **Remove Background** (S): BiRefNet_dynamic / RMBG-2.0 matting → mask or cut layer
- [ ] **Content-aware Remove / Inpaint** (M): LaMa ONNX (feeds Phase 6 Remove/Content-Aware Fill)
- [ ] **Super-Resolution / Enhance** (M): Real-ESRGAN / SwinIR 2–4×, tiled with overlap-blend; detail/denoise toggles
- [ ] **Denoise / Sharpen (AI)** (S): learned NR for high-ISO RAW
- [ ] **Colorize** B&W (S), **Neural-style** presets (S, optional)
- [ ] **Face-aware Liquify** (M): landmark model drives Phase-6 Liquify sliders (eyes/nose/smile/face-width)
- [ ] **Generative Fill / Expand** (L, **optional + pluggable** — suite AI policy): prompt → fill selection / extend canvas via a provider abstraction with **two interchangeable backends — local diffusion (`candle`/ONNX) AND a user-configured cloud endpoint (bring-your-own API key)** — plus "none". **Never required for core editing**; the app is fully functional with no AI backend configured. (Policy shared across the suite — see [../RESEARCH.md §5](../RESEARCH.md).)
- [ ] Provider abstraction: a `prism-ai` trait so local / cloud / none are swappable at runtime; cancellation + progress + VRAM guard; model license surfaced on first fetch
- [ ] Tests: deterministic mask IoU vs fixtures (seeded), graceful no-model path

### Phase 11 — Automation, extensibility & plugins  *(pro workflows)*
- [ ] **Actions** (M): record/play user operations as a serialized step list; action panel, sets, conditional steps, insert-stop/menu-item; modal toggles
- [ ] **Batch processing** (M): run an action over a folder/open docs; **Image Processor** (resize + format + ICC convert in bulk); droplet-style export
- [ ] **Scripting** (L): embedded `rhai` sandbox exposing the document/layer/selection API; run-script + script-events; optional JS engine (`boa`/`deno_core`) for Photoshop-script familiarity
- [ ] **Variables / data-driven** (S): bind text/visibility/replace-image to a dataset (CSV) → export N variants
- [ ] **Plugin API** (L, shared `prism-fx`): OpenFX-style effect plugins (load once, run in Pigment/Contour/Pulse); stable C ABI or wasm-component plugins; plugin-defined panels later
- [ ] **Presets manager** (S): brushes, gradients, patterns, styles, swatches, shapes, LUTs — import/export `.abr`/`.grd`/`.pat`/`.asl` where feasible, native format otherwise; shared suite asset library
- [ ] Tests: action record→replay determinism; script API surface; plugin load/exec sandbox

### Phase 12 — Reliability, performance & ease-of-use  *(the polish that earns trust)*
Parity isn't only features — it's that the app is fast, hard to lose work in, and obvious to drive.
- [ ] **Out-of-core / huge docs** (L): finish the sparse-virtual-texture streaming (atlas + page table + LRU + RAM/disk spill) the tile model was designed for; scratch-disk; >VRAM and >RAM documents stay interactive
- [ ] **True tiling** (M): split the current single-texture-per-layer into real 256² tiles for dirty-granular composite + COW (foundations in `prism_core::tile`)
- [ ] **Autosave + crash recovery** (M): periodic snapshot to a recovery file; restore-on-relaunch; never-lose-work
- [ ] **Performance** (M): GPU-memory budget + eviction; multithread tile ops (`rayon`); brush-latency/composite/large-open **benchmarks** in CI; frame-time HUD
- [ ] **Color/precision audit** (S): verify linear-light premultiplied path end-to-end; no double-gamma; HDR > 1.0 preserved
- [ ] **Preferences** (M): performance/scratch/cursors/units/UI-scale/theme/file-handling; persisted
- [ ] **Keyboard shortcuts** (M): full default map matching Photoshop muscle-memory; fully remappable; searchable command palette
- [ ] **Workspace & panels** (M): dockable/floating/collapsible panels; save/load workspaces (Photography/Painting/Web…); reset; tool options bar + contextual task bar
- [ ] **Multi-document tabs** (M): multiple open docs, tear-off windows, "match zoom/location", arrange/tile
- [ ] **Navigation & info** (S): Navigator panel, rotate-view, bird's-eye zoom, Info panel, color sampler points, measure tool, ruler, count tool
- [ ] **Guides/grid/snapping** (M): rulers, manual + smart guides, grid, snapping (guides/grid/layers/doc bounds), guide layout, lock/clear
- [ ] **Artboards** (M): multiple artboards in one doc; per-artboard export; useful for UI/social work
- [ ] **Onboarding & ease-of-use** (S): tool tooltips with shortcut, contextual hints, recent files, templates/new-doc presets, non-modal dialogs, und/redo history scrubbing, in-canvas HUD for tool params
- [ ] **Accessibility** (S): keyboard-drivable, high-contrast theme, scalable UI, screen-reader labels where egui allows
- [ ] Tests: autosave/recovery round-trip; streaming residency under VRAM pressure; benchmark regression gates

---

## 4c. Parity coverage matrix (vs Photoshop surface)

Rough coverage by category. **Done** = shipped (Phases 0–5). **Planned** = in Phases 6–12. **Won't** =
intentionally a sibling-app concern. Target ≥85% of the *weighted, real-world-used* surface.

| Category | Photoshop surface | Status | Phase |
|---|---|---|---|
| Canvas / GPU / view | open, pan/zoom, HiDPI, fit/100% | **Done** | 0 |
| Layers + blend modes (18) | full | **Done** | 1,3 |
| Painting / brush / eraser / fill / eyedropper | wet-layer, dynamics, smoothing | **Done** (pressure platform-blocked) | 1 |
| Brush richness | tip shapes, dual/scatter/texture, mixer, history brush, symmetry | **Planned** | 6,11 |
| Selection (marquee/lasso/wand) + edit ops | + magnetic/quick/object/color-range/select-mask | **Done** core; rest **Planned** | 2,10 |
| Transform | move/scale/translate baked | **Done**; rotate/skew/distort/perspective/warp/puppet **Planned** | 2,7,8 |
| Adjustments | 7 core done; +15 more | **Done** core; rest **Planned** | 3,7 |
| Masks | layer masks done; clipping/vector/group + blend-if | **Done** core; rest **Planned** | 3,4,7 |
| Filters | blur/sharpen/pixelate done; full galleries | **Done** core; rest **Planned** | 3,8 |
| Retouch / heal / clone / content-aware / liquify | none yet | **Planned** | 6 |
| Layer styles (FX) | none yet | **Planned** | 7 |
| Smart objects / smart filters | none yet | **Planned** | 7 |
| Channels / alpha / spot | none yet | **Planned** | 7 |
| Text | basic editable; rich type/OpenType/on-path/warp | **Done** basic; rest **Planned** | 4 |
| Vector / shapes / pen | rect/ellipse done; pen/bool/custom | **Done** basic; rest **Planned** | 4 |
| Color mgmt / ICC / CMYK / soft-proof | none (sRGB/linear today) | **Planned** | 9 |
| RAW / Camera Raw | none | **Planned** | 9 |
| Import: PNG/JPEG/…/PSD/EXR/HDR | yes | **Done** | 5 |
| Export: standard rasters | yes | **Done**; PSD-write / save-for-web / artboards **Planned** | 5,9 |
| AI (select/remove/inpaint/super-res/generative) | none | **Planned** | 10 |
| Automation (actions/batch/scripting) | none | **Planned** | 11 |
| Plugins / extensibility | none | **Planned** | 11 |
| Huge docs / streaming / autosave / crash-recovery | partial (region-COW undo) | **Planned** | 12 |
| Prefs / shortcuts / workspaces / multi-doc / guides / artboards | minimal | **Planned** | 12 |
| Timeline / video / motion / 3D | — | **Won't** (Pulse/Reel; frame-anim only) | — |
| Deep vector authoring | — | **Won't** (Contour) | — |

### Cross-cutting (every phase)
- [x] Tests: core unit (color/blend/tile/fill/raster/curve/histogram/shape), io round-trips, headless-GPU pixel assertions (compositor/wet/undo/selection/transform/adjust/mask) — 58 total
- [x] CI: `fmt --check` + `clippy` (-D warnings, all-targets) + `test` on linux/macos/windows (`.github/workflows/ci.yml`); workspace is rustfmt-clean + clippy-clean
- [ ] Input mapping/shortcuts config; preferences
- [ ] Crash recovery / autosave; error surfacing
- [ ] Benchmarks: brush latency, composite time per layer count, large-image open

---

## 5. Milestones (definition of "usable")

| Milestone | Phase done | Capability | Approx parity | Solo | Team 3–4 |
|---|---|---|---|---|---|
| **Spike** | 0 | Open/view/pan/zoom on GPU | ~5% | 2–4 wk | 1–2 wk |
| **MVP** | 2 | Paint, layers, blend, select, transform, undo, save | ~30% | 6–9 mo | 3–4 mo |
| **Beta** | 3 | + adjustments, masks, filters → real photo editing | ~45% | 12–18 mo | 6–9 mo |
| **1.0** | 4 (+pen/type) | + text, vector, smart objects → general-purpose | ~55% | 2–3 yr | 12–18 mo |
| **Pro** | 5 | + color mgmt, PSD, AI, plugins (re-scoped to 9–11) | ~60% | 3+ yr | 2+ yr |
| **Retouch** | 6 | + clone/heal/patch/content-aware/liquify | ~68% | — | +3–5 mo |
| **Depth** | 7 | + layer styles, smart objects/filters, channels, all adjustments | ~76% | — | +4–6 mo |
| **Creative** | 8–9 | + filter galleries, full color mgmt, RAW, PSD-write, export | ~85% | — | +6–9 mo |
| **Parity** | 10–12 | + AI, automation/plugins, streaming, autosave, prefs/workspaces/guides | **≥90%** | — | +9–12 mo |

**The ≥85% target lands at the end of Phase 9** (Creative) — full editing/retouch/filter/color/IO
surface, color-accurate, opens & saves everything. Phases 10–12 push past 85% on AI, extensibility,
and the reliability/ergonomics that make it production-grade.

Realistic solo target remains **MVP → Beta** with depth; the parity phases assume the project grows
past solo or runs long. Sequencing within 6–12 is flexible, but **6 (retouch) and 7 (layer power +
adjustments) deliver the most felt parity per unit effort** — do them first. Polish > feature count.

---

## 6. Hard Problems (mitigations baked in)

1. **Brush latency** → wet-layer + drain-all-samples + prediction + `frame_latency=1`.
2. **Large docs > VRAM** → sparse tiles + atlas/page-table LRU + RAM/disk spill (Phase 12).
3. **Color correctness** → linear-light premultiplied everywhere; lcms2 ICC; f16 working buffers (8-bit linear bands).
4. **PSD compat** → import via `psd` + our compositor; export is a custom serializer (budget real effort) — ship `.ora` interchange first.
5. **Undo memory** → tile-COW diffs, not full snapshots; compress cold history (lz4).
6. **Non-destructive + undoable** → document is a node-graph/command log; pixels re-derived & cached. Smart objects / smart filters / layer styles re-evaluate as cached render-graph nodes (Phase 7).
7. **Healing seams** → heal/patch transplant *gradients* (Poisson/gradient-domain solve), not raw pixels, so texture matches but tone blends; solve only over the dirty region.
8. **Content-aware quality** → PatchMatch handles high-res texture; optional LaMa-ONNX adds structure; hybrid (LaMa-then-PatchMatch-upsample) beats either alone — and the classic path needs **no bundled model**.
9. **AI shipping** → models are *not* bundled: fetched to a cache on first use behind a feature flag, license surfaced; `ort` with CoreML/DirectML/CUDA EPs + `candle`/`tract` fallback; every AI tool degrades gracefully with no model/GPU. Generative fill stays optional (local diffusion *or* BYO cloud key).
10. **Shared-crate discipline** → promote only generic primitives to `prism-*`; never raster-couple them, or Contour/Pulse break. New raster-only code stays in `pigment-app`.
11. **Filter sprawl** → all filters/effects go through one `prism-fx` OpenFX-style pass registry (author once, reuse across the suite, run destructive or as smart filters) instead of bespoke pipelines per filter.

---

## 7. Immediate Next Steps (toward parity — Phases 0–5 shipped)

Highest felt-parity-per-effort first. Pick up from the Phase-4 completion tasks, then Phase 6/7.

1. [ ] **Finish Phase 4** — pen tool (kurbo handles), Curves UI, multi-stop gradient editor, rich type. Unblocks vector masks (Ph7) and Camera-Raw curve (Ph9).
2. [ ] **Phase 6 retouch core** — Clone Stamp → Healing Brush (Poisson) → Spot Healing → Patch → Content-Aware Fill (PatchMatch). This is the biggest single "feels like Photoshop" jump.
3. [ ] **Phase 7 layer power** — Layer styles (FX passes), Smart Objects + Smart Filters, clipping/vector masks, Channels panel, remaining adjustments. Turns it into a non-destructive pro tool.
4. [ ] **Phase 8/9 in parallel where possible** — filter galleries via `prism-fx`; color management (`lcms2`) + RAW (`rawler`) + PSD-write + Export-As. End of 9 = the ≥85% line.
5. [ ] Stand up the `prism-fx` pass registry early (Phase 8 dep, suite-shared) so filters/styles/smart-filters share one path.
6. [ ] Keep the cross-cutting reliability work (autosave, benchmarks, shortcuts) trickling in alongside features, not all deferred to Phase 12.

*Foundations are free. The product is the polish — and parity is mostly polish at scale.*
