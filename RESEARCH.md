# Pigment — Research Findings (June 2026)

Cited findings backing [PLAN.md](./PLAN.md). Verify all crate versions against crates.io at build time — third-party version metadata is sometimes stale.

---

## 1. GPU app stack (wgpu + egui)

**Versions (mutually compatible):** `wgpu` 29.0.x (pin major `"29"`), `egui`/`egui-winit`/`egui-wgpu` 0.34.x (lockstep), `eframe` 0.34.2 (pulls winit ^0.30.13 + wgpu ^29.0.1), `winit` 0.30.x (`ApplicationHandler` trait pattern).

**Recommendation:** use **eframe with wgpu backend** (default). It exposes `frame.wgpu_render_state() → RenderState { device, queue, target_format, renderer }` — the same device egui renders with, so the canvas shares the GPU context (no second device). Go manual (winit + egui-winit + egui-wgpu) only for multi-window, custom swapchain, headless, or custom frame pacing.

**Compositing canvas into egui — two techniques:**
- **A. Paint callback (`egui_wgpu::CallbackTrait`)** — idiomatic for the live viewport.
  - `prepare(device, queue, screen_desc, encoder, &mut CallbackResources) -> Vec<CommandBuffer>` — update uniforms; can run our own offscreen render pass (the layered composite) here.
  - `paint(info, &mut RenderPass<'static>, &CallbackResources)` — set pipeline/bind group, `draw()` to blit into the egui-allocated rect.
  - Store GPU resources in `wgpu_render_state.renderer.write().callback_resources.insert(...)`; retrieve by type.
  - `let (rect, response) = ui.allocate_exact_size(size, Sense::click_and_drag()); ui.painter().add(Callback::new_paint_callback(rect, MyCallback{..}));` — `response` gives hit-tested pan/zoom/brush input.
- **B. `register_native_texture`** — render doc to an offscreen texture yourself, show via `ui.image((id, size))`. Use for thumbnails/previews. Call `update_egui_texture_from_wgpu_texture` on change.

**Gotchas:**
- **Color space:** surface is typically `Bgra8UnormSrgb`. Do all blending in **linear** in an offscreen `Rgba16Float` buffer (NOT `*Srgb` — hardware would gamma during blends), convert to sRGB only at final blit. Mismatch = classic "too dark" bug.
- **HiDPI:** configure surface in physical px; egui works in logical points — multiply by `ctx.pixels_per_point()`. Allocate canvas texture in physical px for crispness; reconfigure on scale change.
- **Present mode:** `Fifo` only universally guaranteed. `Mailbox`/`Immediate` panic if unsupported → use `AutoVsync`/`AutoNoVsync`. `desired_maximum_frame_latency` default 2; set 1 for snappier brush.
- **MSAA:** egui renders samples=1; keep MSAA out of the swapchain pass. Canvas usually wants pixel-exact (nearest when zoomed in).

Sources: github.com/gfx-rs/wgpu/releases · docs.rs/eframe · github.com/emilk/egui (custom3d_wgpu example, egui-wgpu/renderer.rs) · docs.rs/wgpu (PresentMode, SurfaceConfiguration) · mxs.dev/blog/egui-wgpu-blurry-windows

---

## 2. Tile-based GPU compositing

**Tile size:** Krita & mypaint use 64×64 (CPU-cache tuned, COW, memento undo, LZF compress). For a **GPU** editor use **256×256** (good VRAM granularity; 256² RGBA16F = 512 KB), power-of-two, aligned to workgroups. 128×128 for finer dirty granularity.

**Storage:** CPU-side sparse `HashMap<(layer_id, tx, ty), Arc<TileData>>` — only touched tiles allocated; COW for cheap undo/clone.

**Dirty-tile invalidation (biggest perf lever):** a stroke marks tiles under its dab AABB dirty; recomposite only dirty tiles; dirtiness propagates up the stack. Never recomposite clean tiles.

**Streaming > VRAM = sparse virtual texturing:** fixed GPU tile-atlas (e.g. 16×16 of 256² = 64 MB) + indirection/page-table texture (virtual tile → physical slot), LRU residency, evict cold → RAM → disk. Only visible viewport (+margin, +mip for zoom-out) resident. wgpu has no native sparse API — implement atlas+indirection manually.

**Blend math (premultiplied, linear light):**
```
source-over (premul):  co = cs + cb·(1−αs);   αo = αs + αb·(1−αs)
blended composite:     blended = (1−αb)·Cs + αb·B(Cb,Cs); then source-over
```
Separable modes operate on **straight** (un-premul) RGB → unpremul (rgb/α) → B() → composite → re-premul (skip when α∈{0,1}).
```
Multiply a*b | Screen a+b−a*b | Darken min | Lighten max
LinearDodge(Add) min(1,a+b) | LinearBurn max(0,a+b−1)
Difference |a−b| | Exclusion a+b−2ab
HardLight: b≤.5? mult(a,2b):screen(a,2b−1)   Overlay = HardLight(b,a)
ColorDodge: b==1?1:min(1,a/(1−b))   ColorBurn: b==0?0:1−min(1,(1−a)/b)
SoftLight: b≤.5? a−(1−2b)a(1−a) : a+(2b−1)(D(a)−a), D=a≤.25?((16a−12)a+4)a:√a
```
Non-separable HSL (Lum=.3R+.59G+.11B): Hue/Saturation/Color/Luminosity via SetLum/SetSat.

**Implementation:** simple modes → fixed-function `BlendState`. Complex/HSL → shader sampling both backdrop+source, `blend_mode` uniform + switch, dispatched per dirty tile.

**Linear vs sRGB:** decode sRGB→linear on input, composite in linear, encode→sRGB on output. Krita uses g10 (linear) ICC working space. wgpu: store working buffers `Rgba16Float` (8-bit linear bands; sRGB-unorm can't be storage textures). Premultiply in linear.

**Render graph:** model layer stack as DAG; cache intermediate per-node per-tile; invalidate only dirty tiles + downstream. Groups composite children to own buffer then blend into parent (pass-through skips isolation). Adjustment layers sample running backdrop, no own pixels. Masks = `α *= mask` (R8/R16F).

**Pixel storage:** working/composite = `Rgba16Float` linear-premul (headroom, no banding, HDR >1.0). Imported 8-bit = `Rgba8UnormSrgb` (half VRAM, auto-decode). Surface can't be f16 → final tonemap pass to `*Srgb`. f32 only on demand.

Sources: community.kde.org/Krita/Tile_Data_Format · docs.krita.org (linear_and_gamma, scene_linear_painting) · github.com/mypaint/mypaint (tiledsurface) · w3.org/TR/compositing-1 · en.wikipedia.org/wiki/Blend_modes · realtimerendering.com/blog/gpus-prefer-premultiplication · tonisagrista.com/blog/2023/sparse-virtual-textures · docs.rs/wgpu (TextureFormat) · sotrh.github.io/learn-wgpu (hdr tutorial)

---

## 3. Image IO & PSD

- **`image` 0.25.10** (MSRV 1.88): PNG/JPEG/GIF/WebP/TIFF/BMP/ICO/HDR/EXR/AVIF(ravif)/QOI/DDS. 16-bit yes (`Rgb16`/`Rgba16`). No native CMYK, no >16-bit int in high-level API. Flat single-image — pixel IO foundation, not doc model. Pure-Rust JPEG (slower, baseline only).
- **PSD `psd` 0.3.5**: read-only. Has layer tree/names/RGBA/visibility/opacity/dims, merged composite, blend-mode enum parsed. Lacks: real blend compositing (its flatten ignores blend mode), masks (thin), adjustment layers, smart objects, text layers, layer effects, PSB, 16/32-bit channels. No pure-Rust alternative. Paths: parse + own compositing; FFI; or `psd-tools` Python sidecar (far more complete). **PSD export:** no Rust crate writes PSD — hand-write serializer (documented binary format) or ship `.ora` (zip+PNG, trivial) / TIFF interchange.
- **HDR `exr` 1.74.0**: pure-safe-Rust, f16/f32/u32, full OpenEXR (multi-layer, deep, ZIP/PIZ/PXR24/B44). C-binding `openexr` (vfx-rs) only for DWAA/DWAB parity. Radiant `.hdr` via image's `hdr` feature.
- **Color/ICC `lcms2` 6.1.1** (Little CMS 2.17): ICC v2/v4, CMYK/Lab/XYZ/multichannel, soft-proof, intents. C dep (bundled build). **`qcms` 0.3.0** (Firefox, pure Rust): RGB+Gray only, no CMYK — good for wasm/RGB fast path. Decoders expose embedded ICC via `icc_profile()` → feed `Profile::new_icc`. **Use lcms2 primary, qcms for wasm.**
- **Export:** PNG via image (`new_with_quality`, 8/16-bit) + oxipng. JPEG: `mozjpeg` (libjpeg-turbo, progressive, trellis) beats image's baseline-only pure-Rust. WebP lossy: `webp` (libwebp) — image-webp lossy is limited. AVIF: `ravif` 0.13 (quality/speed/alpha/bitdepth).

Sources: crates.io/crates/image · github.com/chinedufn/psd · github.com/johannesvollmer/exrs · github.com/kornelski/rust-lcms2 · github.com/FirefoxGraphics/qcms · crates.io/crates/ravif

---

## 4. Brush engine & input

- **Input:** winit 0.30 unified Pointer API (`PointerMoved{source}`, `PointerKind`, pen via `Pen/Eraser/Airbrush`). Pressure+tilt on Wayland/Windows/Web; uneven elsewhere. Better: **`octotablet`** (pure Rust, via `raw_window_handle`, alongside winit/eframe) — Wayland full, Windows Ink RealTimeStylus, X11 best-effort; no macOS/iOS/Android yet. Won't do Wintab/Win32 Pointer. Fallbacks: `wintab_lite` (Wacom), `windows` crate `POINTER_PEN_INFO`. **Use octotablet + winit, Wintab fallback for pro Wacom on Win.**
- **Stamp/dab-based** is industry standard (MyPaint/Krita/PS/Procreate). Spacing = dabs every `spacing × radius` (MyPaint default ≈ 0.1). Interpolate: walk arc-length along Catmull-Rom/Bezier fit, emit dab when accumulated dist ≥ spacing, lerp pressure/tilt/color, carry leftover across segments. Smudge = read canvas under dab → blend into brush color state → write (RMW).
- **Low latency:** separate **wet (in-progress)** GPU layer from committed **dry** pixels (Krita/Procreate, patents 9529463/10388055); flatten wet→dry on pen-up; append only new dabs each frame. Input prediction 1–2 dabs ahead. Smoothing: basic / weighted EMA / stabilizer (windowed rope) / prediction (Krita's 4 modes).
- **libmypaint** ("brushlib", C, MIT-ish, reusable — used by MyPaint/Krita/GIMP/Pixelmator): Brush Engine (dynamics) + Surface abstraction. `stroke_to(surface,x,y,pressure,xtilt,ytilt,dtime)` per event (pass every event); you implement `draw_dab(...)` + `get_color(...)`. `.myb` format, input curves on pressure/speed/tilt/random/direction. Cleanest path: `bindgen` over libmypaint, implement `MyPaintSurface` on wgpu wet layer.
- **Timing:** coalesce ALL queued pointer samples per frame (tablets 100–240Hz > frame); feed every sample to dab generator, present at vsync. Use device `dtime`/timestamps (not frame count) for velocity dynamics. Architecture: input poll → lock-free queue → render thread arc-length walker → wgpu draw_dab → wet layer → composite → present.

Sources: github.com/rust-windowing/winit/issues/3833 · github.com/Fuzzyzilla/octotablet · github.com/mypaint/libmypaint/wiki · docs.krita.org (freehand_brush, mypaint_engine) · dl.acm.org/doi/fullHtml/10.1145/3641519.3657418 (Ciallo)

---

## 5. Undo, text, vector, AI, resize

- **Undo:** `undo` (Record linear / History tree), `undo_2` (returns command seq), `undoredo` (sparse deltas/memento). Memory: tile the canvas, record only **dirty tiles** as pre-edit Arc-COW copies; spill/compress cold history (lz4/zstd). Non-destructive: model doc as node-graph/command log — undoable state = graph structure + params (small), pixels re-derived & cached. **Study Graphite (graphite.rs)** — canonical Rust non-destructive node-graph editor. Hybrid: graph/command undo for structure+params, tile-COW for destructive pixel ops.
- **Text:** **`cosmic-text` 0.18.2** — shaping via HarfRust (pure-Rust, no C), raster via swash, bidi, built-in editable `Editor`/`Buffer` (cursor/selection/insert/delete) = text-layer foundation. **`glyphon`** bridges cosmic-text → wgpu (etagere atlas, LRU, renders in existing pass). FreeType/`harfbuzz_rs` only if specific hinting needed.
- **Vector:** **`kurbo` 0.11** (bezier math, arclength, nearest-point, offset — pen-tool model `BezPath`/`CubicBez`). **`lyon` 1.0** (Fill/Stroke tessellation → GPU mesh). Boolean ops: kurbo native WIP (#277) → use **`i_overlay`** (polygon union/intersect/diff; flatten beziers, lose exact curve fidelity — acceptable).
- **AI:** **`ort` 2.0-rc.12** (ONNX Runtime bindings, CUDA/CoreML/DirectML, default for native) vs **`candle` 0.9** (pure-Rust, wasm). Models (all ONNX): bg-removal **BiRefNet (MIT, best edges)** + U²-Net (Apache, fast preview); inpaint **LaMa**; super-res **Real-ESRGAN**. Vet weight licenses (SD weights carry OpenRAIL restrictions; runtime crates are MIT/Apache).
- **Resize:** **`fast_image_resize` 6.0** — SIMD (SSE4/AVX2/NEON/WASM), Lanczos3/Bicubic/Bilinear/Box/Nearest, premultiplied-correct, optional `image` integration. Lanczos3 downscale, Bicubic upscale; >2× upscale → Real-ESRGAN. Alt: `pic-scale`.

Sources: crates.io/crates/undo · docs.rs/undo_2 · github.com/mikwielgus/undoredo · github.com/pop-os/cosmic-text · github.com/grovesNL/glyphon · github.com/linebender/kurbo (+ #277) · docs.rs/lyon_tessellation · lib.rs/crates/i_overlay · github.com/pykeio/ort · github.com/huggingface/candle · github.com/Cykooz/fast_image_resize · graphite.rs

---

# Parity-expansion research (Phases 6–12)

Findings backing the new phases in [PLAN.md](./PLAN.md) §4b. Same caveat: verify crate versions
and model licenses at build time. wgpu confirmed still on **29.0.3** (2026-05-02); no 30/31 yet — the
plan's `"29"` pin holds.

## 6. Retouching & healing (clone / heal / patch / content-aware)

- **Clone Stamp** is the easy one: sample at an offset point, stamp brush dabs reading the source
  region — reuse the existing dab pipeline with a source-texture sampler and offset uniform. Aligned
  vs non-aligned = whether the offset resets per stroke. Sample current / below / all-merged = which
  composite the source reads from.
- **Healing Brush / Patch = gradient-domain (Poisson) blending.** The trick that makes healing look
  seamless: don't copy pixels, copy the *gradient field* of the source patch and solve a Poisson
  equation so the result matches the destination's boundary tone/color while keeping the source's
  texture. This is **Poisson Image Editing** (Pérez et al. 2003) / "seamless cloning"; OpenCV ships it
  as `seamlessClone` (NORMAL_CLONE / MIXED_CLONE). Solve ∇²f = div(guidance) over the masked region
  with Dirichlet boundary = destination, via a sparse linear solve (Jacobi/multigrid on GPU, or a CPU
  sparse solver `nalgebra`/`faer`). Solve only over the dirty rect. Spot-healing auto-picks the source
  from the surrounding ring.
- **Content-Aware Fill = PatchMatch** (Barnes et al. 2009): randomized nearest-neighbor patch
  correspondence that fills a hole by iteratively copying best-matching patches from a sampling region;
  fast, GPU-able, and — critically — **needs no trained model** and excels at high-resolution repeating
  texture. Modern hybrid (ECCV'22 "Guided PatchMatch", arXiv 2208.03552): run a deep model (LaMa) to
  hallucinate plausible structure, then guided-PatchMatch to synthesize at full resolution — beats
  either alone (users preferred it, +7.4 metric over LaMa) and avoids LaMa's blur/tiling at high res.
  So: ship PatchMatch as the always-available baseline; LaMa-ONNX (Phase 10) is an optional structure
  prior. Photoshop's "Remove tool" is the same idea behind a single brush.
- **Liquify = forward-mapping mesh warp.** Maintain a displacement grid (or full-res displacement
  texture); push/pull/twirl/pucker/bloat tools accumulate offsets under the brush; sample the source
  through the inverse displacement at composite time (GPU). Freeze mask protects regions; reconstruct
  blends back toward identity. Face-aware Liquify just drives those sliders from face landmarks (Ph10).

Sources: en.wikipedia.org/wiki/Gradient-domain_image_processing · Pérez/Gangnet/Blake "Poisson Image
Editing" (SIGGRAPH 2003) · docs.opencv.org (photo: seamlessClone) · Barnes et al. "PatchMatch"
(SIGGRAPH 2009) · arxiv.org/abs/2208.03552 (Guided PatchMatch, ECCV 2022) · medium.com PatchMatch-vs-AI-inpainting

## 7. Layer styles, smart objects, blend-if

- **Layer styles (FX)** are deterministic GPU passes computed from the layer's alpha/shape, layered
  in a fixed order (Photoshop's stacking: Drop Shadow behind, then Outer Glow, Stroke, the layer,
  Inner Shadow/Glow, overlays, Bevel, Satin). Each effect = a small pass: shadows/glows are a blurred,
  offset, colorized alpha; stroke is a distance-field/dilate of the alpha; overlays multiply a
  color/gradient/pattern inside the alpha; bevel/emboss derives a normal from the alpha's distance
  field and shades it. All cacheable per layer, re-run only when the layer or params change — fits the
  existing dirty-node compositor.
- **Smart objects** = embed a child document (or external reference) as a layer whose *rendered output*
  is what composites; transforms/filters apply to the render, not the source, so they stay editable
  ("edit contents" reopens the child). This is exactly the suite's render-graph node model — a smart
  object is a graph node that evaluates a linked document on demand and caches its tiles (same
  machinery as Dynamic Link in [SUITE.md](https://github.com/KwaminaWhyte/prism-suite-prism/blob/main/SUITE.md)). **Smart filters** = filters parented to a
  smart object's node, stored as params, re-evaluated on change, individually masked/toggled.
- **Blend-If / advanced blending**: per-layer "this layer" + "underlying" gray (and per-channel)
  range sliders that gate the layer's contribution by luma/channel value, with a split-slider feathered
  falloff — a cheap extra term in the composite shader. Plus fill-vs-opacity (fill excludes layer
  styles), knockout, and interior-effects-blend flags.
- **Channels**: the doc already stores RGBA in linear-premul; a channels panel exposes per-channel
  view/edit, plus named **alpha channels** = stored selection masks (R8/R16F, the same texture type as
  the selection mask) and **spot channels** for print. Store as extra mask layers in `prism-core`
  (app-agnostic). Split/merge channels = trivial reshuffle.

Sources: helpx.adobe.com/photoshop (layer effects, smart objects, advanced blending) · w3.org/TR/compositing-1 · [SUITE.md](https://github.com/KwaminaWhyte/prism-suite-prism/blob/main/SUITE.md) (render-graph node = Dynamic Link / smart object)

## 8. Filters, blur galleries, distortion, lens correction

- Structure all filters as a **`prism-fx` OpenFX-style pass registry** (suite-shared): each filter
  declares params + a GPU pass (compute preferred); runs destructively on a layer or as a Phase-7
  smart filter. OpenFX is the established host/plugin contract (also used by Pulse/Natron-style apps),
  so authoring once gives suite-wide reuse.
- **Blur gallery**: field/iris/tilt-shift = spatially-varying Gaussian driven by a control-point
  "blur map"; path/spin = directional/rotational sampling; **lens blur** = depth/alpha-weighted
  gather with a polygonal bokeh kernel (bright-spot bloom for highlights). Motion blur = directional
  convolution. All are gather passes over the f16 composite.
- **Distort**: analytic inverse-coordinate warps (pinch/spherize/polar/twirl/ripple/wave) sample the
  source through a remap; **Displace** uses a displacement map texture; **Warp** uses a Bézier/mesh
  grid (reuse Liquify's displacement machinery).
- **Lens Correction**: **`lensfun`** — a pure-Rust port of the LensFun project — corrects geometric
  distortion, transverse chromatic aberration, and vignetting from the LensFun camera/lens database,
  bit-exact-tested (1e-3) against the C++ reference. Pre-alpha API, so wrap it behind our own trait and
  keep a polynomial-model fallback (Adobe LCP-style coefficients) for unknown lenses. Feeds both the
  Lens Correction filter and the RAW develop module (Phase 9).
- **Render** (clouds/fibers/flare/lighting) = procedural noise (Perlin/Simplex) + analytic generators;
  **Stylize/Artistic** (emboss/find-edges/oil-paint/Filter-Gallery set) = convolution + edge operators
  + kuwahara-style smoothing. All standard, no external deps.

Sources: openfx.readthedocs.io · helpx.adobe.com/photoshop (blur gallery, filters, lens correction) ·
docs.rs/lensfun · github.com/lensfun/lensfun · lensfun.github.io/manual (corrections)

## 9. Pro color, RAW develop, PSD export

- **Color management**: `lcms2` 6.1 (Little CMS 2.17) is the primary engine — ICC v2/v4, CMYK/Lab/XYZ,
  rendering intents, soft-proof transforms; build a working-space abstraction in `prism-color`
  (sRGB/AdobeRGB/Display-P3/ProPhoto/linear) and do device-link transforms at import/export and for
  soft-proof preview (with gamut-warning overlay). `qcms` (pure-Rust, RGB+gray only) is the wasm fast
  path. CMYK mode + separations needs an N-channel doc path (extend `prism-core` color types,
  app-agnostic).
- **RAW / Camera Raw**: **`rawler`** (mature; demosaics Bayer via Malvar-He-Cutler, X-Trans via
  bilinear; parses metadata for CR2/CR3/NEF/ARW/RAF/DNG…) with **`rawloader`** as the broader-format
  sibling; **`zenraw`** is a safe-Rust alternative emitting scene-referred linear f32 (clean handoff
  into our linear pipeline). Develop module = scene-linear in → WB/exposure/tone-curve/HSL/detail/lens
  (lensfun)/grain → our working space; expose as a re-editable smart-object source so RAW edits stay
  non-destructive. The "Camera Raw filter" reuses the same control set as a filter on any layer.
- **PSD export**: no Rust crate writes PSD (the `psd` crate is read-only); hand-write a serializer
  against the documented Adobe PSD/PSB binary spec — layers, groups, masks, blend modes, opacity, text
  as layers, basic layer styles, merged composite for compatibility. Validate by export→reimport
  fidelity. Ship **`.ora`** (OpenRaster: zip + per-layer PNG + stack.xml — trivial) and **layered
  TIFF** first as easy faithful interchange while the PSD serializer matures.
- **Export As / Save for Web**: reuse the better-than-baseline encoders already in stack (`mozjpeg`,
  `webp`, `ravif`, `oxipng`) with per-format quality/size preview, multi-scale (@1x/@2x/@3x), metadata
  strip/keep, and profile embed.

Sources: github.com/kornelski/rust-lcms2 · github.com/FirefoxGraphics/qcms · lib.rs/crates/rawler ·
github.com/pedrocr/rawloader · github.com/imazen/zenraw · adobe PSD file format spec (the documented
binary format) · openraster.org · helpx.adobe.com/photoshop (export-as / save-for-web)

## 10. AI / neural tools (ort, models on demand)

- Runtime: **`ort` 2.0** (ONNX Runtime) with **CoreML** (macOS), **DirectML** (Windows), **CUDA/TensorRT**
  (NVIDIA) execution providers; **`candle`**/**`tract`** pure-Rust for wasm/no-EP fallback. Models are
  **not bundled** — fetch to a cache on first use behind a feature flag, surface each weight's license
  (segmentation/restoration weights are mostly MIT/Apache; diffusion weights carry OpenRAIL terms).
- **Select Subject / Object Select**: **SAM2 / SAM3** (Meta Segment Anything, prompt-driven by
  click/box; SAM3 added late-2025) for interactive object selection; **BiRefNet** for class-agnostic
  saliency. Output → editable selection/mask.
- **Remove Background / matting**: **BiRefNet_dynamic** (released 2025-03-31, trained 256²–2304²,
  robust at any resolution, best edge fidelity) or **RMBG-2.0 / BEN2** — all ONNX.
- **Inpaint / Remove**: **LaMa** (Fourier-conv inpainting) — feeds Phase-6 content-aware/Remove as the
  optional structure prior.
- **Super-resolution**: **Real-ESRGAN** (robust, widely used) or **SwinIR** (transformer, higher
  fidelity); tile with overlap-blend for large images; >2× upscale path.
- **Generative fill/expand**: optional and pluggable — local diffusion (`candle`/ONNX SD-class) *or* a
  user-configured cloud endpoint (BYO key). Behind a provider trait so local/cloud/none are swappable;
  never required for core editing.

Sources: github.com/pykeio/ort · github.com/ZhengPeng7/BiRefNet · github.com/1038lab/ComfyUI-RMBG
(RMBG-2.0/BEN2/SAM2/SAM3, v3.0.0 2026-01-01) · github.com/advimman/lama · github.com/xinntao/Real-ESRGAN ·
github.com/JingyunLiang/SwinIR · arxiv.org/abs/2503.14757 (real-time HR inpainting)

## 11. Automation, scripting, plugins

- **Actions** = a serialized list of recorded operations (each command already exists as a reversible
  `Command` in the undo system — record the same descriptors to a JSON action file). Sets, conditional
  steps, modal toggles, insert-stop/menu-item. **Batch** = run an action over a folder/open docs;
  **Image Processor** = bulk resize+format+ICC-convert; droplet = an action bundled as an executable
  drop target.
- **Scripting**: embed **`rhai`** (pure-Rust, sandboxed, easy to bind) exposing a document/layer/
  selection API; optional **`boa`** or **`deno_core`** JS engine to court Photoshop's ExtendScript/UXP
  script familiarity. Run-script + script-event hooks.
- **Plugins**: an **OpenFX-style** effect plugin host in the suite-shared `prism-fx` (stable C ABI or
  wasm-component plugins) so a filter authored once loads in Pigment, Contour, and Pulse. Panel/UI
  plugins are a later, larger surface.
- **Presets / assets**: brushes/gradients/patterns/styles/swatches/shapes/LUTs in a shared suite asset
  library; import Adobe formats where feasible (`.abr` brushes, `.grd` gradients, `.pat` patterns,
  `.asl` styles, `.aco` swatches, `.csh` shapes, `.cube`/`.3dl` LUTs), native format otherwise.

Sources: rhai.rs · github.com/boa-dev/boa · github.com/denoland/deno_core · openfx.readthedocs.io ·
helpx.adobe.com/photoshop (actions, scripting, batch, presets) · [SUITE.md](https://github.com/KwaminaWhyte/prism-suite-prism/blob/main/SUITE.md) (shared asset library, prism-fx)
