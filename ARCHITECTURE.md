# Pigment — Architecture (living doc)

Module/data-flow detail, filled in as code lands. High-level rationale lives in
[PLAN.md](./PLAN.md) §2; cited research in [RESEARCH.md](./RESEARCH.md).

## Current state: Phase 0

```
crates/
  pigment-core/   # GPU-agnostic document model
    geometry.rs   # Size, Rect
    color.rs      # Rgba + sRGB<->linear helpers (the gamma boundary)
    blend.rs      # BlendMode enum; stable shader_id() discriminants
    tile.rs       # TileCoord, Tile (256² RGBA f32, Arc-shared COW) — types only in P0
    layer.rs      # Layer, LayerKind {Raster|Group}, LayerTree (id alloc, add/get)
    document.rs   # Document {size, layers, active_layer}
  pigment-io/     # file <-> pixels
    lib.rs        # load_image() -> LoadedImage{size, rgba8}; placeholder(); supported exts
  pigment-app/    # eframe binary `pigment`
    main.rs       # NativeOptions (Wgpu renderer), run_native
    app.rs        # PigmentApp: eframe::App::ui — menu/tools/layers panels + central canvas;
                  #   pan/zoom input, fit, File→Open
    canvas.rs     # ViewTransform; CanvasRenderer (pipelines/uniform/sampler, in
                  #   callback_resources); CanvasPaint (egui_wgpu::CallbackTrait)
    shaders/canvas.wgsl  # shared vs_main + fs_checker + fs_image
```

### Frame data flow (Phase 0)
1. `PigmentApp::ui` lays out panels; central panel allocates the viewport `rect`.
2. Input: drag → `view.pan`; scroll → `view.zoom_to(anchor)`.
3. Compute `doc_rect` (document quad) in egui points from image size × zoom + pan.
4. Add `egui_wgpu::Callback` with `CanvasPaint { doc_rect, image }`.
5. `CanvasPaint::prepare` (has wgpu device/queue + ScreenDescriptor): lazily upload
   image texture on generation change; convert `doc_rect` points → clip space; write uniform.
6. `CanvasPaint::paint`: inside egui's render pass (scissored to `rect`), draw checker quad
   then image quad.

### Key decisions locked in P0
- eframe **wgpu** backend; share egui's `wgpu::Device` (no second GPU context).
- GPU resources live in egui `callback_resources` (outlive frames).
- View transform folded CPU-side into a clip-space quad → trivial shader.
- egui target is non-sRGB `Bgra8Unorm`; P0 passes sRGB-encoded bytes straight through
  (`Rgba8Unorm` sample). Real linear compositing → f16 offscreen + encode-at-blit in P1.

## Phase 1 (in progress) — GPU compositor + brush

`canvas.rs` replaced the Phase 0 single-quad renderer with a real compositor
(`CanvasGpu`, one egui callback resource):

- **Layers** — each raster layer is one canvas-sized `Rgba16Float` linear-premul
  texture (`GpuLayer`). Tiling/atlas is the next refinement (a layer is currently
  a degenerate single tile).
- **Brush** (`dab.wgsl`) — app's arc-length walker emits instanced `Dab`s in doc
  space; `paint_dabs` renders them into the active layer with premultiplied-over
  blending. `app.rs` maps screen→doc via `doc_rect` and brackets strokes with
  drag_started/stopped.
- **Compositor** (`composite.wgsl`) — ping-pong over two targets; each visible
  layer samples (backdrop, layer) and writes the blended result. Per-layer
  opacity/blend supplied via one uniform buffer with **dynamic offsets**
  (256-byte stride). Blend modes in-shader (Normal/Multiply/Screen/Overlay/
  Darken/Lighten/Add so far).
- **Display** (`display.wgsl`) — runs inside egui's pass; samples the final
  composite, composites over the checkerboard in linear light, sRGB-encodes to
  the non-sRGB egui target.

All compositor passes are recorded into egui's own `CommandEncoder` in
`prepare()` (no separate submit); `paint()` only issues the display draw.

Known shortcuts to revisit: recomposite-every-frame (no dirty tracking), no
wet-layer/pressure, no undo yet, no tile streaming. See PLAN.md §4 Phase 1.

### Save / load + CPU tools (added)
- **GPU access from the app** — `with_gpu(frame, |gpu, device, queue| …)` reaches
  the `CanvasGpu` callback resource + device/queue via `frame.wgpu_render_state()`,
  so the app can read back / upload textures outside the paint callback.
- **`.pigment`** (`pigment_io::document_file`) — `b"PIGMENT1"` magic + JSON
  `DocMeta` + per-layer lz4 RGBA16F blobs. Save = `read_layer` (copy_texture_to_buffer,
  256-aligned rows, mapped readback) per layer. Open = rebuild `LayerTree` from
  meta, stage pixels in `PendingUpload`, upload next frame.
- **Bucket fill** (`pigment_core::fill::flood_fill_mask`) — readback active layer →
  f32 → 4-connected (or global) flood vs seed within tolerance → write fill color →
  re-upload. Snapshots first (`begin_command_now`) for undo.
- **Eyedropper** — 1×1 `read_pixel`, unpremultiply, linear→sRGB.
- **Layers** — add / delete / reorder (vec swap) / inline rename; delete frees the
  GPU layer (`drop_layer`).

Both fill and eyedropper currently sample the **active layer only** (no
sample-all-layers yet).

## Next
Tile model + `TileCache` (replace full-canvas layer textures) + dirty-tile
invalidation; wet-layer + pen pressure; History panel.
