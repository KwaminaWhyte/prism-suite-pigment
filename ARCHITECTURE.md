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

## Next: Phase 1 (planned)
Tile model on GPU (atlas + page table), compositor over dirty tiles (f16 linear,
premultiplied), layers panel wired to real layers, brush engine v1 (wet layer),
undo command stack. See PLAN.md §4 Phase 1.
