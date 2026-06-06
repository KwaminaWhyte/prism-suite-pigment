# The Prism Suite — an open source creative suite

Pigment is app #1 of a planned **four-app suite** that works together the way
Adobe's Creative Cloud apps do (Dynamic Link, smart objects, shared color and
assets). The shared engine crates Pigment is built on (`pigment-core`,
`pigment-gpu` compositor, color management, render graph) are designed from the
start to be reused across all four.

> Names below are provisional. The umbrella is **Prism** (light → pigments).

## The four apps

| # | App | Adobe analog | Domain | Open source we build on / fork from |
|---|-----|--------------|--------|--------------------------------------|
| 1 | **Pigment** | Photoshop | Raster image editing | greenfield (Rust/wgpu); ideas from Krita/mypaint |
| 2 | **Contour** | Illustrator | Vector graphics | `kurbo` + `lyon` + `i_overlay`; ideas from Inkscape |
| 3 | **Reel** | Premiere Pro | Video editing (NLE) | FFmpeg, MLT-style engine; ideas from Kdenlive |
| 4 | **Pulse** | After Effects | Motion graphics / VFX / compositing | OpenFX, OpenColorIO, OpenEXR; ideas from Natron |

## Why a suite, not four apps

Adobe's real moat isn't any single app — it's that they **interoperate**: a
shape pasted from Illustrator stays editable, an After Effects comp drops into a
Premiere timeline via Dynamic Link and updates live, everything shares the same
color. We get this for free *if* the shared layer/compositor/color engine is one
codebase. That is the whole architectural bet.

## Shared foundation (one engine, four apps)

```
                ┌───────────────────────────────────────────┐
   Pigment ─┐   │  prism-core    layer/scene graph, tiles,   │
   Contour ─┤   │                command/undo, doc model     │
   Reel    ─┼──▶│  prism-gpu     wgpu render graph,           │
   Pulse   ─┘   │                blend/effect passes, tiles   │
                │  prism-color   linear-light, ICC/OCIO, CMYK │
                │  prism-media   FFmpeg decode/encode, audio  │
                │  prism-fx      OpenFX-style effect plugins   │
                │  prism-io      file formats + interchange    │
                └───────────────────────────────────────────┘
```

Today these live as `pigment-core` / `pigment-gpu` / etc. When app #2 starts,
the reusable parts get promoted to `prism-*` crates and Pigment depends on them.
The render graph, tile model, blend math, and color pipeline are **identical**
needs for raster, vector raster preview, video frames, and comp layers.

## Interop mechanisms (the Adobe-parity features)

1. **Dynamic Link** — a Pulse comp referenced in a Reel timeline renders live;
   editing the comp updates the edit. Same for a Contour artboard placed in
   Pigment. Implemented as a shared render-graph node that evaluates the linked
   document on demand (cached per frame/tile).
2. **Smart objects / live placement** — place a `.contour` vector doc or a
   `.pigment` doc inside another app; it stays editable at its source resolution,
   re-rasterized on transform.
3. **Common interchange format** — a `prism-doc` container (layer tree + scene
   graph + media refs) that every app reads. Lossy-but-faithful import/export to
   PSD/AI/SVG/Premiere XML/AEP-ish as bridges to Adobe.
4. **Shared clipboard** — copy a path, layer, keyframe, or color in one app,
   paste editable in another (shared in-memory model + serialized fallback).
5. **One color pipeline** — `prism-color` (linear-light, ICC + OpenColorIO)
   means a color/look is identical across all four apps and on export.
6. **Shared effects** — `prism-fx` (OpenFX-style) effects run in any app that
   composites: blurs, grades, distortions authored once.
7. **Shared asset library** — brushes, gradients, LUTs, fonts, templates in a
   common store all apps see.

## Suite roadmap (high level)

- **Now:** Pigment to MVP/Beta (see [PLAN.md](./PLAN.md)). Keep engine crates
  app-agnostic so promotion to `prism-*` is mechanical.
- **Next:** promote shared crates; start **Contour** (vector) — closest to
  Pigment, reuses the compositor + color + render graph directly.
- **Then:** **Pulse** (motion/VFX) — adds time/keyframes + OpenFX to the same
  render graph; OpenColorIO/OpenEXR for film-grade color.
- **Then:** **Reel** (NLE) — adds FFmpeg media engine + audio + timeline; Dynamic
  Link to Pulse comps closes the suite loop.

Each app is independently useful; the value compounds as interop lands.

*Foundations are free. The product is the polish — and the glue between apps.*
