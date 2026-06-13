# Changelog

All notable changes to **Pigment** (the Prism suite's raster editor) are documented
here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project is pre-1.0, so versions are `0.x` milestones.

## [Unreleased]

### Added
- **Layer comps** (Layer power). The Layers panel gains a **Layer Comps** section
  where you can **Capture** a named snapshot of every layer's appearance —
  visibility, opacity, and blend mode — and later **Restore** it with one click,
  plus inline **rename** and **delete**. The capture/restore logic is a pure
  function over the layer list (`app::comps`), keyed by stable `LayerId`, so
  restoring is robust to layers being reordered, added, or removed since capture
  (added layers are left untouched; entries for removed layers are ignored).
  Position/transform is intentionally out of scope — Pigment layers carry no
  persistent position in their model (Move/Transform bakes into pixels), so a
  comp captures the appearance attributes that live on the layer. Comps persist
  in the `.pigment` document via a new additive `DocMeta.comps` field
  (`#[serde(default)]` + `skip_serializing_if`), so existing documents round-trip
  unchanged and comp-free documents stay byte-compatible. On load, saved layer
  ids are remapped to the freshly allocated ids. Tests cover capture→restore
  round-trips, restore reverting edits, robustness to add/remove/reorder, the
  runtime↔serde conversion with id remap, and the doc serde round-trip incl. the
  legacy (no-`comps`) case.

### Fixed
- **Default gradient background now shows on first load.** A freshly staged
  document (the startup default, plus any opened image/`.pigment`) came up as a
  blank/transparent canvas until the user's first edit. Root cause: the per-frame
  composite is recorded into egui's paint-callback (`prepare`) command encoder,
  which doesn't reliably execute while egui is **idle** immediately after load —
  so the uploaded layer pixels were never composited into the displayed buffer
  until an edit (which composites through a self-submitting path) forced it. (The
  layer texture was correct all along — verified by GPU read-back — but the
  composite output stayed transparent.) Staging a document now runs an initial
  composite through the **self-submitting** `composite_now` (its own encoder +
  `queue.submit`, the same path edit operations use) and builds the display bind
  group from it, so the document — including the sample gradient background — is
  presented on the very first frame. The pixel upload is also no longer consumed
  until the GPU state is actually ready (it retries next frame), so a not-yet-
  initialized render state can't silently drop the document. A headless test
  asserts the default background's upload buffer is a real, varying gradient.

## [0.3.0] - 2026-06-13

### Added
- **Font-family selection for Text layers** (Type richness). The Text layer
  panel gains a **font family** dropdown next to the text/size/color/align
  controls. It lists "Default" (the renderer's default sans-serif face) plus
  every family in the system font database, enumerated once via the shared
  `prism_io::text::available_families()` and cached (the DB scan is expensive).
  Picking a family sets the layer's `TextDef.family`, which changes the layer
  fingerprint and re-rasterizes the text in the chosen face through
  `render_text(..., family)`. Back-compatible: `family` defaults to `None`
  (`#[serde(default)]`), so existing text defs and `.pigment` documents
  round-trip unchanged. Tests cover default- and selected-family rasterization
  and the `TextDef` serde round-trip incl. the legacy (no-`family`) case.

### Fixed
- **Text layers keep their position when a property changes (font family, size,
  color, alignment).** Text/Vector layers carry no position in their definition
  — their pixels rasterize at the canvas origin — so a Move/Transform *bake*
  lived only in the layer's pixels. Any property edit re-rasterized at the origin
  and overwrote the whole texture, discarding the bake and snapping the text back
  to the top-left. The recently-added font-family change merely made the reset
  visible; the underlying defect was that re-rasterize ignored the layer's
  placement. The Move/Transform bake of a generated layer now records its
  accumulated translate (`gen_offset`, doc px), and `sync_generated_layers`
  re-applies that offset to the freshly-rasterized buffer (via the existing
  `reposition` helper) before upload — so *every* re-raster (family/size/color/
  align) preserves position, undoable as before. The placement is general (not
  font-family-specific) and needs no model change, so `.pigment` files round-trip
  unchanged. Tests: a pure unit test that a moved text layer keeps its position
  across a property/font change (and that a zero/absent offset is a no-op),
  exercising the exact buffer-placement seam without a GPU.
- **Move / Transform no longer snaps a layer back to its origin on release**
  (most visible with **Text** layers, which a user cannot reposition at all
  without this). The Move/Transform tools translate the active layer with a
  composite-time uv affine and *bake* it into the layer's pixels on pointer
  release. The drag-stop frame turns off `xform_active` *before* the frame's
  affine is computed, so the bake frame was sending `set_layer_transform(None)`
  — clearing the GPU's `xform_layer` so `bake_transform` returned early (a
  no-op) while the live preview affine was also dropped, snapping the layer back
  to where the drag started. The affine is now kept live for the bake frame too
  (`send_layer_xform(active, bake)`), so the move bakes and persists for every
  layer kind, undoable exactly as before. Tests: a pure unit test that the
  affine is still sent on the bake frame (and the translate→uv-offset mapping),
  plus a GPU regression that a baked move persists (gated on a GPU adapter,
  skip-on-no-adapter, so the default headless `cargo test` stays green). No
  model/serde change, so `.pigment` files round-trip unchanged.

## [0.2.0] - 2026-06-09

### Added
- **Render filter — Clouds & Difference Clouds** (Phase 8). A new pair of
  destructive *generator* filters wired through the exact existing filter pattern
  (GPU shader pass keyed by `kind` in `filter.wgsl` → `apply_clouds` on the
  compositor → `do_clouds` in the app → a new **Filter ▸ Render** submenu →
  tests), undoable (region-COW) like the existing blur / distort / stylize /
  noise / pixelate filters. Both paint the active layer with a deterministic
  multi-octave value-noise (fBm) field — a soft cloud texture — built on the same
  `hash21` lattice the noise/diffuse filters use, so it is **reproducible for a
  given seed** and reproduced bit-for-bit by the test-only `canvas::filter_math`
  CPU reference. **Clouds** (kind 25) fills the layer with the field (ignoring the
  source); **Difference Clouds** (kind 26) composites the field against the
  existing pixels via per-channel absolute difference (Photoshop-style), so
  repeated application folds the field and builds the characteristic veins. The
  fBm sums `octaves` value-noise layers (each doubling frequency, scaling
  amplitude by `roughness`), renormalised into `[0,1]`; controls expose **seed**,
  **scale** (base feature size px), **roughness** (per-octave falloff) and
  **octaves** under the new Render submenu. (Also tightened the shared CPU
  `hash21` to WGSL's floor-based `fract` so negative samples match the shader
  exactly.) Tests: the CPU reference module gains `value_noise` / `fbm` plus
  `clouds` / `difference_clouds` and **6 CPU unit tests** (determinism for a
  fixed seed; different seeds differ; output in range / opaque / gray; the field
  is spatially smooth, not white noise; difference-clouds = |base − noise|; the
  fold keeps transforming on repeat) — all pass under a normal `cargo test` — plus
  **1 GPU pixel test** gated on a GPU adapter (skip-on-no-adapter), so the default
  headless `cargo test` stays green. The new control fields are runtime UI state
  on the (non-serialized) app, so `.pigment` files round-trip unchanged.

## [0.1.0] - 2026-06-09

### Added
- **Sharpen filter — High Pass** (Phase 8). A new destructive filter wired
  through the exact existing filter pattern (GPU shader pass keyed by `kind` in
  `filter.wgsl` → `apply_high_pass` on the compositor → `do_high_pass` in the app
  → Filter ▸ Sharpen submenu → tests), undoable (region-COW) like the existing
  blur / distort / stylize / noise / pixelate filters, all in linear-premultiplied
  working space (the difference is taken on the unpremultiplied colour, then
  re-premultiplied, so a transparent edge doesn't bias it). **High Pass** (kind
  24) — the classic Photoshop sharpen prep: subtract a Gaussian-blurred copy from
  the original and re-centre at mid-gray, so flat areas go neutral gray (0.5) and
  only the high-frequency detail/edges survive as a signed deviation about 0.5.
  It reuses the existing separable Gaussian (kind 1, two passes) for the blur,
  saving the untouched source in the `ping` buffer, then a two-input combine pass
  (the filter bind group gains a back-compatible secondary texture at binding 3,
  aliased to the primary input for every single-input kind) subtracts the blur
  from the original. A `radius` (blur scale → coarser detail kept) and an
  `amount` (detail gain; 1 = identity high pass, 0 = flat mid-gray) control it,
  exposed under a new **Filter ▸ Sharpen** submenu (alongside the existing
  Sharpen). Tests: the test-only `canvas::filter_math` CPU reference module gains
  a separable `gaussian_blur` (matching the kind-1 shader weights bit-for-bit)
  and `high_pass` with **5 new deterministic unit tests** — Gaussian preserves a
  flat field, High Pass flattens locally-uniform areas to mid-gray while the edge
  carries detail, the two sides of an edge deviate in opposite directions about
  mid-gray (signed), amount 0 is a flat mid-gray field, and a larger amount
  scales the detail — plus **1 new headless-GPU pixel test**
  (`high_pass_flattens_flats_and_keeps_edges`) mirroring the existing GPU
  filter-test pattern (skip-on-no-adapter). App test count 109 → 115. No
  shared-crate changes (raster-only, pigment-app per PLAN §0a). *Still open
  (Sharpen family backlog):* Smart Sharpen, Unsharp Mask refinement.
- **Pixelate filters — Mosaic, Crystallize, Color Halftone, Mezzotint**
  (Phase 8). Four new destructive filters wired through the exact existing
  filter pattern (GPU shader pass keyed by `kind` in `filter.wgsl` → `apply_*`
  on the compositor → `do_*` in the app → Filter ▸ Pixelate menu → tests), all
  undoable (region-COW) like the existing blur / distort / stylize / noise
  filters, all in linear-premultiplied working space. The legacy **Pixelate**
  (kind 3, which point-samples each block's centre) is preserved and moved into
  the new submenu alongside the cell-based family.
  **Mosaic** (kind 20) — averages each `cell`×`cell` block to one colour (the
  true block mean, alpha-weighted in premultiplied space), so every pixel in a
  cell shares that average. **Crystallize** (kind 21) — Voronoi-like cells: each
  pixel snaps to the colour of its nearest jittered seed (one seed per
  `cell`×`cell` block, offset within the block by a hash of the block index +
  `seed`; the 3×3 block neighbourhood is searched so adjacent seeds can win),
  giving irregular polygons. It snaps to the seed's exact texel centre (a true
  snap to one source colour, never a blend) and is seeded-deterministic per the
  `diffuse` hash convention. **Color Halftone** (kind 22) — a per-channel dot
  screen: tile into `cell`-px cells rotated by a screen `angle` (with a 22.5°
  per-channel offset, CMY-rosette style); each cell's channel average sets a dot
  radius (darker channel → bigger dot of full ink, brighter → smaller), output
  binary per channel (full ink / paper). **Mezzotint** (kind 23) — a seeded
  threshold dither of Rec.709 luma against a per-pixel hashed threshold (biased
  by `amount`) to pure black/white grain, stable for a given seed. A new
  **Filter ▸ Pixelate** submenu hosts the legacy Pixelate plus the four with
  their controls (mosaic/crystallize cell size + crystallize seed; halftone cell
  + screen angle; mezzotint threshold + seed). Tests: the test-only
  `canvas::filter_math` CPU reference module gains the four filters (`mosaic`,
  `crystallize`, `color_halftone`, `mezzotint`) with **7 new deterministic unit
  tests** — mosaic cell = block average (each cell uniform = its mean) + flat
  identity, crystallize determinism (same seed ≡, different seed ≠) + cell
  snapping (output ⊂ source colours, never blended) + flat identity, color
  halftone dot grows as the cell darkens (more ink) + binary per channel,
  mezzotint binary + deterministic + brightness-tracking — plus **4 new
  headless-GPU pixel tests** (`mosaic_cell_is_uniform_block_average`,
  `crystallize_is_deterministic_and_snaps`,
  `color_halftone_dot_tracks_brightness`,
  `mezzotint_is_binary_and_tracks_brightness`) mirroring the existing GPU
  filter-test pattern. App test count 98 → 109. No shared-crate changes
  (raster-only, pigment-app per PLAN §0a). *Still open (Phase 8 pixelate
  backlog):* Fragment, Pointillize, selection-clipped pixelate, non-destructive
  smart-filter form.
- **Noise filters — Add Noise, Median, Dust & Scratches** (Phase 8). Three new
  destructive filters wired through the exact existing filter pattern (GPU
  shader pass keyed by `kind` in `filter.wgsl` → `apply_*` on the compositor →
  `do_*` in the app → Filter ▸ Noise menu → tests), all undoable (region-COW)
  like the existing blur / distort / stylize filters, all in
  linear-premultiplied working space (noise added to the unpremultiplied colour,
  then re-premultiplied, so a transparent edge doesn't bias the result).
  **Add Noise** (kind 17) — seeded-deterministic per-pixel noise, **gaussian**
  (Box–Muller) or **uniform** (a symmetric difference of two i.i.d. hashes so
  it's zero-mean: the raw `fract(sin)` hash is biased on a regular grid), with an
  `amount`, a **monochromatic** toggle (same noise on R/G/B), and a `seed`. It
  follows the `diffuse` hash philosophy exactly — stable for a given seed, no
  temporal randomness — and is **zero-mean so the channel average is preserved**.
  **Median** (kind 18) — per-channel median over a `(2·radius+1)²` window
  (despeckle / salt-pepper / impulse removal); radius param. **Dust &
  Scratches** (kind 19) — a thresholded median: a channel is replaced by the
  window median only when the original differs from it by more than the
  `threshold`, so specks are removed while sub-threshold detail is preserved.
  A new **Filter ▸ Noise** submenu hosts the three with their parameter
  controls (add-noise amount + gaussian/uniform + monochromatic + seed; median
  radius; dust threshold). Tests: the test-only `canvas::filter_math` CPU
  reference module gains the three filters (`add_noise`, `median`,
  `dust_and_scratches`) with **8 new deterministic unit tests** — add-noise
  amount-0 identity / determinism (same seed ≡, different seed ≠) / monochromatic
  equal-RGB-delta / mean-preserved (gaussian + uniform) + perturbation, median
  removes an impulse / radius grows the window, dust & scratches changes only
  above-threshold pixels / high-threshold identity — plus **3 new headless-GPU
  pixel tests** (`add_noise_is_deterministic_and_zero_mean` covering
  determinism + zero-mean + monochromatic R=G=B, `median_removes_an_impulse`,
  `dust_scratches_only_changes_above_threshold`) mirroring the existing GPU
  filter-test pattern. App test count 87 → 98. No shared-crate changes
  (raster-only, pigment-app per PLAN §0a). *Still open (Phase 8 noise backlog):*
  Reduce Noise, Despeckle, selection-clipped noise, non-destructive smart-filter
  form.
- **Distort filters — Twirl, Pinch/Spherize, Ripple/Wave, Polar Coordinates**
  (Phase 8). Four new destructive coordinate-displacement filters, wired through
  the exact existing filter pattern (GPU shader pass keyed by `kind` in
  `filter.wgsl` → `apply_distort` on the compositor → `do_*` in the app →
  Filter ▸ Distort menu → tests), all undoable (region-COW) like the existing
  blur / Gaussian / Sharpen / Pixelate filters. Each is a per-pixel
  coordinate-remap sampling filter (sample the source at a displaced coordinate,
  edge-clamp), working in pixel space about the canvas center.
  **Twirl** (kind 8) — rotates the image about the center by up to an `angle`,
  falling off quadratically to 0 at `radius` (untouched outside). **Pinch /
  Spherize** (kind 9) — a signed radial remap: positive pulls toward the center
  (pinch), negative pushes outward (spherize/bulge), with a smooth falloff to
  `radius`. **Ripple / Wave** (kind 10) — sinusoidal displacement where each axis
  is offset by a sine of the other, parameterised by `amplitude` (px) and
  `wavelength` (px). **Polar Coordinates** (kinds 11/12) — rectangular→polar
  (x = angle, y = radius) and the inverse polar→rectangular, an exact (modulo
  resampling) round-trip pair. A new **Filter ▸ Distort** submenu hosts the four
  with their parameter sliders (twirl angle + radius; pinch signed amount +
  radius; ripple amplitude + wavelength; polar rect↔polar toggle). All reuse the
  existing edge-clamped filter sampler and run in linear-premultiplied working
  space. Tests: the test-only `canvas::filter_math` CPU reference module gains
  the four remaps with **13 new deterministic unit tests** — twirl identity at
  angle 0 / outside-radius + center fixed / rotates a probe off its column /
  determinism, pinch identity at amount 0 / signed-opposite displacement /
  center fixed, ripple identity at amplitude 0 / wavelength periodicity /
  non-trivial warp, polar round-trip recovery + radius→rows mapping +
  determinism — plus **4 new headless-GPU pixel tests** (`twirl_rotates_within_
  radius`, `pinch_and_bulge_are_signed`, `ripple_is_periodic`,
  `polar_round_trips`) mirroring the existing GPU filter-test pattern. App test
  count 66 → 83. No shared-crate changes (raster-only, pigment-app per PLAN §0a).
  *Still open (Phase 8 distort backlog):* Warp (mesh + presets), Shear, Displace
  (displacement map), Lens Correction (`lensfun`), Adaptive Wide Angle,
  on-canvas center/handle UI, selection-clipped distort, non-destructive
  smart-filter form.
- **Blur family filters — Motion, Box, Radial** (Phase 8). Three new
  destructive blur filters, wired through the exact existing filter pattern
  (GPU shader pass keyed by `kind` in `filter.wgsl` → `apply_*` on the
  compositor → `do_*` in the app → Filter ▸ Blur menu → tests), all undoable
  (region-COW) like the existing Gaussian / Sharpen / Pixelate filters.
  **Motion Blur** (kind 4) — a directional/linear box average of `2·distance+1`
  taps along an `angle`, the classic streak blur (single pass). **Box Blur**
  (kind 5) — a flat, correctly-normalized kernel run separably (horizontal then
  vertical pass, reusing the Gaussian two-pass path) for a fast even blur.
  **Radial Blur** (kinds 6/7) — about the canvas center, in two modes:
  **Spin** (rotational, amount = degrees; smears tangentially) and **Zoom**
  (radial, amount = % ; smears toward/from the center), with a quality
  (sample-count) control; spin corrects for non-square pixels so the rotation
  is circular in pixel space. All operate in linear-premultiplied working space
  (averaging premultiplied samples is a correct blur) and reuse the existing
  edge-clamped filter sampler. A new **Filter ▸ Blur** submenu hosts the three
  with their parameter sliders (box radius; motion angle + distance; radial
  spin/zoom toggle + amount + samples). Tests: a new CPU reference module
  (`canvas::filter_math`, test-only) gives **12 deterministic unit tests** of
  the kernel math the shader implements — motion-blur axis-only smearing +
  energy conservation + radius-0 identity, box-blur separability/normalization
  (flat-field preserved, impulse → 3×3, two-axis equivalence, radius-0
  identity), radial spin-vs-zoom directionality + both-mode identity at amount 0
  + determinism — plus **3 new headless-GPU pixel tests** (`motion_blur_smears_
  along_angle`, `box_blur_normalizes_and_spreads`, `radial_spin_vs_zoom`)
  mirroring the existing GPU filter-test pattern. App test count 51 → 66.
  *Still open (Phase 8 blur backlog):* Surface Blur, Smart Blur, the Blur
  Gallery (Field/Iris/Tilt-Shift/Path/Lens), on-canvas radial-center handle,
  selection-clipped blur, non-destructive smart-filter form.
- **Gradient editor / gradient fill** (Phase 4 completion). The gradient tool is
  now a full multi-stop gradient editor: an independent **color rail** and
  **opacity rail** (Photoshop's two-rail model — add/remove/position color stops
  and opacity stops separately), all five **gradient geometries** (Linear,
  Radial, Angle, Reflected, Diamond), and **ordered dithering** to suppress 8-bit
  banding. A drag defines the gradient axis (`start→end`) which every geometry
  reinterprets (radial uses the drag length as the radius, diamond as the
  half-extent, angle as the reference direction, etc.); a **"fill layer"** toggle
  + button fills the whole layer across the canvas without dragging. Fills are
  clipped to the active selection and composited source-over onto the active
  layer (region-COW undo, like every other fill). Built-in **presets**
  (Foreground→Transparent, Black→White, Spectrum, Sunset) seed the editor. The
  gradient sampling/rasterization/dither math lives in the shared, app-agnostic
  `prism_core::gradient` (multi-stop interpolation in the working/linear space,
  premultiplied output matching `shape.rs`); the app converts the editor's sRGB
  stops to linear at fill time. Because the fill writes pixels directly (the
  established CPU fill path), gradient fills **persist to `.pigment`** as layer
  pixels with no format change; the shared `Gradient` type is also serde-ready
  (serialized round-trip tested in prism-io) so saved gradient presets/fills can
  be embedded later. Tests: 18 new `prism-core` unit tests (stop interpolation in
  the working space incl. multi-stop and unsorted stops, independent opacity
  rail, each geometry's parameterization, seeded/deterministic dither + presence
  + average-preservation, premultiplied render, id/zero-dim edge cases), 1 new
  prism-io serde round-trip, 3 new app editor tests (sRGB→linear conversion,
  preset load), and 1 new headless-GPU pixel test (read→render→upload gradient
  fill across a real f16 layer). *Still open:* on-canvas draggable stop handles,
  per-effect blend/reverse, noise gradients, `.grd` import.
- **Adjustment layers persist to `.pigment`** (Phase 7). Closes a known
  data-loss gap: adjustment layers were runtime-only — saving and reopening a
  document silently dropped every adjustment layer's parameters (and the
  adjustment layer itself reloaded as a blank raster layer). They now round-trip
  losslessly for **every** adjustment kind (Brightness/Contrast, Levels,
  Hue/Saturation, Exposure, Invert, Threshold, Black & White, Curves, Vibrance,
  Photo Filter, Posterize, Gradient Map, Color Balance, Channel Mixer — kinds up
  to 14). Follows the proven layer-styles persistence pattern: the `.pigment`
  doc model (`prism-io::document_file`) gains an optional per-layer `adjustment`
  payload (`Option<prism_core::Adjustment>`), reusing the shared `Adjustment`
  enum's own serde derive verbatim so the kind + every param (including Curves'
  variable-length per-channel control points and Channel Mixer's 3×4 matrix)
  serialize unchanged. The field uses `#[serde(default)]` + `skip_serializing_if`
  so **old documents (no `adjustment` key) still load** (as raster, as before)
  and non-adjustment layers stay byte-compact. On save, each
  `LayerKind::Adjustment` is written to `LayerMeta.adjustment`; on open, layers
  with that payload are rebuilt as adjustment layers and the existing recomposite
  path (`sync_curve_luts` / compositor params) rebuilds their LUTs/matrices so
  the restored adjustment renders immediately. Save→load round-trip unit-tested
  for Curves, Color Balance, and Channel Mixer (kind + every param), plus a
  raster back-compat case; prism-io adds a full-payload serde round-trip + an
  old-doc back-compat test. This was the last open item on the Phase-7
  adjustment work.
- **Adjustments: Color Balance + Channel Mixer** (Phase 7). Two more
  non-destructive adjustment layers, wired end-to-end through the established
  pattern (model variant in `prism-core::adjust` → composite-shader kind →
  inspector UI → GPU/unit tests). **Color Balance** (shader kind 13) applies a
  per-tonal-range RGB push — independent `cyan↔red / magenta↔green / yellow↔blue`
  sliders for Shadows / Midtones / Highlights, plus a *preserve luminosity*
  toggle. Because each output channel depends only on that same input channel, it
  rasterizes to a per-channel transfer LUT (reusing the Curves/Gradient-Map LUT
  texture + `curve_luts` slot), built CPU-side by `ColorBalanceLuts::build`
  (shadows weight darks, highlights weight lights, midtones a bell at 0.5).
  **Channel Mixer** (shader kind 14) computes each output channel as a linear mix
  of all input channels plus a constant (`[from_r, from_g, from_b, const]` per
  output), with a *monochrome* mode that collapses to a single weighted gray —
  output mixes all inputs, so it can't use a 1-D LUT and instead rides a small
  3-row matrix added to the compositor params (`CompositeParams` grew from 352 to
  400 bytes, still within the 512-byte `PARAMS_STRIDE` slot). New kinds appear
  automatically in the Add-Adjustment menu (it iterates `Adjustment::defaults()`).
  Tests: 7 new `prism-core` unit tests (LUT identity/shadow/highlight weighting,
  mixer swap/monochrome/clamp, encode kind+name stability) and 2 new
  headless-GPU pixel tests (shadow red-push lifts red; red↔blue mixer swap turns
  red into blue). *Still open:* Selective Color, multi-stop Gradient Map, Color
  Lookup (`.cube`/`.3dl` LUT), Shadows/Highlights, HDR Toning, Equalize,
  Replace/Match Color. (Adjustment params — including Color Balance / Channel
  Mixer — now persist to the `.pigment` doc; see the adjustment-persistence
  entry above.)
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
