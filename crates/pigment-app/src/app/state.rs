use super::*;

impl PigmentApp {
    /// Re-rasterize any text/vector layer whose definition changed, uploading
    /// the result to its layer texture (keeps them editable after creation).
    /// Build + upload the GPU tone-curve LUT for every Curves adjustment layer.
    /// Cheap (a few 256-entry tables) and only called on a recomposite frame.
    pub(crate) fn sync_curve_luts(&mut self, frame: &mut eframe::Frame) {
        let curves: Vec<(LayerId, CurvePoints)> = self
            .doc
            .layers
            .layers
            .iter()
            .filter_map(|l| match &l.kind {
                LayerKind::Adjustment(Adjustment::Curves(cp)) => Some((l.id, cp.clone())),
                _ => None,
            })
            .collect();
        let gradmaps: Vec<(LayerId, [f32; 3], [f32; 3])> = self
            .doc
            .layers
            .layers
            .iter()
            .filter_map(|l| match &l.kind {
                LayerKind::Adjustment(Adjustment::GradientMap { low, high }) => {
                    Some((l.id, *low, *high))
                }
                _ => None,
            })
            .collect();
        #[allow(clippy::type_complexity)]
        let balances: Vec<(LayerId, [f32; 3], [f32; 3], [f32; 3])> = self
            .doc
            .layers
            .layers
            .iter()
            .filter_map(|l| match &l.kind {
                LayerKind::Adjustment(Adjustment::ColorBalance {
                    shadows,
                    midtones,
                    highlights,
                    ..
                }) => Some((l.id, *shadows, *midtones, *highlights)),
                _ => None,
            })
            .collect();
        if curves.is_empty() && gradmaps.is_empty() && balances.is_empty() {
            return;
        }
        with_gpu(frame, |gpu, device, queue| {
            for (id, cp) in &curves {
                gpu.set_curve_lut(device, queue, *id, &cp.rgb, &cp.r, &cp.g, &cp.b);
            }
            for (id, low, high) in &gradmaps {
                gpu.set_gradient_lut(device, queue, *id, *low, *high);
            }
            for (id, sh, mid, hi) in &balances {
                gpu.set_color_balance_lut(device, queue, *id, *sh, *mid, *hi);
            }
        });
    }

    pub(crate) fn sync_generated_layers(&mut self, frame: &mut eframe::Frame) {
        use std::hash::{Hash, Hasher};
        let (w, h) = (self.doc.size.width, self.doc.size.height);
        let mut jobs: Vec<(LayerId, Vec<u8>)> = Vec::new();
        for l in &self.doc.layers.layers {
            let fp = match &l.kind {
                LayerKind::Text(t) => {
                    let mut hsh = std::collections::hash_map::DefaultHasher::new();
                    t.text.hash(&mut hsh);
                    t.font_px.to_bits().hash(&mut hsh);
                    t.align.hash(&mut hsh);
                    t.family.hash(&mut hsh);
                    for c in t.color {
                        c.to_bits().hash(&mut hsh);
                    }
                    (w, h).hash(&mut hsh);
                    hsh.finish()
                }
                LayerKind::Vector(v) => {
                    let mut hsh = std::collections::hash_map::DefaultHasher::new();
                    v.kind.hash(&mut hsh);
                    for x in v.rect {
                        x.to_bits().hash(&mut hsh);
                    }
                    for c in v.color {
                        c.to_bits().hash(&mut hsh);
                    }
                    (w, h).hash(&mut hsh);
                    hsh.finish()
                }
                _ => continue,
            };
            if self.gen_fp.get(&l.id) == Some(&fp) {
                continue;
            }
            self.gen_fp.insert(l.id, fp);
            let px = match &l.kind {
                LayerKind::Text(t) => {
                    let align = match t.align {
                        1 => TextAlign::Center,
                        2 => TextAlign::Right,
                        _ => TextAlign::Left,
                    };
                    text::render_text(&t.text, t.font_px, t.color, w, h, align, t.family.as_deref())
                }
                LayerKind::Vector(v) => {
                    let kind = if v.kind == 1 {
                        ShapeKind::Ellipse
                    } else {
                        ShapeKind::Rectangle
                    };
                    let c = v.color;
                    let lin = [
                        srgb_to_linear(c[0]),
                        srgb_to_linear(c[1]),
                        srgb_to_linear(c[2]),
                        c[3],
                    ];
                    shape::fill_shape(kind, v.rect, lin, w, h)
                }
                _ => continue,
            };
            // Generated pixels rasterize at the canvas origin; re-apply any
            // baked Move/Transform translate so a property edit keeps the layer
            // where the user placed it instead of snapping to the top-left.
            let px = place_generated(px, w, h, self.gen_offset.get(&l.id).copied());
            jobs.push((l.id, f32_to_f16_bytes(&px)));
        }
        if jobs.is_empty() {
            return;
        }
        with_gpu(frame, |gpu, device, queue| {
            for (id, bytes) in &jobs {
                gpu.ensure_layer(device, *id);
                gpu.upload_layer(queue, *id, bytes);
            }
        });
        self.force_composite = true;
    }

    pub(crate) fn refresh_histogram(&mut self, frame: &mut eframe::Frame) {
        let order = self.layer_order();
        let comp = with_gpu(frame, |gpu, d, q| {
            let p = gpu.composite_now(d, q, &order);
            gpu.read_composite(d, q, p)
        })
        .flatten();
        self.force_composite = true;
        if let Some(b) = comp {
            self.hist = Some(histogram::histogram(&f16_bytes_to_f32(&b), 256));
        }
    }

    /// Grow the current stroke's dirty bounding box by a dab at `p` of radius `r`.
    pub(crate) fn expand_dirty(&mut self, p: egui::Vec2, r: f32) {
        let lo = egui::vec2(p.x - r, p.y - r);
        let hi = egui::vec2(p.x + r, p.y + r);
        self.stroke_dirty = Some(match self.stroke_dirty {
            None => (lo, hi),
            Some((a, b)) => (
                egui::vec2(a.x.min(lo.x), a.y.min(lo.y)),
                egui::vec2(b.x.max(hi.x), b.y.max(hi.y)),
            ),
        });
    }

    /// The stroke's dirty box as a clamped `[x, y, w, h]` (whole canvas if unset).
    pub(crate) fn dirty_rect(&self) -> [u32; 4] {
        let (w, h) = (self.doc.size.width, self.doc.size.height);
        match self.stroke_dirty {
            Some((mn, mx)) => {
                let x = mn.x.floor().clamp(0.0, w as f32) as u32;
                let y = mn.y.floor().clamp(0.0, h as f32) as u32;
                let rw = ((mx.x.ceil().clamp(0.0, w as f32) as u32).saturating_sub(x)).max(1);
                let rh = ((mx.y.ceil().clamp(0.0, h as f32) as u32).saturating_sub(y)).max(1);
                [x, y, rw, rh]
            }
            None => [0, 0, w, h],
        }
    }

    pub(crate) fn layer_fingerprint(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.active_id().0.hash(&mut h);
        for l in &self.doc.layers.layers {
            l.id.0.hash(&mut h);
            l.visible.hash(&mut h);
            l.blend.shader_id().hash(&mut h);
            l.opacity.to_bits().hash(&mut h);
            if let LayerKind::Adjustment(a) = &l.kind {
                let (k, p) = a.encode();
                k.hash(&mut h);
                for v in p {
                    v.to_bits().hash(&mut h);
                }
                // encode() can't carry Curves control points — hash them directly.
                if let Adjustment::Curves(cp) = a {
                    for ch in [&cp.rgb, &cp.r, &cp.g, &cp.b] {
                        ch.len().hash(&mut h);
                        for (x, y) in ch {
                            x.to_bits().hash(&mut h);
                            y.to_bits().hash(&mut h);
                        }
                    }
                }
                if let Adjustment::GradientMap { low, high } = a {
                    for v in low.iter().chain(high.iter()) {
                        v.to_bits().hash(&mut h);
                    }
                }
                if let Adjustment::ColorBalance {
                    shadows,
                    midtones,
                    highlights,
                    preserve_luminosity,
                } = a
                {
                    for v in shadows.iter().chain(midtones.iter()).chain(highlights.iter()) {
                        v.to_bits().hash(&mut h);
                    }
                    preserve_luminosity.hash(&mut h);
                }
                if let Adjustment::ChannelMixer {
                    r,
                    g,
                    b,
                    monochrome,
                } = a
                {
                    for v in r.iter().chain(g.iter()).chain(b.iter()) {
                        v.to_bits().hash(&mut h);
                    }
                    monochrome.hash(&mut h);
                }
            }
            if let Some(bi) = self.blend_if.get(&l.id) {
                for v in bi {
                    v.to_bits().hash(&mut h);
                }
            }
            self.clipped_layers.contains(&l.id).hash(&mut h);
            if let Some((c, w)) = self.layer_strokes.get(&l.id) {
                for v in c {
                    v.to_bits().hash(&mut h);
                }
                w.to_bits().hash(&mut h);
            }
            if let Some((c, o, b)) = self.layer_shadows.get(&l.id) {
                for v in c.iter().chain(o.iter()).chain(std::iter::once(b)) {
                    v.to_bits().hash(&mut h);
                }
            }
            if let Some(c) = self.layer_overlays.get(&l.id) {
                for v in c {
                    v.to_bits().hash(&mut h);
                }
            }
            if let Some((c, o, b)) = self.layer_inner_shadows.get(&l.id) {
                for v in c.iter().chain(o.iter()).chain(std::iter::once(b)) {
                    v.to_bits().hash(&mut h);
                }
            }
            if let Some((c, sz)) = self.layer_outer_glows.get(&l.id) {
                for v in c.iter().chain(std::iter::once(sz)) {
                    v.to_bits().hash(&mut h);
                }
            }
            if let Some((c, sz)) = self.layer_inner_glows.get(&l.id) {
                for v in c.iter().chain(std::iter::once(sz)) {
                    v.to_bits().hash(&mut h);
                }
            }
            if let Some((a, b, ang, op)) = self.layer_grad_overlays.get(&l.id) {
                for v in a.iter().chain(b.iter()).chain([ang, op]) {
                    v.to_bits().hash(&mut h);
                }
            }
            if let Some((hi, sh, sz, soft, ang, alt)) = self.layer_bevels.get(&l.id) {
                for v in hi.iter().chain(sh.iter()).chain([sz, soft, ang, alt]) {
                    v.to_bits().hash(&mut h);
                }
            }
        }
        h.finish()
    }

    pub(crate) fn active_id(&self) -> LayerId {
        self.doc.active_layer.unwrap_or(self.background_id)
    }

    /// Whether `id` is a generated (text/vector) layer — one whose pixels are
    /// re-rasterized from a definition rather than painted. These carry no
    /// position field, so their placement is tracked in `gen_offset`.
    pub(crate) fn is_generated_layer(&self, id: LayerId) -> bool {
        self.doc
            .layers
            .layers
            .iter()
            .any(|l| l.id == id && matches!(l.kind, LayerKind::Text(_) | LayerKind::Vector(_)))
    }

    pub(crate) fn layer_order(&self) -> Vec<LayerDraw> {
        self.doc
            .layers
            .layers
            .iter()
            .map(|l| {
                let (adjust_kind, adjust) = match &l.kind {
                    LayerKind::Adjustment(a) => a.encode(),
                    _ => (0, [0.0; 4]),
                };
                // Channel-Mixer matrix (identity unless this is a mixer layer).
                let mixer = match &l.kind {
                    LayerKind::Adjustment(a) => a.channel_mixer_matrix(),
                    _ => None,
                };
                let (mix_r, mix_g, mix_b) = mixer
                    .map(|m| (m.r, m.g, m.b))
                    .unwrap_or(([1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]));
                let blend_if = self.blend_if.get(&l.id).copied();
                let stroke = self.layer_strokes.get(&l.id);
                let shadow = self.layer_shadows.get(&l.id);
                let overlay = self.layer_overlays.get(&l.id).copied();
                let inner_shadow = self.layer_inner_shadows.get(&l.id);
                let outer_glow = self.layer_outer_glows.get(&l.id);
                let inner_glow = self.layer_inner_glows.get(&l.id);
                let grad = self.layer_grad_overlays.get(&l.id);
                let bevel = self.layer_bevels.get(&l.id);
                let cw = self.doc.size.width.max(1) as f32;
                LayerDraw {
                    id: l.id,
                    opacity: l.opacity,
                    blend: l.blend.shader_id(),
                    visible: l.visible,
                    adjust_kind,
                    adjust,
                    mix_r,
                    mix_g,
                    mix_b,
                    has_blend_if: blend_if.is_some(),
                    blend_if: blend_if.unwrap_or([0.0, 1.0, 0.0, 1.0]),
                    clipped: self.clipped_layers.contains(&l.id),
                    has_stroke: stroke.is_some(),
                    stroke_color: stroke.map(|(c, _)| *c).unwrap_or([0.0; 4]),
                    stroke_width: stroke.map(|(_, w)| *w / cw).unwrap_or(0.0),
                    has_shadow: shadow.is_some(),
                    shadow_color: shadow.map(|(c, _, _)| *c).unwrap_or([0.0; 4]),
                    shadow_offset: shadow
                        .map(|(_, o, _)| [o[0] / cw, o[1] / cw])
                        .unwrap_or([0.0; 2]),
                    shadow_blur: shadow.map(|(_, _, b)| *b / cw).unwrap_or(0.0),
                    has_overlay: overlay.is_some(),
                    overlay_color: overlay.unwrap_or([0.0; 4]),
                    has_inner_shadow: inner_shadow.is_some(),
                    inner_shadow_color: inner_shadow.map(|(c, _, _)| *c).unwrap_or([0.0; 4]),
                    inner_shadow_offset: inner_shadow
                        .map(|(_, o, _)| [o[0] / cw, o[1] / cw])
                        .unwrap_or([0.0; 2]),
                    inner_shadow_blur: inner_shadow.map(|(_, _, b)| *b / cw).unwrap_or(0.0),
                    has_outer_glow: outer_glow.is_some(),
                    outer_glow_color: outer_glow.map(|(c, _)| *c).unwrap_or([0.0; 4]),
                    outer_glow_size: outer_glow.map(|(_, sz)| *sz / cw).unwrap_or(0.0),
                    has_inner_glow: inner_glow.is_some(),
                    inner_glow_color: inner_glow.map(|(c, _)| *c).unwrap_or([0.0; 4]),
                    inner_glow_size: inner_glow.map(|(_, sz)| *sz / cw).unwrap_or(0.0),
                    has_grad_overlay: grad.is_some(),
                    grad_color0: grad.map(|(a, _, _, _)| *a).unwrap_or([0.0; 4]),
                    grad_color1: grad.map(|(_, b, _, _)| *b).unwrap_or([0.0; 4]),
                    grad_angle: grad.map(|(_, _, a, _)| a.to_radians()).unwrap_or(0.0),
                    grad_opacity: grad.map(|(_, _, _, o)| *o).unwrap_or(0.0),
                    has_bevel: bevel.is_some(),
                    bevel_highlight: bevel.map(|(hi, _, _, _, _, _)| *hi).unwrap_or([0.0; 4]),
                    bevel_shadow: bevel.map(|(_, sh, _, _, _, _)| *sh).unwrap_or([0.0; 4]),
                    bevel_size: bevel.map(|(_, _, sz, _, _, _)| *sz / cw).unwrap_or(0.0),
                    bevel_soften: bevel.map(|(_, _, _, so, _, _)| *so / cw).unwrap_or(0.0),
                    bevel_angle: bevel.map(|(_, _, _, _, a, _)| a.to_radians()).unwrap_or(0.0),
                    bevel_altitude: bevel.map(|(_, _, _, _, _, al)| al.to_radians()).unwrap_or(0.0),
                }
            })
            .collect()
    }

    pub(crate) fn fit(&mut self, viewport: egui::Rect) {
        let img = egui::vec2(self.doc.size.width as f32, self.doc.size.height as f32);
        if img.x <= 0.0 || img.y <= 0.0 {
            return;
        }
        let scale = (viewport.width() / img.x).min(viewport.height() / img.y) * 0.95;
        self.view = ViewTransform {
            pan: egui::Vec2::ZERO,
            zoom: scale.clamp(0.02, 64.0),
        };
    }

    pub(crate) fn dab_at(&self, doc_pos: egui::Vec2, alpha: f32, size_scale: f32) -> Dab {
        let c = self.brush_color;
        Dab {
            center: [doc_pos.x, doc_pos.y],
            radius: (self.brush_size * 0.5 * size_scale).max(0.5),
            hardness: self.brush_hardness.clamp(0.0, 0.99),
            color: [
                srgb_to_linear(c.r() as f32 / 255.0),
                srgb_to_linear(c.g() as f32 / 255.0),
                srgb_to_linear(c.b() as f32 / 255.0),
                alpha,
            ],
        }
    }
}

/// Shift a freshly-rasterized generated layer (full-canvas RGBA-f32) by its
/// accumulated placement `offset` (doc px). Generated layers rasterize at the
/// canvas origin, so this re-applies the translate a Move/Transform bake left in
/// the layer's pixels — keeping moved text/shapes in place across re-rasters.
/// `None` or a zero offset returns the buffer unchanged (no copy).
fn place_generated(px: Vec<f32>, w: u32, h: u32, offset: Option<egui::Vec2>) -> Vec<f32> {
    let Some(off) = offset.filter(|o| *o != egui::Vec2::ZERO) else {
        return px;
    };
    // `reposition` samples src at (x + ox, y + oy); to move content by +off we
    // read from −off (rounded to whole pixels, matching the bake's integer copy).
    reposition(&px, w, h, w, h, -off.x.round() as i32, -off.y.round() as i32)
}

#[cfg(test)]
mod text_family_tests {
    use super::*;

    /// Rasterize a `TextDef` the way `sync_generated_layers` does: map `align`
    /// and forward `family` into `render_text`. Pure (no GPU), so it exercises
    /// the same family-routing logic without a device.
    fn rasterize(t: &TextDef, w: u32, h: u32) -> Vec<f32> {
        let align = match t.align {
            1 => TextAlign::Center,
            2 => TextAlign::Right,
            _ => TextAlign::Left,
        };
        text::render_text(&t.text, t.font_px, t.color, w, h, align, t.family.as_deref())
    }

    /// A text layer with no family renders via the default face and still draws.
    #[test]
    fn default_family_renders() {
        let t = TextDef {
            text: "Ag".into(),
            ..TextDef::default()
        };
        let buf = rasterize(&t, 96, 64);
        let lit = buf.iter().skip(3).step_by(4).filter(|&&a| a > 0.1).count();
        assert!(lit > 0, "default-family text should rasterize glyphs");
    }

    /// Selecting an available system family is honored: the layer rasterizes in
    /// that face and the family survives the doc model's serde round-trip
    /// (legacy docs without the key deserialize to `None`).
    #[test]
    fn selected_family_rasterizes_and_round_trips() {
        let families = text::available_families();
        let chosen = families.first().cloned().unwrap_or_else(|| "Serif".into());
        let t = TextDef {
            text: "Ag".into(),
            family: Some(chosen.clone()),
            ..TextDef::default()
        };
        let buf = rasterize(&t, 96, 64);
        let lit = buf.iter().skip(3).step_by(4).filter(|&&a| a > 0.1).count();
        assert!(lit > 0, "text with family {chosen:?} should rasterize glyphs");

        // Doc-model round trip: family persists; a legacy doc (no `family` key)
        // loads as `None` so old `.pigment` text defs stay back-compatible.
        let json = serde_json::to_string(&t).expect("serialize TextDef");
        let back: TextDef = serde_json::from_str(&json).expect("deserialize TextDef");
        assert_eq!(back.family, Some(chosen));

        let legacy = r#"{"text":"Hi","font_px":48.0,"color":[1.0,1.0,1.0,1.0],"align":0}"#;
        let old: TextDef = serde_json::from_str(legacy).expect("deserialize legacy TextDef");
        assert_eq!(old.family, None);
    }

    /// Top-left of the lit region (alpha > 0.1) in a full-canvas RGBA-f32 buffer,
    /// as `(min_x, min_y)`; `None` if nothing is lit.
    fn lit_origin(buf: &[f32], w: u32, h: u32) -> Option<(u32, u32)> {
        let mut mn: Option<(u32, u32)> = None;
        for y in 0..h {
            for x in 0..w {
                if buf[((y * w + x) * 4 + 3) as usize] > 0.1 {
                    let (mx, my) = mn.unwrap_or((x, y));
                    mn = Some((mx.min(x), my.min(y)));
                }
            }
        }
        mn
    }

    /// Regression for "moved text snaps back to the top-left after a property
    /// change". The placement seam (`place_generated`) must re-apply the layer's
    /// baked Move offset on every re-rasterize, so changing a property (here the
    /// font family — but the seam is property-agnostic) keeps the glyphs where
    /// the user dragged them instead of resetting to the canvas origin. Pure: no
    /// GPU, exercises the exact buffer-placement logic `sync_generated_layers`
    /// uses.
    #[test]
    fn moved_text_keeps_position_after_property_change() {
        let (w, h) = (160u32, 96u32);
        let off = egui::vec2(40.0, 24.0);

        // Initial placement: rasterize at origin, then bake a Move of `off`.
        let base = rasterize(&TextDef::default(), w, h);
        let base_org = lit_origin(&base, w, h).expect("base text lights pixels");
        let moved = place_generated(base, w, h, Some(off));
        let moved_org = lit_origin(&moved, w, h).expect("moved text lights pixels");
        assert_eq!(
            moved_org,
            (base_org.0 + off.x as u32, base_org.1 + off.y as u32),
            "move shifts the lit region by the offset"
        );

        // Re-rasterize after a property change (new family) and re-place using the
        // same stored offset, as `sync_generated_layers` does. The glyphs must
        // land at the moved origin, NOT back at the (smaller) un-moved origin.
        let edited = TextDef {
            family: Some(
                text::available_families()
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "Serif".into()),
            ),
            ..TextDef::default()
        };
        let reraster = rasterize(&edited, w, h);
        let reraster_org = lit_origin(&reraster, w, h).expect("re-rasterized text lights pixels");
        let replaced = place_generated(reraster, w, h, Some(off));
        let replaced_org = lit_origin(&replaced, w, h).expect("replaced text lights pixels");

        assert_eq!(
            replaced_org,
            (reraster_org.0 + off.x as u32, reraster_org.1 + off.y as u32),
            "re-rasterized text after a property change stays at the moved position"
        );
        assert!(
            replaced_org.0 >= off.x as u32 && replaced_org.1 >= off.y as u32,
            "moved text did not snap back toward the top-left: {replaced_org:?}"
        );
    }

    /// A zero (or absent) offset is a no-op: an un-moved generated layer
    /// re-rasterizes at the origin exactly as before.
    #[test]
    fn unmoved_generated_layer_places_at_origin() {
        let (w, h) = (96u32, 64u32);
        let buf = rasterize(&TextDef::default(), w, h);
        let placed_none = place_generated(buf.clone(), w, h, None);
        let placed_zero = place_generated(buf.clone(), w, h, Some(egui::Vec2::ZERO));
        assert_eq!(placed_none, buf, "None offset leaves the buffer unchanged");
        assert_eq!(
            placed_zero, buf,
            "zero offset leaves the buffer unchanged"
        );
    }
}
