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
        if curves.is_empty() {
            return;
        }
        with_gpu(frame, |gpu, device, queue| {
            for (id, cp) in &curves {
                gpu.set_curve_lut(device, queue, *id, &cp.rgb, &cp.r, &cp.g, &cp.b);
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
                    text::render_text(&t.text, t.font_px, t.color, w, h, align)
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
            }
            if let Some(bi) = self.blend_if.get(&l.id) {
                for v in bi {
                    v.to_bits().hash(&mut h);
                }
            }
            self.clipped_layers.contains(&l.id).hash(&mut h);
        }
        h.finish()
    }

    pub(crate) fn active_id(&self) -> LayerId {
        self.doc.active_layer.unwrap_or(self.background_id)
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
                let blend_if = self.blend_if.get(&l.id).copied();
                LayerDraw {
                    id: l.id,
                    opacity: l.opacity,
                    blend: l.blend.shader_id(),
                    visible: l.visible,
                    adjust_kind,
                    adjust,
                    has_blend_if: blend_if.is_some(),
                    blend_if: blend_if.unwrap_or([0.0, 1.0, 0.0, 1.0]),
                    clipped: self.clipped_layers.contains(&l.id),
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
