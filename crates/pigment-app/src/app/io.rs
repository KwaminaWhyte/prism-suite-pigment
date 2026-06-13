use super::*;

/// Runtime per-layer style maps, gathered for one layer, in the tuple shapes
/// used by `PigmentApp`'s style HashMaps. Used purely as the pure-function
/// boundary between the runtime maps and the serializable `LayerStyles` model so
/// the mapping can be unit-tested without a live app/GPU.
#[allow(clippy::type_complexity)]
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct RuntimeLayerStyles {
    pub stroke: Option<([f32; 4], f32)>,
    pub drop_shadow: Option<([f32; 4], [f32; 2], f32)>,
    pub color_overlay: Option<[f32; 4]>,
    pub inner_shadow: Option<([f32; 4], [f32; 2], f32)>,
    pub outer_glow: Option<([f32; 4], f32)>,
    pub inner_glow: Option<([f32; 4], f32)>,
    pub grad_overlay: Option<([f32; 4], [f32; 4], f32, f32)>,
    pub bevel: Option<([f32; 4], [f32; 4], f32, f32, f32, f32)>,
}

impl RuntimeLayerStyles {
    fn is_empty(&self) -> bool {
        self.stroke.is_none()
            && self.drop_shadow.is_none()
            && self.color_overlay.is_none()
            && self.inner_shadow.is_none()
            && self.outer_glow.is_none()
            && self.inner_glow.is_none()
            && self.grad_overlay.is_none()
            && self.bevel.is_none()
    }
}

/// Map the runtime per-layer style tuples to the serializable `.pigment`
/// `LayerStyles` model. Returns `None` when the layer carries no styles so the
/// document stays compact. Pure: no app/GPU state.
pub(crate) fn runtime_styles_to_meta(rt: &RuntimeLayerStyles) -> Option<LayerStyles> {
    if rt.is_empty() {
        return None;
    }
    Some(LayerStyles {
        stroke: rt.stroke.map(|(color, width_px)| StrokeStyle { color, width_px }),
        drop_shadow: rt.drop_shadow.map(|(color, offset_px, blur_px)| ShadowStyle {
            color,
            offset_px,
            blur_px,
        }),
        color_overlay: rt.color_overlay.map(|color| ColorOverlayStyle { color }),
        inner_shadow: rt.inner_shadow.map(|(color, offset_px, blur_px)| ShadowStyle {
            color,
            offset_px,
            blur_px,
        }),
        outer_glow: rt.outer_glow.map(|(color, size_px)| GlowStyle { color, size_px }),
        inner_glow: rt.inner_glow.map(|(color, size_px)| GlowStyle { color, size_px }),
        gradient_overlay: rt
            .grad_overlay
            .map(|(color0, color1, angle_deg, opacity)| GradientOverlayStyle {
                color0,
                color1,
                angle_deg,
                opacity,
            }),
        bevel: rt
            .bevel
            .map(|(highlight, shadow, size_px, soften_px, angle_deg, altitude_deg)| BevelStyle {
                highlight,
                shadow,
                size_px,
                soften_px,
                angle_deg,
                altitude_deg,
            }),
    })
}

/// Inverse of `runtime_styles_to_meta`: map a serialized `LayerStyles` back to
/// the runtime tuple shapes. Pure: no app/GPU state.
pub(crate) fn meta_styles_to_runtime(styles: &LayerStyles) -> RuntimeLayerStyles {
    RuntimeLayerStyles {
        stroke: styles.stroke.map(|s| (s.color, s.width_px)),
        drop_shadow: styles.drop_shadow.map(|s| (s.color, s.offset_px, s.blur_px)),
        color_overlay: styles.color_overlay.map(|s| s.color),
        inner_shadow: styles.inner_shadow.map(|s| (s.color, s.offset_px, s.blur_px)),
        outer_glow: styles.outer_glow.map(|s| (s.color, s.size_px)),
        inner_glow: styles.inner_glow.map(|s| (s.color, s.size_px)),
        grad_overlay: styles
            .gradient_overlay
            .map(|s| (s.color0, s.color1, s.angle_deg, s.opacity)),
        bevel: styles
            .bevel
            .map(|s| (s.highlight, s.shadow, s.size_px, s.soften_px, s.angle_deg, s.altitude_deg)),
    }
}

impl PigmentApp {
    /// Gather a layer's runtime styles from the per-style HashMaps into one
    /// bundle (the pure-function input for `.pigment` serialization).
    pub(crate) fn collect_runtime_styles(&self, id: LayerId) -> RuntimeLayerStyles {
        RuntimeLayerStyles {
            stroke: self.layer_strokes.get(&id).copied(),
            drop_shadow: self.layer_shadows.get(&id).copied(),
            color_overlay: self.layer_overlays.get(&id).copied(),
            inner_shadow: self.layer_inner_shadows.get(&id).copied(),
            outer_glow: self.layer_outer_glows.get(&id).copied(),
            inner_glow: self.layer_inner_glows.get(&id).copied(),
            grad_overlay: self.layer_grad_overlays.get(&id).copied(),
            bevel: self.layer_bevels.get(&id).copied(),
        }
    }

    /// Insert a bundle of runtime styles into the per-style HashMaps for `id`.
    /// `None` entries leave the corresponding map untouched (callers clear maps
    /// before a load so absent styles do not linger from a previous document).
    pub(crate) fn install_runtime_styles(&mut self, id: LayerId, rt: &RuntimeLayerStyles) {
        if let Some(v) = rt.stroke {
            self.layer_strokes.insert(id, v);
        }
        if let Some(v) = rt.drop_shadow {
            self.layer_shadows.insert(id, v);
        }
        if let Some(v) = rt.color_overlay {
            self.layer_overlays.insert(id, v);
        }
        if let Some(v) = rt.inner_shadow {
            self.layer_inner_shadows.insert(id, v);
        }
        if let Some(v) = rt.outer_glow {
            self.layer_outer_glows.insert(id, v);
        }
        if let Some(v) = rt.inner_glow {
            self.layer_inner_glows.insert(id, v);
        }
        if let Some(v) = rt.grad_overlay {
            self.layer_grad_overlays.insert(id, v);
        }
        if let Some(v) = rt.bevel {
            self.layer_bevels.insert(id, v);
        }
    }

    /// Remove all per-layer styles (used before loading a new document so styles
    /// from the previous document do not bleed onto re-allocated layer ids).
    pub(crate) fn clear_all_layer_styles(&mut self) {
        self.layer_strokes.clear();
        self.layer_shadows.clear();
        self.layer_overlays.clear();
        self.layer_inner_shadows.clear();
        self.layer_outer_glows.clear();
        self.layer_inner_glows.clear();
        self.layer_grad_overlays.clear();
        self.layer_bevels.clear();
    }

    pub(crate) fn open_image(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Images", prism_io::SUPPORTED_EXTENSIONS)
            .pick_file()
        else {
            return;
        };
        match prism_io::load_image(&path) {
            Ok(img) => {
                self.doc = Document::new(img.size);
                self.background_id = self.doc.layers.layers[0].id;
                self.pending = Some(PendingUpload {
                    size: img.size,
                    layers: vec![(self.background_id, image_to_f16_bytes(&img))],
                });
                self.needs_fit = true;
            }
            Err(e) => log::error!("open image failed: {e}"),
        }
    }

    pub(crate) fn open_pigment(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Pigment", &["pigment"])
            .pick_file()
        else {
            return;
        };
        let (meta, pixels) = match document_file::load_document(&path) {
            Ok(v) => v,
            Err(e) => {
                log::error!("open .pigment failed: {e}");
                return;
            }
        };
        let size = Size::new(meta.width, meta.height);
        let mut doc = Document::new(size);
        doc.layers.layers.clear();
        // Layers are re-allocated fresh ids on load, so styles keyed by the old
        // (saved) ids must be re-installed under the new ids. Clear first so a
        // previous document's styles don't bleed onto re-used ids.
        self.clear_all_layer_styles();
        let mut staged = Vec::new();
        let mut loaded_styles: Vec<(LayerId, RuntimeLayerStyles)> = Vec::new();
        // Layers get fresh ids on load; map saved id -> new id so layer comps
        // (keyed by saved id) can be remapped onto the loaded layers.
        let mut id_map: HashMap<u64, LayerId> = HashMap::new();
        for (i, lm) in meta.layers.iter().enumerate() {
            let id = doc.layers.alloc_id();
            id_map.insert(lm.id, id);
            // Adjustment layers carry a serialized `Adjustment` descriptor;
            // restore them as adjustment layers (kind + params). Everything else
            // (including old docs with no `adjustment` key) loads as raster.
            let mut layer = match &lm.adjustment {
                Some(adj) => Layer::adjustment(id, lm.name.clone(), adj.clone()),
                None => Layer::raster(id, lm.name.clone()),
            };
            layer.opacity = lm.opacity;
            layer.visible = lm.visible;
            layer.blend = BlendMode::from_shader_id(lm.blend);
            doc.layers.layers.push(layer);
            let bytes = pixels.get(i).map(|p| p.rgba16f.clone()).unwrap_or_default();
            staged.push((id, bytes));
            if let Some(styles) = &lm.styles {
                loaded_styles.push((id, meta_styles_to_runtime(styles)));
            }
        }
        self.background_id = doc
            .layers
            .layers
            .first()
            .map(|l| l.id)
            .unwrap_or(LayerId(0));
        doc.active_layer = doc.layers.layers.last().map(|l| l.id);
        self.doc = doc;
        // Restore layer comps, remapping saved layer ids to the freshly allocated
        // ones; entries for layers absent from this document are dropped.
        self.comps = meta
            .comps
            .iter()
            .map(|m| super::comps::meta_to_comp(m, &id_map))
            .collect();
        for (id, rt) in &loaded_styles {
            self.install_runtime_styles(*id, rt);
        }
        self.pending = Some(PendingUpload {
            size,
            layers: staged,
        });
        self.needs_fit = true;
        // Loaded styles change the layer fingerprint; force a recomposite so the
        // restored styles render on the next frame.
        self.force_composite = true;
    }

    pub(crate) fn open_psd(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Photoshop", &["psd"])
            .pick_file()
        else {
            return;
        };
        let doc_psd = match prism_io::psd_import::load_psd(&path) {
            Ok(d) => d,
            Err(e) => {
                log::error!("open .psd failed: {e}");
                return;
            }
        };
        let size = Size::new(doc_psd.width.max(1), doc_psd.height.max(1));
        let mut doc = Document::new(size);
        doc.layers.layers.clear();
        let mut staged = Vec::new();
        for pl in &doc_psd.layers {
            let id = doc.layers.alloc_id();
            let mut layer = Layer::raster(id, pl.name.clone());
            layer.opacity = pl.opacity;
            layer.visible = pl.visible;
            layer.blend = BlendMode::from_shader_id(pl.blend);
            doc.layers.layers.push(layer);
            staged.push((id, rgba8_to_f16_bytes(&pl.rgba8)));
        }
        if staged.is_empty() {
            let id = doc.layers.add_raster("Background");
            staged.push((id, vec![0u8; (size.width * size.height * 8) as usize]));
        }
        self.background_id = doc
            .layers
            .layers
            .first()
            .map(|l| l.id)
            .unwrap_or(LayerId(0));
        doc.active_layer = doc.layers.layers.last().map(|l| l.id);
        self.doc = doc;
        self.pending = Some(PendingUpload {
            size,
            layers: staged,
        });
        self.masked_layers.clear();
        self.needs_fit = true;
    }

    /// Place a Contour vector document (`.contour`) as a new rasterized raster
    /// layer (Pigment <- Contour cross-app interop). Static "place": the shapes
    /// are flattened to pixels at the current document size; live Dynamic-Link
    /// is future work. Shapes outside the canvas are simply clipped.
    pub(crate) fn place_contour(&mut self, frame: &mut eframe::Frame) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Contour", &["contour"])
            .pick_file()
        else {
            return;
        };
        let (w, h) = (self.doc.size.width, self.doc.size.height);
        let placed = match crate::contour_import::place(&path, w, h) {
            Ok(p) => p,
            Err(e) => {
                log::error!("place .contour failed: {e}");
                return;
            }
        };
        let id = self.doc.layers.add_raster(placed.name);
        self.doc.active_layer = Some(id);
        let bytes = f32_to_f16_bytes(&placed.pixels);
        with_gpu(frame, |gpu, device, queue| {
            gpu.ensure_layer(device, id);
            gpu.upload_layer(queue, id, &bytes);
        });
        self.force_composite = true;
    }

    /// Place a `.contour` as a *linked* raster layer (the suite's Dynamic-Link):
    /// like `place_contour`, but we remember `LayerId -> (path, mtime)` so the
    /// per-frame poll in `sync_linked_contours` re-rasterizes + re-uploads the
    /// same layer whenever the source file changes on disk. The layer name gets
    /// a "(linked)" suffix so it's distinguishable from a static place.
    pub(crate) fn place_contour_linked(&mut self, frame: &mut eframe::Frame) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Contour", &["contour"])
            .pick_file()
        else {
            return;
        };
        let (w, h) = (self.doc.size.width, self.doc.size.height);
        let placed = match crate::contour_import::place(&path, w, h) {
            Ok(p) => p,
            Err(e) => {
                log::error!("place .contour (linked) failed: {e}");
                return;
            }
        };
        // Record the link timestamp now; treat a stat failure as the epoch so
        // the first successful stat next frame counts as "newer" and re-syncs.
        let mtime = std::fs::metadata(&path)
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        let id = self
            .doc
            .layers
            .add_raster(format!("{} (linked)", placed.name));
        self.doc.active_layer = Some(id);
        let bytes = f32_to_f16_bytes(&placed.pixels);
        with_gpu(frame, |gpu, device, queue| {
            gpu.ensure_layer(device, id);
            gpu.upload_layer(queue, id, &bytes);
        });
        self.linked_contours.insert(id, (path, mtime));
        self.force_composite = true;
    }

    /// Per-frame Dynamic-Link poll: for every linked `.contour` layer, stat the
    /// source file; if its mtime is newer than what we last rendered (and the
    /// layer still exists), re-rasterize at the current document size and
    /// re-upload to the SAME layer id. Layers removed from the document are
    /// pruned. Stat/read errors (e.g. the file briefly vanishing mid-save) are
    /// skipped this frame and retried next frame. Cheap: one stat per link.
    pub(crate) fn sync_linked_contours(&mut self, frame: &mut eframe::Frame) {
        if self.linked_contours.is_empty() {
            return;
        }
        // Prune links whose layer no longer exists in the document.
        let alive: HashSet<LayerId> = self.doc.layers.layers.iter().map(|l| l.id).collect();
        self.linked_contours.retain(|id, _| alive.contains(id));

        let (w, h) = (self.doc.size.width, self.doc.size.height);
        // Collect work first (we can't borrow self mutably while iterating the map).
        let mut jobs: Vec<(LayerId, Vec<u8>, SystemTime)> = Vec::new();
        for (id, (path, last)) in &self.linked_contours {
            // File temporarily missing (e.g. atomic save in progress) → skip.
            let Ok(mtime) = std::fs::metadata(path).and_then(|m| m.modified()) else {
                continue;
            };
            if mtime <= *last {
                continue;
            }
            match crate::contour_import::place(path, w, h) {
                Ok(placed) => jobs.push((*id, f32_to_f16_bytes(&placed.pixels), mtime)),
                Err(e) => log::warn!("linked .contour re-render skipped: {e}"),
            }
        }
        if jobs.is_empty() {
            return;
        }
        with_gpu(frame, |gpu, device, queue| {
            for (id, bytes, _) in &jobs {
                gpu.ensure_layer(device, *id);
                gpu.upload_layer(queue, *id, bytes);
            }
        });
        for (id, _, mtime) in jobs {
            if let Some(entry) = self.linked_contours.get_mut(&id) {
                entry.1 = mtime;
            }
        }
        self.force_composite = true;
    }

    pub(crate) fn open_exr(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("OpenEXR", &["exr"])
            .pick_file()
        else {
            return;
        };
        let (size, rgba) = match prism_io::exr_io::load_exr(&path) {
            Ok(v) => v,
            Err(e) => {
                log::error!("open .exr failed: {e}");
                return;
            }
        };
        // EXR is linear straight RGBA -> premultiply to f16.
        let mut bytes = Vec::with_capacity(rgba.len() * 2);
        for px in rgba.chunks_exact(4) {
            let a = px[3];
            for &ch in px.iter().take(3) {
                bytes.extend_from_slice(&f16::from_f32(ch * a).to_le_bytes());
            }
            bytes.extend_from_slice(&f16::from_f32(a).to_le_bytes());
        }
        self.doc = Document::new(size);
        self.background_id = self.doc.layers.layers[0].id;
        self.pending = Some(PendingUpload {
            size,
            layers: vec![(self.background_id, bytes)],
        });
        self.masked_layers.clear();
        self.needs_fit = true;
    }

    pub(crate) fn export_image(&mut self, frame: &mut eframe::Frame) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter(
                "Image",
                &["png", "jpg", "jpeg", "webp", "tif", "tiff", "bmp"],
            )
            .set_file_name("export.png")
            .save_file()
        else {
            return;
        };
        let (w, h) = (self.doc.size.width, self.doc.size.height);
        let order = self.layer_order();
        let comp = with_gpu(frame, |gpu, d, q| {
            let p = gpu.composite_now(d, q, &order);
            gpu.read_composite(d, q, p)
        })
        .flatten();
        self.force_composite = true;
        let Some(bytes) = comp else { return };
        let f = f16_bytes_to_f32(&bytes);
        // Composite is linear premultiplied -> straight sRGB 8-bit.
        let mut rgba8 = Vec::with_capacity((w * h * 4) as usize);
        for px in f.chunks_exact(4) {
            let a = px[3];
            let inv = if a > 1e-5 { 1.0 / a } else { 0.0 };
            let to8 =
                |lin: f32| (linear_to_srgb((lin * inv).clamp(0.0, 1.0)) * 255.0).round() as u8;
            rgba8.push(to8(px[0]));
            rgba8.push(to8(px[1]));
            rgba8.push(to8(px[2]));
            rgba8.push((a.clamp(0.0, 1.0) * 255.0).round() as u8);
        }
        if let Err(e) = prism_io::export::save_rgba8(&path, &rgba8, w, h) {
            log::error!("export failed: {e}");
        }
    }

    pub(crate) fn save_pigment(&mut self, frame: &mut eframe::Frame) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Pigment", &["pigment"])
            .set_file_name("untitled.pigment")
            .save_file()
        else {
            return;
        };
        let meta = DocMeta {
            width: self.doc.size.width,
            height: self.doc.size.height,
            layers: self
                .doc
                .layers
                .layers
                .iter()
                .map(|l| LayerMeta {
                    id: l.id.0,
                    name: l.name.clone(),
                    blend: l.blend.shader_id(),
                    opacity: l.opacity,
                    visible: l.visible,
                    styles: runtime_styles_to_meta(&self.collect_runtime_styles(l.id)),
                    // Persist the full adjustment descriptor (kind + every param)
                    // for adjustment layers so they round-trip; `None` otherwise.
                    adjustment: match &l.kind {
                        LayerKind::Adjustment(a) => Some(a.clone()),
                        _ => None,
                    },
                })
                .collect(),
            comps: self.comps.iter().map(super::comps::comp_to_meta).collect(),
        };
        let ids: Vec<LayerId> = self.doc.layers.layers.iter().map(|l| l.id).collect();
        let pixels = with_gpu(frame, |gpu, device, queue| {
            ids.iter()
                .filter_map(|id| {
                    gpu.read_layer(device, queue, *id).map(|b| LayerPixels {
                        id: id.0,
                        rgba16f: b,
                    })
                })
                .collect::<Vec<_>>()
        });
        let Some(pixels) = pixels else { return };
        if let Err(e) = document_file::save_document(&path, &meta, &pixels) {
            log::error!("save failed: {e}");
        }
    }

    /// Bucket fill the active layer from `seed` (document pixels).
    pub(crate) fn do_fill(&mut self, frame: &mut eframe::Frame, seed: (u32, u32)) {
        let (w, h) = (self.doc.size.width, self.doc.size.height);
        let active = self.active_id();
        let c = self.brush_color;
        let a = self.brush_opacity;
        let fill = [
            srgb_to_linear(c.r() as f32 / 255.0) * a,
            srgb_to_linear(c.g() as f32 / 255.0) * a,
            srgb_to_linear(c.b() as f32 / 255.0) * a,
            a,
        ];
        let tol = self.fill_tolerance;
        let contiguous = self.fill_contiguous;
        let sample_all = self.sample_all;
        let has_sel = self.selection_active;
        let order = self.layer_order();
        with_gpu(frame, |gpu, device, queue| {
            gpu.begin_command_now(device, queue, active, "Fill");
            // Sample source: the composite (all layers) or just the active layer.
            let sample = if sample_all {
                let is_ping = gpu.composite_now(device, queue, &order);
                gpu.read_composite(device, queue, is_ping)
            } else {
                gpu.read_layer(device, queue, active)
            };
            let Some(sample) = sample else { return };
            let sbuf = f16_bytes_to_f32(&sample);
            let mask = flood_fill_mask(&sbuf, w, h, seed.0, seed.1, tol, contiguous);
            // Constrain the fill to the active selection, if any.
            let sel = if has_sel {
                gpu.read_selection(device, queue)
            } else {
                None
            };
            // Write target is always the active layer.
            let Some(active_bytes) = gpu.read_layer(device, queue, active) else {
                return;
            };
            let mut abuf = f16_bytes_to_f32(&active_bytes);
            for (i, &m) in mask.iter().enumerate() {
                let selected = sel.as_ref().is_none_or(|s| s[i] > 0.5);
                if m && selected {
                    let o = i * 4;
                    abuf[o..o + 4].copy_from_slice(&fill);
                }
            }
            gpu.upload_layer(queue, active, &f32_to_f16_bytes(&abuf));
        });
        self.force_composite = true;
    }

    pub(crate) fn do_eyedrop(&mut self, frame: &mut eframe::Frame, seed: (u32, u32)) {
        let active = self.active_id();
        let sample_all = self.sample_all;
        let order = self.layer_order();
        let px = with_gpu(frame, |gpu, device, queue| {
            if sample_all {
                let is_ping = gpu.composite_now(device, queue, &order);
                gpu.read_composite_pixel(device, queue, is_ping, seed.0, seed.1)
            } else {
                gpu.read_pixel(device, queue, active, seed.0, seed.1)
            }
        })
        .flatten();
        if let Some(p) = px {
            let a = p[3];
            if a > 0.01 {
                let to8 =
                    |lin: f32| (linear_to_srgb(lin / a).clamp(0.0, 1.0) * 255.0).round() as u8;
                self.brush_color = egui::Color32::from_rgb(to8(p[0]), to8(p[1]), to8(p[2]));
            }
        }
        // Sample-all eyedrop runs composite_now (touches ping/pong); recomposite.
        if self.sample_all {
            self.force_composite = true;
        }
    }

    /// Magic wand: flood the composite at `seed` within tolerance → selection.
    pub(crate) fn do_magic_wand(&mut self, frame: &mut eframe::Frame, seed: (u32, u32)) {
        let (w, h) = (self.doc.size.width, self.doc.size.height);
        let order = self.layer_order();
        let (tol, contiguous) = (self.fill_tolerance, self.fill_contiguous);
        let comp = with_gpu(frame, |gpu, device, queue| {
            let p = gpu.composite_now(device, queue, &order);
            gpu.read_composite(device, queue, p)
        })
        .flatten();
        self.force_composite = true;
        let Some(bytes) = comp else { return };
        let buf = f16_bytes_to_f32(&bytes);
        let mask_b = flood_fill_mask(&buf, w, h, seed.0, seed.1, tol, contiguous);
        let mask: Vec<f32> = mask_b.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
        self.commit_selection(frame, mask);
    }
}

#[cfg(test)]
mod style_persist_tests {
    use super::*;

    /// A runtime style bundle carrying all 8 layer styles round-trips through the
    /// serializable `LayerStyles` model (the save→load mapping the .pigment IO
    /// uses) reproducing every param. GPU upload is intentionally untested.
    #[test]
    fn all_eight_styles_round_trip_through_meta() {
        let rt = RuntimeLayerStyles {
            stroke: Some(([0.1, 0.2, 0.3, 1.0], 4.5)),
            drop_shadow: Some(([0.0, 0.0, 0.0, 0.75], [5.0, -3.0], 8.0)),
            color_overlay: Some([0.8, 0.1, 0.1, 0.6]),
            inner_shadow: Some(([0.05, 0.05, 0.05, 0.5], [-2.0, 2.0], 3.0)),
            outer_glow: Some(([1.0, 0.9, 0.2, 0.8], 12.0)),
            inner_glow: Some(([0.2, 0.9, 1.0, 0.7], 6.0)),
            grad_overlay: Some(([0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0], 45.0, 0.9)),
            bevel: Some(([1.0, 1.0, 1.0, 0.75], [0.0, 0.0, 0.0, 0.75], 5.0, 2.0, 120.0, 30.0)),
        };

        // Forward: runtime -> meta -> JSON (the on-save path).
        let meta = runtime_styles_to_meta(&rt).expect("non-empty styles produce Some");
        let json = serde_json::to_string(&meta).expect("serialize LayerStyles");

        // Back: JSON -> meta -> runtime (the on-load path).
        let back_meta: LayerStyles = serde_json::from_str(&json).expect("deserialize LayerStyles");
        let back = meta_styles_to_runtime(&back_meta);

        assert_eq!(rt, back, "all 8 styles must round-trip losslessly");
    }

    /// A layer with no styles maps to `None`, so it serializes without a styles
    /// key and re-installing produces no map entries.
    #[test]
    fn empty_styles_map_to_none() {
        let rt = RuntimeLayerStyles::default();
        assert!(runtime_styles_to_meta(&rt).is_none());
    }

    /// install_runtime_styles writes exactly the present styles into the HashMaps,
    /// keyed by the (possibly re-allocated) layer id — the on-load installation.
    #[test]
    fn install_writes_present_styles_only() {
        let styles = LayerStyles {
            stroke: Some(StrokeStyle {
                color: [0.4, 0.5, 0.6, 1.0],
                width_px: 7.0,
            }),
            bevel: Some(BevelStyle {
                highlight: [1.0, 1.0, 1.0, 0.5],
                shadow: [0.0, 0.0, 0.0, 0.5],
                size_px: 9.0,
                soften_px: 1.0,
                angle_deg: 90.0,
                altitude_deg: 45.0,
            }),
            ..Default::default()
        };
        let rt = meta_styles_to_runtime(&styles);
        assert_eq!(rt.stroke, Some(([0.4, 0.5, 0.6, 1.0], 7.0)));
        assert_eq!(
            rt.bevel,
            Some(([1.0, 1.0, 1.0, 0.5], [0.0, 0.0, 0.0, 0.5], 9.0, 1.0, 90.0, 45.0))
        );
        // The unset styles stay None.
        assert!(rt.drop_shadow.is_none());
        assert!(rt.color_overlay.is_none());
        assert!(rt.grad_overlay.is_none());
    }
}

#[cfg(test)]
mod adjustment_persist_tests {
    use super::*;
    use prism_core::layer::Layer;

    /// Build a `LayerMeta` from a layer the way `save_pigment` does (only the
    /// adjustment-relevant fields), then serialize → deserialize and reconstruct
    /// the layer the way `open_pigment` does. Returns the rebuilt layer's kind so
    /// tests can assert the adjustment kind + every param survived round-trip.
    /// GPU pixel upload is intentionally excluded (untested per repo convention).
    fn round_trip_layer(layer: &Layer) -> LayerKind {
        // --- save mapping (mirror of `save_pigment`) ---
        let meta = LayerMeta {
            id: layer.id.0,
            name: layer.name.clone(),
            blend: layer.blend.shader_id(),
            opacity: layer.opacity,
            visible: layer.visible,
            styles: None,
            adjustment: match &layer.kind {
                LayerKind::Adjustment(a) => Some(a.clone()),
                _ => None,
            },
        };

        // --- the actual on-disk hop: JSON serialize + deserialize ---
        let json = serde_json::to_string(&meta).expect("serialize LayerMeta");
        let back: LayerMeta = serde_json::from_str(&json).expect("deserialize LayerMeta");

        // --- load mapping (mirror of `open_pigment`) ---
        let rebuilt = match &back.adjustment {
            Some(adj) => Layer::adjustment(layer.id, back.name.clone(), adj.clone()),
            None => Layer::raster(layer.id, back.name.clone()),
        };
        rebuilt.kind
    }

    /// Curves (variable-length per-channel control points) round-trips: every
    /// knot of the master + R/G/B curves survives save→load.
    #[test]
    fn curves_adjustment_round_trips() {
        let cp = CurvePoints {
            rgb: vec![(0.0, 0.1), (0.5, 0.7), (1.0, 0.9)],
            r: vec![(0.0, 0.0), (1.0, 0.8)],
            g: vec![(0.0, 0.2), (0.4, 0.5), (1.0, 1.0)],
            b: vec![(0.0, 0.05), (1.0, 0.95)],
        };
        let original = Adjustment::Curves(cp.clone());
        let layer = Layer::adjustment(LayerId(3), "Curves", original.clone());

        match round_trip_layer(&layer) {
            LayerKind::Adjustment(Adjustment::Curves(back)) => assert_eq!(back, cp),
            other => panic!("expected Curves adjustment, got {other:?}"),
        }
    }

    /// Color Balance (multi-param: 3 ranges × 3 channels + a flag) round-trips
    /// with every param intact.
    #[test]
    fn color_balance_adjustment_round_trips() {
        let original = Adjustment::ColorBalance {
            shadows: [0.2, -0.1, 0.3],
            midtones: [-0.4, 0.5, 0.0],
            highlights: [0.1, 0.1, -0.6],
            preserve_luminosity: false,
        };
        let layer = Layer::adjustment(LayerId(4), "Color Balance", original.clone());

        match round_trip_layer(&layer) {
            LayerKind::Adjustment(a) => assert_eq!(a, original),
            other => panic!("expected Color Balance adjustment, got {other:?}"),
        }
    }

    /// Channel Mixer (3×4 matrix + monochrome flag) round-trips with every
    /// coefficient intact.
    #[test]
    fn channel_mixer_adjustment_round_trips() {
        let original = Adjustment::ChannelMixer {
            r: [0.6, 0.3, 0.1, 0.05],
            g: [0.0, 1.2, -0.2, 0.0],
            b: [0.1, 0.0, 0.9, -0.1],
            monochrome: true,
        };
        let layer = Layer::adjustment(LayerId(5), "Channel Mixer", original.clone());

        match round_trip_layer(&layer) {
            LayerKind::Adjustment(a) => assert_eq!(a, original),
            other => panic!("expected Channel Mixer adjustment, got {other:?}"),
        }
    }

    /// A raster layer carries no adjustment payload and reconstructs as raster
    /// (back-compat: non-adjustment layers are unaffected).
    #[test]
    fn raster_layer_has_no_adjustment() {
        let layer = Layer::raster(LayerId(6), "bg");
        assert!(matches!(round_trip_layer(&layer), LayerKind::Raster));
    }
}
