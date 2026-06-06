use super::*;

impl PigmentApp {
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
        let mut staged = Vec::new();
        for (i, lm) in meta.layers.iter().enumerate() {
            let id = doc.layers.alloc_id();
            let mut layer = Layer::raster(id, lm.name.clone());
            layer.opacity = lm.opacity;
            layer.visible = lm.visible;
            layer.blend = BlendMode::from_shader_id(lm.blend);
            doc.layers.layers.push(layer);
            let bytes = pixels.get(i).map(|p| p.rgba16f.clone()).unwrap_or_default();
            staged.push((id, bytes));
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
        self.needs_fit = true;
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
                })
                .collect(),
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
