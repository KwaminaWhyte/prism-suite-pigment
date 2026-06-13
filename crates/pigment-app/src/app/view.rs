use super::*;

impl eframe::App for PigmentApp {
    fn ui(&mut self, root: &mut egui::Ui, frame: &mut eframe::Frame) {
        // Process any staged GPU uploads (new/opened document) before painting.
        // Only clear `pending` once the upload actually ran: if the wgpu render
        // state / CanvasGpu isn't ready yet (can happen on the first frame),
        // `with_gpu` is a no-op returning None — consuming `pending` then would
        // drop the document's pixels permanently (blank canvas until an edit).
        if self.pending.is_some() {
            // The per-frame composite is recorded into egui's paint-callback
            // (`prepare`) encoder, which doesn't reliably execute while egui is
            // idle immediately after load — so the default document (e.g. the
            // gradient background) could stay blank until the first edit forced
            // egui to render. Composite once here through the **self-submitting**
            // `composite_now` (its own encoder + `queue.submit`, the same path the
            // edit operations use) and build the display bind group from it, so
            // the document is presented on the very first frame.
            let order = self.layer_order();
            // Uploaded layers that carry a non-destructive smart-filter stack must
            // have it re-applied: the saved pixels are the *source* (un-filtered),
            // and the stack regenerates the displayed result on top of them.
            let smart: Vec<(LayerId, Vec<super::smart_filter::SmartPass>)> = self
                .smart_filters
                .iter()
                .filter(|(_, s)| !s.is_empty())
                .map(|(id, s)| (*id, s.enabled_passes()))
                .collect();
            let uploaded = with_gpu(frame, |gpu, device, queue| {
                let pend = self.pending.as_ref().expect("pending checked above");
                gpu.ensure_canvas(device, pend.size);
                for (id, bytes) in &pend.layers {
                    gpu.ensure_layer(device, *id);
                    gpu.upload_layer(queue, *id, bytes);
                }
                for (id, passes) in &smart {
                    gpu.reapply_smart_filters(device, queue, *id, passes);
                }
                let final_is_ping = gpu.composite_now(device, queue, &order);
                gpu.build_display_bind_group(device, final_is_ping);
            })
            .is_some();
            if !uploaded {
                // GPU not ready this frame — keep `pending` and try next frame.
                root.ctx().request_repaint();
                return;
            }
            self.pending = None;
            self.force_composite = true;
            root.ctx().request_repaint();
        }

        // A "fill layer with gradient" request from the gradient editor button.
        if let Some((p0, p1)) = self.grad_fill_pending.take() {
            self.do_gradient(frame, p0, p1);
        }

        // Dynamic-Link: poll linked `.contour` sources every frame. When a link
        // exists, keep the UI repainting (egui idles otherwise) so the mtime
        // poll actually runs and the layer tracks its vector source live.
        if !self.linked_contours.is_empty() {
            self.sync_linked_contours(frame);
            root.ctx().request_repaint();
        }

        egui::TopBottomPanel::top("menu").show_inside(root, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Open image…").clicked() {
                        self.open_image();
                        ui.close_menu();
                    }
                    if ui.button("Open .pigment…").clicked() {
                        self.open_pigment();
                        ui.close_menu();
                    }
                    if ui.button("Open .psd…").clicked() {
                        self.open_psd();
                        ui.close_menu();
                    }
                    if ui.button("Open .exr…").clicked() {
                        self.open_exr();
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Place .contour…").clicked() {
                        self.place_contour(frame);
                        ui.close_menu();
                    }
                    if ui.button("Place .contour (linked)…").clicked() {
                        self.place_contour_linked(frame);
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Save .pigment…").clicked() {
                        self.save_pigment(frame);
                        ui.close_menu();
                    }
                    if ui.button("Export image…").clicked() {
                        self.export_image(frame);
                        ui.close_menu();
                    }
                });
                ui.menu_button("Edit", |ui| {
                    if ui.button("Undo").clicked() {
                        self.undo_count += 1;
                        ui.close_menu();
                    }
                    if ui.button("Redo").clicked() {
                        self.redo_count += 1;
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Cut").clicked() {
                        self.cut_selection(frame);
                        ui.close_menu();
                    }
                    if ui.button("Copy").clicked() {
                        self.copy_selection(frame);
                        ui.close_menu();
                    }
                    if ui
                        .add_enabled(self.clipboard.is_some(), egui::Button::new("Paste"))
                        .clicked()
                    {
                        self.paste(frame);
                        ui.close_menu();
                    }
                    if ui.button("Layer from selection").clicked() {
                        self.layer_from_selection(frame);
                        ui.close_menu();
                    }
                });
                ui.menu_button("Image", |ui| {
                    ui.horizontal(|ui| {
                        ui.add(egui::DragValue::new(&mut self.resize_w).range(1..=16384));
                        ui.label("×");
                        ui.add(egui::DragValue::new(&mut self.resize_h).range(1..=16384));
                    });
                    if ui.button("Resize image (resample)").clicked() {
                        self.resize_image(frame, self.resize_w, self.resize_h);
                        ui.close_menu();
                    }
                    if ui.button("Canvas size (no resample)").clicked() {
                        self.resize_canvas(frame, self.resize_w, self.resize_h);
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Crop to selection").clicked() {
                        self.crop_to_selection(frame);
                        ui.close_menu();
                    }
                    if ui.button("Flip layer horizontal").clicked() {
                        self.flip_active(frame, true);
                        ui.close_menu();
                    }
                    if ui.button("Flip layer vertical").clicked() {
                        self.flip_active(frame, false);
                        ui.close_menu();
                    }
                });
                ui.menu_button("Filter", |ui| {
                    ui.add(
                        egui::Slider::new(&mut self.filter_radius, 0.0..=40.0).text("blur radius"),
                    );
                    if ui.button("Gaussian blur").clicked() {
                        self.do_filter(frame, 1, self.filter_radius, 0.0);
                        ui.close_menu();
                    }
                    ui.separator();
                    ui.add(
                        egui::Slider::new(&mut self.filter_amount, 0.0..=4.0)
                            .text("sharpen amount"),
                    );
                    if ui.button("Sharpen").clicked() {
                        self.do_filter(frame, 2, 0.0, self.filter_amount);
                        ui.close_menu();
                    }
                    ui.separator();
                    ui.menu_button("Sharpen", |ui| {
                        // High Pass (Sharpen family): orig − Gaussian blur,
                        // re-centred at mid-gray. Radius = blur scale, amount =
                        // detail gain.
                        ui.add(
                            egui::Slider::new(&mut self.high_pass_radius, 0.0..=40.0)
                                .text("high pass radius"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.high_pass_amount, 0.0..=4.0)
                                .text("amount"),
                        );
                        if ui.button("High Pass").clicked() {
                            self.do_high_pass(
                                frame,
                                self.high_pass_radius,
                                self.high_pass_amount,
                            );
                            ui.close_menu();
                        }
                    });
                    ui.separator();
                    ui.menu_button("Pixelate", |ui| {
                        // Pixelate (legacy kind 3: snap to the block-centre
                        // sample) and the cell-based family below it.
                        ui.add(
                            egui::Slider::new(&mut self.filter_block, 1.0..=40.0)
                                .text("pixel size"),
                        );
                        if ui.button("Pixelate").clicked() {
                            self.do_filter(frame, 3, self.filter_block, 0.0);
                            ui.close_menu();
                        }
                        ui.separator();
                        // Mosaic (true block average) + Crystallize (jittered
                        // Voronoi cells) share the cell-size slider.
                        ui.add(
                            egui::Slider::new(&mut self.mosaic_cell, 2.0..=64.0).text("cell size"),
                        );
                        if ui.button("Mosaic").clicked() {
                            self.do_mosaic(frame, self.mosaic_cell);
                            ui.close_menu();
                        }
                        ui.add(
                            egui::Slider::new(&mut self.crystallize_seed, 0.0..=100.0)
                                .text("crystallize seed"),
                        );
                        if ui.button("Crystallize").clicked() {
                            self.do_crystallize(frame, self.mosaic_cell, self.crystallize_seed);
                            ui.close_menu();
                        }
                        ui.separator();
                        // Color Halftone (per-channel dot screen: cell + angle).
                        ui.add(
                            egui::Slider::new(&mut self.halftone_cell, 2.0..=32.0)
                                .text("max radius (cell)"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.halftone_angle, 0.0..=90.0)
                                .text("screen angle°"),
                        );
                        if ui.button("Color Halftone").clicked() {
                            self.do_color_halftone(frame, self.halftone_cell, self.halftone_angle);
                            ui.close_menu();
                        }
                        ui.separator();
                        // Mezzotint (seeded threshold dither to black/white).
                        ui.add(
                            egui::Slider::new(&mut self.mezzotint_amount, 0.0..=1.0)
                                .text("threshold"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.mezzotint_seed, 0.0..=100.0).text("seed"),
                        );
                        if ui.button("Mezzotint").clicked() {
                            self.do_mezzotint(frame, self.mezzotint_amount, self.mezzotint_seed);
                            ui.close_menu();
                        }
                    });
                    ui.separator();
                    ui.menu_button("Blur", |ui| {
                        // Box blur (separable flat kernel).
                        ui.add(
                            egui::Slider::new(&mut self.box_radius, 1.0..=40.0).text("box radius"),
                        );
                        if ui.button("Box Blur").clicked() {
                            self.do_filter(frame, 5, self.box_radius, 0.0);
                            ui.close_menu();
                        }
                        ui.separator();
                        // Motion blur (angle + distance).
                        ui.add(
                            egui::Slider::new(&mut self.motion_angle, -180.0..=180.0)
                                .text("angle°"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.motion_distance, 1.0..=100.0)
                                .text("distance"),
                        );
                        if ui.button("Motion Blur").clicked() {
                            self.do_motion_blur(frame, self.motion_angle, self.motion_distance);
                            ui.close_menu();
                        }
                        ui.separator();
                        // Radial blur (spin / zoom about the canvas center).
                        ui.horizontal(|ui| {
                            ui.selectable_value(&mut self.radial_spin, true, "Spin");
                            ui.selectable_value(&mut self.radial_spin, false, "Zoom");
                        });
                        let amt_label = if self.radial_spin {
                            "angle°"
                        } else {
                            "zoom %"
                        };
                        let amt_range = if self.radial_spin {
                            0.0..=90.0
                        } else {
                            0.0..=100.0
                        };
                        ui.add(
                            egui::Slider::new(&mut self.radial_amount, amt_range).text(amt_label),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.radial_samples, 4..=64)
                                .text("quality (samples)"),
                        );
                        if ui.button("Radial Blur").clicked() {
                            self.do_radial_blur(
                                frame,
                                self.radial_spin,
                                self.radial_amount,
                                self.radial_samples,
                            );
                            ui.close_menu();
                        }
                        ui.separator();
                        // Tilt-Shift (Blur Gallery): graduated focus blur — sharp
                        // band, blur falls off above/below it.
                        ui.add(
                            egui::Slider::new(&mut self.tilt_center, 0.0..=1.0)
                                .text("focus center"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.tilt_half_band, 0.0..=400.0)
                                .text("band half-width"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.tilt_feather, 1.0..=600.0)
                                .text("feather"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.tilt_radius, 0.0..=40.0)
                                .text("max blur"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.tilt_angle, -45.0..=45.0)
                                .text("tilt angle°"),
                        );
                        if ui.button("Tilt-Shift").clicked() {
                            self.do_tilt_shift(
                                frame,
                                self.tilt_center,
                                self.tilt_half_band,
                                self.tilt_feather,
                                self.tilt_radius,
                                self.tilt_angle,
                            );
                            ui.close_menu();
                        }
                    });
                    ui.menu_button("Distort", |ui| {
                        // Twirl (angle + radius about the canvas center).
                        ui.add(
                            egui::Slider::new(&mut self.twirl_angle, -360.0..=360.0)
                                .text("twirl angle°"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.twirl_radius, 1.0..=1000.0)
                                .text("twirl radius"),
                        );
                        if ui.button("Twirl").clicked() {
                            self.do_twirl(frame, self.twirl_angle, self.twirl_radius);
                            ui.close_menu();
                        }
                        ui.separator();
                        // Pinch (+) / Spherize-bulge (−), signed amount + radius.
                        ui.add(
                            egui::Slider::new(&mut self.pinch_amount, -1.0..=1.0)
                                .text("pinch (− bulge)"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.pinch_radius, 1.0..=1000.0)
                                .text("pinch radius"),
                        );
                        if ui.button("Pinch / Spherize").clicked() {
                            self.do_pinch(frame, self.pinch_amount, self.pinch_radius);
                            ui.close_menu();
                        }
                        ui.separator();
                        // Ripple / Wave (amplitude + wavelength).
                        ui.add(
                            egui::Slider::new(&mut self.ripple_amplitude, 0.0..=64.0)
                                .text("amplitude"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.ripple_wavelength, 2.0..=200.0)
                                .text("wavelength"),
                        );
                        if ui.button("Ripple / Wave").clicked() {
                            self.do_ripple(frame, self.ripple_amplitude, self.ripple_wavelength);
                            ui.close_menu();
                        }
                        ui.separator();
                        // Polar Coordinates (rectangular <-> polar).
                        ui.horizontal(|ui| {
                            ui.selectable_value(&mut self.polar_to_polar, true, "Rect→Polar");
                            ui.selectable_value(&mut self.polar_to_polar, false, "Polar→Rect");
                        });
                        if ui.button("Polar Coordinates").clicked() {
                            self.do_polar(frame, self.polar_to_polar);
                            ui.close_menu();
                        }
                    });
                    ui.menu_button("Stylize", |ui| {
                        // Find Edges (Sobel magnitude inverted → dark edges on white).
                        ui.add(
                            egui::Slider::new(&mut self.edge_width, 1.0..=8.0)
                                .text("edge width"),
                        );
                        if ui.button("Find Edges").clicked() {
                            self.do_find_edges(frame, self.edge_width);
                            ui.close_menu();
                        }
                        ui.separator();
                        // Emboss (directional gray relief: angle + height/amount).
                        ui.add(
                            egui::Slider::new(&mut self.emboss_angle, -180.0..=180.0)
                                .text("emboss angle°"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.emboss_amount, 0.1..=20.0)
                                .text("height/amount"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.emboss_width, 1.0..=8.0)
                                .text("emboss width"),
                        );
                        if ui.button("Emboss").clicked() {
                            self.do_emboss(
                                frame,
                                self.emboss_angle,
                                self.emboss_amount,
                                self.emboss_width,
                            );
                            ui.close_menu();
                        }
                        ui.separator();
                        // Glowing Edges (bright colored edges on black).
                        ui.add(
                            egui::Slider::new(&mut self.glow_brightness, 0.5..=20.0)
                                .text("brightness"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.glow_width, 1.0..=8.0)
                                .text("edge width"),
                        );
                        if ui.button("Glowing Edges").clicked() {
                            self.do_glowing_edges(frame, self.glow_brightness, self.glow_width);
                            ui.close_menu();
                        }
                        ui.separator();
                        // Oil Paint (Kuwahara quadrant filter → painterly patches).
                        ui.add(
                            egui::Slider::new(&mut self.oil_paint_radius, 1.0..=8.0)
                                .text("brush radius"),
                        );
                        if ui.button("Oil Paint").clicked() {
                            self.do_oil_paint(frame, self.oil_paint_radius);
                            ui.close_menu();
                        }
                        ui.separator();
                        // Diffuse (seeded-deterministic neighbour scramble).
                        ui.add(
                            egui::Slider::new(&mut self.diffuse_amount, 0.0..=32.0)
                                .text("amount"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.diffuse_seed, 0.0..=100.0)
                                .text("seed"),
                        );
                        if ui.button("Diffuse").clicked() {
                            self.do_diffuse(frame, self.diffuse_amount, self.diffuse_seed);
                            ui.close_menu();
                        }
                    });
                    ui.menu_button("Noise", |ui| {
                        // Add Noise (seeded-deterministic; gaussian/uniform,
                        // amount, monochromatic toggle).
                        ui.add(egui::Slider::new(&mut self.noise_amount, 0.0..=1.0).text("amount"));
                        ui.checkbox(&mut self.noise_gaussian, "gaussian (else uniform)");
                        ui.checkbox(&mut self.noise_mono, "monochromatic");
                        ui.add(egui::Slider::new(&mut self.noise_seed, 0.0..=100.0).text("seed"));
                        if ui.button("Add Noise").clicked() {
                            self.do_add_noise(
                                frame,
                                self.noise_amount,
                                self.noise_mono,
                                self.noise_gaussian,
                                self.noise_seed,
                            );
                            ui.close_menu();
                        }
                        ui.separator();
                        // Median (per-channel median over a (2r+1)² window).
                        ui.add(
                            egui::Slider::new(&mut self.median_radius, 1.0..=4.0).text("radius"),
                        );
                        if ui.button("Median").clicked() {
                            self.do_median(frame, self.median_radius);
                            ui.close_menu();
                        }
                        ui.separator();
                        // Dust & Scratches (thresholded median).
                        ui.add(
                            egui::Slider::new(&mut self.dust_threshold, 0.0..=1.0)
                                .text("threshold"),
                        );
                        if ui.button("Dust & Scratches").clicked() {
                            self.do_dust_and_scratches(
                                frame,
                                self.median_radius,
                                self.dust_threshold,
                            );
                            ui.close_menu();
                        }
                    });
                    ui.menu_button("Render", |ui| {
                        // Clouds / Difference Clouds (deterministic fBm value-noise
                        // generator): seed, base feature scale, per-octave
                        // roughness, octave count. Clouds fills the layer; Diff.
                        // Clouds differences the field against the existing pixels.
                        ui.add(egui::Slider::new(&mut self.clouds_seed, 0.0..=100.0).text("seed"));
                        ui.add(
                            egui::Slider::new(&mut self.clouds_scale, 2.0..=256.0).text("scale"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.clouds_roughness, 0.05..=0.95)
                                .text("roughness"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.clouds_octaves, 1..=10).text("octaves"),
                        );
                        if ui.button("Clouds").clicked() {
                            self.do_clouds(
                                frame,
                                false,
                                self.clouds_seed,
                                self.clouds_scale,
                                self.clouds_roughness,
                                self.clouds_octaves,
                            );
                            ui.close_menu();
                        }
                        if ui.button("Difference Clouds").clicked() {
                            self.do_clouds(
                                frame,
                                true,
                                self.clouds_seed,
                                self.clouds_scale,
                                self.clouds_roughness,
                                self.clouds_octaves,
                            );
                            ui.close_menu();
                        }
                    });
                    ui.menu_button("Adjustments", |ui| {
                        // Destructive, undoable tonal ops (vs the non-destructive
                        // Adjustment layers). Posterize quantizes each channel to N
                        // display-space levels; Threshold collapses to black/white
                        // at a luma cutoff. Both bake into the active layer.
                        let mut lv = self.posterize_levels as f32;
                        if ui
                            .add(egui::Slider::new(&mut lv, 2.0..=32.0).text("levels"))
                            .changed()
                        {
                            self.posterize_levels = lv.round() as u32;
                        }
                        if ui.button("Posterize").clicked() {
                            self.do_posterize(frame, self.posterize_levels);
                            ui.close_menu();
                        }
                        ui.separator();
                        ui.add(
                            egui::Slider::new(&mut self.threshold_level, 0.0..=1.0)
                                .text("threshold"),
                        );
                        if ui.button("Threshold").clicked() {
                            self.do_threshold(frame, self.threshold_level);
                            ui.close_menu();
                        }
                    });
                });
                ui.menu_button("Select", |ui| {
                    if ui.button("All").clicked() {
                        self.sel_op_pending = Some(SelectionOp::All);
                        self.selection_active = true; // show marching ants on whole canvas
                        ui.close_menu();
                    }
                    if ui.button("None").clicked() {
                        self.sel_op_pending = Some(SelectionOp::None);
                        self.selection_active = false; // deselect: hide marching ants
                        ui.close_menu();
                    }
                    if ui.button("Invert").clicked() {
                        self.sel_op_pending = Some(SelectionOp::Invert);
                        self.selection_active = true;
                        ui.close_menu();
                    }
                    ui.separator();
                    ui.add(
                        egui::Slider::new(&mut self.feather_radius, 0.0..=30.0).text("feather px"),
                    );
                    if ui.button("Feather").clicked() {
                        let r = self.feather_radius;
                        self.map_selection(frame, move |m, w, h| raster::feather(m, w, h, r));
                        ui.close_menu();
                    }
                    if ui.button("Grow 1px").clicked() {
                        self.map_selection(frame, |m, w, h| raster::grow_shrink(m, w, h, 1));
                        ui.close_menu();
                    }
                    if ui.button("Shrink 1px").clicked() {
                        self.map_selection(frame, |m, w, h| raster::grow_shrink(m, w, h, -1));
                        ui.close_menu();
                    }
                });
                ui.menu_button("View", |ui| {
                    if ui.button("Fit to screen").clicked() {
                        self.needs_fit = true;
                        ui.close_menu();
                    }
                    if ui.button("100%").clicked() {
                        self.view = ViewTransform::default();
                        ui.close_menu();
                    }
                });
                ui.menu_button("Window", |ui| {
                    ui.checkbox(&mut self.panels.tool_options, "Tool options bar");
                    ui.checkbox(&mut self.panels.tools, "Tools palette");
                    ui.checkbox(&mut self.panels.properties, "Properties panel");
                    ui.separator();
                    if ui
                        .add_enabled(
                            !self.panels.all_shown(),
                            egui::Button::new("Show all panels"),
                        )
                        .clicked()
                    {
                        self.panels.show_all();
                        ui.close_menu();
                    }
                });
                ui.separator();
                ui.label(format!("{} %", (self.view.zoom * 100.0).round() as i32));
            });
        });

        // Contextual tool-options bar (Affinity-style): per-tool controls live
        // here, under the menu, instead of cluttering the right properties panel.
        if self.panels.tool_options {
            egui::TopBottomPanel::top("tool_options").show_inside(root, |ui| {
                ui.horizontal_wrapped(|ui| {
                    let brushy = matches!(
                        self.tool,
                        Tool::Brush
                            | Tool::Eraser
                            | Tool::Clone
                            | Tool::Heal
                            | Tool::SpotHeal
                            | Tool::ContentFill
                            | Tool::Dodge
                            | Tool::Detail
                            | Tool::Liquify
                    );
                    if brushy {
                        ui.add(egui::Slider::new(&mut self.brush_size, 1.0..=400.0).text("size"));
                        ui.add(
                            egui::Slider::new(&mut self.brush_hardness, 0.0..=0.99)
                                .text("hardness"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.brush_opacity, 0.0..=1.0).text("opacity"),
                        );
                        ui.checkbox(&mut self.speed_dynamics, "speed→size");
                        if self.speed_dynamics {
                            ui.add(
                                egui::Slider::new(&mut self.min_size_scale, 0.05..=1.0).text("min"),
                            );
                        }
                    }
                    if self.tool == Tool::Liquify {
                        ui.separator();
                        for (m, name) in [(0u8, "Push"), (1, "Twirl"), (2, "Pucker"), (3, "Bloat")]
                        {
                            if ui.selectable_label(self.liquify_mode == m, name).clicked() {
                                self.liquify_mode = m;
                            }
                        }
                    }
                    if self.tool == Tool::Detail {
                        ui.separator();
                        for (m, name) in [
                            (0u8, "Saturate"),
                            (1, "Desaturate"),
                            (2, "Blur"),
                            (3, "Sharpen"),
                        ] {
                            if ui.selectable_label(self.detail_mode == m, name).clicked() {
                                self.detail_mode = m;
                            }
                        }
                    }
                    if self.tool == Tool::Fill {
                        ui.separator();
                        ui.add(
                            egui::Slider::new(&mut self.fill_tolerance, 0.0..=1.0)
                                .text("tolerance"),
                        );
                        ui.checkbox(&mut self.fill_contiguous, "contiguous");
                    }
                    if self.tool == Tool::Gradient {
                        self.gradient_options_ui(ui);
                    }
                    if matches!(self.tool, Tool::Fill | Tool::Eyedropper) {
                        ui.checkbox(&mut self.sample_all, "sample all layers");
                    }
                    if matches!(self.tool, Tool::MoveLayer | Tool::Transform) {
                        ui.separator();
                        ui.label("Drag to move the active layer; Shift+drag scales (Transform).");
                    }
                    if matches!(
                        self.tool,
                        Tool::SelectRect | Tool::SelectEllipse | Tool::Lasso | Tool::MagicWand
                    ) {
                        ui.separator();
                        ui.label("Shift: add · Alt: subtract");
                    }
                    if matches!(self.tool, Tool::Clone | Tool::Heal | Tool::SpotHeal) {
                        ui.separator();
                        ui.label("Alt-click sets the source");
                    }
                    if self.tool == Tool::Patch {
                        ui.separator();
                        ui.selectable_value(&mut self.patch_source_mode, true, "Source");
                        ui.selectable_value(&mut self.patch_source_mode, false, "Destination");
                        ui.label("Lasso a region, then drag it onto the texture to clone");
                    }
                    if matches!(self.tool, Tool::Pen | Tool::DirectSelect) {
                        ui.separator();
                        let n = self.work_path.anchors.len();
                        let can_fill = n >= 2;
                        if ui
                            .add_enabled(can_fill, egui::Button::new("Path → selection"))
                            .clicked()
                        {
                            self.path_to_selection(frame);
                        }
                        if ui
                            .add_enabled(can_fill, egui::Button::new("Apply as vector mask"))
                            .clicked()
                        {
                            self.path_to_mask(frame);
                        }
                        if ui
                            .add_enabled(n > 0, egui::Button::new("Clear path"))
                            .clicked()
                        {
                            self.work_path.clear();
                            self.pen_grab = None;
                        }
                        let closed = self.work_path.closed;
                        ui.label(if self.tool == Tool::Pen {
                            if closed {
                                "Path closed · drag anchors with Direct Select"
                            } else {
                                "Click to add · drag for curves · click first anchor to close"
                            }
                        } else {
                            "Drag anchors (●) or handles (○) to edit"
                        });
                    }
                });
            });
        }

        if self.panels.tools {
            egui::SidePanel::left("tools")
                .exact_width(48.0)
                .show_inside(root, |ui| {
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            ui.add_space(6.0);
                            use crate::icons;
                            // Tool families (Affinity-style): one button per group; a
                            // multi-tool group opens a flyout menu of its variants. The
                            // group button shows the active tool's icon when active.
                            type T = Tool;
                            let groups: &[&[(Tool, &str, &str)]] = &[
                                &[
                                    (T::Move, icons::PAN, "Pan view"),
                                    (T::MoveLayer, icons::MOVE, "Move layer"),
                                ],
                                &[
                                    (T::SelectRect, icons::RECT_SELECT, "Rectangle select"),
                                    (T::SelectEllipse, icons::ELLIPSE_SELECT, "Ellipse select"),
                                ],
                                &[(T::Lasso, icons::LASSO, "Lasso select")],
                                &[(T::MagicWand, icons::MAGIC_WAND, "Magic wand")],
                                &[(T::Crop, icons::CROP, "Crop")],
                                &[(T::Transform, icons::TRANSFORM, "Transform")],
                                &[
                                    (T::Brush, icons::BRUSH, "Brush"),
                                    (T::Eraser, icons::ERASER, "Eraser"),
                                ],
                                &[
                                    (T::Clone, icons::CLONE, "Clone stamp"),
                                    (T::Heal, icons::HEAL, "Healing brush"),
                                    (T::SpotHeal, icons::SPOT_HEAL, "Spot heal"),
                                    (T::ContentFill, icons::CONTENT_FILL, "Content-aware fill"),
                                    (T::Patch, icons::PATCH, "Patch"),
                                    (T::Dodge, icons::DODGE, "Dodge / burn"),
                                    (T::Detail, icons::DETAIL, "Detail (sponge/blur/sharpen)"),
                                    (T::Liquify, icons::LIQUIFY, "Liquify"),
                                ],
                                &[
                                    (T::Fill, icons::FILL, "Bucket fill"),
                                    (T::Gradient, icons::GRADIENT, "Gradient"),
                                ],
                                &[(T::Eyedropper, icons::EYEDROPPER, "Eyedropper")],
                                &[(T::Text, icons::TEXT, "Text")],
                                &[
                                    (T::Pen, icons::PEN, "Pen (work path)"),
                                    (T::DirectSelect, icons::DIRECT_SELECT, "Direct select (edit path)"),
                                ],
                                &[
                                    (T::ShapeRect, icons::SHAPE, "Rectangle shape"),
                                    (T::ShapeEllipse, icons::ELLIPSE_SELECT, "Ellipse shape"),
                                ],
                            ];
                            for tools in groups {
                                let active = tools.iter().find(|(t, _, _)| *t == self.tool);
                                let rep = active.map(|(_, ic, _)| *ic).unwrap_or(tools[0].1);
                                if tools.len() == 1 {
                                    let (t, ic, name) = tools[0];
                                    let btn = egui::SelectableLabel::new(
                                        self.tool == t,
                                        egui::RichText::new(ic).size(20.0),
                                    );
                                    if ui
                                        .add_sized([36.0, 30.0], btn)
                                        .on_hover_text(name)
                                        .clicked()
                                    {
                                        self.tool = t;
                                    }
                                } else {
                                    // Flyout menu; button shows the active variant's icon.
                                    let label = egui::RichText::new(rep).size(20.0).color(
                                        if active.is_some() {
                                            ui.visuals().selection.stroke.color
                                        } else {
                                            ui.visuals().text_color()
                                        },
                                    );
                                    ui.menu_button(label, |ui| {
                                        for (t, ic, name) in tools.iter() {
                                            if ui
                                                .selectable_label(
                                                    self.tool == *t,
                                                    format!("{ic}  {name}"),
                                                )
                                                .clicked()
                                            {
                                                self.tool = *t;
                                                ui.close_menu();
                                            }
                                        }
                                    });
                                }
                            }
                            ui.add_space(6.0);
                            ui.separator();
                            ui.vertical_centered(|ui| {
                                ui.color_edit_button_srgba(&mut self.brush_color)
                            });
                        });
                });
        }

        if self.panels.properties {
            egui::SidePanel::right("panels")
                .default_width(250.0)
                .show_inside(root, |ui| {
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            // Tool options moved to the contextual bar (top). This
                            // panel holds layer/document properties.
                            {
                                let id = self.active_id();
                                let mut clip = self.clipped_layers.contains(&id);
                                if ui
                                    .checkbox(&mut clip, "Clip to layer below")
                                    .on_hover_text(
                                        "Clipping mask: show only over the layer beneath's pixels",
                                    )
                                    .changed()
                                {
                                    if clip {
                                        self.clipped_layers.insert(id);
                                    } else {
                                        self.clipped_layers.remove(&id);
                                    }
                                    self.force_composite = true;
                                }
                            }

                            ui.separator();
                            egui::CollapsingHeader::new("Blend-If").show(ui, |ui| {
                                let id = self.active_id();
                                let mut enabled = self.blend_if.contains_key(&id);
                                if ui.checkbox(&mut enabled, "enable (active layer)").changed() {
                                    if enabled {
                                        self.blend_if.insert(id, [0.0, 1.0, 0.0, 1.0]);
                                    } else {
                                        self.blend_if.remove(&id);
                                    }
                                }
                                if let Some(bi) = self.blend_if.get_mut(&id) {
                                    ui.label("This layer");
                                    ui.add(egui::Slider::new(&mut bi[0], 0.0..=1.0).text("black"));
                                    ui.add(egui::Slider::new(&mut bi[1], 0.0..=1.0).text("white"));
                                    ui.label("Underlying");
                                    ui.add(egui::Slider::new(&mut bi[2], 0.0..=1.0).text("black"));
                                    ui.add(egui::Slider::new(&mut bi[3], 0.0..=1.0).text("white"));
                                }
                            });

                            ui.separator();
                            egui::CollapsingHeader::new("Layer style: Stroke").show(ui, |ui| {
                                let id = self.active_id();
                                let mut on = self.layer_strokes.contains_key(&id);
                                if ui.checkbox(&mut on, "enable (active layer)").changed() {
                                    if on {
                                        self.layer_strokes.insert(id, ([0.0, 0.0, 0.0, 1.0], 4.0));
                                    } else {
                                        self.layer_strokes.remove(&id);
                                    }
                                    self.force_composite = true;
                                }
                                if let Some((color, width)) = self.layer_strokes.get_mut(&id) {
                                    let mut rgb = [color[0], color[1], color[2]];
                                    if ui.color_edit_button_rgb(&mut rgb).changed() {
                                        *color = [rgb[0], rgb[1], rgb[2], color[3]];
                                        self.force_composite = true;
                                    }
                                    if ui
                                        .add(egui::Slider::new(width, 1.0..=40.0).text("width px"))
                                        .changed()
                                    {
                                        self.force_composite = true;
                                    }
                                }
                            });

                            ui.separator();
                            egui::CollapsingHeader::new("Layer style: Drop shadow").show(
                                ui,
                                |ui| {
                                    let id = self.active_id();
                                    let mut on = self.layer_shadows.contains_key(&id);
                                    if ui.checkbox(&mut on, "enable (active layer)").changed() {
                                        if on {
                                            self.layer_shadows.insert(
                                                id,
                                                ([0.0, 0.0, 0.0, 0.7], [6.0, 6.0], 6.0),
                                            );
                                        } else {
                                            self.layer_shadows.remove(&id);
                                        }
                                        self.force_composite = true;
                                    }
                                    if let Some((color, off, blur)) =
                                        self.layer_shadows.get_mut(&id)
                                    {
                                        let mut rgba = *color;
                                        if ui
                                            .color_edit_button_rgba_premultiplied(&mut rgba)
                                            .changed()
                                        {
                                            *color = rgba;
                                            self.force_composite = true;
                                        }
                                        let mut ch = false;
                                        ch |= ui
                                            .add(
                                                egui::Slider::new(&mut off[0], -50.0..=50.0)
                                                    .text("dx"),
                                            )
                                            .changed();
                                        ch |= ui
                                            .add(
                                                egui::Slider::new(&mut off[1], -50.0..=50.0)
                                                    .text("dy"),
                                            )
                                            .changed();
                                        ch |= ui
                                            .add(
                                                egui::Slider::new(blur, 0.0..=40.0).text("blur px"),
                                            )
                                            .changed();
                                        if ch {
                                            self.force_composite = true;
                                        }
                                    }
                                },
                            );

                            ui.separator();
                            egui::CollapsingHeader::new("Layer style: Color overlay").show(
                                ui,
                                |ui| {
                                    let id = self.active_id();
                                    let mut on = self.layer_overlays.contains_key(&id);
                                    if ui.checkbox(&mut on, "enable (active layer)").changed() {
                                        if on {
                                            self.layer_overlays.insert(id, [0.8, 0.1, 0.1, 1.0]);
                                        } else {
                                            self.layer_overlays.remove(&id);
                                        }
                                        self.force_composite = true;
                                    }
                                    if let Some(c) = self.layer_overlays.get_mut(&id) {
                                        if ui.color_edit_button_rgba_premultiplied(c).changed() {
                                            self.force_composite = true;
                                        }
                                    }
                                },
                            );

                            ui.separator();
                            egui::CollapsingHeader::new("Layer style: Gradient overlay").show(
                                ui,
                                |ui| {
                                    let id = self.active_id();
                                    let mut on = self.layer_grad_overlays.contains_key(&id);
                                    if ui.checkbox(&mut on, "enable (active layer)").changed() {
                                        if on {
                                            self.layer_grad_overlays.insert(
                                                id,
                                                (
                                                    [0.0, 0.0, 0.0, 1.0],
                                                    [1.0, 1.0, 1.0, 1.0],
                                                    0.0,
                                                    1.0,
                                                ),
                                            );
                                        } else {
                                            self.layer_grad_overlays.remove(&id);
                                        }
                                        self.force_composite = true;
                                    }
                                    if let Some((c0, c1, angle, opacity)) =
                                        self.layer_grad_overlays.get_mut(&id)
                                    {
                                        let mut ch = false;
                                        ui.horizontal(|ui| {
                                            let mut a = [c0[0], c0[1], c0[2]];
                                            if ui.color_edit_button_rgb(&mut a).changed() {
                                                *c0 = [a[0], a[1], a[2], 1.0];
                                                ch = true;
                                            }
                                            ui.label("to");
                                            let mut b = [c1[0], c1[1], c1[2]];
                                            if ui.color_edit_button_rgb(&mut b).changed() {
                                                *c1 = [b[0], b[1], b[2], 1.0];
                                                ch = true;
                                            }
                                        });
                                        ch |= ui
                                            .add(
                                                egui::Slider::new(angle, -180.0..=180.0)
                                                    .text("angle°"),
                                            )
                                            .changed();
                                        ch |= ui
                                            .add(
                                                egui::Slider::new(opacity, 0.0..=1.0)
                                                    .text("opacity"),
                                            )
                                            .changed();
                                        if ch {
                                            self.force_composite = true;
                                        }
                                    }
                                },
                            );

                            ui.separator();
                            egui::CollapsingHeader::new("Layer style: Inner shadow").show(
                                ui,
                                |ui| {
                                    let id = self.active_id();
                                    let mut on = self.layer_inner_shadows.contains_key(&id);
                                    if ui.checkbox(&mut on, "enable (active layer)").changed() {
                                        if on {
                                            self.layer_inner_shadows.insert(
                                                id,
                                                ([0.0, 0.0, 0.0, 0.7], [4.0, 4.0], 5.0),
                                            );
                                        } else {
                                            self.layer_inner_shadows.remove(&id);
                                        }
                                        self.force_composite = true;
                                    }
                                    if let Some((color, off, blur)) =
                                        self.layer_inner_shadows.get_mut(&id)
                                    {
                                        let mut rgba = *color;
                                        if ui
                                            .color_edit_button_rgba_premultiplied(&mut rgba)
                                            .changed()
                                        {
                                            *color = rgba;
                                            self.force_composite = true;
                                        }
                                        let mut ch = false;
                                        ch |= ui
                                            .add(
                                                egui::Slider::new(&mut off[0], -50.0..=50.0)
                                                    .text("dx"),
                                            )
                                            .changed();
                                        ch |= ui
                                            .add(
                                                egui::Slider::new(&mut off[1], -50.0..=50.0)
                                                    .text("dy"),
                                            )
                                            .changed();
                                        ch |= ui
                                            .add(
                                                egui::Slider::new(blur, 0.0..=40.0).text("blur px"),
                                            )
                                            .changed();
                                        if ch {
                                            self.force_composite = true;
                                        }
                                    }
                                },
                            );

                            ui.separator();
                            egui::CollapsingHeader::new("Layer style: Outer glow").show(ui, |ui| {
                                let id = self.active_id();
                                let mut on = self.layer_outer_glows.contains_key(&id);
                                if ui.checkbox(&mut on, "enable (active layer)").changed() {
                                    if on {
                                        self.layer_outer_glows
                                            .insert(id, ([1.0, 0.85, 0.4, 0.75], 8.0));
                                    } else {
                                        self.layer_outer_glows.remove(&id);
                                    }
                                    self.force_composite = true;
                                }
                                if let Some((color, size)) = self.layer_outer_glows.get_mut(&id) {
                                    let mut rgba = *color;
                                    if ui
                                        .color_edit_button_rgba_premultiplied(&mut rgba)
                                        .changed()
                                    {
                                        *color = rgba;
                                        self.force_composite = true;
                                    }
                                    if ui
                                        .add(egui::Slider::new(size, 1.0..=40.0).text("size px"))
                                        .changed()
                                    {
                                        self.force_composite = true;
                                    }
                                }
                            });

                            ui.separator();
                            egui::CollapsingHeader::new("Layer style: Inner glow").show(ui, |ui| {
                                let id = self.active_id();
                                let mut on = self.layer_inner_glows.contains_key(&id);
                                if ui.checkbox(&mut on, "enable (active layer)").changed() {
                                    if on {
                                        self.layer_inner_glows
                                            .insert(id, ([1.0, 0.95, 0.6, 0.75], 8.0));
                                    } else {
                                        self.layer_inner_glows.remove(&id);
                                    }
                                    self.force_composite = true;
                                }
                                if let Some((color, size)) = self.layer_inner_glows.get_mut(&id) {
                                    let mut rgba = *color;
                                    if ui
                                        .color_edit_button_rgba_premultiplied(&mut rgba)
                                        .changed()
                                    {
                                        *color = rgba;
                                        self.force_composite = true;
                                    }
                                    if ui
                                        .add(egui::Slider::new(size, 1.0..=40.0).text("size px"))
                                        .changed()
                                    {
                                        self.force_composite = true;
                                    }
                                }
                            });

                            ui.separator();
                            egui::CollapsingHeader::new("Layer style: Bevel & emboss").show(
                                ui,
                                |ui| {
                                    let id = self.active_id();
                                    let mut on = self.layer_bevels.contains_key(&id);
                                    if ui.checkbox(&mut on, "enable (active layer)").changed() {
                                        if on {
                                            // (highlight, shadow, size px, soften px, angle deg, altitude deg)
                                            self.layer_bevels.insert(
                                                id,
                                                (
                                                    [1.0, 1.0, 1.0, 0.75],
                                                    [0.0, 0.0, 0.0, 0.75],
                                                    5.0,
                                                    2.0,
                                                    120.0,
                                                    30.0,
                                                ),
                                            );
                                        } else {
                                            self.layer_bevels.remove(&id);
                                        }
                                        self.force_composite = true;
                                    }
                                    if let Some((hi, sh, size, soft, angle, alt)) =
                                        self.layer_bevels.get_mut(&id)
                                    {
                                        let mut ch = false;
                                        ui.label("highlight");
                                        let mut hrgba = *hi;
                                        if ui
                                            .color_edit_button_rgba_premultiplied(&mut hrgba)
                                            .changed()
                                        {
                                            *hi = hrgba;
                                            ch = true;
                                        }
                                        ui.label("shadow");
                                        let mut srgba = *sh;
                                        if ui
                                            .color_edit_button_rgba_premultiplied(&mut srgba)
                                            .changed()
                                        {
                                            *sh = srgba;
                                            ch = true;
                                        }
                                        ch |= ui
                                            .add(
                                                egui::Slider::new(size, 1.0..=40.0).text("size px"),
                                            )
                                            .changed();
                                        ch |= ui
                                            .add(
                                                egui::Slider::new(soft, 0.0..=20.0)
                                                    .text("soften px"),
                                            )
                                            .changed();
                                        ch |= ui
                                            .add(
                                                egui::Slider::new(angle, 0.0..=360.0)
                                                    .text("angle°"),
                                            )
                                            .changed();
                                        ch |= ui
                                            .add(
                                                egui::Slider::new(alt, 0.0..=90.0)
                                                    .text("altitude°"),
                                            )
                                            .changed();
                                        if ch {
                                            self.force_composite = true;
                                        }
                                    }
                                },
                            );

                            ui.separator();
                            egui::CollapsingHeader::new("Channels").show(ui, |ui| {
                                let names = with_gpu(frame, |gpu, _, _| gpu.channel_names())
                                    .unwrap_or_default();
                                if ui
                                    .add_enabled(
                                        self.selection_active,
                                        egui::Button::new("Save selection as channel"),
                                    )
                                    .clicked()
                                {
                                    let name = format!("Alpha {}", names.len() + 1);
                                    with_gpu(frame, |gpu, d, q| {
                                        gpu.save_selection_as_channel(d, q, name)
                                    });
                                }
                                for name in names {
                                    ui.horizontal(|ui| {
                                        ui.label(&name);
                                        if ui.small_button("Load").clicked() {
                                            let n = name.clone();
                                            with_gpu(frame, |gpu, d, q| gpu.load_channel(d, q, &n));
                                            self.selection_active = true;
                                            self.force_composite = true;
                                        }
                                        if ui.small_button("✕").clicked() {
                                            let n = name.clone();
                                            with_gpu(frame, |gpu, _, _| gpu.delete_channel(&n));
                                        }
                                    });
                                }
                            });

                            ui.separator();
                            egui::CollapsingHeader::new("Histogram").show(ui, |ui| {
                                if ui.button("Refresh").clicked() {
                                    self.refresh_histogram(frame);
                                }
                                if let Some(h) = &self.hist {
                                    let (rect, _) = ui.allocate_exact_size(
                                        egui::vec2(ui.available_width(), 64.0),
                                        egui::Sense::hover(),
                                    );
                                    let painter = ui.painter_at(rect);
                                    painter.rect_filled(rect, 2.0, egui::Color32::from_gray(18));
                                    let max =
                                        h.luma.iter().copied().max().unwrap_or(1).max(1) as f32;
                                    let n = h.luma.len().max(1);
                                    for (i, &c) in h.luma.iter().enumerate() {
                                        let x = rect.left() + rect.width() * i as f32 / n as f32;
                                        let bh = rect.height() * (c as f32 / max);
                                        painter.line_segment(
                                            [
                                                egui::pos2(x, rect.bottom()),
                                                egui::pos2(x, rect.bottom() - bh),
                                            ],
                                            egui::Stroke::new(1.0, egui::Color32::from_gray(200)),
                                        );
                                    }
                                }
                            });

                            ui.separator();
                            let (undos, redos) = with_gpu(frame, |gpu, _, _| gpu.history_labels())
                                .unwrap_or_default();
                            egui::CollapsingHeader::new(format!(
                                "History  ({} / {})",
                                undos.len(),
                                redos.len()
                            ))
                            .show(ui, |ui| {
                                // Future states (redoable), furthest first.
                                for (i, l) in redos.iter().enumerate().rev() {
                                    if ui.small_button(format!("redo  {l}")).clicked() {
                                        self.redo_count += (i + 1) as u32;
                                    }
                                }
                                ui.label("—— now ——");
                                // Past states (undoable), newest first.
                                for (i, l) in undos.iter().rev().enumerate() {
                                    if ui.small_button(format!("undo  {l}")).clicked() {
                                        self.undo_count += (i + 1) as u32;
                                    }
                                }
                            });

                            ui.separator();
                            egui::CollapsingHeader::new(format!(
                                "Layer Comps  ({})",
                                self.comps.len()
                            ))
                            .show(ui, |ui| {
                                // Create a comp from the current layer state.
                                ui.horizontal(|ui| {
                                    ui.add(
                                        egui::TextEdit::singleline(&mut self.new_comp_name)
                                            .desired_width(120.0)
                                            .hint_text("Comp name"),
                                    );
                                    if ui.button("Capture").clicked() {
                                        let name = if self.new_comp_name.trim().is_empty() {
                                            format!("Comp {}", self.comps.len() + 1)
                                        } else {
                                            self.new_comp_name.trim().to_string()
                                        };
                                        let comp = super::comps::capture_comp(
                                            name,
                                            &self.doc.layers.layers,
                                        );
                                        self.comps.push(comp);
                                        self.new_comp_name.clear();
                                    }
                                });
                                // List existing comps: restore / rename / delete.
                                let mut to_apply: Option<usize> = None;
                                let mut to_delete: Option<usize> = None;
                                for (i, comp) in self.comps.iter_mut().enumerate() {
                                    ui.horizontal(|ui| {
                                        if ui
                                            .small_button("▶")
                                            .on_hover_text("Restore this comp")
                                            .clicked()
                                        {
                                            to_apply = Some(i);
                                        }
                                        ui.add(
                                            egui::TextEdit::singleline(&mut comp.name)
                                                .desired_width(120.0),
                                        );
                                        if ui
                                            .small_button("✕")
                                            .on_hover_text("Delete this comp")
                                            .clicked()
                                        {
                                            to_delete = Some(i);
                                        }
                                    });
                                }
                                if let Some(i) = to_apply {
                                    super::comps::apply_comp(
                                        &self.comps[i],
                                        &mut self.doc.layers.layers,
                                    );
                                    self.force_composite = true;
                                }
                                if let Some(i) = to_delete {
                                    self.comps.remove(i);
                                }
                            });

                            // ---- Smart Filters (non-destructive filter stack) ----
                            // Operates on the active layer; only raster layers can
                            // carry pixels to filter. Each edit is deferred into a
                            // `SmartFilterUi` action and dispatched after the panel
                            // section so the mutators (which take `&mut self` +
                            // `frame` and re-apply the stack on the GPU) don't
                            // conflict with the borrows used to draw the list.
                            {
                                use super::smart_filter::{SmartFilterKind as SfK, SmartFilterUi};
                                let active = self.active_id();
                                let is_raster = matches!(
                                    self.doc.layers.get(active).map(|l| &l.kind),
                                    Some(LayerKind::Raster)
                                );
                                let mut sf_action = SmartFilterUi::None;
                                egui::CollapsingHeader::new("Smart Filters")
                                    .default_open(false)
                                    .show(ui, |ui| {
                                        if !is_raster {
                                            ui.label("Select a raster layer.");
                                            return;
                                        }
                                        ui.menu_button("Add filter", |ui| {
                                            if ui.button("Gaussian Blur").clicked() {
                                                sf_action = SmartFilterUi::Add(
                                                    SfK::GaussianBlur { radius: 4.0 },
                                                );
                                                ui.close_menu();
                                            }
                                            if ui.button("Sharpen").clicked() {
                                                sf_action =
                                                    SmartFilterUi::Add(SfK::Sharpen { amount: 1.0 });
                                                ui.close_menu();
                                            }
                                            if ui.button("Posterize").clicked() {
                                                sf_action = SmartFilterUi::Add(
                                                    SfK::Posterize { levels: 6 },
                                                );
                                                ui.close_menu();
                                            }
                                        });
                                        let stack = self.smart_stack(active);
                                        if stack.is_empty() {
                                            ui.weak("No smart filters on this layer.");
                                        }
                                        for (i, f) in stack.filters.iter().enumerate() {
                                            ui.separator();
                                            ui.horizontal(|ui| {
                                                let mut on = f.enabled;
                                                if ui.checkbox(&mut on, "").changed() {
                                                    sf_action = SmartFilterUi::Toggle(i);
                                                }
                                                ui.label(f.kind.label());
                                                if ui
                                                    .small_button(crate::icons::ARROW_UP)
                                                    .on_hover_text("Apply earlier")
                                                    .clicked()
                                                {
                                                    sf_action = SmartFilterUi::Up(i);
                                                }
                                                if ui
                                                    .small_button(crate::icons::ARROW_DOWN)
                                                    .on_hover_text("Apply later")
                                                    .clicked()
                                                {
                                                    sf_action = SmartFilterUi::Down(i);
                                                }
                                                if ui
                                                    .small_button(crate::icons::TRASH)
                                                    .on_hover_text("Remove")
                                                    .clicked()
                                                {
                                                    sf_action = SmartFilterUi::Remove(i);
                                                }
                                            });
                                            // Params editor — committed on release so
                                            // dragging a slider doesn't re-run the whole
                                            // stack every pixel of the drag.
                                            match f.kind {
                                                SfK::GaussianBlur { radius } => {
                                                    let mut r = radius;
                                                    let resp = ui.add(
                                                        egui::Slider::new(&mut r, 0.0..=40.0)
                                                            .text("radius"),
                                                    );
                                                    if resp.changed() || resp.drag_stopped() {
                                                        sf_action = SmartFilterUi::Edit(
                                                            i,
                                                            SfK::GaussianBlur { radius: r },
                                                        );
                                                    }
                                                }
                                                SfK::Sharpen { amount } => {
                                                    let mut a = amount;
                                                    let resp = ui.add(
                                                        egui::Slider::new(&mut a, 0.0..=4.0)
                                                            .text("amount"),
                                                    );
                                                    if resp.changed() || resp.drag_stopped() {
                                                        sf_action = SmartFilterUi::Edit(
                                                            i,
                                                            SfK::Sharpen { amount: a },
                                                        );
                                                    }
                                                }
                                                SfK::Posterize { levels } => {
                                                    let mut l = levels as f32;
                                                    let resp = ui.add(
                                                        egui::Slider::new(&mut l, 2.0..=32.0)
                                                            .text("levels"),
                                                    );
                                                    if resp.changed() || resp.drag_stopped() {
                                                        sf_action = SmartFilterUi::Edit(
                                                            i,
                                                            SfK::Posterize {
                                                                levels: l.round() as u32,
                                                            },
                                                        );
                                                    }
                                                }
                                            }
                                        }
                                    });
                                match sf_action {
                                    SmartFilterUi::None => {}
                                    SmartFilterUi::Add(k) => self.add_smart_filter(frame, k),
                                    SmartFilterUi::Remove(i) => self.remove_smart_filter(frame, i),
                                    SmartFilterUi::Toggle(i) => self.toggle_smart_filter(frame, i),
                                    SmartFilterUi::Up(i) => self.move_smart_filter_up(frame, i),
                                    SmartFilterUi::Down(i) => self.move_smart_filter_down(frame, i),
                                    SmartFilterUi::Edit(i, k) => self.edit_smart_filter(frame, i, k),
                                }
                            }

                            ui.separator();
                            ui.horizontal(|ui| {
                                ui.heading("Layers");
                                if ui
                                    .button(
                                        egui::RichText::new(crate::icons::PLUS_LAYER).size(18.0),
                                    )
                                    .on_hover_text("New layer")
                                    .clicked()
                                {
                                    let id = self.doc.layers.add_raster(format!(
                                        "Layer {}",
                                        self.doc.layers.layers.len()
                                    ));
                                    self.doc.active_layer = Some(id);
                                }
                                ui.menu_button("Adj", |ui| {
                                    for adj in Adjustment::defaults() {
                                        if ui.button(adj.name()).clicked() {
                                            let id = self.doc.layers.add_adjustment(adj);
                                            self.doc.active_layer = Some(id);
                                            ui.close_menu();
                                        }
                                    }
                                })
                                .response
                                .on_hover_text("New adjustment layer");
                                ui.menu_button("Mask", |ui| {
                                    let has = self.masked_layers.contains(&self.active_id());
                                    if ui.button("Add white mask").clicked() {
                                        self.add_mask(frame, false);
                                        ui.close_menu();
                                    }
                                    if ui
                                        .add_enabled(
                                            self.selection_active,
                                            egui::Button::new("Add from selection"),
                                        )
                                        .clicked()
                                    {
                                        self.add_mask(frame, true);
                                        ui.close_menu();
                                    }
                                    if ui
                                        .add_enabled(has, egui::Button::new("Delete mask"))
                                        .clicked()
                                    {
                                        self.delete_mask(frame);
                                        ui.close_menu();
                                    }
                                    ui.add_enabled(
                                        has,
                                        egui::Checkbox::new(
                                            &mut self.edit_mask,
                                            "Edit mask (brush=reveal, eraser=hide)",
                                        ),
                                    );
                                });
                            });
                            ui.separator();

                            let active = self.active_id();
                            // Lazily enumerate system font families once (the DB
                            // scan is expensive) so the text-layer font chooser
                            // below can borrow the cached list while iterating
                            // layers without re-borrowing `self` mutably.
                            if self.font_families.is_empty() {
                                self.font_families = text::available_families();
                            }
                            let font_families = &self.font_families;
                            let mut action = LayerAction::None;
                            let ids: Vec<LayerId> =
                                self.doc.layers.layers.iter().rev().map(|l| l.id).collect();
                            for id in ids {
                                let layer = self.doc.layers.get_mut(id).unwrap();
                                let is_active = id == active;
                                egui::Frame::NONE
                                    .fill(if is_active {
                                        egui::Color32::from_rgb(50, 70, 100)
                                    } else {
                                        egui::Color32::TRANSPARENT
                                    })
                                    .inner_margin(4.0)
                                    .show(ui, |ui| {
                                        ui.horizontal(|ui| {
                                            let eye = if layer.visible {
                                                crate::icons::EYE
                                            } else {
                                                " "
                                            };
                                            if ui
                                                .selectable_label(
                                                    layer.visible,
                                                    egui::RichText::new(eye).size(15.0),
                                                )
                                                .on_hover_text("Toggle visibility")
                                                .clicked()
                                            {
                                                layer.visible = !layer.visible;
                                            }
                                            ui.add(
                                                egui::TextEdit::singleline(&mut layer.name)
                                                    .desired_width(90.0),
                                            );
                                            if ui
                                                .small_button(crate::icons::ARROW_UP)
                                                .on_hover_text("Move up")
                                                .clicked()
                                            {
                                                action = LayerAction::MoveUp(id);
                                            }
                                            if ui
                                                .small_button(crate::icons::ARROW_DOWN)
                                                .on_hover_text("Move down")
                                                .clicked()
                                            {
                                                action = LayerAction::MoveDown(id);
                                            }
                                            if ui
                                                .small_button(crate::icons::TRASH)
                                                .on_hover_text("Delete layer")
                                                .clicked()
                                            {
                                                action = LayerAction::Delete(id);
                                            }
                                        });
                                        let is_adjustment =
                                            matches!(layer.kind, LayerKind::Adjustment(_));
                                        ui.horizontal(|ui| {
                                            if ui.selectable_label(is_active, "active").clicked() {
                                                self.doc.active_layer = Some(id);
                                            }
                                            if !is_adjustment {
                                                egui::ComboBox::from_id_salt(("blend", id.0))
                                                    .selected_text(format!("{:?}", layer.blend))
                                                    .width(120.0)
                                                    .show_ui(ui, |ui| {
                                                        for mode in BlendMode::ALL {
                                                            ui.selectable_value(
                                                                &mut layer.blend,
                                                                mode,
                                                                format!("{mode:?}"),
                                                            );
                                                        }
                                                    });
                                            }
                                        });
                                        ui.add(
                                            egui::Slider::new(&mut layer.opacity, 0.0..=1.0)
                                                .show_value(false)
                                                .text("opacity"),
                                        );
                                        match &mut layer.kind {
                                            LayerKind::Adjustment(adj) => adjustment_ui(ui, adj),
                                            LayerKind::Text(t) => {
                                                ui.add(
                                                    egui::TextEdit::singleline(&mut t.text)
                                                        .desired_width(150.0),
                                                );
                                                // Font family chooser: "Default"
                                                // (None → renderer default face)
                                                // plus every enumerated system
                                                // family. Selecting a family sets
                                                // `t.family`, which changes the
                                                // layer fingerprint and triggers a
                                                // re-rasterize in the chosen face.
                                                let current = t
                                                    .family
                                                    .as_deref()
                                                    .unwrap_or("Default");
                                                egui::ComboBox::from_id_salt(("font", id.0))
                                                    .selected_text(current)
                                                    .width(150.0)
                                                    .show_ui(ui, |ui| {
                                                        ui.selectable_value(
                                                            &mut t.family,
                                                            None,
                                                            "Default",
                                                        );
                                                        for fam in font_families {
                                                            ui.selectable_value(
                                                                &mut t.family,
                                                                Some(fam.clone()),
                                                                fam,
                                                            );
                                                        }
                                                    });
                                                ui.add(
                                                    egui::Slider::new(&mut t.font_px, 6.0..=300.0)
                                                        .text("size"),
                                                );
                                                let mut col = srgba_to_color(t.color);
                                                if ui.color_edit_button_srgba(&mut col).changed() {
                                                    t.color = color_to_srgba(col);
                                                }
                                                ui.horizontal(|ui| {
                                                    ui.selectable_value(&mut t.align, 0, "L");
                                                    ui.selectable_value(&mut t.align, 1, "C");
                                                    ui.selectable_value(&mut t.align, 2, "R");
                                                });
                                            }
                                            LayerKind::Vector(v) => {
                                                let mut col = srgba_to_color(v.color);
                                                if ui.color_edit_button_srgba(&mut col).changed() {
                                                    v.color = color_to_srgba(col);
                                                }
                                            }
                                            _ => {}
                                        }
                                    });
                                ui.separator();
                            }

                            // Apply structural layer changes after the loop.
                            let ls = &mut self.doc.layers.layers;
                            match action {
                                LayerAction::None => {}
                                LayerAction::Delete(id) => {
                                    if ls.len() > 1 {
                                        ls.retain(|l| l.id != id);
                                        with_gpu(frame, |gpu, _, _| gpu.drop_layer(id));
                                        if self.doc.active_layer == Some(id) {
                                            self.doc.active_layer = ls.last().map(|l| l.id);
                                        }
                                        self.background_id =
                                            ls.first().map(|l| l.id).unwrap_or(self.background_id);
                                    }
                                }
                                LayerAction::MoveUp(id) => {
                                    if let Some(i) = ls.iter().position(|l| l.id == id) {
                                        if i + 1 < ls.len() {
                                            ls.swap(i, i + 1);
                                        }
                                    }
                                }
                                LayerAction::MoveDown(id) => {
                                    if let Some(i) = ls.iter().position(|l| l.id == id) {
                                        if i > 0 {
                                            ls.swap(i, i - 1);
                                        }
                                    }
                                }
                            }
                        });
                });
        }

        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(egui::Color32::from_gray(30)))
            .show_inside(root, |ui| {
                let rect = ui.available_rect_before_wrap();
                let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());

                if self.needs_fit {
                    self.fit(rect);
                    self.needs_fit = false;
                }

                let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                if scroll != 0.0 {
                    if let Some(cursor) = response.hover_pos() {
                        self.view
                            .zoom_to((scroll * 0.0015).exp(), cursor - rect.center());
                    }
                }

                let (mut undo, mut redo) = (self.undo_count, self.redo_count);
                self.undo_count = 0;
                self.redo_count = 0;
                let (mut do_copy, mut do_cut, mut do_paste) = (false, false, false);
                ui.input(|i| {
                    if i.modifiers.command {
                        if i.key_pressed(egui::Key::Z) {
                            if i.modifiers.shift {
                                redo += 1;
                            } else {
                                undo += 1;
                            }
                        }
                        if i.key_pressed(egui::Key::Y) {
                            redo += 1;
                        }
                        do_copy |= i.key_pressed(egui::Key::C);
                        do_cut |= i.key_pressed(egui::Key::X);
                        do_paste |= i.key_pressed(egui::Key::V);
                    }
                });
                if do_copy {
                    self.copy_selection(frame);
                }
                if do_cut {
                    self.cut_selection(frame);
                }
                if do_paste {
                    self.paste(frame);
                }

                let img = egui::vec2(self.doc.size.width as f32, self.doc.size.height as f32);
                let doc_rect = egui::Rect::from_center_size(
                    rect.center() + self.view.pan,
                    img * self.view.zoom,
                );

                let mut dabs: Vec<Dab> = Vec::new();
                let mut begin_command = false;
                let mut commit_command = false;
                let mut wet_begin = false;
                let mut wet_end = false;
                let mut bake = false;
                let erase = self.tool == Tool::Eraser;
                let paint_mask = self.edit_mask && self.masked_layers.contains(&self.active_id());
                let dirty_radius = self.brush_size * 0.5 + 1.0; // max dab extent
                                                                // Brush paints full-coverage dabs into the wet layer (opacity is
                                                                // applied when flattening). Eraser paints directly at its strength.
                let dab_alpha = if erase { self.brush_opacity } else { 1.0 };
                match self.tool {
                    Tool::Move => {
                        if response.dragged() {
                            self.view.pan += response.drag_delta();
                        }
                    }
                    Tool::MoveLayer | Tool::Transform => {
                        let allow_scale = self.tool == Tool::Transform;
                        if response.drag_started() {
                            self.xform_active = true;
                            self.xform_translate = egui::Vec2::ZERO;
                            self.xform_scale = 1.0;
                            begin_command = true;
                        }
                        if response.dragged() {
                            if allow_scale && ui.input(|i| i.modifiers.shift) {
                                self.xform_scale = (self.xform_scale
                                    * (1.0 - response.drag_delta().y * 0.005))
                                    .clamp(0.05, 20.0);
                            } else {
                                self.xform_translate +=
                                    response.drag_delta() / self.view.zoom.max(1e-3);
                            }
                            ui.ctx().request_repaint();
                        }
                        if response.drag_stopped() && self.xform_active {
                            bake = true;
                            commit_command = true;
                            self.xform_active = false;
                        }
                    }
                    Tool::Crop => {}
                    Tool::Text => {
                        if response.clicked() {
                            let id = self.doc.layers.add_text(TextDef::default());
                            self.doc.active_layer = Some(id);
                        }
                    }
                    Tool::ShapeRect | Tool::ShapeEllipse => {
                        let kind = if self.tool == Tool::ShapeEllipse {
                            1
                        } else {
                            0
                        };
                        if response.drag_started() {
                            if let Some(p) = response.interact_pointer_pos() {
                                let s = screen_to_doc(p, doc_rect, self.doc.size);
                                self.sel_drag_start = Some(s);
                                let c = self.brush_color;
                                let color = [
                                    c.r() as f32 / 255.0,
                                    c.g() as f32 / 255.0,
                                    c.b() as f32 / 255.0,
                                    1.0,
                                ];
                                let id = self.doc.layers.add_vector(
                                    "Shape",
                                    VectorDef {
                                        kind,
                                        rect: [s.x, s.y, 0.0, 0.0],
                                        color,
                                    },
                                );
                                self.doc.active_layer = Some(id);
                                self.shape_drag = Some(id);
                            }
                        }
                        if response.dragged() {
                            if let (Some(s), Some(p), Some(id)) = (
                                self.sel_drag_start,
                                response.interact_pointer_pos(),
                                self.shape_drag,
                            ) {
                                let cur = screen_to_doc(p, doc_rect, self.doc.size);
                                let rect = [
                                    s.x.min(cur.x),
                                    s.y.min(cur.y),
                                    (s.x - cur.x).abs(),
                                    (s.y - cur.y).abs(),
                                ];
                                if let Some(LayerKind::Vector(v)) =
                                    self.doc.layers.get_mut(id).map(|l| &mut l.kind)
                                {
                                    v.rect = rect;
                                }
                            }
                            ui.ctx().request_repaint();
                        }
                        if response.drag_stopped() {
                            self.shape_drag = None;
                            self.sel_drag_start = None;
                        }
                    }
                    Tool::Gradient => {
                        if response.drag_started() {
                            if let Some(p) = response.interact_pointer_pos() {
                                self.grad_start = Some(screen_to_doc(p, doc_rect, self.doc.size));
                            }
                        }
                        if response.dragged() {
                            ui.ctx().request_repaint();
                        }
                        if response.drag_stopped() {
                            if let (Some(s), Some(p)) =
                                (self.grad_start.take(), response.interact_pointer_pos())
                            {
                                let cur = screen_to_doc(p, doc_rect, self.doc.size);
                                self.do_gradient(frame, s, cur);
                            }
                        }
                        // Guide line.
                        if let (Some(s), Some(p)) = (self.grad_start, response.hover_pos()) {
                            let a = doc_to_screen(s, doc_rect, self.doc.size);
                            ui.painter().add(egui::Shape::line_segment(
                                [a, p],
                                egui::Stroke::new(1.5, egui::Color32::WHITE),
                            ));
                        }
                    }
                    Tool::Pen => {
                        // CLOSE_HIT: screen-px radius for clicking the first anchor
                        // to close, and the doc-px radius derived from it.
                        const CLOSE_PX: f32 = 8.0;
                        let close_doc =
                            CLOSE_PX / (doc_rect.width() / self.doc.size.width.max(1) as f32).max(1e-3);
                        if response.drag_started() {
                            // A press that turns into a drag: place an anchor, then
                            // drag to set its symmetric Bézier handles.
                            if let Some(p) = response.interact_pointer_pos() {
                                let d = screen_to_doc(p, doc_rect, self.doc.size);
                                self.pen_place_or_close(d, close_doc);
                                self.pen_dragging = !self.work_path.closed;
                            }
                        }
                        if response.dragged() {
                            if self.pen_dragging {
                                if let Some(p) = response.interact_pointer_pos() {
                                    let d = screen_to_doc(p, doc_rect, self.doc.size);
                                    if let Some(a) = self.work_path.anchors.last_mut() {
                                        a.set_smooth_out([d.x, d.y]);
                                    }
                                }
                            }
                            ui.ctx().request_repaint();
                        }
                        if response.drag_stopped() {
                            self.pen_dragging = false;
                        }
                        // A plain click (no drag) also places a corner anchor.
                        if response.clicked() {
                            if let Some(p) = response.interact_pointer_pos() {
                                let d = screen_to_doc(p, doc_rect, self.doc.size);
                                self.pen_place_or_close(d, close_doc);
                            }
                        }
                        self.draw_path_overlay(ui, doc_rect, true);
                    }
                    Tool::DirectSelect => {
                        let pick_doc =
                            8.0 / (doc_rect.width() / self.doc.size.width.max(1) as f32).max(1e-3);
                        if response.drag_started() {
                            if let Some(p) = response.interact_pointer_pos() {
                                let d = screen_to_doc(p, doc_rect, self.doc.size);
                                self.pen_grab = self.path_pick(d, pick_doc);
                            }
                        }
                        if response.dragged() {
                            if let (Some((idx, part)), Some(p)) =
                                (self.pen_grab, response.interact_pointer_pos())
                            {
                                let d = screen_to_doc(p, doc_rect, self.doc.size);
                                self.path_move_grab(idx, part, [d.x, d.y]);
                            }
                            ui.ctx().request_repaint();
                        }
                        if response.drag_stopped() {
                            self.pen_grab = None;
                        }
                        self.draw_path_overlay(ui, doc_rect, true);
                    }
                    Tool::Brush | Tool::Eraser => {
                        let spacing = (self.brush_size * 0.15).max(0.75);
                        let dt = ui.input(|i| i.stable_dt).max(1e-3);
                        if let Some(p) = response.interact_pointer_pos() {
                            let cur = screen_to_doc(p, doc_rect, self.doc.size);
                            match self.stroke_last {
                                None => {
                                    begin_command = !paint_mask; // mask paint isn't undoable yet
                                    self.stroke_dirty = None;
                                    self.expand_dirty(cur, dirty_radius);
                                    if !erase && !paint_mask {
                                        wet_begin = true;
                                        self.wet_active = true;
                                    }
                                    dabs.push(self.dab_at(cur, dab_alpha, 1.0));
                                    self.stroke_last = Some(cur);
                                    self.stroke_residual = 0.0;
                                }
                                Some(last) => {
                                    let seg = cur - last;
                                    let dist = seg.length();
                                    if dist > 1e-3 {
                                        // Velocity → size: faster strokes taper thinner.
                                        let scale = if self.speed_dynamics {
                                            const SPEED_MAX: f32 = 2500.0; // doc px/sec
                                            let n = (dist / dt / SPEED_MAX).clamp(0.0, 1.0);
                                            1.0 - n * (1.0 - self.min_size_scale)
                                        } else {
                                            1.0
                                        };
                                        let dir = seg / dist;
                                        let mut t = self.stroke_residual;
                                        while t <= dist {
                                            dabs.push(self.dab_at(
                                                last + dir * t,
                                                dab_alpha,
                                                scale,
                                            ));
                                            t += spacing;
                                        }
                                        self.stroke_residual = t - dist;
                                        self.stroke_last = Some(cur);
                                        self.expand_dirty(cur, dirty_radius);
                                    }
                                }
                            }
                            ui.ctx().request_repaint();
                        }
                        if response.drag_stopped()
                            || (self.stroke_last.is_some()
                                && response.interact_pointer_pos().is_none())
                        {
                            if self.wet_active {
                                wet_end = true;
                                self.wet_active = false;
                            }
                            commit_command = true;
                            self.stroke_last = None;
                            self.stroke_residual = 0.0;
                        }
                    }
                    Tool::Clone => {
                        let alt = ui.input(|i| i.modifiers.alt);
                        let spacing = (self.brush_size * 0.15).max(0.75);
                        if let Some(p) = response.interact_pointer_pos() {
                            let cur = screen_to_doc(p, doc_rect, self.doc.size);
                            if alt {
                                // Alt-click sets the clone source anchor; no paint.
                                if response.drag_started() || response.clicked() {
                                    self.clone_source = Some(cur);
                                }
                            } else if let Some(src) = self.clone_source {
                                match self.stroke_last {
                                    None => {
                                        // Aligned: lock the offset at the first dab.
                                        self.clone_offset = [cur.x - src.x, cur.y - src.y];
                                        begin_command = true;
                                        self.stroke_dirty = None;
                                        self.expand_dirty(cur, dirty_radius);
                                        dabs.push(self.dab_at(cur, self.brush_opacity, 1.0));
                                        self.stroke_last = Some(cur);
                                        self.stroke_residual = 0.0;
                                    }
                                    Some(last) => {
                                        let seg = cur - last;
                                        let dist = seg.length();
                                        if dist > 1e-3 {
                                            let dir = seg / dist;
                                            let mut t = self.stroke_residual;
                                            while t <= dist {
                                                dabs.push(self.dab_at(
                                                    last + dir * t,
                                                    self.brush_opacity,
                                                    1.0,
                                                ));
                                                t += spacing;
                                            }
                                            self.stroke_residual = t - dist;
                                            self.stroke_last = Some(cur);
                                            self.expand_dirty(cur, dirty_radius);
                                        }
                                    }
                                }
                                ui.ctx().request_repaint();
                            }
                        }
                        if response.drag_stopped()
                            || (self.stroke_last.is_some()
                                && response.interact_pointer_pos().is_none())
                        {
                            commit_command = true;
                            self.stroke_last = None;
                            self.stroke_residual = 0.0;
                        }
                    }
                    Tool::Heal => {
                        let alt = ui.input(|i| i.modifiers.alt);
                        let r = self.brush_size * 0.5;
                        if let Some(p) = response.interact_pointer_pos() {
                            let cur = screen_to_doc(p, doc_rect, self.doc.size);
                            if alt {
                                if response.drag_started() || response.clicked() {
                                    self.clone_source = Some(cur);
                                }
                            } else if let Some(src) = self.clone_source {
                                match self.stroke_last {
                                    None => {
                                        self.clone_offset = [cur.x - src.x, cur.y - src.y];
                                        let n =
                                            (self.doc.size.width * self.doc.size.height) as usize;
                                        self.heal_mask = vec![false; n];
                                        self.heal_mark(cur, r);
                                        self.stroke_last = Some(cur);
                                    }
                                    Some(last) => {
                                        let seg = cur - last;
                                        let dist = seg.length();
                                        if dist > 1e-3 {
                                            let dir = seg / dist;
                                            let step = (r * 0.5).max(1.0);
                                            let mut t = 0.0;
                                            while t <= dist {
                                                self.heal_mark(last + dir * t, r);
                                                t += step;
                                            }
                                            self.heal_mark(cur, r);
                                            self.stroke_last = Some(cur);
                                        }
                                    }
                                }
                                ui.ctx().request_repaint();
                            }
                        }
                        if response.drag_stopped()
                            || (self.stroke_last.is_some()
                                && response.interact_pointer_pos().is_none())
                        {
                            let off = self.clone_offset;
                            self.do_heal(frame, off);
                            self.stroke_last = None;
                        }
                    }
                    Tool::SpotHeal => {
                        let r = self.brush_size * 0.5;
                        if let Some(p) = response.interact_pointer_pos() {
                            let cur = screen_to_doc(p, doc_rect, self.doc.size);
                            match self.stroke_last {
                                None => {
                                    let n = (self.doc.size.width * self.doc.size.height) as usize;
                                    self.heal_mask = vec![false; n];
                                    self.heal_mark(cur, r);
                                    self.stroke_last = Some(cur);
                                }
                                Some(last) => {
                                    let seg = cur - last;
                                    let dist = seg.length();
                                    if dist > 1e-3 {
                                        let dir = seg / dist;
                                        let step = (r * 0.5).max(1.0);
                                        let mut t = 0.0;
                                        while t <= dist {
                                            self.heal_mark(last + dir * t, r);
                                            t += step;
                                        }
                                        self.heal_mark(cur, r);
                                        self.stroke_last = Some(cur);
                                    }
                                }
                            }
                            ui.ctx().request_repaint();
                        }
                        if response.drag_stopped()
                            || (self.stroke_last.is_some()
                                && response.interact_pointer_pos().is_none())
                        {
                            self.do_spot_heal(frame);
                            self.stroke_last = None;
                        }
                    }
                    Tool::ContentFill => {
                        let r = self.brush_size * 0.5;
                        if let Some(p) = response.interact_pointer_pos() {
                            let cur = screen_to_doc(p, doc_rect, self.doc.size);
                            match self.stroke_last {
                                None => {
                                    let n = (self.doc.size.width * self.doc.size.height) as usize;
                                    self.heal_mask = vec![false; n];
                                    self.heal_mark(cur, r);
                                    self.stroke_last = Some(cur);
                                }
                                Some(last) => {
                                    let seg = cur - last;
                                    let dist = seg.length();
                                    if dist > 1e-3 {
                                        let dir = seg / dist;
                                        let step = (r * 0.5).max(1.0);
                                        let mut t = 0.0;
                                        while t <= dist {
                                            self.heal_mark(last + dir * t, r);
                                            t += step;
                                        }
                                        self.heal_mark(cur, r);
                                        self.stroke_last = Some(cur);
                                    }
                                }
                            }
                            ui.ctx().request_repaint();
                        }
                        if response.drag_stopped()
                            || (self.stroke_last.is_some()
                                && response.interact_pointer_pos().is_none())
                        {
                            self.do_content_fill(frame);
                            self.stroke_last = None;
                        }
                    }
                    Tool::Patch => {
                        let (w, h) = (self.doc.size.width, self.doc.size.height);
                        let has_region = self.patch_region.iter().any(|&m| m);
                        if !has_region {
                            // Phase 1: freehand-lasso the region to transplant.
                            if response.drag_started() {
                                self.patch_points.clear();
                                if let Some(p) = response.interact_pointer_pos() {
                                    self.patch_points
                                        .push(screen_to_doc(p, doc_rect, self.doc.size));
                                }
                            }
                            if response.dragged() {
                                if let Some(p) = response.interact_pointer_pos() {
                                    let d = screen_to_doc(p, doc_rect, self.doc.size);
                                    if self
                                        .patch_points
                                        .last()
                                        .is_none_or(|l| (*l - d).length() > 2.0)
                                    {
                                        self.patch_points.push(d);
                                    }
                                }
                                ui.ctx().request_repaint();
                            }
                            if response.drag_stopped() {
                                if self.patch_points.len() >= 3 {
                                    let pts: Vec<(f32, f32)> =
                                        self.patch_points.iter().map(|v| (v.x, v.y)).collect();
                                    let m = raster::polygon_mask(&pts, w, h);
                                    self.patch_region = m.iter().map(|&v| v > 0.5).collect();
                                    // Centroid = the handle the user grabs to drag.
                                    let (mut sx, mut sy, mut n) = (0.0f32, 0.0f32, 0.0f32);
                                    for (i, &on) in self.patch_region.iter().enumerate() {
                                        if on {
                                            sx += (i as u32 % w) as f32 + 0.5;
                                            sy += (i as u32 / w) as f32 + 0.5;
                                            n += 1.0;
                                        }
                                    }
                                    if n > 0.0 {
                                        self.patch_anchor = Some(egui::vec2(sx / n, sy / n));
                                    }
                                    self.patch_offset = egui::Vec2::ZERO;
                                }
                                self.patch_points.clear();
                            }
                            // Draw the in-progress lasso path.
                            if self.patch_points.len() >= 2 {
                                let pts: Vec<egui::Pos2> = self
                                    .patch_points
                                    .iter()
                                    .map(|v| doc_to_screen(*v, doc_rect, self.doc.size))
                                    .collect();
                                ui.painter().add(egui::Shape::line(
                                    pts,
                                    egui::Stroke::new(1.5, egui::Color32::WHITE),
                                ));
                            }
                        } else {
                            // Phase 2: drag the region onto a source area.
                            if response.drag_started() {
                                if let Some(p) = response.interact_pointer_pos() {
                                    self.stroke_last =
                                        Some(screen_to_doc(p, doc_rect, self.doc.size));
                                    self.patch_offset = egui::Vec2::ZERO;
                                }
                            }
                            if response.dragged() {
                                if let (Some(start), Some(p)) =
                                    (self.stroke_last, response.interact_pointer_pos())
                                {
                                    let cur = screen_to_doc(p, doc_rect, self.doc.size);
                                    self.patch_offset = cur - start;
                                }
                                ui.ctx().request_repaint();
                            }
                            if response.drag_stopped() {
                                let drag = self.patch_offset;
                                if drag.length() >= 1.0 {
                                    let region = self.patch_region.clone();
                                    // Source mode: lasso = destination, drag points at the
                                    // source texture (src[p] = image[p + drag]).
                                    // Destination mode: lasso = source, drag moves it onto
                                    // the destination (dest = region + drag).
                                    if self.patch_source_mode {
                                        self.do_patch(frame, region, [-drag.x, -drag.y]);
                                    } else {
                                        let dest = translate_mask(&region, w, h, drag.x, drag.y);
                                        self.do_patch(frame, dest, [drag.x, drag.y]);
                                    }
                                    // Consume the region after applying.
                                    self.patch_region.clear();
                                    self.patch_anchor = None;
                                }
                                self.patch_offset = egui::Vec2::ZERO;
                                self.stroke_last = None;
                            }
                            // Draw the committed region outline + its dragged ghost.
                            let to_screen =
                                |d: egui::Vec2| doc_to_screen(d, doc_rect, self.doc.size);
                            if let Some(anchor) = self.patch_anchor {
                                let a = to_screen(anchor);
                                let b = to_screen(anchor + self.patch_offset);
                                let col = egui::Color32::from_rgb(80, 200, 255);
                                let painter = ui.painter();
                                painter.circle_stroke(a, 5.0, egui::Stroke::new(1.5, col));
                                if self.patch_offset.length() > 0.5 {
                                    painter.line_segment([a, b], egui::Stroke::new(1.5, col));
                                    painter.circle_filled(b, 3.0, col);
                                }
                            }
                            // Right-click / Esc-style: clicking outside re-arms lasso.
                            if response.clicked_by(egui::PointerButton::Secondary) {
                                self.patch_region.clear();
                                self.patch_anchor = None;
                            }
                        }
                    }
                    Tool::Dodge => {
                        let burn = ui.input(|i| i.modifiers.alt);
                        let r = self.brush_size * 0.5;
                        let flow = (self.brush_opacity * 0.4).clamp(0.02, 1.0);
                        if let Some(p) = response.interact_pointer_pos() {
                            let cur = screen_to_doc(p, doc_rect, self.doc.size);
                            match self.stroke_last {
                                None => {
                                    let n = (self.doc.size.width * self.doc.size.height) as usize;
                                    self.tone_mask = vec![0.0; n];
                                    self.tone_mark(cur, r, flow);
                                    self.stroke_last = Some(cur);
                                }
                                Some(last) => {
                                    let seg = cur - last;
                                    let dist = seg.length();
                                    if dist > 1e-3 {
                                        let dir = seg / dist;
                                        let step = (r * 0.25).max(1.0);
                                        let mut t = 0.0;
                                        while t <= dist {
                                            self.tone_mark(last + dir * t, r, flow);
                                            t += step;
                                        }
                                        self.tone_mark(cur, r, flow);
                                        self.stroke_last = Some(cur);
                                    }
                                }
                            }
                            ui.ctx().request_repaint();
                        }
                        if response.drag_stopped()
                            || (self.stroke_last.is_some()
                                && response.interact_pointer_pos().is_none())
                        {
                            self.do_dodge_burn(frame, burn);
                            self.stroke_last = None;
                        }
                    }
                    Tool::Liquify => {
                        let (w, h) = (self.doc.size.width as usize, self.doc.size.height as usize);
                        let r = self.brush_size * 0.5;
                        let strength = self.brush_opacity.max(0.05);
                        if let Some(p) = response.interact_pointer_pos() {
                            let cur = screen_to_doc(p, doc_rect, self.doc.size);
                            if self.stroke_last.is_none() {
                                self.liquify_capture(frame);
                            }
                            if self.liquify_src.len() == w * h * 4 {
                                use prism_core::warp;
                                match self.liquify_mode {
                                    0 => {
                                        if let Some(last) = self.stroke_last {
                                            let mv = cur - last;
                                            if mv.length() > 1e-3 {
                                                warp::stamp_push(
                                                    &mut self.liquify_disp,
                                                    w,
                                                    h,
                                                    cur.x,
                                                    cur.y,
                                                    r,
                                                    mv.x,
                                                    mv.y,
                                                    strength,
                                                );
                                            }
                                        }
                                    }
                                    1 => warp::stamp_twirl(
                                        &mut self.liquify_disp,
                                        w,
                                        h,
                                        cur.x,
                                        cur.y,
                                        r,
                                        0.08 * strength,
                                        1.0,
                                    ),
                                    2 => warp::stamp_pinch(
                                        &mut self.liquify_disp,
                                        w,
                                        h,
                                        cur.x,
                                        cur.y,
                                        r,
                                        1.0,
                                        0.04 * strength,
                                    ),
                                    _ => warp::stamp_pinch(
                                        &mut self.liquify_disp,
                                        w,
                                        h,
                                        cur.x,
                                        cur.y,
                                        r,
                                        -1.0,
                                        0.04 * strength,
                                    ),
                                }
                                self.liquify_apply(frame);
                            }
                            self.stroke_last = Some(cur);
                            ui.ctx().request_repaint();
                        }
                        if response.drag_stopped()
                            || (self.stroke_last.is_some()
                                && response.interact_pointer_pos().is_none())
                        {
                            self.stroke_last = None;
                            self.liquify_src = Vec::new();
                            self.liquify_disp = Vec::new();
                        }
                    }
                    Tool::Detail => {
                        let r = self.brush_size * 0.5;
                        let flow = (self.brush_opacity * 0.5).clamp(0.02, 1.0);
                        if let Some(p) = response.interact_pointer_pos() {
                            let cur = screen_to_doc(p, doc_rect, self.doc.size);
                            match self.stroke_last {
                                None => {
                                    let n = (self.doc.size.width * self.doc.size.height) as usize;
                                    self.tone_mask = vec![0.0; n];
                                    self.tone_mark(cur, r, flow);
                                    self.stroke_last = Some(cur);
                                }
                                Some(last) => {
                                    let seg = cur - last;
                                    let dist = seg.length();
                                    if dist > 1e-3 {
                                        let dir = seg / dist;
                                        let step = (r * 0.25).max(1.0);
                                        let mut t = 0.0;
                                        while t <= dist {
                                            self.tone_mark(last + dir * t, r, flow);
                                            t += step;
                                        }
                                        self.tone_mark(cur, r, flow);
                                        self.stroke_last = Some(cur);
                                    }
                                }
                            }
                            ui.ctx().request_repaint();
                        }
                        if response.drag_stopped()
                            || (self.stroke_last.is_some()
                                && response.interact_pointer_pos().is_none())
                        {
                            self.do_detail(frame);
                            self.stroke_last = None;
                        }
                    }
                    Tool::Fill => {
                        if response.clicked() {
                            if let Some(p) = response.interact_pointer_pos() {
                                let d = screen_to_doc(p, doc_rect, self.doc.size);
                                if let Some(seed) = clamp_seed(d, self.doc.size) {
                                    self.do_fill(frame, seed);
                                }
                            }
                        }
                    }
                    Tool::Eyedropper => {
                        if response.clicked() {
                            if let Some(p) = response.interact_pointer_pos() {
                                let d = screen_to_doc(p, doc_rect, self.doc.size);
                                if let Some(seed) = clamp_seed(d, self.doc.size) {
                                    self.do_eyedrop(frame, seed);
                                }
                            }
                        }
                    }
                    Tool::SelectRect | Tool::SelectEllipse => {
                        let ellipse = self.tool == Tool::SelectEllipse;
                        if response.drag_started() {
                            if let Some(p) = response.interact_pointer_pos() {
                                self.sel_drag_start =
                                    Some(screen_to_doc(p, doc_rect, self.doc.size));
                                self.sel_mode = mode_from_modifiers(ui.input(|i| i.modifiers));
                                self.sel_base = self.read_selection(frame);
                            }
                        }
                        if response.dragged() {
                            if let (Some(start), Some(p)) =
                                (self.sel_drag_start, response.interact_pointer_pos())
                            {
                                let cur = screen_to_doc(p, doc_rect, self.doc.size);
                                let rect = [
                                    start.x.min(cur.x),
                                    start.y.min(cur.y),
                                    (start.x - cur.x).abs(),
                                    (start.y - cur.y).abs(),
                                ];
                                let (w, h) = (self.doc.size.width, self.doc.size.height);
                                let shape = shape_mask(rect, ellipse, w, h);
                                self.commit_selection(frame, shape);
                            }
                            ui.ctx().request_repaint();
                        }
                        if response.drag_stopped() {
                            self.sel_drag_start = None;
                        }
                    }
                    Tool::Lasso => {
                        if response.drag_started() {
                            self.lasso_points.clear();
                            self.sel_mode = mode_from_modifiers(ui.input(|i| i.modifiers));
                            self.sel_base = self.read_selection(frame);
                            if let Some(p) = response.interact_pointer_pos() {
                                self.lasso_points
                                    .push(screen_to_doc(p, doc_rect, self.doc.size));
                            }
                        }
                        if response.dragged() {
                            if let Some(p) = response.interact_pointer_pos() {
                                let d = screen_to_doc(p, doc_rect, self.doc.size);
                                if self
                                    .lasso_points
                                    .last()
                                    .is_none_or(|l| (*l - d).length() > 2.0)
                                {
                                    self.lasso_points.push(d);
                                }
                            }
                            ui.ctx().request_repaint();
                        }
                        if response.drag_stopped() {
                            if self.lasso_points.len() >= 3 {
                                let (w, h) = (self.doc.size.width, self.doc.size.height);
                                let pts: Vec<(f32, f32)> =
                                    self.lasso_points.iter().map(|v| (v.x, v.y)).collect();
                                let mask = raster::polygon_mask(&pts, w, h);
                                self.commit_selection(frame, mask);
                            }
                            self.lasso_points.clear();
                        }
                        // Draw the in-progress lasso path.
                        if self.lasso_points.len() >= 2 {
                            let pts: Vec<egui::Pos2> = self
                                .lasso_points
                                .iter()
                                .map(|v| doc_to_screen(*v, doc_rect, self.doc.size))
                                .collect();
                            ui.painter().add(egui::Shape::line(
                                pts,
                                egui::Stroke::new(1.5, egui::Color32::WHITE),
                            ));
                        }
                    }
                    Tool::MagicWand => {
                        if response.clicked() {
                            if let Some(p) = response.interact_pointer_pos() {
                                let d = screen_to_doc(p, doc_rect, self.doc.size);
                                if let Some(seed) = clamp_seed(d, self.doc.size) {
                                    self.sel_mode = mode_from_modifiers(ui.input(|i| i.modifiers));
                                    self.sel_base = self.read_selection(frame);
                                    self.do_magic_wand(frame, seed);
                                }
                            }
                        }
                    }
                }
                // Painting into a mask reveals (white); the eraser hides.
                if paint_mask && !erase {
                    for d in &mut dabs {
                        d.color = [1.0, 1.0, 1.0, 1.0];
                    }
                }

                // Re-rasterize edited text/vector layers before compositing.
                self.sync_generated_layers(frame);

                let layers = self.layer_order();

                // Recomposite only when content/structure changed (not on pan/zoom).
                let fp = self.layer_fingerprint();
                let dirty = !dabs.is_empty()
                    || wet_begin
                    || wet_end
                    || bake
                    || self.xform_active
                    || begin_command
                    || undo > 0
                    || redo > 0
                    || self.force_composite
                    || fp != self.last_fingerprint;
                let command_label = match self.tool {
                    Tool::Eraser => "Erase",
                    Tool::MoveLayer => "Move",
                    Tool::Transform => "Transform",
                    Tool::Clone => "Clone Stamp",
                    _ => "Brush",
                };
                // Keep the affine live for the bake frame too: `drag_stopped`
                // clears `xform_active` before this point, but the accumulated
                // translate/scale must still reach `set_layer_transform` so the
                // bake has something to bake (otherwise the move snaps back).
                let xform = if send_layer_xform(self.xform_active, bake) {
                    Some(compute_xform(
                        self.xform_translate,
                        self.xform_scale,
                        self.doc.size,
                    ))
                } else {
                    None
                };
                // A Move/Transform bake of a generated (text/vector) layer shifts
                // its pixels but not its definition. Record the translate so a
                // later re-rasterize (font/size/color/align change) re-places the
                // fresh pixels here instead of snapping back to the canvas origin.
                if bake && self.xform_translate != egui::Vec2::ZERO {
                    let id = self.active_id();
                    if self.is_generated_layer(id) {
                        *self.gen_offset.entry(id).or_default() += self.xform_translate;
                    }
                }
                self.last_fingerprint = fp;
                self.force_composite = false;

                // Upload Curves LUTs before the paint callback composites.
                if dirty {
                    self.sync_curve_luts(frame);
                }

                // Keep marching ants animating while a selection is active.
                if self.selection_active {
                    ui.ctx().request_repaint();
                }
                let time = ui.input(|i| i.time) as f32;

                ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                    rect,
                    CanvasPaint {
                        doc_rect,
                        checker_pts: 10.0,
                        canvas_size: self.doc.size,
                        layers,
                        active_id: self.active_id(),
                        dabs,
                        erase,
                        begin_command,
                        command_label: command_label.into(),
                        commit_command,
                        dirty_rect: self.dirty_rect(),
                        undo,
                        redo,
                        wet_begin,
                        wet_end,
                        wet_opacity: self.brush_opacity,
                        paint_into_wet: self.wet_active,
                        paint_mask,
                        clone: self.tool == Tool::Clone,
                        clone_offset: self.clone_offset,
                        dirty,
                        selection_op: self.sel_op_pending.take(),
                        time,
                        xform,
                        bake,
                    },
                ));

                // Clone / Heal source marker (crosshair at the sampled point).
                if matches!(self.tool, Tool::Clone | Tool::Heal) {
                    if let Some(src) = self.clone_source {
                        let sp = egui::pos2(
                            doc_rect.min.x
                                + src.x / self.doc.size.width.max(1) as f32 * doc_rect.width(),
                            doc_rect.min.y
                                + src.y / self.doc.size.height.max(1) as f32 * doc_rect.height(),
                        );
                        let col = egui::Color32::from_rgb(80, 200, 255);
                        let painter = ui.painter();
                        painter.circle_stroke(sp, 6.0, egui::Stroke::new(1.5, col));
                        painter.line_segment(
                            [sp - egui::vec2(9.0, 0.0), sp + egui::vec2(9.0, 0.0)],
                            egui::Stroke::new(1.0, col),
                        );
                        painter.line_segment(
                            [sp - egui::vec2(0.0, 9.0), sp + egui::vec2(0.0, 9.0)],
                            egui::Stroke::new(1.0, col),
                        );
                    }
                }
            });
    }
}
