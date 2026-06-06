use super::*;

fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::default();
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .ok()?;
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
}

/// A solid `n`x`n` buffer of linear-premultiplied RGBA16F bytes.
fn solid(n: u32, r: f32, g: f32, b: f32, a: f32) -> Vec<u8> {
    let px = [r * a, g * a, b * a, a];
    let mut o = Vec::new();
    for _ in 0..(n * n) {
        for &c in &px {
            o.extend_from_slice(&half::f16::from_f32(c).to_le_bytes());
        }
    }
    o
}

// Drives the real GPU compositor: upload -> composite -> brush(wet)+flatten
// -> undo, asserting pixels via readback. Skips if no GPU adapter.
#[test]
fn compositor_brush_wet_undo() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping compositor_brush_wet_undo");
        return;
    };
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    let size = Size::new(8, 8);
    gpu.ensure_canvas(&device, size);
    let l0 = LayerId(0);
    gpu.ensure_layer(&device, l0);
    gpu.upload_layer(&queue, l0, &solid(8, 1.0, 0.0, 0.0, 1.0)); // red

    let order = vec![LayerDraw {
        id: l0,
        opacity: 1.0,
        blend: 0,
        visible: true,
        adjust_kind: 0,
        adjust: [0.0; 4],
        has_blend_if: false,
        blend_if: [0.0, 1.0, 0.0, 1.0],
        clipped: false,
    }];
    let p = gpu.composite_now(&device, &queue, &order);
    let px = gpu.read_composite_pixel(&device, &queue, p, 4, 4).unwrap();
    assert!(
        px[0] > 0.9 && px[1] < 0.1 && px[3] > 0.9,
        "composite red: {px:?}"
    );

    // Blue wet brush dab over the center, flattened.
    let mut enc = device.create_command_encoder(&Default::default());
    gpu.begin_command(&device, &mut enc, l0, "test");
    gpu.wet_begin(&mut enc, l0, 1.0);
    let dab = Dab {
        center: [2.0, 2.0],
        radius: 2.5,
        hardness: 0.95,
        color: [0.0, 0.0, 1.0, 1.0],
    };
    gpu.paint_dabs(&device, &queue, &mut enc, l0, &[dab], false, true, false);
    gpu.wet_end(&device, &queue, &mut enc);
    // Region-COW: only snapshot the touched corner.
    gpu.commit_command(&device, &mut enc, [0, 0, 5, 5]);
    queue.submit([enc.finish()]);

    let p = gpu.composite_now(&device, &queue, &order);
    let near = gpu.read_composite_pixel(&device, &queue, p, 2, 2).unwrap();
    let far = gpu.read_composite_pixel(&device, &queue, p, 7, 7).unwrap();
    assert!(
        near[2] > 0.9 && near[0] < 0.1,
        "baked blue at stroke: {near:?}"
    );
    assert!(
        far[0] > 0.9 && far[2] < 0.1,
        "far pixel untouched red: {far:?}"
    );

    // Undo restores the region to red.
    let mut enc = device.create_command_encoder(&Default::default());
    gpu.undo(&device, &mut enc);
    queue.submit([enc.finish()]);
    let p = gpu.composite_now(&device, &queue, &order);
    let near = gpu.read_composite_pixel(&device, &queue, p, 2, 2).unwrap();
    assert!(near[0] > 0.9 && near[2] < 0.1, "undo back to red: {near:?}");
}

// A rectangle selection must clip painting to the selected region.
#[test]
fn selection_clips_paint() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping selection_clips_paint");
        return;
    };
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    let size = Size::new(8, 8);
    gpu.ensure_canvas(&device, size);
    let l0 = LayerId(0);
    gpu.ensure_layer(&device, l0);
    gpu.upload_layer(&queue, l0, &solid(8, 1.0, 0.0, 0.0, 1.0)); // red
    let order = vec![LayerDraw {
        id: l0,
        opacity: 1.0,
        blend: 0,
        visible: true,
        adjust_kind: 0,
        adjust: [0.0; 4],
        has_blend_if: false,
        blend_if: [0.0, 1.0, 0.0, 1.0],
        clipped: false,
    }];

    // Select the left half, then paint blue over the whole canvas.
    let mut enc = device.create_command_encoder(&Default::default());
    gpu.apply_selection(
        &device,
        &queue,
        &mut enc,
        &SelectionOp::Marquee {
            rect: [0.0, 0.0, 4.0, 8.0],
            ellipse: false,
        },
    );
    let dab = Dab {
        center: [4.0, 4.0],
        radius: 16.0,
        hardness: 0.99,
        color: [0.0, 0.0, 1.0, 1.0],
    };
    gpu.paint_dabs(&device, &queue, &mut enc, l0, &[dab], false, false, false);
    queue.submit([enc.finish()]);
    assert!(gpu.has_selection());

    let p = gpu.composite_now(&device, &queue, &order);
    let inside = gpu.read_composite_pixel(&device, &queue, p, 1, 4).unwrap();
    let outside = gpu.read_composite_pixel(&device, &queue, p, 6, 4).unwrap();
    assert!(inside[2] > 0.5, "selected area painted blue: {inside:?}");
    assert!(
        outside[0] > 0.5 && outside[2] < 0.2,
        "unselected area untouched red: {outside:?}"
    );
}

// Translating a layer by +half-width then baking should clear the left half
// and keep the right half.
#[test]
fn transform_bake_translates() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping transform_bake_translates");
        return;
    };
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    let size = Size::new(8, 8);
    gpu.ensure_canvas(&device, size);
    let l0 = LayerId(0);
    gpu.ensure_layer(&device, l0);
    gpu.upload_layer(&queue, l0, &solid(8, 1.0, 0.0, 0.0, 1.0)); // red everywhere

    // Move right by half width: layer-from-canvas off.x = -0.5.
    gpu.set_layer_transform(Some(l0), [1.0, 0.0, 0.0, 1.0], [-0.5, 0.0]);
    let mut enc = device.create_command_encoder(&Default::default());
    gpu.bake_transform(&device, &queue, &mut enc);
    queue.submit([enc.finish()]);

    let left = gpu.read_pixel(&device, &queue, l0, 1, 4).unwrap();
    let right = gpu.read_pixel(&device, &queue, l0, 6, 4).unwrap();
    assert!(left[3] < 0.1, "left half cleared after move: {left:?}");
    assert!(
        right[0] > 0.9 && right[3] > 0.9,
        "right half keeps red: {right:?}"
    );
}

// An Invert adjustment layer over a red layer yields cyan in the composite.
#[test]
fn adjustment_invert() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping adjustment_invert");
        return;
    };
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    let size = Size::new(8, 8);
    gpu.ensure_canvas(&device, size);
    let l0 = LayerId(0);
    let l1 = LayerId(1);
    gpu.ensure_layer(&device, l0);
    gpu.ensure_layer(&device, l1); // adjustment layer (no pixels needed)
    gpu.upload_layer(&queue, l0, &solid(8, 1.0, 0.0, 0.0, 1.0)); // red

    let (k, p) = prism_core::adjust::Adjustment::Invert.encode();
    let order = vec![
        LayerDraw {
            id: l0,
            opacity: 1.0,
            blend: 0,
            visible: true,
            adjust_kind: 0,
            adjust: [0.0; 4],
            has_blend_if: false,
            blend_if: [0.0, 1.0, 0.0, 1.0],
            clipped: false,
        },
        LayerDraw {
            id: l1,
            opacity: 1.0,
            blend: 0,
            visible: true,
            adjust_kind: k,
            adjust: p,
            has_blend_if: false,
            blend_if: [0.0, 1.0, 0.0, 1.0],
            clipped: false,
        },
    ];
    let pp = gpu.composite_now(&device, &queue, &order);
    let px = gpu.read_composite_pixel(&device, &queue, pp, 4, 4).unwrap();
    // Inverted red (premultiplied, alpha 1): low red, high green+blue.
    assert!(
        px[0] < 0.2 && px[1] > 0.8 && px[2] > 0.8,
        "invert red -> cyan: {px:?}"
    );
}

// A Curves adjustment layer with an inverting master curve turns red -> cyan,
// exercising the full LUT build + upload + shader-sample path.
#[test]
fn curves_invert_master() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping curves_invert_master");
        return;
    };
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    let size = Size::new(8, 8);
    gpu.ensure_canvas(&device, size);
    let l0 = LayerId(0);
    let l1 = LayerId(1);
    gpu.ensure_layer(&device, l0);
    gpu.ensure_layer(&device, l1); // adjustment layer (no pixels needed)
    gpu.upload_layer(&queue, l0, &solid(8, 1.0, 0.0, 0.0, 1.0)); // red

    // Master curve inverts (0->1, 1->0); per-channel curves stay identity.
    let inv = [(0.0, 1.0), (1.0, 0.0)];
    let idc = [(0.0, 0.0), (1.0, 1.0)];
    gpu.set_curve_lut(&device, &queue, l1, &inv, &idc, &idc, &idc);

    let order = vec![
        LayerDraw {
            id: l0,
            opacity: 1.0,
            blend: 0,
            visible: true,
            adjust_kind: 0,
            adjust: [0.0; 4],
            has_blend_if: false,
            blend_if: [0.0, 1.0, 0.0, 1.0],
            clipped: false,
        },
        LayerDraw {
            id: l1,
            opacity: 1.0,
            blend: 0,
            visible: true,
            adjust_kind: 8, // Curves
            adjust: [0.0; 4],
            has_blend_if: false,
            blend_if: [0.0, 1.0, 0.0, 1.0],
            clipped: false,
        },
    ];
    let pp = gpu.composite_now(&device, &queue, &order);
    let px = gpu.read_composite_pixel(&device, &queue, pp, 4, 4).unwrap();
    assert!(
        px[0] < 0.2 && px[1] > 0.8 && px[2] > 0.8,
        "curves invert red -> cyan: {px:?}"
    );
}

// Clone stamp: source = left-half green / right-half red. Stamping the right
// half with offset +4px samples the green left half, so the dab paints green.
#[test]
fn clone_stamp_copies_source() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping clone_stamp_copies_source");
        return;
    };
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    let size = Size::new(8, 8);
    gpu.ensure_canvas(&device, size); // selection exists, has_selection = false
    let l0 = LayerId(0);
    gpu.ensure_layer(&device, l0);

    // Left half (x<4) green, right half red — premultiplied f16.
    let mut buf = Vec::new();
    for _y in 0..8 {
        for x in 0..8 {
            let px = if x < 4 {
                [0.0, 1.0, 0.0, 1.0]
            } else {
                [1.0, 0.0, 0.0, 1.0]
            };
            for &c in &px {
                buf.extend_from_slice(&half::f16::from_f32(c).to_le_bytes());
            }
        }
    }
    gpu.upload_layer(&queue, l0, &buf);

    // Stamp the right half sampling 4px to the left (green source).
    let dab = Dab {
        center: [6.0, 4.0],
        radius: 2.0,
        hardness: 0.99,
        color: [0.0, 0.0, 0.0, 1.0],
    };
    let mut enc = device.create_command_encoder(&Default::default());
    gpu.snapshot_clone_source(&device, &mut enc, l0);
    gpu.paint_clone_dabs(&device, &queue, &mut enc, l0, &[dab], [4.0, 0.0]);
    queue.submit([enc.finish()]);

    let px = gpu.read_pixel(&device, &queue, l0, 6, 4).unwrap();
    assert!(
        px[1] > 0.8 && px[0] < 0.2,
        "clone stamped green over red: {px:?}"
    );
}

// A Posterize(2) adjustment over a mid-gray layer snaps it to white (the
// sRGB-space value rounds up at 2 levels) — exercises a new adjustment kind.
#[test]
fn posterize_adjustment() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping posterize_adjustment");
        return;
    };
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    let size = Size::new(8, 8);
    gpu.ensure_canvas(&device, size);
    let l0 = LayerId(0);
    let l1 = LayerId(1);
    gpu.ensure_layer(&device, l0);
    gpu.ensure_layer(&device, l1);
    gpu.upload_layer(&queue, l0, &solid(8, 0.6, 0.6, 0.6, 1.0)); // mid gray

    let (k, p) = prism_core::adjust::Adjustment::Posterize { levels: 2 }.encode();
    let order = vec![
        LayerDraw {
            id: l0,
            opacity: 1.0,
            blend: 0,
            visible: true,
            adjust_kind: 0,
            adjust: [0.0; 4],
            has_blend_if: false,
            blend_if: [0.0, 1.0, 0.0, 1.0],
            clipped: false,
        },
        LayerDraw {
            id: l1,
            opacity: 1.0,
            blend: 0,
            visible: true,
            adjust_kind: k,
            adjust: p,
            has_blend_if: false,
            blend_if: [0.0, 1.0, 0.0, 1.0],
            clipped: false,
        },
    ];
    let pp = gpu.composite_now(&device, &queue, &order);
    let px = gpu.read_composite_pixel(&device, &queue, pp, 4, 4).unwrap();
    assert!(
        px[0] > 0.9,
        "posterize(2) snaps mid gray up to white: {px:?}"
    );
}

// Blend-If: a gray top layer with "this layer white" pulled below its luma is
// hidden, revealing the white layer beneath.
#[test]
fn blend_if_hides_bright_source() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping blend_if_hides_bright_source");
        return;
    };
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    let size = Size::new(8, 8);
    gpu.ensure_canvas(&device, size);
    let l0 = LayerId(0);
    let l1 = LayerId(1);
    gpu.ensure_layer(&device, l0);
    gpu.ensure_layer(&device, l1);
    gpu.upload_layer(&queue, l0, &solid(8, 1.0, 1.0, 1.0, 1.0)); // white base
    gpu.upload_layer(&queue, l1, &solid(8, 0.5, 0.5, 0.5, 1.0)); // gray top

    // this_white = 0.45 < gray luma 0.5 -> top fully gated out.
    let order = vec![
        LayerDraw {
            id: l0,
            opacity: 1.0,
            blend: 0,
            visible: true,
            adjust_kind: 0,
            adjust: [0.0; 4],
            has_blend_if: false,
            blend_if: [0.0, 1.0, 0.0, 1.0],
            clipped: false,
        },
        LayerDraw {
            id: l1,
            opacity: 1.0,
            blend: 0,
            visible: true,
            adjust_kind: 0,
            adjust: [0.0; 4],
            has_blend_if: true,
            blend_if: [0.0, 0.45, 0.0, 1.0],
            clipped: false,
        },
    ];
    let pp = gpu.composite_now(&device, &queue, &order);
    let px = gpu.read_composite_pixel(&device, &queue, pp, 4, 4).unwrap();
    assert!(
        px[0] > 0.9,
        "blend-if hides gray top -> white shows through: {px:?}"
    );
}

// Clipping mask: a green top layer clipped to a base that's opaque on the left
// and transparent on the right shows green only on the left.
#[test]
fn clipping_mask_gates_by_base_alpha() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping clipping_mask_gates_by_base_alpha");
        return;
    };
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    let size = Size::new(8, 8);
    gpu.ensure_canvas(&device, size);
    let l0 = LayerId(0);
    let l1 = LayerId(1);
    gpu.ensure_layer(&device, l0);
    gpu.ensure_layer(&device, l1);
    // Base: left half opaque red, right half transparent.
    let mut base = Vec::new();
    for _y in 0..8 {
        for x in 0..8 {
            let px = if x < 4 {
                [1.0, 0.0, 0.0, 1.0]
            } else {
                [0.0, 0.0, 0.0, 0.0]
            };
            for &c in &px {
                base.extend_from_slice(&half::f16::from_f32(c).to_le_bytes());
            }
        }
    }
    gpu.upload_layer(&queue, l0, &base);
    gpu.upload_layer(&queue, l1, &solid(8, 0.0, 1.0, 0.0, 1.0)); // green top

    let order = vec![
        LayerDraw {
            id: l0,
            opacity: 1.0,
            blend: 0,
            visible: true,
            adjust_kind: 0,
            adjust: [0.0; 4],
            has_blend_if: false,
            blend_if: [0.0, 1.0, 0.0, 1.0],
            clipped: false,
        },
        LayerDraw {
            id: l1,
            opacity: 1.0,
            blend: 0,
            visible: true,
            adjust_kind: 0,
            adjust: [0.0; 4],
            has_blend_if: false,
            blend_if: [0.0, 1.0, 0.0, 1.0],
            clipped: true,
        },
    ];
    let pp = gpu.composite_now(&device, &queue, &order);
    let left = gpu.read_composite_pixel(&device, &queue, pp, 2, 4).unwrap();
    let right = gpu.read_composite_pixel(&device, &queue, pp, 6, 4).unwrap();
    assert!(left[1] > 0.8, "green shows over opaque base: {left:?}");
    assert!(
        right[3] < 0.1,
        "clipped out where base is transparent: {right:?}"
    );
}

// A layer mask (0 on the left half) hides those pixels in the composite.
#[test]
fn layer_mask_hides() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping layer_mask_hides");
        return;
    };
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    let size = Size::new(8, 8);
    gpu.ensure_canvas(&device, size);
    let l0 = LayerId(0);
    gpu.ensure_layer(&device, l0);
    gpu.upload_layer(&queue, l0, &solid(8, 1.0, 0.0, 0.0, 1.0)); // red
    let mut mvals = vec![0.0f32; 64];
    for y in 0..8 {
        for x in 4..8 {
            mvals[y * 8 + x] = 1.0; // reveal right half
        }
    }
    gpu.set_mask(&device, &queue, l0, Some(&mvals));
    let order = vec![LayerDraw {
        id: l0,
        opacity: 1.0,
        blend: 0,
        visible: true,
        adjust_kind: 0,
        adjust: [0.0; 4],
        has_blend_if: false,
        blend_if: [0.0, 1.0, 0.0, 1.0],
        clipped: false,
    }];
    let p = gpu.composite_now(&device, &queue, &order);
    let left = gpu.read_composite_pixel(&device, &queue, p, 1, 4).unwrap();
    let right = gpu.read_composite_pixel(&device, &queue, p, 6, 4).unwrap();
    assert!(left[3] < 0.1, "masked-out left is transparent: {left:?}");
    assert!(
        right[0] > 0.9 && right[3] > 0.9,
        "revealed right is red: {right:?}"
    );
}
