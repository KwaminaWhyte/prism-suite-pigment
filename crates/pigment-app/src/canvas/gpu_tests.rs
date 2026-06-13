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

/// A plain, no-style LayerDraw for `id` — new style tests fill only the fields
/// they exercise via struct-update syntax.
fn base_draw(id: LayerId) -> LayerDraw {
    LayerDraw {
        id,
        opacity: 1.0,
        blend: 0,
        visible: true,
        adjust_kind: 0,
        adjust: [0.0; 4],
        mix_r: [1.0, 0.0, 0.0, 0.0],
        mix_g: [0.0, 1.0, 0.0, 0.0],
        mix_b: [0.0, 0.0, 1.0, 0.0],
        has_blend_if: false,
        blend_if: [0.0, 1.0, 0.0, 1.0],
        clipped: false,
        has_stroke: false,
        stroke_color: [0.0; 4],
        stroke_width: 0.0,
        has_shadow: false,
        shadow_color: [0.0; 4],
        shadow_offset: [0.0; 2],
        shadow_blur: 0.0,
        has_overlay: false,
        overlay_color: [0.0; 4],
        has_inner_shadow: false,
        inner_shadow_color: [0.0; 4],
        inner_shadow_offset: [0.0; 2],
        inner_shadow_blur: 0.0,
        has_outer_glow: false,
        outer_glow_color: [0.0; 4],
        outer_glow_size: 0.0,
        has_inner_glow: false,
        inner_glow_color: [0.0; 4],
        inner_glow_size: 0.0,
        has_grad_overlay: false,
        grad_color0: [0.0; 4],
        grad_color1: [0.0; 4],
        grad_angle: 0.0,
        grad_opacity: 0.0,
        has_bevel: false,
        bevel_highlight: [0.0; 4],
        bevel_shadow: [0.0; 4],
        bevel_size: 0.0,
        bevel_soften: 0.0,
        bevel_angle: 0.0,
        bevel_altitude: 0.0,
    }
}

/// A 16x16 layer with a 4x4 opaque white square at x6..10, y6..10; rest empty.
#[cfg(test)]
fn square_16() -> Vec<u8> {
    let mut buf = Vec::new();
    for y in 0..16 {
        for x in 0..16 {
            let px = if (6..10).contains(&x) && (6..10).contains(&y) {
                [1.0f32, 1.0, 1.0, 1.0]
            } else {
                [0.0, 0.0, 0.0, 0.0]
            };
            for &c in &px {
                buf.extend_from_slice(&half::f16::from_f32(c).to_le_bytes());
            }
        }
    }
    buf
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
        mix_r: [1.0, 0.0, 0.0, 0.0],
        mix_g: [0.0, 1.0, 0.0, 0.0],
        mix_b: [0.0, 0.0, 1.0, 0.0],
        has_blend_if: false,
        blend_if: [0.0, 1.0, 0.0, 1.0],
        clipped: false,
        has_stroke: false,
        stroke_color: [0.0; 4],
        stroke_width: 0.0,
        has_shadow: false,
        shadow_color: [0.0; 4],
        shadow_offset: [0.0; 2],
        shadow_blur: 0.0,
        has_overlay: false,
        overlay_color: [0.0; 4],
        has_inner_shadow: false,
        inner_shadow_color: [0.0; 4],
        inner_shadow_offset: [0.0; 2],
        inner_shadow_blur: 0.0,
        has_outer_glow: false,
        outer_glow_color: [0.0; 4],
        outer_glow_size: 0.0,
        has_inner_glow: false,
        inner_glow_color: [0.0; 4],
        inner_glow_size: 0.0,
        has_grad_overlay: false,
        grad_color0: [0.0; 4],
        grad_color1: [0.0; 4],
        grad_angle: 0.0,
        grad_opacity: 0.0,
        has_bevel: false,
        bevel_highlight: [0.0; 4],
        bevel_shadow: [0.0; 4],
        bevel_size: 0.0,
        bevel_soften: 0.0,
        bevel_angle: 0.0,
        bevel_altitude: 0.0,
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
        mix_r: [1.0, 0.0, 0.0, 0.0],
        mix_g: [0.0, 1.0, 0.0, 0.0],
        mix_b: [0.0, 0.0, 1.0, 0.0],
        has_blend_if: false,
        blend_if: [0.0, 1.0, 0.0, 1.0],
        clipped: false,
        has_stroke: false,
        stroke_color: [0.0; 4],
        stroke_width: 0.0,
        has_shadow: false,
        shadow_color: [0.0; 4],
        shadow_offset: [0.0; 2],
        shadow_blur: 0.0,
        has_overlay: false,
        overlay_color: [0.0; 4],
        has_inner_shadow: false,
        inner_shadow_color: [0.0; 4],
        inner_shadow_offset: [0.0; 2],
        inner_shadow_blur: 0.0,
        has_outer_glow: false,
        outer_glow_color: [0.0; 4],
        outer_glow_size: 0.0,
        has_inner_glow: false,
        inner_glow_color: [0.0; 4],
        inner_glow_size: 0.0,
        has_grad_overlay: false,
        grad_color0: [0.0; 4],
        grad_color1: [0.0; 4],
        grad_angle: 0.0,
        grad_opacity: 0.0,
        has_bevel: false,
        bevel_highlight: [0.0; 4],
        bevel_shadow: [0.0; 4],
        bevel_size: 0.0,
        bevel_soften: 0.0,
        bevel_angle: 0.0,
        bevel_altitude: 0.0,
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

// Regression for "move snaps back to origin": the drag-stop frame fires the
// bake. The bug was that the app cleared the live affine (set_layer_transform
// with None) on that same frame *before* baking — turning the bake into a no-op
// (`xform_layer` was None, so `bake_transform` returned early) and snapping the
// layer back to where it started. The fix keeps the affine live for the bake
// frame, so the bake must land. This is the GPU half of the seam; the gating
// fix itself lives in `app::view` (egui interaction, not unit-testable).
#[test]
fn move_persists_when_baked_on_drag_stop() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping move_persists_when_baked_on_drag_stop");
        return;
    };
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    let size = Size::new(8, 8);
    gpu.ensure_canvas(&device, size);
    let l0 = LayerId(0);
    gpu.ensure_layer(&device, l0);
    gpu.upload_layer(&queue, l0, &solid(8, 1.0, 0.0, 0.0, 1.0)); // red everywhere

    // Drag frames keep the live affine (right by half width) on the layer; the
    // fixed drag-stop frame re-sends that same affine instead of clearing it,
    // then issues the bake. With the bug (a None here) the bake was a no-op.
    gpu.set_layer_transform(Some(l0), [1.0, 0.0, 0.0, 1.0], [-0.5, 0.0]);
    let mut enc = device.create_command_encoder(&Default::default());
    gpu.bake_transform(&device, &queue, &mut enc);
    queue.submit([enc.finish()]);

    let left = gpu.read_pixel(&device, &queue, l0, 1, 4).unwrap();
    let right = gpu.read_pixel(&device, &queue, l0, 6, 4).unwrap();
    assert!(left[3] < 0.1, "left half cleared after move persists: {left:?}");
    assert!(
        right[0] > 0.9 && right[3] > 0.9,
        "right half keeps red after move persists: {right:?}"
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
            mix_r: [1.0, 0.0, 0.0, 0.0],
            mix_g: [0.0, 1.0, 0.0, 0.0],
            mix_b: [0.0, 0.0, 1.0, 0.0],
            has_blend_if: false,
            blend_if: [0.0, 1.0, 0.0, 1.0],
            clipped: false,
            has_stroke: false,
            stroke_color: [0.0; 4],
            stroke_width: 0.0,
            has_shadow: false,
            shadow_color: [0.0; 4],
            shadow_offset: [0.0; 2],
            shadow_blur: 0.0,
            has_overlay: false,
            overlay_color: [0.0; 4],
            has_inner_shadow: false,
            inner_shadow_color: [0.0; 4],
            inner_shadow_offset: [0.0; 2],
            inner_shadow_blur: 0.0,
            has_outer_glow: false,
            outer_glow_color: [0.0; 4],
            outer_glow_size: 0.0,
            has_inner_glow: false,
            inner_glow_color: [0.0; 4],
            inner_glow_size: 0.0,
            has_grad_overlay: false,
            grad_color0: [0.0; 4],
            grad_color1: [0.0; 4],
            grad_angle: 0.0,
            grad_opacity: 0.0,
            has_bevel: false,
            bevel_highlight: [0.0; 4],
            bevel_shadow: [0.0; 4],
            bevel_size: 0.0,
            bevel_soften: 0.0,
            bevel_angle: 0.0,
            bevel_altitude: 0.0,
        },
        LayerDraw {
            id: l1,
            opacity: 1.0,
            blend: 0,
            visible: true,
            adjust_kind: k,
            adjust: p,
            mix_r: [1.0, 0.0, 0.0, 0.0],
            mix_g: [0.0, 1.0, 0.0, 0.0],
            mix_b: [0.0, 0.0, 1.0, 0.0],
            has_blend_if: false,
            blend_if: [0.0, 1.0, 0.0, 1.0],
            clipped: false,
            has_stroke: false,
            stroke_color: [0.0; 4],
            stroke_width: 0.0,
            has_shadow: false,
            shadow_color: [0.0; 4],
            shadow_offset: [0.0; 2],
            shadow_blur: 0.0,
            has_overlay: false,
            overlay_color: [0.0; 4],
            has_inner_shadow: false,
            inner_shadow_color: [0.0; 4],
            inner_shadow_offset: [0.0; 2],
            inner_shadow_blur: 0.0,
            has_outer_glow: false,
            outer_glow_color: [0.0; 4],
            outer_glow_size: 0.0,
            has_inner_glow: false,
            inner_glow_color: [0.0; 4],
            inner_glow_size: 0.0,
            has_grad_overlay: false,
            grad_color0: [0.0; 4],
            grad_color1: [0.0; 4],
            grad_angle: 0.0,
            grad_opacity: 0.0,
            has_bevel: false,
            bevel_highlight: [0.0; 4],
            bevel_shadow: [0.0; 4],
            bevel_size: 0.0,
            bevel_soften: 0.0,
            bevel_angle: 0.0,
            bevel_altitude: 0.0,
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
            mix_r: [1.0, 0.0, 0.0, 0.0],
            mix_g: [0.0, 1.0, 0.0, 0.0],
            mix_b: [0.0, 0.0, 1.0, 0.0],
            has_blend_if: false,
            blend_if: [0.0, 1.0, 0.0, 1.0],
            clipped: false,
            has_stroke: false,
            stroke_color: [0.0; 4],
            stroke_width: 0.0,
            has_shadow: false,
            shadow_color: [0.0; 4],
            shadow_offset: [0.0; 2],
            shadow_blur: 0.0,
            has_overlay: false,
            overlay_color: [0.0; 4],
            has_inner_shadow: false,
            inner_shadow_color: [0.0; 4],
            inner_shadow_offset: [0.0; 2],
            inner_shadow_blur: 0.0,
            has_outer_glow: false,
            outer_glow_color: [0.0; 4],
            outer_glow_size: 0.0,
            has_inner_glow: false,
            inner_glow_color: [0.0; 4],
            inner_glow_size: 0.0,
            has_grad_overlay: false,
            grad_color0: [0.0; 4],
            grad_color1: [0.0; 4],
            grad_angle: 0.0,
            grad_opacity: 0.0,
            has_bevel: false,
            bevel_highlight: [0.0; 4],
            bevel_shadow: [0.0; 4],
            bevel_size: 0.0,
            bevel_soften: 0.0,
            bevel_angle: 0.0,
            bevel_altitude: 0.0,
        },
        LayerDraw {
            id: l1,
            opacity: 1.0,
            blend: 0,
            visible: true,
            adjust_kind: 8, // Curves
            adjust: [0.0; 4],
            mix_r: [1.0, 0.0, 0.0, 0.0],
            mix_g: [0.0, 1.0, 0.0, 0.0],
            mix_b: [0.0, 0.0, 1.0, 0.0],
            has_blend_if: false,
            blend_if: [0.0, 1.0, 0.0, 1.0],
            clipped: false,
            has_stroke: false,
            stroke_color: [0.0; 4],
            stroke_width: 0.0,
            has_shadow: false,
            shadow_color: [0.0; 4],
            shadow_offset: [0.0; 2],
            shadow_blur: 0.0,
            has_overlay: false,
            overlay_color: [0.0; 4],
            has_inner_shadow: false,
            inner_shadow_color: [0.0; 4],
            inner_shadow_offset: [0.0; 2],
            inner_shadow_blur: 0.0,
            has_outer_glow: false,
            outer_glow_color: [0.0; 4],
            outer_glow_size: 0.0,
            has_inner_glow: false,
            inner_glow_color: [0.0; 4],
            inner_glow_size: 0.0,
            has_grad_overlay: false,
            grad_color0: [0.0; 4],
            grad_color1: [0.0; 4],
            grad_angle: 0.0,
            grad_opacity: 0.0,
            has_bevel: false,
            bevel_highlight: [0.0; 4],
            bevel_shadow: [0.0; 4],
            bevel_size: 0.0,
            bevel_soften: 0.0,
            bevel_angle: 0.0,
            bevel_altitude: 0.0,
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
            mix_r: [1.0, 0.0, 0.0, 0.0],
            mix_g: [0.0, 1.0, 0.0, 0.0],
            mix_b: [0.0, 0.0, 1.0, 0.0],
            has_blend_if: false,
            blend_if: [0.0, 1.0, 0.0, 1.0],
            clipped: false,
            has_stroke: false,
            stroke_color: [0.0; 4],
            stroke_width: 0.0,
            has_shadow: false,
            shadow_color: [0.0; 4],
            shadow_offset: [0.0; 2],
            shadow_blur: 0.0,
            has_overlay: false,
            overlay_color: [0.0; 4],
            has_inner_shadow: false,
            inner_shadow_color: [0.0; 4],
            inner_shadow_offset: [0.0; 2],
            inner_shadow_blur: 0.0,
            has_outer_glow: false,
            outer_glow_color: [0.0; 4],
            outer_glow_size: 0.0,
            has_inner_glow: false,
            inner_glow_color: [0.0; 4],
            inner_glow_size: 0.0,
            has_grad_overlay: false,
            grad_color0: [0.0; 4],
            grad_color1: [0.0; 4],
            grad_angle: 0.0,
            grad_opacity: 0.0,
            has_bevel: false,
            bevel_highlight: [0.0; 4],
            bevel_shadow: [0.0; 4],
            bevel_size: 0.0,
            bevel_soften: 0.0,
            bevel_angle: 0.0,
            bevel_altitude: 0.0,
        },
        LayerDraw {
            id: l1,
            opacity: 1.0,
            blend: 0,
            visible: true,
            adjust_kind: k,
            adjust: p,
            mix_r: [1.0, 0.0, 0.0, 0.0],
            mix_g: [0.0, 1.0, 0.0, 0.0],
            mix_b: [0.0, 0.0, 1.0, 0.0],
            has_blend_if: false,
            blend_if: [0.0, 1.0, 0.0, 1.0],
            clipped: false,
            has_stroke: false,
            stroke_color: [0.0; 4],
            stroke_width: 0.0,
            has_shadow: false,
            shadow_color: [0.0; 4],
            shadow_offset: [0.0; 2],
            shadow_blur: 0.0,
            has_overlay: false,
            overlay_color: [0.0; 4],
            has_inner_shadow: false,
            inner_shadow_color: [0.0; 4],
            inner_shadow_offset: [0.0; 2],
            inner_shadow_blur: 0.0,
            has_outer_glow: false,
            outer_glow_color: [0.0; 4],
            outer_glow_size: 0.0,
            has_inner_glow: false,
            inner_glow_color: [0.0; 4],
            inner_glow_size: 0.0,
            has_grad_overlay: false,
            grad_color0: [0.0; 4],
            grad_color1: [0.0; 4],
            grad_angle: 0.0,
            grad_opacity: 0.0,
            has_bevel: false,
            bevel_highlight: [0.0; 4],
            bevel_shadow: [0.0; 4],
            bevel_size: 0.0,
            bevel_soften: 0.0,
            bevel_angle: 0.0,
            bevel_altitude: 0.0,
        },
    ];
    let pp = gpu.composite_now(&device, &queue, &order);
    let px = gpu.read_composite_pixel(&device, &queue, pp, 4, 4).unwrap();
    assert!(
        px[0] > 0.9,
        "posterize(2) snaps mid gray up to white: {px:?}"
    );
}

// Gradient Map (red shadows -> blue highlights) over a white backdrop maps the
// high luminance to blue.
#[test]
fn gradient_map_adjustment() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping gradient_map_adjustment");
        return;
    };
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    let size = Size::new(8, 8);
    gpu.ensure_canvas(&device, size);
    let l0 = LayerId(0);
    let l1 = LayerId(1);
    gpu.ensure_layer(&device, l0);
    gpu.ensure_layer(&device, l1);
    gpu.upload_layer(&queue, l0, &solid(8, 1.0, 1.0, 1.0, 1.0)); // white backdrop
    gpu.set_gradient_lut(&device, &queue, l1, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]);

    let (k, p) = prism_core::adjust::Adjustment::GradientMap {
        low: [1.0, 0.0, 0.0],
        high: [0.0, 0.0, 1.0],
    }
    .encode();
    let order = vec![
        LayerDraw {
            id: l0,
            opacity: 1.0,
            blend: 0,
            visible: true,
            adjust_kind: 0,
            adjust: [0.0; 4],
            mix_r: [1.0, 0.0, 0.0, 0.0],
            mix_g: [0.0, 1.0, 0.0, 0.0],
            mix_b: [0.0, 0.0, 1.0, 0.0],
            has_blend_if: false,
            blend_if: [0.0, 1.0, 0.0, 1.0],
            clipped: false,
            has_stroke: false,
            stroke_color: [0.0; 4],
            stroke_width: 0.0,
            has_shadow: false,
            shadow_color: [0.0; 4],
            shadow_offset: [0.0; 2],
            shadow_blur: 0.0,
            has_overlay: false,
            overlay_color: [0.0; 4],
            has_inner_shadow: false,
            inner_shadow_color: [0.0; 4],
            inner_shadow_offset: [0.0; 2],
            inner_shadow_blur: 0.0,
            has_outer_glow: false,
            outer_glow_color: [0.0; 4],
            outer_glow_size: 0.0,
            has_inner_glow: false,
            inner_glow_color: [0.0; 4],
            inner_glow_size: 0.0,
            has_grad_overlay: false,
            grad_color0: [0.0; 4],
            grad_color1: [0.0; 4],
            grad_angle: 0.0,
            grad_opacity: 0.0,
            has_bevel: false,
            bevel_highlight: [0.0; 4],
            bevel_shadow: [0.0; 4],
            bevel_size: 0.0,
            bevel_soften: 0.0,
            bevel_angle: 0.0,
            bevel_altitude: 0.0,
        },
        LayerDraw {
            id: l1,
            opacity: 1.0,
            blend: 0,
            visible: true,
            adjust_kind: k,
            adjust: p,
            mix_r: [1.0, 0.0, 0.0, 0.0],
            mix_g: [0.0, 1.0, 0.0, 0.0],
            mix_b: [0.0, 0.0, 1.0, 0.0],
            has_blend_if: false,
            blend_if: [0.0, 1.0, 0.0, 1.0],
            clipped: false,
            has_stroke: false,
            stroke_color: [0.0; 4],
            stroke_width: 0.0,
            has_shadow: false,
            shadow_color: [0.0; 4],
            shadow_offset: [0.0; 2],
            shadow_blur: 0.0,
            has_overlay: false,
            overlay_color: [0.0; 4],
            has_inner_shadow: false,
            inner_shadow_color: [0.0; 4],
            inner_shadow_offset: [0.0; 2],
            inner_shadow_blur: 0.0,
            has_outer_glow: false,
            outer_glow_color: [0.0; 4],
            outer_glow_size: 0.0,
            has_inner_glow: false,
            inner_glow_color: [0.0; 4],
            inner_glow_size: 0.0,
            has_grad_overlay: false,
            grad_color0: [0.0; 4],
            grad_color1: [0.0; 4],
            grad_angle: 0.0,
            grad_opacity: 0.0,
            has_bevel: false,
            bevel_highlight: [0.0; 4],
            bevel_shadow: [0.0; 4],
            bevel_size: 0.0,
            bevel_soften: 0.0,
            bevel_angle: 0.0,
            bevel_altitude: 0.0,
        },
    ];
    let pp = gpu.composite_now(&device, &queue, &order);
    let px = gpu.read_composite_pixel(&device, &queue, pp, 4, 4).unwrap();
    assert!(
        px[2] > 0.8 && px[0] < 0.2,
        "white backdrop maps to the highlight color (blue): {px:?}"
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
            mix_r: [1.0, 0.0, 0.0, 0.0],
            mix_g: [0.0, 1.0, 0.0, 0.0],
            mix_b: [0.0, 0.0, 1.0, 0.0],
            has_blend_if: false,
            blend_if: [0.0, 1.0, 0.0, 1.0],
            clipped: false,
            has_stroke: false,
            stroke_color: [0.0; 4],
            stroke_width: 0.0,
            has_shadow: false,
            shadow_color: [0.0; 4],
            shadow_offset: [0.0; 2],
            shadow_blur: 0.0,
            has_overlay: false,
            overlay_color: [0.0; 4],
            has_inner_shadow: false,
            inner_shadow_color: [0.0; 4],
            inner_shadow_offset: [0.0; 2],
            inner_shadow_blur: 0.0,
            has_outer_glow: false,
            outer_glow_color: [0.0; 4],
            outer_glow_size: 0.0,
            has_inner_glow: false,
            inner_glow_color: [0.0; 4],
            inner_glow_size: 0.0,
            has_grad_overlay: false,
            grad_color0: [0.0; 4],
            grad_color1: [0.0; 4],
            grad_angle: 0.0,
            grad_opacity: 0.0,
            has_bevel: false,
            bevel_highlight: [0.0; 4],
            bevel_shadow: [0.0; 4],
            bevel_size: 0.0,
            bevel_soften: 0.0,
            bevel_angle: 0.0,
            bevel_altitude: 0.0,
        },
        LayerDraw {
            id: l1,
            opacity: 1.0,
            blend: 0,
            visible: true,
            adjust_kind: 0,
            adjust: [0.0; 4],
            mix_r: [1.0, 0.0, 0.0, 0.0],
            mix_g: [0.0, 1.0, 0.0, 0.0],
            mix_b: [0.0, 0.0, 1.0, 0.0],
            has_blend_if: true,
            blend_if: [0.0, 0.45, 0.0, 1.0],
            clipped: false,
            has_stroke: false,
            stroke_color: [0.0; 4],
            stroke_width: 0.0,
            has_shadow: false,
            shadow_color: [0.0; 4],
            shadow_offset: [0.0; 2],
            shadow_blur: 0.0,
            has_overlay: false,
            overlay_color: [0.0; 4],
            has_inner_shadow: false,
            inner_shadow_color: [0.0; 4],
            inner_shadow_offset: [0.0; 2],
            inner_shadow_blur: 0.0,
            has_outer_glow: false,
            outer_glow_color: [0.0; 4],
            outer_glow_size: 0.0,
            has_inner_glow: false,
            inner_glow_color: [0.0; 4],
            inner_glow_size: 0.0,
            has_grad_overlay: false,
            grad_color0: [0.0; 4],
            grad_color1: [0.0; 4],
            grad_angle: 0.0,
            grad_opacity: 0.0,
            has_bevel: false,
            bevel_highlight: [0.0; 4],
            bevel_shadow: [0.0; 4],
            bevel_size: 0.0,
            bevel_soften: 0.0,
            bevel_angle: 0.0,
            bevel_altitude: 0.0,
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
            mix_r: [1.0, 0.0, 0.0, 0.0],
            mix_g: [0.0, 1.0, 0.0, 0.0],
            mix_b: [0.0, 0.0, 1.0, 0.0],
            has_blend_if: false,
            blend_if: [0.0, 1.0, 0.0, 1.0],
            clipped: false,
            has_stroke: false,
            stroke_color: [0.0; 4],
            stroke_width: 0.0,
            has_shadow: false,
            shadow_color: [0.0; 4],
            shadow_offset: [0.0; 2],
            shadow_blur: 0.0,
            has_overlay: false,
            overlay_color: [0.0; 4],
            has_inner_shadow: false,
            inner_shadow_color: [0.0; 4],
            inner_shadow_offset: [0.0; 2],
            inner_shadow_blur: 0.0,
            has_outer_glow: false,
            outer_glow_color: [0.0; 4],
            outer_glow_size: 0.0,
            has_inner_glow: false,
            inner_glow_color: [0.0; 4],
            inner_glow_size: 0.0,
            has_grad_overlay: false,
            grad_color0: [0.0; 4],
            grad_color1: [0.0; 4],
            grad_angle: 0.0,
            grad_opacity: 0.0,
            has_bevel: false,
            bevel_highlight: [0.0; 4],
            bevel_shadow: [0.0; 4],
            bevel_size: 0.0,
            bevel_soften: 0.0,
            bevel_angle: 0.0,
            bevel_altitude: 0.0,
        },
        LayerDraw {
            id: l1,
            opacity: 1.0,
            blend: 0,
            visible: true,
            adjust_kind: 0,
            adjust: [0.0; 4],
            mix_r: [1.0, 0.0, 0.0, 0.0],
            mix_g: [0.0, 1.0, 0.0, 0.0],
            mix_b: [0.0, 0.0, 1.0, 0.0],
            has_blend_if: false,
            blend_if: [0.0, 1.0, 0.0, 1.0],
            clipped: true,
            has_stroke: false,
            stroke_color: [0.0; 4],
            stroke_width: 0.0,
            has_shadow: false,
            shadow_color: [0.0; 4],
            shadow_offset: [0.0; 2],
            shadow_blur: 0.0,
            has_overlay: false,
            overlay_color: [0.0; 4],
            has_inner_shadow: false,
            inner_shadow_color: [0.0; 4],
            inner_shadow_offset: [0.0; 2],
            inner_shadow_blur: 0.0,
            has_outer_glow: false,
            outer_glow_color: [0.0; 4],
            outer_glow_size: 0.0,
            has_inner_glow: false,
            inner_glow_color: [0.0; 4],
            inner_glow_size: 0.0,
            has_grad_overlay: false,
            grad_color0: [0.0; 4],
            grad_color1: [0.0; 4],
            grad_angle: 0.0,
            grad_opacity: 0.0,
            has_bevel: false,
            bevel_highlight: [0.0; 4],
            bevel_shadow: [0.0; 4],
            bevel_size: 0.0,
            bevel_soften: 0.0,
            bevel_angle: 0.0,
            bevel_altitude: 0.0,
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

// Channels: save a left-half selection, clear it, reload from the channel, then
// paint — the restored selection clips the paint to the left half.
#[test]
fn channel_save_load_roundtrip() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping channel_save_load_roundtrip");
        return;
    };
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    let size = Size::new(8, 8);
    gpu.ensure_canvas(&device, size);
    let l0 = LayerId(0);
    gpu.ensure_layer(&device, l0);
    gpu.upload_layer(&queue, l0, &solid(8, 1.0, 0.0, 0.0, 1.0)); // red

    // Select the left half and save it as a channel.
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
    queue.submit([enc.finish()]);
    gpu.save_selection_as_channel(&device, &queue, "a".to_string());

    // Clear the selection, then reload it from the channel.
    let mut enc = device.create_command_encoder(&Default::default());
    gpu.apply_selection(&device, &queue, &mut enc, &SelectionOp::None);
    queue.submit([enc.finish()]);
    gpu.load_channel(&device, &queue, "a");
    assert!(gpu.has_selection(), "selection restored from channel");

    // Paint blue over the whole canvas; the restored selection clips it left.
    let dab = Dab {
        center: [4.0, 4.0],
        radius: 16.0,
        hardness: 0.99,
        color: [0.0, 0.0, 1.0, 1.0],
    };
    let mut enc = device.create_command_encoder(&Default::default());
    gpu.paint_dabs(&device, &queue, &mut enc, l0, &[dab], false, false, false);
    queue.submit([enc.finish()]);

    let order = vec![LayerDraw {
        id: l0,
        opacity: 1.0,
        blend: 0,
        visible: true,
        adjust_kind: 0,
        adjust: [0.0; 4],
        mix_r: [1.0, 0.0, 0.0, 0.0],
        mix_g: [0.0, 1.0, 0.0, 0.0],
        mix_b: [0.0, 0.0, 1.0, 0.0],
        has_blend_if: false,
        blend_if: [0.0, 1.0, 0.0, 1.0],
        clipped: false,
        has_stroke: false,
        stroke_color: [0.0; 4],
        stroke_width: 0.0,
        has_shadow: false,
        shadow_color: [0.0; 4],
        shadow_offset: [0.0; 2],
        shadow_blur: 0.0,
        has_overlay: false,
        overlay_color: [0.0; 4],
        has_inner_shadow: false,
        inner_shadow_color: [0.0; 4],
        inner_shadow_offset: [0.0; 2],
        inner_shadow_blur: 0.0,
        has_outer_glow: false,
        outer_glow_color: [0.0; 4],
        outer_glow_size: 0.0,
        has_inner_glow: false,
        inner_glow_color: [0.0; 4],
        inner_glow_size: 0.0,
        has_grad_overlay: false,
        grad_color0: [0.0; 4],
        grad_color1: [0.0; 4],
        grad_angle: 0.0,
        grad_opacity: 0.0,
        has_bevel: false,
        bevel_highlight: [0.0; 4],
        bevel_shadow: [0.0; 4],
        bevel_size: 0.0,
        bevel_soften: 0.0,
        bevel_angle: 0.0,
        bevel_altitude: 0.0,
    }];
    let p = gpu.composite_now(&device, &queue, &order);
    let left = gpu.read_composite_pixel(&device, &queue, p, 1, 4).unwrap();
    let right = gpu.read_composite_pixel(&device, &queue, p, 6, 4).unwrap();
    assert!(
        left[2] > 0.5,
        "restored selection paints left blue: {left:?}"
    );
    assert!(
        right[0] > 0.5 && right[2] < 0.2,
        "right half untouched red: {right:?}"
    );
}

// Outer-stroke layer style: a small opaque square gets a red stroke ring just
// outside its edge; far pixels stay empty, the interior stays white.
#[test]
fn layer_stroke_outlines_edge() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping layer_stroke_outlines_edge");
        return;
    };
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    let size = Size::new(16, 16);
    gpu.ensure_canvas(&device, size);
    let l0 = LayerId(0);
    gpu.ensure_layer(&device, l0);
    // 4x4 white opaque square at x6..10, y6..10; rest transparent.
    let mut buf = Vec::new();
    for y in 0..16 {
        for x in 0..16 {
            let px = if (6..10).contains(&x) && (6..10).contains(&y) {
                [1.0, 1.0, 1.0, 1.0]
            } else {
                [0.0, 0.0, 0.0, 0.0]
            };
            for &c in &px {
                buf.extend_from_slice(&half::f16::from_f32(c).to_le_bytes());
            }
        }
    }
    gpu.upload_layer(&queue, l0, &buf);

    let order = vec![LayerDraw {
        id: l0,
        opacity: 1.0,
        blend: 0,
        visible: true,
        adjust_kind: 0,
        adjust: [0.0; 4],
        mix_r: [1.0, 0.0, 0.0, 0.0],
        mix_g: [0.0, 1.0, 0.0, 0.0],
        mix_b: [0.0, 0.0, 1.0, 0.0],
        has_blend_if: false,
        blend_if: [0.0, 1.0, 0.0, 1.0],
        clipped: false,
        has_stroke: true,
        stroke_color: [1.0, 0.0, 0.0, 1.0], // red
        stroke_width: 2.0 / 16.0,           // ~2px in uv
        has_shadow: false,
        shadow_color: [0.0; 4],
        shadow_offset: [0.0; 2],
        shadow_blur: 0.0,
        has_overlay: false,
        overlay_color: [0.0; 4],
        has_inner_shadow: false,
        inner_shadow_color: [0.0; 4],
        inner_shadow_offset: [0.0; 2],
        inner_shadow_blur: 0.0,
        has_outer_glow: false,
        outer_glow_color: [0.0; 4],
        outer_glow_size: 0.0,
        has_inner_glow: false,
        inner_glow_color: [0.0; 4],
        inner_glow_size: 0.0,
        has_grad_overlay: false,
        grad_color0: [0.0; 4],
        grad_color1: [0.0; 4],
        grad_angle: 0.0,
        grad_opacity: 0.0,
        has_bevel: false,
        bevel_highlight: [0.0; 4],
        bevel_shadow: [0.0; 4],
        bevel_size: 0.0,
        bevel_soften: 0.0,
        bevel_angle: 0.0,
        bevel_altitude: 0.0,
    }];
    let pp = gpu.composite_now(&device, &queue, &order);
    let edge = gpu.read_composite_pixel(&device, &queue, pp, 5, 8).unwrap(); // just left of square
    let far = gpu.read_composite_pixel(&device, &queue, pp, 0, 0).unwrap();
    let inside = gpu.read_composite_pixel(&device, &queue, pp, 7, 7).unwrap();
    assert!(
        edge[0] > 0.5 && edge[1] < 0.3 && edge[3] > 0.3,
        "red stroke just outside the edge: {edge:?}"
    );
    assert!(far[3] < 0.1, "far pixel stays empty: {far:?}");
    assert!(
        inside[0] > 0.8 && inside[1] > 0.8,
        "interior stays white: {inside:?}"
    );
}

// Drop shadow: an opaque square casts a dark, offset shadow down-right; the
// shadow region (outside the square) is dark and semi-opaque, far stays empty.
#[test]
fn layer_drop_shadow_offsets_behind() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping layer_drop_shadow_offsets_behind");
        return;
    };
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    let size = Size::new(16, 16);
    gpu.ensure_canvas(&device, size);
    let l0 = LayerId(0);
    gpu.ensure_layer(&device, l0);
    let mut buf = Vec::new();
    for y in 0..16 {
        for x in 0..16 {
            let px = if (6..10).contains(&x) && (6..10).contains(&y) {
                [1.0, 1.0, 1.0, 1.0]
            } else {
                [0.0, 0.0, 0.0, 0.0]
            };
            for &c in &px {
                buf.extend_from_slice(&half::f16::from_f32(c).to_le_bytes());
            }
        }
    }
    gpu.upload_layer(&queue, l0, &buf);

    let order = vec![LayerDraw {
        id: l0,
        opacity: 1.0,
        blend: 0,
        visible: true,
        adjust_kind: 0,
        adjust: [0.0; 4],
        mix_r: [1.0, 0.0, 0.0, 0.0],
        mix_g: [0.0, 1.0, 0.0, 0.0],
        mix_b: [0.0, 0.0, 1.0, 0.0],
        has_blend_if: false,
        blend_if: [0.0, 1.0, 0.0, 1.0],
        clipped: false,
        has_stroke: false,
        stroke_color: [0.0; 4],
        stroke_width: 0.0,
        has_shadow: true,
        shadow_color: [0.0, 0.0, 0.0, 0.8], // black, mostly opaque
        shadow_offset: [4.0 / 16.0, 4.0 / 16.0], // down-right ~4px
        shadow_blur: 2.0 / 16.0,
        has_overlay: false,
        overlay_color: [0.0; 4],
        has_inner_shadow: false,
        inner_shadow_color: [0.0; 4],
        inner_shadow_offset: [0.0; 2],
        inner_shadow_blur: 0.0,
        has_outer_glow: false,
        outer_glow_color: [0.0; 4],
        outer_glow_size: 0.0,
        has_inner_glow: false,
        inner_glow_color: [0.0; 4],
        inner_glow_size: 0.0,
        has_grad_overlay: false,
        grad_color0: [0.0; 4],
        grad_color1: [0.0; 4],
        grad_angle: 0.0,
        grad_opacity: 0.0,
        has_bevel: false,
        bevel_highlight: [0.0; 4],
        bevel_shadow: [0.0; 4],
        bevel_size: 0.0,
        bevel_soften: 0.0,
        bevel_angle: 0.0,
        bevel_altitude: 0.0,
    }];
    let pp = gpu.composite_now(&device, &queue, &order);
    // (12,12): square shifted +4 lands here, outside the square itself.
    let shadow = gpu
        .read_composite_pixel(&device, &queue, pp, 12, 12)
        .unwrap();
    let far = gpu.read_composite_pixel(&device, &queue, pp, 1, 1).unwrap();
    assert!(
        shadow[3] > 0.2 && shadow[0] < 0.3,
        "dark offset shadow present: {shadow:?}"
    );
    assert!(far[3] < 0.1, "far corner stays empty: {far:?}");
}

// Color overlay: a white layer with a full-strength red overlay composites red.
#[test]
fn layer_color_overlay_recolors() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping layer_color_overlay_recolors");
        return;
    };
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    let size = Size::new(8, 8);
    gpu.ensure_canvas(&device, size);
    let l0 = LayerId(0);
    gpu.ensure_layer(&device, l0);
    gpu.upload_layer(&queue, l0, &solid(8, 1.0, 1.0, 1.0, 1.0)); // white

    let order = vec![LayerDraw {
        id: l0,
        opacity: 1.0,
        blend: 0,
        visible: true,
        adjust_kind: 0,
        adjust: [0.0; 4],
        mix_r: [1.0, 0.0, 0.0, 0.0],
        mix_g: [0.0, 1.0, 0.0, 0.0],
        mix_b: [0.0, 0.0, 1.0, 0.0],
        has_blend_if: false,
        blend_if: [0.0, 1.0, 0.0, 1.0],
        clipped: false,
        has_stroke: false,
        stroke_color: [0.0; 4],
        stroke_width: 0.0,
        has_shadow: false,
        shadow_color: [0.0; 4],
        shadow_offset: [0.0; 2],
        shadow_blur: 0.0,
        has_overlay: true,
        overlay_color: [1.0, 0.0, 0.0, 1.0], // full red
        has_inner_shadow: false,
        inner_shadow_color: [0.0; 4],
        inner_shadow_offset: [0.0; 2],
        inner_shadow_blur: 0.0,
        has_outer_glow: false,
        outer_glow_color: [0.0; 4],
        outer_glow_size: 0.0,
        has_inner_glow: false,
        inner_glow_color: [0.0; 4],
        inner_glow_size: 0.0,
        has_grad_overlay: false,
        grad_color0: [0.0; 4],
        grad_color1: [0.0; 4],
        grad_angle: 0.0,
        grad_opacity: 0.0,
        has_bevel: false,
        bevel_highlight: [0.0; 4],
        bevel_shadow: [0.0; 4],
        bevel_size: 0.0,
        bevel_soften: 0.0,
        bevel_angle: 0.0,
        bevel_altitude: 0.0,
    }];
    let pp = gpu.composite_now(&device, &queue, &order);
    let px = gpu.read_composite_pixel(&device, &queue, pp, 4, 4).unwrap();
    assert!(
        px[0] > 0.8 && px[1] < 0.2 && px[2] < 0.2,
        "white layer recolored to red overlay: {px:?}"
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
        mix_r: [1.0, 0.0, 0.0, 0.0],
        mix_g: [0.0, 1.0, 0.0, 0.0],
        mix_b: [0.0, 0.0, 1.0, 0.0],
        has_blend_if: false,
        blend_if: [0.0, 1.0, 0.0, 1.0],
        clipped: false,
        has_stroke: false,
        stroke_color: [0.0; 4],
        stroke_width: 0.0,
        has_shadow: false,
        shadow_color: [0.0; 4],
        shadow_offset: [0.0; 2],
        shadow_blur: 0.0,
        has_overlay: false,
        overlay_color: [0.0; 4],
        has_inner_shadow: false,
        inner_shadow_color: [0.0; 4],
        inner_shadow_offset: [0.0; 2],
        inner_shadow_blur: 0.0,
        has_outer_glow: false,
        outer_glow_color: [0.0; 4],
        outer_glow_size: 0.0,
        has_inner_glow: false,
        inner_glow_color: [0.0; 4],
        inner_glow_size: 0.0,
        has_grad_overlay: false,
        grad_color0: [0.0; 4],
        grad_color1: [0.0; 4],
        grad_angle: 0.0,
        grad_opacity: 0.0,
        has_bevel: false,
        bevel_highlight: [0.0; 4],
        bevel_shadow: [0.0; 4],
        bevel_size: 0.0,
        bevel_soften: 0.0,
        bevel_angle: 0.0,
        bevel_altitude: 0.0,
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

// Inner shadow: an opaque white square gets a dark band INSIDE its edge on the
// offset side; the square's coverage never extends past its own bounds.
#[test]
fn layer_inner_shadow_darkens_inside_edge() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping layer_inner_shadow_darkens_inside_edge");
        return;
    };
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    let size = Size::new(16, 16);
    gpu.ensure_canvas(&device, size);
    let l0 = LayerId(0);
    gpu.ensure_layer(&device, l0);
    gpu.upload_layer(&queue, l0, &square_16());

    let order = vec![LayerDraw {
        has_inner_shadow: true,
        inner_shadow_color: [0.0, 0.0, 0.0, 0.9], // dark
        inner_shadow_offset: [3.0 / 16.0, 3.0 / 16.0], // down-right
        inner_shadow_blur: 1.0 / 16.0,
        ..base_draw(l0)
    }];
    let pp = gpu.composite_now(&device, &queue, &order);
    // (6,6): the top-left interior corner — the inverse-alpha cast from up-left
    // lands here, so this pixel is darkened relative to plain white.
    let near = gpu.read_composite_pixel(&device, &queue, pp, 6, 6).unwrap();
    // (9,9): the down-right interior corner — away from the cast, stays bright.
    let far = gpu.read_composite_pixel(&device, &queue, pp, 9, 9).unwrap();
    let outside = gpu.read_composite_pixel(&device, &queue, pp, 2, 2).unwrap();
    assert!(
        near[3] > 0.9 && near[0] < far[0],
        "inside edge darkened by inner shadow: near={near:?} far={far:?}"
    );
    assert!(
        outside[3] < 0.1,
        "inner shadow never paints outside the shape: {outside:?}"
    );
}

// Outer glow: an opaque square emits a colored halo just OUTSIDE its edge that
// fades with distance; far corners stay empty and the interior is unchanged.
#[test]
fn layer_outer_glow_halos_outside_edge() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping layer_outer_glow_halos_outside_edge");
        return;
    };
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    let size = Size::new(16, 16);
    gpu.ensure_canvas(&device, size);
    let l0 = LayerId(0);
    gpu.ensure_layer(&device, l0);
    gpu.upload_layer(&queue, l0, &square_16());

    let order = vec![LayerDraw {
        has_outer_glow: true,
        outer_glow_color: [0.0, 1.0, 0.0, 1.0], // green
        outer_glow_size: 3.0 / 16.0,
        ..base_draw(l0)
    }];
    let pp = gpu.composite_now(&device, &queue, &order);
    let edge = gpu.read_composite_pixel(&device, &queue, pp, 5, 8).unwrap(); // just left of square
    let far = gpu.read_composite_pixel(&device, &queue, pp, 0, 0).unwrap();
    let inside = gpu.read_composite_pixel(&device, &queue, pp, 7, 7).unwrap();
    assert!(
        edge[1] > 0.15 && edge[3] > 0.1,
        "green glow just outside the edge: {edge:?}"
    );
    assert!(far[3] < 0.15, "far corner barely lit: {far:?}");
    assert!(
        inside[0] > 0.8 && inside[1] > 0.8 && inside[2] > 0.8,
        "interior stays white: {inside:?}"
    );
}

// Inner glow: an opaque square gets a colored glow INSIDE its edge that fades
// toward the center; coverage never extends past the shape's own bounds.
#[test]
fn layer_inner_glow_lights_inside_edge() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping layer_inner_glow_lights_inside_edge");
        return;
    };
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    let size = Size::new(16, 16);
    gpu.ensure_canvas(&device, size);
    let l0 = LayerId(0);
    gpu.ensure_layer(&device, l0);
    gpu.upload_layer(&queue, l0, &square_16());

    let order = vec![LayerDraw {
        has_inner_glow: true,
        inner_glow_color: [0.0, 0.0, 1.0, 1.0], // blue
        inner_glow_size: 2.0 / 16.0,
        ..base_draw(l0)
    }];
    let pp = gpu.composite_now(&device, &queue, &order);
    // (6,6): an interior edge pixel — picks up the blue glow.
    let edge = gpu.read_composite_pixel(&device, &queue, pp, 6, 6).unwrap();
    let outside = gpu.read_composite_pixel(&device, &queue, pp, 2, 2).unwrap();
    assert!(
        edge[2] > 0.3 && edge[3] > 0.9,
        "inner edge tinted blue by inner glow: {edge:?}"
    );
    assert!(
        outside[3] < 0.1,
        "inner glow never paints outside the shape: {outside:?}"
    );
}

// Gradient overlay: a full-opacity black->white horizontal gradient over a white
// square reads dark on the left and bright on the right.
#[test]
fn layer_gradient_overlay_ramps_across() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping layer_gradient_overlay_ramps_across");
        return;
    };
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    let size = Size::new(16, 16);
    gpu.ensure_canvas(&device, size);
    let l0 = LayerId(0);
    gpu.ensure_layer(&device, l0);
    // Full white opaque canvas so the gradient is sampled everywhere.
    gpu.upload_layer(&queue, l0, &solid(16, 1.0, 1.0, 1.0, 1.0));

    let order = vec![LayerDraw {
        has_grad_overlay: true,
        grad_color0: [0.0, 0.0, 0.0, 1.0], // black at t=0 (left)
        grad_color1: [1.0, 1.0, 1.0, 1.0], // white at t=1 (right)
        grad_angle: 0.0,                    // along +x
        grad_opacity: 1.0,
        ..base_draw(l0)
    }];
    let pp = gpu.composite_now(&device, &queue, &order);
    let left = gpu.read_composite_pixel(&device, &queue, pp, 1, 8).unwrap();
    let right = gpu.read_composite_pixel(&device, &queue, pp, 14, 8).unwrap();
    assert!(
        left[0] < right[0] && left[0] < 0.3,
        "gradient overlay dark on the left: left={left:?} right={right:?}"
    );
    assert!(right[0] > 0.7, "gradient overlay bright on the right: {right:?}");
}

// Bevel & Emboss (Inner Bevel): a gray square lit from the +x side (angle 0)
// reads a bright highlight on its light-facing (right) edge and a dark shadow on
// the opposite (left) edge, while the flat interior is unchanged.
#[test]
fn layer_bevel_lights_facing_edge() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping layer_bevel_lights_facing_edge");
        return;
    };
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    let size = Size::new(16, 16);
    gpu.ensure_canvas(&device, size);
    let l0 = LayerId(0);
    gpu.ensure_layer(&device, l0);
    // A mid-gray 8x8 square at x4..12, y4..12 so the bevel band has room to read.
    let mut buf = Vec::new();
    for y in 0..16 {
        for x in 0..16 {
            let px = if (4..12).contains(&x) && (4..12).contains(&y) {
                [0.5f32, 0.5, 0.5, 1.0]
            } else {
                [0.0, 0.0, 0.0, 0.0]
            };
            for &c in &px {
                buf.extend_from_slice(&half::f16::from_f32(c).to_le_bytes());
            }
        }
    }
    gpu.upload_layer(&queue, l0, &buf);

    let order = vec![LayerDraw {
        has_bevel: true,
        bevel_highlight: [1.0, 1.0, 1.0, 1.0], // white highlight
        bevel_shadow: [0.0, 0.0, 0.0, 1.0],    // black shadow
        bevel_size: 2.0 / 16.0,
        bevel_soften: 0.0,
        bevel_angle: 0.0,                      // light from +x (right)
        bevel_altitude: std::f32::consts::FRAC_PI_6, // 30°
        ..base_draw(l0)
    }];
    let pp = gpu.composite_now(&device, &queue, &order);
    // Interior edge pixels just inside the left and right borders, mid-height.
    let right = gpu.read_composite_pixel(&device, &queue, pp, 10, 8).unwrap();
    let left = gpu.read_composite_pixel(&device, &queue, pp, 5, 8).unwrap();
    let core = gpu.read_composite_pixel(&device, &queue, pp, 8, 8).unwrap();
    let outside = gpu.read_composite_pixel(&device, &queue, pp, 1, 1).unwrap();
    assert!(
        right[0] > core[0] && right[0] > left[0],
        "light-facing (right) edge brightened: right={right:?} core={core:?} left={left:?}"
    );
    assert!(
        left[0] < core[0],
        "shadowed (left) edge darkened: left={left:?} core={core:?}"
    );
    assert!(
        outside[3] < 0.1,
        "bevel never paints outside the shape: {outside:?}"
    );
}

// Color Balance: a strong red shadow shift over a dark-gray backdrop lifts the
// red channel (a per-channel transfer LUT, shader kind 13), leaving the brighter
// channels comparatively untouched.
#[test]
fn color_balance_shadow_red_push() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping color_balance_shadow_red_push");
        return;
    };
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    let size = Size::new(8, 8);
    gpu.ensure_canvas(&device, size);
    let l0 = LayerId(0);
    let l1 = LayerId(1);
    gpu.ensure_layer(&device, l0);
    gpu.ensure_layer(&device, l1);
    // Dark gray backdrop so the shadow range dominates the LUT weighting.
    gpu.upload_layer(&queue, l0, &solid(8, 0.15, 0.15, 0.15, 1.0));
    gpu.set_color_balance_lut(&device, &queue, l1, [1.0, 0.0, 0.0], [0.0; 3], [0.0; 3]);

    let (k, p) = prism_core::adjust::Adjustment::ColorBalance {
        shadows: [1.0, 0.0, 0.0],
        midtones: [0.0; 3],
        highlights: [0.0; 3],
        preserve_luminosity: false,
    }
    .encode();
    let order = vec![
        base_draw(l0),
        LayerDraw {
            adjust_kind: k,
            adjust: p,
            ..base_draw(l1)
        },
    ];
    let pp = gpu.composite_now(&device, &queue, &order);
    let px = gpu.read_composite_pixel(&device, &queue, pp, 4, 4).unwrap();
    // Composite is linear-premultiplied; 0.15 sRGB ≈ 0.0185 linear. Red must end
    // up clearly above the (unshifted) green/blue.
    assert!(
        px[0] > px[1] + 0.05 && px[0] > px[2] + 0.05,
        "shadow red push lifts the red channel above green/blue: {px:?}"
    );
}

// Channel Mixer: a red↔blue swap matrix (output R = input B, output B = input R)
// over a pure-red layer yields blue (shader kind 14).
#[test]
fn channel_mixer_swaps_red_blue() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping channel_mixer_swaps_red_blue");
        return;
    };
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    let size = Size::new(8, 8);
    gpu.ensure_canvas(&device, size);
    let l0 = LayerId(0);
    let l1 = LayerId(1);
    gpu.ensure_layer(&device, l0);
    gpu.ensure_layer(&device, l1);
    gpu.upload_layer(&queue, l0, &solid(8, 1.0, 0.0, 0.0, 1.0)); // pure red

    let adj = prism_core::adjust::Adjustment::ChannelMixer {
        r: [0.0, 0.0, 1.0, 0.0], // out R = in B
        g: [0.0, 1.0, 0.0, 0.0], // out G = in G
        b: [1.0, 0.0, 0.0, 0.0], // out B = in R
        monochrome: false,
    };
    let (k, p) = adj.encode();
    let m = adj.channel_mixer_matrix().unwrap();
    let order = vec![
        base_draw(l0),
        LayerDraw {
            adjust_kind: k,
            adjust: p,
            mix_r: m.r,
            mix_g: m.g,
            mix_b: m.b,
            ..base_draw(l1)
        },
    ];
    let pp = gpu.composite_now(&device, &queue, &order);
    let px = gpu.read_composite_pixel(&device, &queue, pp, 4, 4).unwrap();
    assert!(
        px[2] > 0.8 && px[0] < 0.2,
        "red/blue swap turns red into blue: {px:?}"
    );
}

// The gradient-fill path exactly as `do_gradient` runs it on the GPU: upload a
// solid layer, read it back, render a multi-stop gradient (shared
// `prism_core::gradient`) over it source-over, upload, then read pixels back.
// Asserts the gradient took effect (left vs right of a linear black→white ramp)
// and that the f16 layer round-trip preserves it. Skips if no GPU adapter.
#[test]
fn gradient_fill_writes_layer() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping gradient_fill_writes_layer");
        return;
    };
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    let size = Size::new(16, 4);
    gpu.ensure_canvas(&device, size);
    let l0 = LayerId(0);
    gpu.ensure_layer(&device, l0);
    // Start fully transparent so the opaque gradient is what we read back.
    gpu.upload_layer(&queue, l0, &solid(16, 0.0, 0.0, 0.0, 0.0));

    // Black→white opaque linear gradient across the 16px width, dither off so
    // the assertion is exact.
    let grad = prism_core::gradient::Gradient::two_color(
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        prism_core::gradient::GradientType::Linear,
    );
    let grad = prism_core::gradient::Gradient {
        dither: false,
        ..grad
    };
    let g = grad.render((0.0, 0.0), (16.0, 0.0), 16, 4); // premultiplied linear

    // Read the (transparent) layer back, composite the gradient source-over.
    let bytes = gpu.read_layer(&device, &queue, l0).unwrap();
    let mut base: Vec<f32> = bytes
        .chunks_exact(2)
        .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect();
    for i in 0..(16 * 4) {
        let ga = g[i * 4 + 3];
        for c in 0..4 {
            base[i * 4 + c] = g[i * 4 + c] + base[i * 4 + c] * (1.0 - ga);
        }
    }
    let mut out = Vec::with_capacity(base.len() * 2);
    for &c in &base {
        out.extend_from_slice(&half::f16::from_f32(c).to_le_bytes());
    }
    gpu.upload_layer(&queue, l0, &out);

    // Leftmost is ~black, rightmost ~white; both opaque.
    let left = gpu.read_pixel(&device, &queue, l0, 0, 2).unwrap();
    let right = gpu.read_pixel(&device, &queue, l0, 15, 2).unwrap();
    assert!(left[3] > 0.9 && right[3] > 0.9, "opaque: {left:?} {right:?}");
    assert!(left[0] < 0.2, "left ~black, got {left:?}");
    assert!(right[0] > 0.8, "right ~white, got {right:?}");
    assert!(right[0] > left[0] + 0.5, "ramp increases L→R");
}

// ---- Blur-family filters (Phase 8) ---------------------------------------

/// Build an `n`x`n` RGBA16F layer buffer from a per-pixel closure returning
/// straight-linear `[r,g,b,a]`; stored premultiplied (the working space).
#[cfg(test)]
fn layer_from(n: u32, f: impl Fn(u32, u32) -> [f32; 4]) -> Vec<u8> {
    let mut buf = Vec::with_capacity((n * n * 8) as usize);
    for y in 0..n {
        for x in 0..n {
            let p = f(x, y);
            let a = p[3];
            for c in &[p[0] * a, p[1] * a, p[2] * a, a] {
                buf.extend_from_slice(&half::f16::from_f32(*c).to_le_bytes());
            }
        }
    }
    buf
}

// Motion blur smears a single opaque pixel along the angle, not across it.
#[test]
fn motion_blur_smears_along_angle() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping motion_blur_smears_along_angle");
        return;
    };
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    let n = 17;
    let size = Size::new(n, n);
    gpu.ensure_canvas(&device, size);
    let l0 = LayerId(0);
    gpu.ensure_layer(&device, l0);
    // One opaque white pixel at the center, rest transparent.
    let c = n / 2;
    gpu.upload_layer(
        &queue,
        l0,
        &layer_from(n, |x, y| {
            if x == c && y == c {
                [1.0, 1.0, 1.0, 1.0]
            } else {
                [0.0; 4]
            }
        }),
    );

    // Horizontal motion blur (angle 0), 3 taps each side.
    gpu.apply_motion_blur(&device, &queue, l0, 0.0, 3.0);

    let on_axis = gpu.read_pixel(&device, &queue, l0, c + 2, c).unwrap();
    let off_axis = gpu.read_pixel(&device, &queue, l0, c, c + 2).unwrap();
    assert!(
        on_axis[3] > 0.05,
        "pixel 2 along the streak picks up signal: {on_axis:?}"
    );
    assert!(
        off_axis[3] < 0.02,
        "off-axis (perpendicular) stays empty: {off_axis:?}"
    );
    assert!(
        on_axis[3] > off_axis[3] + 0.05,
        "streak axis >> perpendicular"
    );
}

// Box blur preserves a flat field exactly (kernel normalized) and spreads an
// impulse symmetrically (separable H+V).
#[test]
fn box_blur_normalizes_and_spreads() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping box_blur_normalizes_and_spreads");
        return;
    };
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    let n = 16;
    let size = Size::new(n, n);
    gpu.ensure_canvas(&device, size);

    // Flat opaque gray must survive a box blur unchanged (normalization).
    let l0 = LayerId(0);
    gpu.ensure_layer(&device, l0);
    gpu.upload_layer(&queue, l0, &solid(n, 0.5, 0.5, 0.5, 1.0));
    gpu.apply_filter(&device, &queue, l0, 5, 3.0, 0.0); // kind 5 = box blur
    let flat = gpu.read_pixel(&device, &queue, l0, n / 2, n / 2).unwrap();
    assert!(
        (flat[0] - 0.5).abs() < 0.02 && (flat[3] - 1.0).abs() < 0.02,
        "flat field preserved by box blur: {flat:?}"
    );

    // Impulse spreads into the neighbourhood symmetrically.
    let l1 = LayerId(1);
    gpu.ensure_layer(&device, l1);
    let c = n / 2;
    gpu.upload_layer(
        &queue,
        l1,
        &layer_from(n, |x, y| {
            if x == c && y == c {
                [1.0, 1.0, 1.0, 1.0]
            } else {
                [0.0; 4]
            }
        }),
    );
    gpu.apply_filter(&device, &queue, l1, 5, 1.0, 0.0); // radius 1 -> 3x3 box
    let center = gpu.read_pixel(&device, &queue, l1, c, c).unwrap();
    let right = gpu.read_pixel(&device, &queue, l1, c + 1, c).unwrap();
    let down = gpu.read_pixel(&device, &queue, l1, c, c + 1).unwrap();
    let far = gpu.read_pixel(&device, &queue, l1, c + 3, c + 3).unwrap();
    assert!(center[3] > 0.05, "center retains some signal: {center:?}");
    assert!(right[3] > 0.05, "spread right: {right:?}");
    assert!(down[3] > 0.05, "spread down: {down:?}");
    assert!((right[3] - down[3]).abs() < 0.02, "spread symmetric H vs V");
    assert!(far[3] < 0.02, "no signal outside the 3x3 box: {far:?}");
}

// Radial spin smears tangentially; radial zoom smears radially. An off-center
// bright pixel distinguishes the two modes.
#[test]
fn radial_spin_vs_zoom() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping radial_spin_vs_zoom");
        return;
    };
    let n = 31u32;
    let size = Size::new(n, n);
    let c = (n / 2) as i32;
    let bx = (c + 8) as u32; // right of center, same row
    let by = c as u32;
    let bright = |x: u32, y: u32| {
        if x == bx && y == by {
            [1.0, 1.0, 1.0, 1.0]
        } else {
            [0.0; 4]
        }
    };

    // --- Spin: smears vertically (tangent), not horizontally (radius). ---
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    gpu.ensure_canvas(&device, size);
    let ls = LayerId(0);
    gpu.ensure_layer(&device, ls);
    gpu.upload_layer(&queue, ls, &layer_from(n, bright));
    // ~17° spin, plenty of samples.
    gpu.apply_radial_blur(&device, &queue, ls, c as f32, c as f32, true, 0.3, 32);
    let tang = gpu.read_pixel(&device, &queue, ls, bx, by - 2).unwrap()[3]
        + gpu.read_pixel(&device, &queue, ls, bx, by + 2).unwrap()[3];
    let radial = gpu.read_pixel(&device, &queue, ls, bx - 2, by).unwrap()[3]
        + gpu.read_pixel(&device, &queue, ls, bx + 2, by).unwrap()[3];
    assert!(
        tang > radial + 0.02,
        "spin smears tangentially (vert {tang}) > radially (horiz {radial})"
    );

    // --- Zoom: smears horizontally (radius), not vertically (tangent). ---
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    gpu.ensure_canvas(&device, size);
    let lz = LayerId(0);
    gpu.ensure_layer(&device, lz);
    gpu.upload_layer(&queue, lz, &layer_from(n, bright));
    gpu.apply_radial_blur(&device, &queue, lz, c as f32, c as f32, false, 0.4, 32);
    let z_radial = gpu.read_pixel(&device, &queue, lz, bx - 2, by).unwrap()[3]
        + gpu.read_pixel(&device, &queue, lz, bx + 2, by).unwrap()[3];
    let z_tang = gpu.read_pixel(&device, &queue, lz, bx, by - 2).unwrap()[3]
        + gpu.read_pixel(&device, &queue, lz, bx, by + 2).unwrap()[3];
    assert!(
        z_radial > z_tang + 0.02,
        "zoom smears radially (horiz {z_radial}) > tangentially (vert {z_tang})"
    );
}

// ---- Distort filters (Phase 8) -------------------------------------------

// A horizontal black→white opaque ramp: the sampled red value encodes which
// source column a coordinate remap pulled from.
#[cfg(test)]
fn ramp_layer(n: u32) -> Vec<u8> {
    layer_from(n, |x, _y| {
        let v = x as f32 / (n - 1) as f32;
        [v, v, v, 1.0]
    })
}

// Twirl rotates the source about the center inside its radius; pixels outside
// the radius are untouched, and a pixel above the center samples off its column.
#[test]
fn twirl_rotates_within_radius() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping twirl_rotates_within_radius");
        return;
    };
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    let n = 41u32;
    gpu.ensure_canvas(&device, Size::new(n, n));
    let l0 = LayerId(0);
    gpu.ensure_layer(&device, l0);
    gpu.upload_layer(&queue, l0, &ramp_layer(n));

    let c = n / 2;
    // ~86° twirl over a generous radius.
    gpu.apply_distort(
        &device, &queue, l0, 8, c as f32, c as f32, 1.5, 18.0, [0.0; 2],
    );

    // A point above the center departs from the 0.5 center-column value.
    let above = gpu.read_pixel(&device, &queue, l0, c, c - 6).unwrap();
    assert!(
        (above[0] - 0.5).abs() > 0.04,
        "twirl rotated the sample off the center column: {above:?}"
    );
    // A far corner (outside the radius) is unchanged: ~its source ramp value.
    let corner = gpu.read_pixel(&device, &queue, l0, 1, 1).unwrap();
    let expect = 1.0 / (n - 1) as f32;
    assert!(
        (corner[0] - expect).abs() < 0.05,
        "outside-radius pixel untouched: {corner:?} (expect {expect})"
    );
}

// Pinch and bulge (signed amount) displace the source in opposite radial
// directions: on a left→right ramp, a pixel right of center reads lower under
// pinch and higher under bulge.
#[test]
fn pinch_and_bulge_are_signed() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping pinch_and_bulge_are_signed");
        return;
    };
    let n = 41u32;
    let c = n / 2;
    let px = c + 6;
    let radius = 18.0;

    let mut gp = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    gp.ensure_canvas(&device, Size::new(n, n));
    let lp = LayerId(0);
    gp.ensure_layer(&device, lp);
    gp.upload_layer(&queue, lp, &ramp_layer(n));
    gp.apply_distort(
        &device, &queue, lp, 9, c as f32, c as f32, 0.6, radius, [0.0; 2],
    );
    let pinch = gp.read_pixel(&device, &queue, lp, px, c).unwrap()[0];

    let mut gb = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    gb.ensure_canvas(&device, Size::new(n, n));
    let lb = LayerId(0);
    gb.ensure_layer(&device, lb);
    gb.upload_layer(&queue, lb, &ramp_layer(n));
    gb.apply_distort(
        &device, &queue, lb, 9, c as f32, c as f32, -0.6, radius, [0.0; 2],
    );
    let bulge = gb.read_pixel(&device, &queue, lb, px, c).unwrap()[0];

    let identity = px as f32 / (n - 1) as f32;
    assert!(
        pinch < identity - 0.01,
        "pinch pulls source inward: {pinch}"
    );
    assert!(
        bulge > identity + 0.01,
        "bulge pushes source outward: {bulge}"
    );
    assert!(bulge > pinch, "bulge and pinch are signed-opposite");
}

// Ripple displaces sinusoidally with a wavelength period: rows one wavelength
// apart receive the same x-displacement, so they read back identical.
#[test]
fn ripple_is_periodic() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping ripple_is_periodic");
        return;
    };
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    let n = 60u32;
    let wl = 20.0f32;
    gpu.ensure_canvas(&device, Size::new(n, n));
    let l0 = LayerId(0);
    gpu.ensure_layer(&device, l0);
    gpu.upload_layer(&queue, l0, &ramp_layer(n));
    gpu.apply_distort(&device, &queue, l0, 10, 0.0, 0.0, 0.0, 0.0, [6.0, wl]);

    let probe_x = n / 3;
    let y0 = 12u32;
    let y1 = y0 + wl as u32;
    let a = gpu.read_pixel(&device, &queue, l0, probe_x, y0).unwrap()[0];
    let b = gpu.read_pixel(&device, &queue, l0, probe_x, y1).unwrap()[0];
    assert!(
        (a - b).abs() < 0.03,
        "rows one wavelength apart match: {a} vs {b}"
    );

    // And the warp is non-trivial: some pixel differs from its source ramp.
    let mid = gpu.read_pixel(&device, &queue, l0, n / 2, 3).unwrap()[0];
    let src = (n / 2) as f32 / (n - 1) as f32;
    assert!(
        (mid - src).abs() > 0.005,
        "ripple displaced a sample: {mid}"
    );
}

// Polar round-trip: ToRect ∘ ToPolar recovers a smooth radial image near the
// center, within resampling tolerance.
#[test]
fn polar_round_trips() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping polar_round_trips");
        return;
    };
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    let n = 64u32;
    let c = (n / 2) as f32;
    gpu.ensure_canvas(&device, Size::new(n, n));
    let l0 = LayerId(0);
    gpu.ensure_layer(&device, l0);
    // Smooth radial bump (bright center → dark edge).
    gpu.upload_layer(
        &queue,
        l0,
        &layer_from(n, |x, y| {
            let dx = x as f32 - c;
            let dy = y as f32 - c;
            let r = (dx * dx + dy * dy).sqrt() / c;
            let v = (1.0 - r).clamp(0.0, 1.0);
            [v, v, v, 1.0]
        }),
    );

    let probe = |g: &CanvasGpu, dev: &wgpu::Device, q: &wgpu::Queue| {
        g.read_pixel(dev, q, l0, n / 2, n / 2 - n / 4).unwrap()[0]
    };
    let before = probe(&gpu, &device, &queue);

    gpu.apply_distort(&device, &queue, l0, 11, c, c, 0.0, 0.0, [0.0; 2]); // rect->polar
    gpu.apply_distort(&device, &queue, l0, 12, c, c, 0.0, 0.0, [0.0; 2]); // polar->rect
    let after = probe(&gpu, &device, &queue);
    assert!(
        (after - before).abs() < 0.12,
        "polar round-trip recovers the value: {before} -> {after}"
    );
}

// ---- Noise filters (Phase 8) ---------------------------------------------

// Add Noise is seeded-deterministic (same seed → identical result), zero-mean
// (the average is preserved), and monochromatic mode applies the SAME delta to
// R/G/B (so R==G==B on a gray field).
#[test]
fn add_noise_is_deterministic_and_zero_mean() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping add_noise_is_deterministic_and_zero_mean");
        return;
    };
    let n = 32u32;
    let base = 0.5f32;
    let gray = layer_from(n, |_x, _y| [base, base, base, 1.0]);

    let run = |seed: f32, mono: bool| -> Vec<[f32; 4]> {
        let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
        gpu.ensure_canvas(&device, Size::new(n, n));
        let l0 = LayerId(0);
        gpu.ensure_layer(&device, l0);
        gpu.upload_layer(&queue, l0, &gray);
        gpu.apply_noise(&device, &queue, l0, 0.2, mono, true, seed);
        let mut px = Vec::with_capacity((n * n) as usize);
        for y in 0..n {
            for x in 0..n {
                px.push(gpu.read_pixel(&device, &queue, l0, x, y).unwrap());
            }
        }
        px
    };

    let a = run(5.0, false);
    let b = run(5.0, false);
    // Determinism: same seed → identical (no temporal randomness).
    for (pa, pb) in a.iter().zip(b.iter()) {
        for c in 0..4 {
            assert!(
                (pa[c] - pb[c]).abs() < 1e-3,
                "deterministic for a fixed seed"
            );
        }
    }
    // Zero-mean: per-channel average ≈ the base.
    for ch in 0..3 {
        let mean: f32 = a.iter().map(|p| p[ch]).sum::<f32>() / a.len() as f32;
        assert!(
            (mean - base).abs() < 0.03,
            "noise preserves the mean (ch {ch}): {mean}"
        );
    }
    // The field is actually perturbed.
    assert!(
        a.iter().any(|p| (p[0] - base).abs() > 1e-2),
        "noise perturbs the field"
    );

    // Monochromatic: R == G == B for every pixel.
    let m = run(5.0, true);
    for p in &m {
        assert!(
            (p[0] - p[1]).abs() < 2e-3 && (p[1] - p[2]).abs() < 2e-3,
            "monochromatic: equal RGB delta: {p:?}"
        );
    }
}

// Median removes a single bright impulse on a flat field, replacing it with the
// surrounding value, while leaving flat areas untouched.
#[test]
fn median_removes_an_impulse() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping median_removes_an_impulse");
        return;
    };
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    let n = 11u32;
    gpu.ensure_canvas(&device, Size::new(n, n));
    let l0 = LayerId(0);
    gpu.ensure_layer(&device, l0);
    let c = n / 2;
    gpu.upload_layer(
        &queue,
        l0,
        &layer_from(n, |x, y| {
            if x == c && y == c {
                [1.0, 1.0, 1.0, 1.0] // bright impulse / outlier
            } else {
                [0.4, 0.4, 0.4, 1.0]
            }
        }),
    );
    gpu.apply_median(&device, &queue, l0, 1.0, None);

    let center = gpu.read_pixel(&device, &queue, l0, c, c).unwrap();
    assert!(
        (center[0] - 0.4).abs() < 0.03,
        "median removed the impulse: {center:?}"
    );
    let flat = gpu.read_pixel(&device, &queue, l0, 1, 1).unwrap();
    assert!(
        (flat[0] - 0.4).abs() < 0.03,
        "flat area preserved: {flat:?}"
    );
}

// Dust & Scratches only replaces pixels that differ from the window median by
// more than the threshold: a strong speck is removed, a sub-threshold ripple is
// preserved.
#[test]
fn dust_scratches_only_changes_above_threshold() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping dust_scratches_only_changes_above_threshold");
        return;
    };
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    let n = 11u32;
    gpu.ensure_canvas(&device, Size::new(n, n));
    let l0 = LayerId(0);
    gpu.ensure_layer(&device, l0);
    let base = 0.5f32;
    let c = n / 2;
    let (sx, sy) = (2u32, 2u32); // a sub-threshold pixel
    gpu.upload_layer(
        &queue,
        l0,
        &layer_from(n, |x, y| {
            if x == c && y == c {
                [0.95, 0.95, 0.95, 1.0] // big deviation (0.45)
            } else if x == sx && y == sy {
                [base + 0.05, base + 0.05, base + 0.05, 1.0] // tiny (0.05)
            } else {
                [base, base, base, 1.0]
            }
        }),
    );
    gpu.apply_median(&device, &queue, l0, 1.0, Some(0.2));

    let speck = gpu.read_pixel(&device, &queue, l0, c, c).unwrap();
    assert!(
        (speck[0] - base).abs() < 0.03,
        "above-threshold speck replaced by the median: {speck:?}"
    );
    let small = gpu.read_pixel(&device, &queue, l0, sx, sy).unwrap();
    assert!(
        (small[0] - (base + 0.05)).abs() < 0.03,
        "below-threshold pixel preserved: {small:?}"
    );
}

// ---- Pixelate family (Phase 8) -------------------------------------------

// Mosaic averages each cell to one colour: every pixel inside a cell is equal
// (the cell is uniform), and that value is the block mean — here a half-black /
// half-white cell averages to mid-gray.
#[test]
fn mosaic_cell_is_uniform_block_average() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping mosaic_cell_is_uniform_block_average");
        return;
    };
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    let n = 8u32;
    gpu.ensure_canvas(&device, Size::new(n, n));
    let l0 = LayerId(0);
    gpu.ensure_layer(&device, l0);
    // A single 8×8 cell whose left half is black, right half white → mean 0.5.
    gpu.upload_layer(
        &queue,
        l0,
        &layer_from(n, |x, _y| {
            if x < n / 2 {
                [0.0, 0.0, 0.0, 1.0]
            } else {
                [1.0, 1.0, 1.0, 1.0]
            }
        }),
    );
    gpu.apply_mosaic(&device, &queue, l0, n as f32);
    // Every pixel is the same mid-gray average.
    let a = gpu.read_pixel(&device, &queue, l0, 0, 0).unwrap();
    let b = gpu.read_pixel(&device, &queue, l0, n - 1, n - 1).unwrap();
    assert!(
        (a[0] - 0.5).abs() < 0.02,
        "cell averages to mid-gray: {a:?}"
    );
    for c in 0..4 {
        assert!((a[c] - b[c]).abs() < 0.01, "cell is uniform: {a:?} {b:?}");
    }
}

// Crystallize is seeded-deterministic and snaps to source colours: every output
// pixel equals some input pixel (no blending), and a flat field is unchanged.
#[test]
fn crystallize_is_deterministic_and_snaps() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping crystallize_is_deterministic_and_snaps");
        return;
    };
    let n = 16u32;
    let make = |x: u32, _y: u32| {
        let v = x as f32 / (n - 1) as f32;
        [v, 1.0 - v, 0.5, 1.0]
    };
    let run = |seed: f32| -> Vec<[f32; 4]> {
        let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
        gpu.ensure_canvas(&device, Size::new(n, n));
        let l0 = LayerId(0);
        gpu.ensure_layer(&device, l0);
        gpu.upload_layer(&queue, l0, &layer_from(n, make));
        gpu.apply_crystallize(&device, &queue, l0, 4.0, seed);
        let mut px = Vec::new();
        for y in 0..n {
            for x in 0..n {
                px.push(gpu.read_pixel(&device, &queue, l0, x, y).unwrap());
            }
        }
        px
    };
    let a = run(7.0);
    let b = run(7.0);
    for (pa, pb) in a.iter().zip(b.iter()) {
        for c in 0..4 {
            assert!((pa[c] - pb[c]).abs() < 1e-3, "deterministic for a seed");
        }
    }
    // Snapping: every output red value matches one of the source's red values
    // (the source reds are evenly spaced k/(n-1)).
    for p in &a {
        let near = (0..n).any(|k| (p[0] - k as f32 / (n - 1) as f32).abs() < 0.03);
        assert!(near, "snaps to a source colour, no blend: {p:?}");
    }
    let c = run(8.0);
    assert!(a != c, "different seed → different cells");
}

// Color Halftone makes bigger dots (more ink) for darker cells: a dark field
// produces more ink pixels than a bright one.
#[test]
fn color_halftone_dot_tracks_brightness() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping color_halftone_dot_tracks_brightness");
        return;
    };
    let n = 32u32;
    let ink = |v: f32| -> usize {
        let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
        gpu.ensure_canvas(&device, Size::new(n, n));
        let l0 = LayerId(0);
        gpu.ensure_layer(&device, l0);
        gpu.upload_layer(&queue, l0, &layer_from(n, |_x, _y| [v, v, v, 1.0]));
        gpu.apply_color_halftone(&device, &queue, l0, 8.0, 0.0);
        let mut count = 0;
        for y in 0..n {
            for x in 0..n {
                if gpu.read_pixel(&device, &queue, l0, x, y).unwrap()[0] < 0.5 {
                    count += 1; // red-channel ink
                }
            }
        }
        count
    };
    let dark = ink(0.2);
    let bright = ink(0.8);
    assert!(
        dark > bright,
        "darker cell → larger dot → more ink: dark {dark} bright {bright}"
    );
}

// Mezzotint produces a pure black/white field that is brighter (more white) for
// a brighter input and reproducible per seed.
#[test]
fn mezzotint_is_binary_and_tracks_brightness() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping mezzotint_is_binary_and_tracks_brightness");
        return;
    };
    let n = 32u32;
    let white = |v: f32| -> usize {
        let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
        gpu.ensure_canvas(&device, Size::new(n, n));
        let l0 = LayerId(0);
        gpu.ensure_layer(&device, l0);
        gpu.upload_layer(&queue, l0, &layer_from(n, |_x, _y| [v, v, v, 1.0]));
        gpu.apply_mezzotint(&device, &queue, l0, 0.5, 4.0);
        let mut count = 0;
        for y in 0..n {
            for x in 0..n {
                let p = gpu.read_pixel(&device, &queue, l0, x, y).unwrap();
                assert!(
                    p[0] < 0.05 || p[0] > 0.95,
                    "binary black/white output: {p:?}"
                );
                if p[0] > 0.5 {
                    count += 1;
                }
            }
        }
        count
    };
    let dark = white(0.25);
    let bright = white(0.75);
    assert!(bright > dark, "brighter → more white: {bright} > {dark}");
}

// High Pass flattens locally-uniform areas to mid-gray (0.5) and keeps the edge
// detail as a signed deviation about it (an edge has two opposite-signed sides).
#[test]
fn high_pass_flattens_flats_and_keeps_edges() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping high_pass_flattens_flats_and_keeps_edges");
        return;
    };
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    let n = 32u32;
    gpu.ensure_canvas(&device, Size::new(n, n));
    let l0 = LayerId(0);
    gpu.ensure_layer(&device, l0);
    // Vertical black/white split at the mid column.
    let c = n / 2;
    gpu.upload_layer(
        &queue,
        l0,
        &layer_from(n, |x, _y| {
            if x >= c {
                [1.0, 1.0, 1.0, 1.0]
            } else {
                [0.0, 0.0, 0.0, 1.0]
            }
        }),
    );
    gpu.apply_high_pass(&device, &queue, l0, 3.0, 1.0);

    let row = n / 2;
    // A column far from the edge is locally flat → high pass ≈ mid-gray.
    let flat = gpu.read_pixel(&device, &queue, l0, 2, row).unwrap();
    assert!(
        (flat[0] - 0.5).abs() < 0.05 && (flat[3] - 1.0).abs() < 0.02,
        "flat area → mid-gray, alpha kept: {flat:?}"
    );
    // The two sides of the edge deviate in opposite directions about mid-gray.
    let dark_side = gpu.read_pixel(&device, &queue, l0, c - 1, row).unwrap();
    let light_side = gpu.read_pixel(&device, &queue, l0, c, row).unwrap();
    assert!(
        dark_side[0] < 0.5,
        "dark side of edge dips below mid-gray: {dark_side:?}"
    );
    assert!(
        light_side[0] > 0.5,
        "light side of edge rises above mid-gray: {light_side:?}"
    );
}

// ---- Render family (Phase 8) ----------------------------------------------

// Clouds fills the layer with a deterministic fBm field: opaque, gray, in range,
// reproducible per seed, and actually varying (not a flat fill). Difference
// Clouds = |source − cloud|, so on a black field it reproduces the raw cloud.
#[test]
fn clouds_are_deterministic_opaque_and_vary() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping clouds_are_deterministic_opaque_and_vary");
        return;
    };
    let n = 32u32;
    let run = |difference: bool, base: f32, seed: f32| -> Vec<[f32; 4]> {
        let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
        gpu.ensure_canvas(&device, Size::new(n, n));
        let l0 = LayerId(0);
        gpu.ensure_layer(&device, l0);
        gpu.upload_layer(&queue, l0, &layer_from(n, |_x, _y| [base, base, base, 1.0]));
        gpu.apply_clouds(&device, &queue, l0, difference, seed, 16.0, 0.5, 5);
        let mut px = Vec::with_capacity((n * n) as usize);
        for y in 0..n {
            for x in 0..n {
                px.push(gpu.read_pixel(&device, &queue, l0, x, y).unwrap());
            }
        }
        px
    };

    // Clouds: deterministic for a fixed seed.
    let a = run(false, 0.5, 7.0);
    let b = run(false, 0.5, 7.0);
    for (pa, pb) in a.iter().zip(b.iter()) {
        for c in 0..4 {
            assert!((pa[c] - pb[c]).abs() < 1e-3, "deterministic for a fixed seed");
        }
    }
    // Opaque, gray, in range, and varying.
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for p in &a {
        assert!((p[3] - 1.0).abs() < 0.02, "opaque: {p:?}");
        assert!((p[0] - p[1]).abs() < 2e-3 && (p[1] - p[2]).abs() < 2e-3, "gray: {p:?}");
        for c in 0..3 {
            assert!((-0.01..=1.01).contains(&p[c]), "in range: {p:?}");
        }
        min = min.min(p[0]);
        max = max.max(p[0]);
    }
    assert!(max - min > 0.05, "field varies (not flat): {min}..{max}");

    // Different seed → a different field.
    let c = run(false, 0.5, 9.0);
    assert!(
        a.iter().zip(c.iter()).any(|(p, q)| (p[0] - q[0]).abs() > 0.02),
        "different seeds → different clouds"
    );

    // Difference Clouds on a black field reproduces the raw cloud (|0 − n| = n).
    let from_black = run(true, 0.0, 7.0);
    for (d, p) in from_black.iter().zip(a.iter()) {
        assert!(
            (d[0] - p[0]).abs() < 5e-3,
            "diff clouds on black = clouds: {} vs {}",
            d[0],
            p[0]
        );
    }
}

// ---- Oil Paint (Kuwahara quadrant filter, Phase 8) -----------------------

// A flat field has zero variance in every quadrant, so the chosen quadrant's
// mean equals the (constant) source colour: Oil Paint leaves it unchanged.
#[test]
fn oil_paint_flat_field_is_identity() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping oil_paint_flat_field_is_identity");
        return;
    };
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    let n = 8u32;
    gpu.ensure_canvas(&device, Size::new(n, n));
    let l0 = LayerId(0);
    gpu.ensure_layer(&device, l0);
    gpu.upload_layer(&queue, l0, &layer_from(n, |_x, _y| [0.3, 0.6, 0.2, 1.0]));
    gpu.apply_oil_paint(&device, &queue, l0, 2.0);
    let p = gpu.read_pixel(&device, &queue, l0, n / 2, n / 2).unwrap();
    assert!(
        (p[0] - 0.3).abs() < 0.02 && (p[1] - 0.6).abs() < 0.02 && (p[2] - 0.2).abs() < 0.02,
        "flat field unchanged: {p:?}"
    );
}

// On a hard vertical black/white step, Oil Paint snaps each pixel to a pure
// side (the flattest quadrant wins) — never the ~0.5 average a box blur gives.
#[test]
fn oil_paint_snaps_to_a_side_of_an_edge() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping oil_paint_snaps_to_a_side_of_an_edge");
        return;
    };
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    let n = 16u32;
    gpu.ensure_canvas(&device, Size::new(n, n));
    let l0 = LayerId(0);
    gpu.ensure_layer(&device, l0);
    gpu.upload_layer(
        &queue,
        l0,
        &layer_from(n, |x, _y| {
            if x < n / 2 {
                [0.0, 0.0, 0.0, 1.0]
            } else {
                [1.0, 1.0, 1.0, 1.0]
            }
        }),
    );
    gpu.apply_oil_paint(&device, &queue, l0, 2.0);
    let row = n / 2;
    for x in 0..n {
        let v = gpu.read_pixel(&device, &queue, l0, x, row).unwrap()[0];
        assert!(
            !(0.05..=0.95).contains(&v),
            "snaps to a pure side, never a blend: {v} at x={x}"
        );
    }
    assert!(
        gpu.read_pixel(&device, &queue, l0, 0, row).unwrap()[0] < 0.05,
        "left stays black"
    );
    assert!(
        gpu.read_pixel(&device, &queue, l0, n - 1, row).unwrap()[0] > 0.95,
        "right stays white"
    );
}

// ---- Posterize / Threshold (destructive tonal filters, Phase 8) ----------

// 2-level posterize snaps every channel to its 0 or 1 extreme (the endpoints
// are identical in linear and sRGB space, so the assertion is space-agnostic).
#[test]
fn posterize_2_levels_snaps_to_extremes() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping posterize_2_levels_snaps_to_extremes");
        return;
    };
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    let n = 8u32;
    gpu.ensure_canvas(&device, Size::new(n, n));
    let l0 = LayerId(0);
    gpu.ensure_layer(&device, l0);
    // A per-pixel ramp of distinct grays so several intermediate values are tested.
    gpu.upload_layer(
        &queue,
        l0,
        &layer_from(n, |x, y| {
            let v = (y * n + x) as f32 / (n * n - 1) as f32;
            [v, v, v, 1.0]
        }),
    );
    gpu.apply_posterize(&device, &queue, l0, 2);
    for y in 0..n {
        for x in 0..n {
            let p = gpu.read_pixel(&device, &queue, l0, x, y).unwrap();
            assert!(
                !(0.02..=0.98).contains(&p[0]),
                "2-level posterize → pure extreme: {} at ({x},{y})",
                p[0]
            );
            assert!((p[3] - 1.0).abs() < 0.02, "alpha preserved");
        }
    }
    // Corners: the darkest pixel stays black, the brightest stays white.
    assert!(gpu.read_pixel(&device, &queue, l0, 0, 0).unwrap()[0] < 0.02, "black stays");
    assert!(
        gpu.read_pixel(&device, &queue, l0, n - 1, n - 1).unwrap()[0] > 0.98,
        "white stays"
    );
}

// A flat mid-gray field is (very nearly) a fixed point of a 4-level posterize
// only at lattice points; here we just assert the result is one of the 4 levels
// and is uniform across the field (the op is a pure per-pixel transfer).
#[test]
fn posterize_is_a_uniform_per_pixel_transfer() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping posterize_is_a_uniform_per_pixel_transfer");
        return;
    };
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    let n = 8u32;
    gpu.ensure_canvas(&device, Size::new(n, n));
    let l0 = LayerId(0);
    gpu.ensure_layer(&device, l0);
    gpu.upload_layer(&queue, l0, &layer_from(n, |_x, _y| [0.18, 0.5, 0.82, 1.0]));
    gpu.apply_posterize(&device, &queue, l0, 4);
    let a = gpu.read_pixel(&device, &queue, l0, 0, 0).unwrap();
    let b = gpu.read_pixel(&device, &queue, l0, n - 1, n - 1).unwrap();
    for c in 0..3 {
        assert!((a[c] - b[c]).abs() < 0.01, "uniform field → uniform output");
    }
}

// Threshold collapses a gray ramp to pure black/white at the luma cutoff: the
// dark end goes black, the bright end white, and the output is strictly binary.
#[test]
fn threshold_splits_to_black_and_white() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping threshold_splits_to_black_and_white");
        return;
    };
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    let n = 16u32;
    gpu.ensure_canvas(&device, Size::new(n, n));
    let l0 = LayerId(0);
    gpu.ensure_layer(&device, l0);
    // Horizontal black→white step so each side is clearly on one side of cutoff.
    gpu.upload_layer(
        &queue,
        l0,
        &layer_from(n, |x, _y| {
            if x < n / 2 {
                [0.0, 0.0, 0.0, 1.0]
            } else {
                [1.0, 1.0, 1.0, 1.0]
            }
        }),
    );
    gpu.apply_threshold(&device, &queue, l0, 0.5);
    let row = n / 2;
    for x in 0..n {
        let v = gpu.read_pixel(&device, &queue, l0, x, row).unwrap()[0];
        assert!(!(0.02..=0.98).contains(&v), "output is binary: {v} at x={x}");
    }
    assert!(gpu.read_pixel(&device, &queue, l0, 0, row).unwrap()[0] < 0.02, "left → black");
    assert!(
        gpu.read_pixel(&device, &queue, l0, n - 1, row).unwrap()[0] > 0.98,
        "right → white"
    );
}

// ---- Smart Filters (non-destructive filter stack) ------------------------

// A smart Gaussian blur softens a hard edge, and *disabling* it (re-applying an
// empty pass list, then clearing the source) restores the original pixels
// exactly — proving the stack is non-destructive (the source is never baked).
#[test]
fn smart_gaussian_blur_softens_then_disabling_restores_original() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping smart_gaussian_blur_softens_then_disabling_restores_original");
        return;
    };
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    let n = 16u32;
    gpu.ensure_canvas(&device, Size::new(n, n));
    let l0 = LayerId(0);
    gpu.ensure_layer(&device, l0);
    // Hard vertical black/white edge at x = n/2.
    gpu.upload_layer(
        &queue,
        l0,
        &layer_from(n, |x, _y| {
            if x < n / 2 {
                [0.0, 0.0, 0.0, 1.0]
            } else {
                [1.0, 1.0, 1.0, 1.0]
            }
        }),
    );
    let row = n / 2;
    // The two pixels straddling the edge start fully black / fully white.
    let pre_l = gpu.read_pixel(&device, &queue, l0, n / 2 - 1, row).unwrap()[0];
    let pre_r = gpu.read_pixel(&device, &queue, l0, n / 2, row).unwrap()[0];
    assert!(pre_l < 0.02 && pre_r > 0.98, "hard edge before blur: {pre_l} {pre_r}");

    // Apply a smart Gaussian blur (shader kind 1) over the source. This snapshots
    // the un-filtered source and writes the blurred result into the layer.
    gpu.reapply_smart_filters(&device, &queue, l0, &[(1, 6.0, 0.0)]);
    assert!(gpu.has_smart_source(l0), "blur snapshotted the source");
    let blur_l = gpu.read_pixel(&device, &queue, l0, n / 2 - 1, row).unwrap()[0];
    let blur_r = gpu.read_pixel(&device, &queue, l0, n / 2, row).unwrap()[0];
    // The edge is now soft: the dark side brightened, the bright side darkened,
    // both into intermediate values (a true blend, never the hard 0/1 step).
    assert!(blur_l > 0.05, "dark side softened up: {blur_l}");
    assert!(blur_r < 0.95, "bright side softened down: {blur_r}");
    assert!(blur_l < blur_r, "still brighter on the right: {blur_l} {blur_r}");

    // Re-applying with the filter *disabled* (empty pass list) resets to source.
    gpu.reapply_smart_filters(&device, &queue, l0, &[]);
    let off_l = gpu.read_pixel(&device, &queue, l0, n / 2 - 1, row).unwrap()[0];
    let off_r = gpu.read_pixel(&device, &queue, l0, n / 2, row).unwrap()[0];
    assert!(
        off_l < 0.02 && off_r > 0.98,
        "disabling restores the hard edge: {off_l} {off_r}"
    );

    // Clearing the source (last filter removed) leaves the original pixels and
    // drops the snapshot — the layer is plain, destructively-editable again.
    gpu.clear_smart_source(&device, &queue, l0);
    assert!(!gpu.has_smart_source(l0), "source dropped after clear");
    for x in 0..n {
        let v = gpu.read_pixel(&device, &queue, l0, x, row).unwrap()[0];
        let want = if x < n / 2 { 0.0 } else { 1.0 };
        assert!((v - want).abs() < 0.02, "original restored at x={x}: {v}");
    }
}

// The smart-filter source is captured once and never re-snapshotted: editing the
// stack (here, swapping a blur radius) still re-applies from the pristine source,
// so a strong blur followed by a zero-radius blur returns the hard edge.
#[test]
fn smart_filter_edit_reapplies_from_pristine_source() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping smart_filter_edit_reapplies_from_pristine_source");
        return;
    };
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    let n = 16u32;
    gpu.ensure_canvas(&device, Size::new(n, n));
    let l0 = LayerId(0);
    gpu.ensure_layer(&device, l0);
    gpu.upload_layer(
        &queue,
        l0,
        &layer_from(n, |x, _y| {
            if x < n / 2 {
                [0.0, 0.0, 0.0, 1.0]
            } else {
                [1.0, 1.0, 1.0, 1.0]
            }
        }),
    );
    let row = n / 2;
    // Strong blur, then edit the same entry down to radius 0 (a no-op pass).
    gpu.reapply_smart_filters(&device, &queue, l0, &[(1, 8.0, 0.0)]);
    gpu.reapply_smart_filters(&device, &queue, l0, &[(1, 0.0, 0.0)]);
    // Because the source was never overwritten, the hard edge is back exactly.
    for x in 0..n {
        let v = gpu.read_pixel(&device, &queue, l0, x, row).unwrap()[0];
        let want = if x < n / 2 { 0.0 } else { 1.0 };
        assert!(
            (v - want).abs() < 0.03,
            "edit re-applies from pristine source at x={x}: {v}"
        );
    }
}

// Tilt-Shift (Blur Gallery, kind 30): a vertical hard edge runs the full canvas
// height; a horizontal focus band sits on the middle rows. Inside the band the
// edge stays a hard step; far above the band the edge is blurred (bleeds), so
// the column just left of the step is no longer pure black. Skips if no adapter.
#[test]
fn tilt_shift_band_sharp_outside_blurred() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping tilt_shift_band_sharp_outside_blurred");
        return;
    };
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    let n = 32u32;
    gpu.ensure_canvas(&device, Size::new(n, n));
    let l0 = LayerId(0);
    gpu.ensure_layer(&device, l0);
    // Black left half, white right half (hard vertical edge at x = n/2).
    gpu.upload_layer(
        &queue,
        l0,
        &layer_from(n, |x, _y| {
            if x < n / 2 {
                [0.0, 0.0, 0.0, 1.0]
            } else {
                [1.0, 1.0, 1.0, 1.0]
            }
        }),
    );
    // Focus band centred on the middle row, narrow band + feather, strong blur.
    let center = 0.5; // uv (mid height)
    gpu.apply_tilt_shift(&device, &queue, l0, center, 3.0, 4.0, 8.0, 0.0);

    let col = n / 2; // x just right of the step is white, x-1 is black
    // In-band row: the edge is untouched (still a hard black→white step).
    let mid = n / 2;
    let in_l = gpu.read_pixel(&device, &queue, l0, col - 1, mid).unwrap()[0];
    let in_r = gpu.read_pixel(&device, &queue, l0, col, mid).unwrap()[0];
    assert!(in_l < 0.05, "in-band left of edge stays black: {in_l}");
    assert!(in_r > 0.95, "in-band right of edge stays white: {in_r}");
    // Far-from-band row (top): the edge is blurred, so the black side bleeds up.
    let far_l = gpu.read_pixel(&device, &queue, l0, col - 1, 0).unwrap()[0];
    assert!(
        far_l > 0.05,
        "far-from-band edge should blur (bleed), got {far_l}"
    );
}

// Iris Blur (Blur Gallery, kind 31): a vertical hard edge runs the full canvas;
// a small focus ellipse sits at the canvas center. The center pixel stays a hard
// step; a corner pixel (well outside the ellipse) is blurred (bleeds), so the
// column just left of the step is no longer pure black. Skips if no adapter.
#[test]
fn iris_blur_center_sharp_outside_blurred() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping iris_blur_center_sharp_outside_blurred");
        return;
    };
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    let n = 41u32;
    gpu.ensure_canvas(&device, Size::new(n, n));
    let l0 = LayerId(0);
    gpu.ensure_layer(&device, l0);
    // Black left half, white right half (hard vertical edge at x = n/2).
    gpu.upload_layer(
        &queue,
        l0,
        &layer_from(n, |x, _y| {
            if x < n / 2 {
                [0.0, 0.0, 0.0, 1.0]
            } else {
                [1.0, 1.0, 1.0, 1.0]
            }
        }),
    );
    let c = (n / 2) as f32;
    // Small focus ellipse at the center, strong blur outside.
    gpu.apply_iris_blur(&device, &queue, l0, c, c, 5.0, 5.0, 0.5, 8.0);

    let col = n / 2; // x just right of the step is white, x-1 is black
    let mid = n / 2;
    // Center (inside the ellipse): the edge is untouched (hard step).
    let in_l = gpu.read_pixel(&device, &queue, l0, col - 1, mid).unwrap()[0];
    let in_r = gpu.read_pixel(&device, &queue, l0, col, mid).unwrap()[0];
    assert!(in_l < 0.05, "center left of edge stays black: {in_l}");
    assert!(in_r > 0.95, "center right of edge stays white: {in_r}");
    // Corner (top, far outside the ellipse): the edge is blurred, so it bleeds.
    let far_l = gpu.read_pixel(&device, &queue, l0, col - 1, 0).unwrap()[0];
    assert!(
        far_l > 0.05,
        "far-from-center edge should blur (bleed), got {far_l}"
    );
}

// Spin Blur (Blur Gallery): rotational motion blur of a flat (constant) image is
// the identity — spinning a constant color about any center reproduces it; an
// off-center bright pixel smears tangentially (vertically), not radially.
// Skips if no adapter.
#[test]
fn spin_blur_flat_identity_and_smears_tangentially() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping spin_blur_flat_identity_and_smears_tangentially");
        return;
    };
    let n = 31u32;
    let size = Size::new(n, n);
    let c = (n / 2) as f32;

    // --- Flat image: spin blur is identity. ---
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    gpu.ensure_canvas(&device, size);
    let lf = LayerId(0);
    gpu.ensure_layer(&device, lf);
    gpu.upload_layer(&queue, lf, &layer_from(n, |_x, _y| [0.4, 0.6, 0.8, 1.0]));
    gpu.apply_radial_blur(&device, &queue, lf, c, c, true, 30.0_f32.to_radians(), 32);
    let p = gpu.read_pixel(&device, &queue, lf, n / 2, n / 4).unwrap();
    assert!(
        (p[0] - 0.4).abs() < 0.02
            && (p[1] - 0.6).abs() < 0.02
            && (p[2] - 0.8).abs() < 0.02,
        "spin blur of a flat image is identity, got {p:?}"
    );

    // --- Off-center bright pixel: smears tangentially, not radially. ---
    let bx = n / 2 + 8; // right of center, same row
    let by = n / 2;
    let mut gpu = CanvasGpu::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    gpu.ensure_canvas(&device, size);
    let ls = LayerId(0);
    gpu.ensure_layer(&device, ls);
    gpu.upload_layer(
        &queue,
        ls,
        &layer_from(n, |x, y| {
            if x == bx && y == by {
                [1.0, 1.0, 1.0, 1.0]
            } else {
                [0.0; 4]
            }
        }),
    );
    gpu.apply_radial_blur(&device, &queue, ls, c, c, true, 0.3, 32);
    let tang = gpu.read_pixel(&device, &queue, ls, bx, by - 2).unwrap()[3]
        + gpu.read_pixel(&device, &queue, ls, bx, by + 2).unwrap()[3];
    let radial = gpu.read_pixel(&device, &queue, ls, bx - 2, by).unwrap()[3]
        + gpu.read_pixel(&device, &queue, ls, bx + 2, by).unwrap()[3];
    assert!(
        tang > radial + 0.02,
        "spin smears tangentially (vert {tang}) > radially (horiz {radial})"
    );
}
