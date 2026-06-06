//! OpenEXR loading: decode a `.exr` file into linear-light RGBA `f32`.

use pigment_core::Size;

/// Load an OpenEXR file as linear-light RGBA f32 (len == w*h*4), full canvas.
///
/// Pixels are stored row-major, top-left origin, four interleaved channels per
/// pixel (R, G, B, A). Missing channels default to `0.0` for RGB and `1.0` for
/// alpha (the `exr` crate substitutes `1.0` when no alpha channel exists).
/// Returns the layer `Size` together with the packed pixel buffer.
pub fn load_exr(path: &std::path::Path) -> Result<(Size, Vec<f32>), exr::error::Error> {
    // `read_first_rgba_layer_from_file` drives two closures:
    //  - `create` builds our pixel container once the resolution is known,
    //  - `set_pixel` writes a single converted RGBA sample into that container.
    let image = exr::prelude::read_first_rgba_layer_from_file(
        path,
        // create: allocate a flat w*h*4 buffer for the whole layer.
        |resolution: exr::math::Vec2<usize>, _channels: &_| {
            let w = resolution.width();
            let h = resolution.height();
            // RGB default to 0.0, alpha to 1.0.
            let mut pixels = vec![0.0f32; w * h * 4];
            for a in pixels.iter_mut().skip(3).step_by(4) {
                *a = 1.0;
            }
            (w, pixels)
        },
        // set_pixel: store one converted f32 RGBA sample at (x, y).
        |(w, pixels): &mut (usize, Vec<f32>),
         position: exr::math::Vec2<usize>,
         (r, g, b, a): (f32, f32, f32, f32)| {
            let idx = (position.y() * *w + position.x()) * 4;
            pixels[idx] = r;
            pixels[idx + 1] = g;
            pixels[idx + 2] = b;
            pixels[idx + 3] = a;
        },
    )?;

    let resolution = image.layer_data.size;
    let size = Size::new(resolution.width() as u32, resolution.height() as u32);
    let (_w, pixels) = image.layer_data.channel_data.pixels;
    Ok((size, pixels))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_exr_missing_path_errors() {
        let path = std::env::temp_dir().join("pigment_does_not_exist_42.exr");
        let _ = std::fs::remove_file(&path);
        let result = load_exr(&path);
        assert!(result.is_err(), "loading a nonexistent EXR must return Err");
    }

    #[test]
    fn load_exr_roundtrips_written_file() {
        // Write a tiny RGBA exr, then read it back and check size + samples.
        let path = std::env::temp_dir().join("pigment_roundtrip_test.exr");
        let _ = std::fs::remove_file(&path);

        exr::prelude::write_rgba_file(&path, 2, 2, |x, y| (x as f32, y as f32, 0.5, 1.0))
            .expect("write tiny exr");

        let (size, pixels) = load_exr(&path).expect("read tiny exr");
        assert_eq!(size, Size::new(2, 2));
        assert_eq!(pixels.len(), 2 * 2 * 4);

        // Pixel (x=1, y=0) in a 2x2 image -> (1.0, 0.0, 0.5, 1.0).
        let (x, y, w) = (1usize, 0usize, 2usize);
        let idx = (y * w + x) * 4;
        assert_eq!(pixels[idx], 1.0);
        assert_eq!(pixels[idx + 1], 0.0);
        assert_eq!(pixels[idx + 2], 0.5);
        assert_eq!(pixels[idx + 3], 1.0);

        let _ = std::fs::remove_file(&path);
    }
}
