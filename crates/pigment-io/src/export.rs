//! Image export: encode an 8-bit sRGB RGBA buffer to a file on disk.
//!
//! The output format is chosen from the destination file extension and is
//! handled by the `image` crate (png, jpg/jpeg, webp, tif/tiff, bmp).

use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    /// Underlying filesystem failure while writing the file.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// The `image` crate failed to encode or dispatch the chosen format.
    #[error("image encode failed: {0}")]
    Encode(#[from] image::ImageError),

    /// The buffer length, dimensions, or requested format were invalid.
    #[error("unsupported export request: {0}")]
    Unsupported(String),
}

/// Encode an 8-bit sRGB RGBA buffer (len == w*h*4) to `path`; the format is
/// chosen from the file extension (png, jpg/jpeg, webp, tif/tiff, bmp).
pub fn save_rgba8(path: &Path, rgba8: &[u8], w: u32, h: u32) -> Result<(), ExportError> {
    if w == 0 || h == 0 {
        return Err(ExportError::Unsupported(format!("zero dimension: {w}x{h}")));
    }

    let expected = (w as usize)
        .checked_mul(h as usize)
        .and_then(|p| p.checked_mul(4))
        .ok_or_else(|| ExportError::Unsupported(format!("dimensions overflow: {w}x{h}")))?;

    if rgba8.len() != expected {
        return Err(ExportError::Unsupported(format!(
            "buffer length {} does not match {w}x{h}x4 = {expected}",
            rgba8.len()
        )));
    }

    let img = image::RgbaImage::from_raw(w, h, rgba8.to_vec()).ok_or_else(|| {
        ExportError::Unsupported(format!("could not build {w}x{h} image from buffer"))
    })?;

    // `save` dispatches on the file extension and returns an ImageError for
    // unknown/unsupported extensions, which we surface as Encode.
    img.save(path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_rgba8_png_roundtrips() {
        let (w, h) = (2u32, 2u32);
        // Distinct pixels so we can sample one back.
        let rgba8: Vec<u8> = vec![
            255, 0, 0, 255, // (0,0) red
            0, 255, 0, 255, // (1,0) green
            0, 0, 255, 255, // (0,1) blue
            10, 20, 30, 40, // (1,1) arbitrary
        ];

        let path = std::env::temp_dir().join("pigment_export_test.png");
        let _ = std::fs::remove_file(&path);

        save_rgba8(&path, &rgba8, w, h).expect("save png");

        let decoded = image::open(&path).expect("reopen png").to_rgba8();
        assert_eq!(decoded.dimensions(), (w, h));
        // Sample pixel (1,1) round-trips exactly (PNG is lossless).
        assert_eq!(decoded.get_pixel(1, 1).0, [10, 20, 30, 40]);
        // And pixel (1,0).
        assert_eq!(decoded.get_pixel(1, 0).0, [0, 255, 0, 255]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn save_rgba8_rejects_zero_dims() {
        let path = std::env::temp_dir().join("pigment_export_zero.png");
        let err = save_rgba8(&path, &[], 0, 4).unwrap_err();
        assert!(matches!(err, ExportError::Unsupported(_)));
        assert!(!path.exists(), "must not write a file for zero dims");
    }

    #[test]
    fn save_rgba8_rejects_wrong_length() {
        let path = std::env::temp_dir().join("pigment_export_badlen.png");
        // 2x2 needs 16 bytes; give 12.
        let err = save_rgba8(&path, &[0u8; 12], 2, 2).unwrap_err();
        assert!(matches!(err, ExportError::Unsupported(_)));
        assert!(!path.exists(), "must not write a file for bad length");
    }
}
