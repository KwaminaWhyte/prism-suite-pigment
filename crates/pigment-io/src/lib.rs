//! pigment-io — load/save bridge between files and the document model.
//!
//! Phase 0: decode common raster formats into a flat 8-bit sRGB RGBA buffer the
//! app can upload to a GPU texture. PSD/EXR/ICC and `.pigment` save/load follow
//! in later phases (PLAN.md §4).

pub mod document_file;
pub mod exr_io;
pub mod export;
pub mod psd_import;
pub mod resize;
pub mod text;

use std::path::Path;

use pigment_core::Size;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum IoError {
    #[error("image decode failed: {0}")]
    Decode(#[from] image::ImageError),
}

/// A decoded image ready to hand to the GPU: tightly packed 8-bit sRGB RGBA,
/// `width * height * 4` bytes, top-left origin.
pub struct LoadedImage {
    pub size: Size,
    pub rgba8: Vec<u8>,
}

/// File extensions we currently accept in the open dialog.
pub const SUPPORTED_EXTENSIONS: &[&str] =
    &["png", "jpg", "jpeg", "webp", "tif", "tiff", "bmp", "gif"];

/// Decode an image file to 8-bit sRGB RGBA.
pub fn load_image(path: impl AsRef<Path>) -> Result<LoadedImage, IoError> {
    let img = image::open(path.as_ref())?;
    let rgba = img.to_rgba8();
    let size = Size::new(rgba.width(), rgba.height());
    Ok(LoadedImage { size, rgba8: rgba.into_raw() })
}

/// Build a placeholder gradient so the canvas shows something on first launch.
pub fn placeholder(size: Size) -> LoadedImage {
    let (w, h) = (size.width, size.height);
    let mut rgba8 = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = (255 * x / w.max(1)) as u8;
            let g = (255 * y / h.max(1)) as u8;
            let b = 128u8;
            rgba8.extend_from_slice(&[r, g, b, 255]);
        }
    }
    LoadedImage { size, rgba8 }
}
