//! Photoshop `.psd` import.
//!
//! Read-only decode of a PSD into per-layer full-canvas RGBA8 buffers, plus the
//! blend/opacity/visibility metadata the renderer needs. Layers are returned
//! bottom-to-top (the order `psd::Psd::layers()` yields them). If the file has
//! no usable layer records we fall back to a single layer built from the
//! merged/flattened image so the document is never empty.

use std::path::Path;

use psd::Psd;
use thiserror::Error;

/// A single PSD layer flattened onto a document-sized RGBA8 canvas.
pub struct PsdLayer {
    /// Layer name as stored in the file.
    pub name: String,
    /// Layer opacity, normalised to `0.0..=1.0`.
    pub opacity: f32,
    /// Blend mode mapped to our shader ids (see [`map_blend_mode`]).
    pub blend: u32,
    /// Whether the layer is marked visible.
    pub visible: bool,
    /// Full-canvas RGBA8, `width * height * 4` bytes. The layer's pixels sit at
    /// their offset within the canvas; everything else is transparent.
    pub rgba8: Vec<u8>,
}

/// A decoded PSD document.
pub struct PsdDoc {
    /// Canvas width in pixels.
    pub width: u32,
    /// Canvas height in pixels.
    pub height: u32,
    /// Layers ordered bottom-to-top.
    pub layers: Vec<PsdLayer>,
}

/// Errors produced while importing a `.psd`.
#[derive(Debug, Error)]
pub enum PsdError {
    /// The file could not be read from disk.
    #[error("failed to read PSD file: {0}")]
    Io(#[from] std::io::Error),
    /// The bytes could not be parsed as a PSD.
    #[error("failed to parse PSD: {0}")]
    Parse(String),
}

/// Map a `psd::BlendMode` to our shader's blend id.
///
/// The `psd` crate's `BlendMode` enum is not re-exported at its crate root, so
/// we cannot name the type here. It is a field-less C-like enum with explicit
/// discriminants, so we cast the returned value to its discriminant and map
/// from that. Discriminants are taken from the psd 0.3.5 source:
/// `PassThrough=0, Normal=1, Dissolve=2, Darken=3, Multiply=4, ColorBurn=5,
///  LinearBurn=6, DarkerColor=7, Lighten=8, Screen=9, ColorDodge=10,
///  LinearDodge=11, LighterColor=12, Overlay=13, SoftLight=14, HardLight=15,
///  VividLight=16, LinearLight=17, PinLight=18, HardMix=19, Difference=20,
///  Exclusion=21, Subtract=22, Divide=23, Hue=24, Saturation=25, Color=26,
///  Luminosity=27`.
///
/// Output ids match our shader: Normal=0, Multiply=1, Screen=2, Overlay=3,
/// Darken=4, Lighten=5, ColorDodge=6, ColorBurn=7, HardLight=8, SoftLight=9,
/// Difference=10, Exclusion=11, LinearDodge(Add)=12, LinearBurn=13, Hue=20,
/// Saturation=21, Color=22, Luminosity=23; anything else maps to 0 (Normal).
fn map_blend_discriminant(d: i64) -> u32 {
    match d {
        1 => 0,   // Normal
        4 => 1,   // Multiply
        9 => 2,   // Screen
        13 => 3,  // Overlay
        3 => 4,   // Darken
        8 => 5,   // Lighten
        10 => 6,  // ColorDodge
        5 => 7,   // ColorBurn
        15 => 8,  // HardLight
        14 => 9,  // SoftLight
        20 => 10, // Difference
        21 => 11, // Exclusion
        11 => 12, // LinearDodge (Add)
        6 => 13,  // LinearBurn
        24 => 20, // Hue
        25 => 21, // Saturation
        26 => 22, // Color
        27 => 23, // Luminosity
        _ => 0,   // PassThrough, Dissolve, DarkerColor, etc. -> Normal
    }
}

/// Load a `.psd`, returning its layers bottom-to-top.
///
/// Falls back to a single flattened layer if no layer records are present.
pub fn load_psd(path: &Path) -> Result<PsdDoc, PsdError> {
    let bytes = std::fs::read(path)?;
    let psd = Psd::from_bytes(&bytes).map_err(|e| PsdError::Parse(e.to_string()))?;

    let width = psd.width();
    let height = psd.height();

    let mut layers = Vec::new();
    for layer in psd.layers() {
        // `blend_mode()` returns the crate's (unnameable) BlendMode enum; cast
        // its discriminant for mapping.
        let blend = map_blend_discriminant(layer.blend_mode() as i64);

        // `PsdLayer::rgba()` returns a document-sized (width*height*4) RGBA8
        // buffer with the layer placed at its offset, transparent elsewhere.
        // It can panic if a required channel is missing, so guard it and fall
        // back to a fully transparent layer rather than failing the load.
        let canvas_len = (width as usize) * (height as usize) * 4;
        let rgba8 = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| layer.rgba()))
            .unwrap_or_else(|_| vec![0u8; canvas_len]);
        let rgba8 = if rgba8.len() == canvas_len {
            rgba8
        } else {
            vec![0u8; canvas_len]
        };

        layers.push(PsdLayer {
            name: layer.name().to_string(),
            opacity: layer.opacity() as f32 / 255.0,
            blend,
            visible: layer.visible(),
            rgba8,
        });
    }

    if layers.is_empty() {
        // No usable layer records: fall back to the merged/flattened image.
        layers.push(PsdLayer {
            name: "Background".to_string(),
            opacity: 1.0,
            blend: 0,
            visible: true,
            rgba8: psd.rgba(),
        });
    }

    Ok(PsdDoc {
        width,
        height,
        layers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_psd_missing_path_errors() {
        let result = load_psd(Path::new("/nonexistent/path/does-not-exist.psd"));
        assert!(result.is_err());
        assert!(matches!(result.err(), Some(PsdError::Io(_))));
    }

    #[test]
    fn blend_mode_mapping() {
        // discriminant -> shader id
        assert_eq!(map_blend_discriminant(1), 0); // Normal
        assert_eq!(map_blend_discriminant(4), 1); // Multiply
        assert_eq!(map_blend_discriminant(9), 2); // Screen
        assert_eq!(map_blend_discriminant(13), 3); // Overlay
        assert_eq!(map_blend_discriminant(11), 12); // LinearDodge (Add)
        assert_eq!(map_blend_discriminant(27), 23); // Luminosity
                                                    // unmapped / pass-through -> Normal
        assert_eq!(map_blend_discriminant(0), 0); // PassThrough
        assert_eq!(map_blend_discriminant(2), 0); // Dissolve
        assert_eq!(map_blend_discriminant(999), 0);
    }
}
