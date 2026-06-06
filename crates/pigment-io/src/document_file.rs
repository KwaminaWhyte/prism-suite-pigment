//! `.pigment` document container: metadata JSON + per-layer lz4-compressed
//! linear-premultiplied RGBA16F pixel blobs.
//!
//! Container layout:
//!   magic b"PIGMENT1" (8 bytes)
//!   u32 LE metadata-json-length
//!   metadata JSON (serde_json)
//!   then for each layer in `meta.layers` order:
//!     u64 LE id
//!     u32 LE compressed-len
//!     lz4-compressed rgba16f bytes (length-prepended via lz4_flex)

use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

/// 8-byte container magic / version tag.
const MAGIC: &[u8; 8] = b"PIGMENT1";

/// Bytes per pixel of the RGBA16F payload (4 channels * 2 bytes).
const BYTES_PER_PIXEL: usize = 4 * 2;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerMeta {
    pub id: u64,
    pub name: String,
    pub blend: u32, // BlendMode shader id
    pub opacity: f32,
    pub visible: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DocMeta {
    pub width: u32,
    pub height: u32,
    pub layers: Vec<LayerMeta>, // bottom-to-top order
}

/// Raw pixel payload for one layer: linear-premultiplied RGBA16F as little-endian
/// bytes, length == width*height*4*2.
pub struct LayerPixels {
    pub id: u64,
    pub rgba16f: Vec<u8>,
}

#[derive(Debug, thiserror::Error)]
pub enum DocError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("metadata json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("bad file magic: expected b\"PIGMENT1\"")]
    BadMagic,

    #[error("unsupported format version")]
    Version,

    #[error("lz4 decompress failed: {0}")]
    Decompress(#[from] lz4_flex::block::DecompressError),

    #[error(
        "pixel length mismatch for layer {id}: expected {expected} bytes, got {actual}"
    )]
    LengthMismatch { id: u64, expected: usize, actual: usize },
}

/// Serialize the document to `path`.
pub fn save_document(
    path: &Path,
    meta: &DocMeta,
    pixels: &[LayerPixels],
) -> Result<(), DocError> {
    let file = std::fs::File::create(path)?;
    let mut w = BufWriter::new(file);

    // magic
    w.write_all(MAGIC)?;

    // metadata JSON, length-prefixed
    let json = serde_json::to_vec(meta)?;
    w.write_all(&(json.len() as u32).to_le_bytes())?;
    w.write_all(&json)?;

    // layers in meta order; look pixels up by id
    for layer in &meta.layers {
        let blob = pixels
            .iter()
            .find(|p| p.id == layer.id)
            .map(|p| p.rgba16f.as_slice())
            .unwrap_or(&[]);

        let compressed = lz4_flex::compress_prepend_size(blob);

        w.write_all(&layer.id.to_le_bytes())?;
        w.write_all(&(compressed.len() as u32).to_le_bytes())?;
        w.write_all(&compressed)?;
    }

    w.flush()?;
    Ok(())
}

/// Inverse of `save_document`. Decompress each layer blob and validate magic plus
/// that each decompressed blob length == width*height*4*2.
pub fn load_document(path: &Path) -> Result<(DocMeta, Vec<LayerPixels>), DocError> {
    let file = std::fs::File::open(path)?;
    let mut r = BufReader::new(file);

    // magic
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        // distinguish a same-length-but-wrong tag (version) from arbitrary bytes.
        if magic.starts_with(b"PIGMENT") {
            return Err(DocError::Version);
        }
        return Err(DocError::BadMagic);
    }

    // metadata JSON
    let json_len = read_u32(&mut r)? as usize;
    let mut json = vec![0u8; json_len];
    r.read_exact(&mut json)?;
    let meta: DocMeta = serde_json::from_slice(&json)?;

    let expected = meta.width as usize * meta.height as usize * BYTES_PER_PIXEL;

    let mut pixels = Vec::with_capacity(meta.layers.len());
    for _ in 0..meta.layers.len() {
        let id = read_u64(&mut r)?;
        let comp_len = read_u32(&mut r)? as usize;

        let mut compressed = vec![0u8; comp_len];
        r.read_exact(&mut compressed)?;

        let rgba16f = lz4_flex::decompress_size_prepended(&compressed)?;

        if rgba16f.len() != expected {
            return Err(DocError::LengthMismatch {
                id,
                expected,
                actual: rgba16f.len(),
            });
        }

        pixels.push(LayerPixels { id, rgba16f });
    }

    Ok((meta, pixels))
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32, DocError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64, DocError> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let (w, h) = (4u32, 4u32);
        let len = w as usize * h as usize * BYTES_PER_PIXEL;

        let meta = DocMeta {
            width: w,
            height: h,
            layers: vec![
                LayerMeta {
                    id: 1,
                    name: "background".to_string(),
                    blend: 0,
                    opacity: 1.0,
                    visible: true,
                },
                LayerMeta {
                    id: 2,
                    name: "paint".to_string(),
                    blend: 3,
                    opacity: 0.5,
                    visible: false,
                },
            ],
        };

        let px0: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
        let px1: Vec<u8> = (0..len).map(|i| (i.wrapping_mul(7) % 253) as u8).collect();
        let pixels = vec![
            LayerPixels { id: 1, rgba16f: px0.clone() },
            LayerPixels { id: 2, rgba16f: px1.clone() },
        ];

        let mut path = std::env::temp_dir();
        path.push("pigment_doc_round_trip.pigment");

        save_document(&path, &meta, &pixels).expect("save");
        let (rmeta, rpix) = load_document(&path).expect("load");

        // metadata
        assert_eq!(rmeta.width, w);
        assert_eq!(rmeta.height, h);
        assert_eq!(rmeta.layers.len(), 2);
        assert_eq!(rmeta.layers[0].id, 1);
        assert_eq!(rmeta.layers[0].name, "background");
        assert_eq!(rmeta.layers[0].blend, 0);
        assert_eq!(rmeta.layers[0].opacity, 1.0);
        assert!(rmeta.layers[0].visible);
        assert_eq!(rmeta.layers[1].id, 2);
        assert_eq!(rmeta.layers[1].name, "paint");
        assert_eq!(rmeta.layers[1].blend, 3);
        assert_eq!(rmeta.layers[1].opacity, 0.5);
        assert!(!rmeta.layers[1].visible);

        // pixels
        assert_eq!(rpix.len(), 2);
        assert_eq!(rpix[0].id, 1);
        assert_eq!(rpix[0].rgba16f, px0);
        assert_eq!(rpix[1].id, 2);
        assert_eq!(rpix[1].rgba16f, px1);

        let _ = std::fs::remove_file(&path);
    }
}
