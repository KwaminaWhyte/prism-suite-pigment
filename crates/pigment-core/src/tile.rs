//! Sparse tile model. Pixels live in 256x256 tiles, allocated only where the
//! document is actually painted, and shared copy-on-write via `Arc` so undo
//! snapshots and layer clones are cheap (PLAN.md §2, RESEARCH.md §2).
//!
//! Phase 0 only sketches the types; the GPU tile cache (atlas + page table) and
//! the compositor that consumes these arrive in Phase 1.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

/// Edge length of a square tile, in pixels.
pub const TILE_SIZE: u32 = 256;

/// Address of a tile within a single layer's infinite tile grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TileCoord {
    pub tx: i32,
    pub ty: i32,
}

impl TileCoord {
    pub const fn new(tx: i32, ty: i32) -> Self {
        Self { tx, ty }
    }

    /// Tile covering pixel (x, y).
    pub fn from_pixel(x: i32, y: i32) -> Self {
        Self {
            tx: x.div_euclid(TILE_SIZE as i32),
            ty: y.div_euclid(TILE_SIZE as i32),
        }
    }
}

/// A single tile of linear-light premultiplied RGBA, one `f32` per channel in
/// CPU memory (the GPU mirror is `Rgba16Float`). Shared COW via `Arc<Tile>`.
#[derive(Clone, Debug)]
pub struct Tile {
    /// `TILE_SIZE * TILE_SIZE * 4` channels, row-major RGBA.
    pub pixels: Box<[f32]>,
}

impl Tile {
    pub fn transparent() -> Arc<Tile> {
        let len = (TILE_SIZE * TILE_SIZE * 4) as usize;
        Arc::new(Tile { pixels: vec![0.0; len].into_boxed_slice() })
    }
}

impl Default for Tile {
    fn default() -> Self {
        let len = (TILE_SIZE * TILE_SIZE * 4) as usize;
        Tile { pixels: vec![0.0; len].into_boxed_slice() }
    }
}
