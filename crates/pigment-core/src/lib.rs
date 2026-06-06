//! pigment-core — GPU-agnostic document model for the Pigment image editor.
//!
//! This crate owns the *state*: documents, the layer tree, blend modes, the
//! sparse tile model, and (later) the command/undo stack. It deliberately knows
//! nothing about wgpu — rendering lives in `pigment-gpu`, and the app wires the
//! two together. See PLAN.md §2.

pub mod blend;
pub mod color;
pub mod document;
pub mod fill;
pub mod geometry;
pub mod layer;
pub mod tile;

pub use blend::BlendMode;
pub use color::Rgba;
pub use document::Document;
pub use geometry::{Rect, Size};
pub use layer::{Layer, LayerId, LayerKind, LayerTree};
pub use tile::{Tile, TileCoord, TILE_SIZE};
