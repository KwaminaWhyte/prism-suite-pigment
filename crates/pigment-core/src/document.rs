//! The top-level document: canvas size, layer tree, and (later) color profile,
//! selection, and command/undo stack.

use crate::geometry::Size;
use crate::layer::{LayerId, LayerTree};

#[derive(Debug)]
pub struct Document {
    pub size: Size,
    pub layers: LayerTree,
    pub active_layer: Option<LayerId>,
}

impl Document {
    /// Create a blank document with a single empty raster layer.
    pub fn new(size: Size) -> Self {
        let mut layers = LayerTree::new();
        let bg = layers.add_raster("Background");
        Self { size, layers, active_layer: Some(bg) }
    }
}
