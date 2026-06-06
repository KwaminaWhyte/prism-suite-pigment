//! Layer tree. A recursive tree of raster / group / adjustment / text / vector
//! layers, each carrying a blend mode, opacity, visibility and (later) a mask.
//! Phase 0 ships the structure + raster layers; richer kinds fill in per phase.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::adjust::Adjustment;
use crate::blend::BlendMode;
use crate::tile::{Tile, TileCoord};

/// Stable per-document layer identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LayerId(pub u64);

/// What a layer actually contains.
#[derive(Debug, Default)]
pub enum LayerKind {
    /// Painted pixels, stored sparsely as tiles.
    #[default]
    Raster,
    /// A container compositing its children before blending into the parent.
    Group { children: Vec<Layer> },
    /// A non-destructive adjustment applied to the backdrop below it.
    Adjustment(Adjustment),
    // Text / Vector / SmartObject arrive in later phases.
}

/// One node in the layer tree.
#[derive(Debug)]
pub struct Layer {
    pub id: LayerId,
    pub name: String,
    pub kind: LayerKind,
    pub blend: BlendMode,
    /// 0.0..=1.0
    pub opacity: f32,
    pub visible: bool,
    /// Sparse pixel storage for raster layers (empty for groups).
    pub tiles: HashMap<TileCoord, Arc<Tile>>,
}

impl Layer {
    pub fn raster(id: LayerId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            kind: LayerKind::Raster,
            blend: BlendMode::Normal,
            opacity: 1.0,
            visible: true,
            tiles: HashMap::new(),
        }
    }

    pub fn adjustment(id: LayerId, name: impl Into<String>, adj: Adjustment) -> Self {
        Self {
            id,
            name: name.into(),
            kind: LayerKind::Adjustment(adj),
            blend: BlendMode::Normal,
            opacity: 1.0,
            visible: true,
            tiles: HashMap::new(),
        }
    }

    pub fn group(id: LayerId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            kind: LayerKind::Group { children: Vec::new() },
            blend: BlendMode::Normal,
            opacity: 1.0,
            visible: true,
            tiles: HashMap::new(),
        }
    }
}

/// The document's ordered stack of layers (front of the vec = bottom of stack).
#[derive(Debug, Default)]
pub struct LayerTree {
    pub layers: Vec<Layer>,
    next_id: u64,
}

impl LayerTree {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn alloc_id(&mut self) -> LayerId {
        let id = LayerId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Push a new empty raster layer on top and return its id.
    pub fn add_raster(&mut self, name: impl Into<String>) -> LayerId {
        let id = self.alloc_id();
        self.layers.push(Layer::raster(id, name));
        id
    }

    /// Push a new adjustment layer on top and return its id.
    pub fn add_adjustment(&mut self, adj: Adjustment) -> LayerId {
        let id = self.alloc_id();
        self.layers.push(Layer::adjustment(id, adj.name(), adj));
        id
    }

    pub fn get(&self, id: LayerId) -> Option<&Layer> {
        self.layers.iter().find(|l| l.id == id)
    }

    pub fn get_mut(&mut self, id: LayerId) -> Option<&mut Layer> {
        self.layers.iter_mut().find(|l| l.id == id)
    }
}
