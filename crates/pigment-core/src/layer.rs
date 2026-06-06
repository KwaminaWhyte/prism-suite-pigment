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
    /// An editable text layer; pixels are re-rasterized from this definition.
    Text(TextDef),
    /// An editable vector shape; pixels are re-rasterized from this definition.
    Vector(VectorDef),
    // SmartObject arrives in a later phase.
}

/// Editable text-layer definition. Colors are straight sRGB; `align` is
/// 0=left, 1=center, 2=right.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextDef {
    pub text: String,
    pub font_px: f32,
    pub color: [f32; 4],
    pub align: u8,
}

impl Default for TextDef {
    fn default() -> Self {
        Self { text: "Text".into(), font_px: 48.0, color: [1.0, 1.0, 1.0, 1.0], align: 0 }
    }
}

/// Editable vector-shape definition. `kind` 0=rectangle, 1=ellipse; `rect` is
/// `[x, y, w, h]` in document px; `color` is straight sRGB.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct VectorDef {
    pub kind: u8,
    pub rect: [f32; 4],
    pub color: [f32; 4],
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

    pub fn text(id: LayerId, name: impl Into<String>, def: TextDef) -> Self {
        Self {
            id,
            name: name.into(),
            kind: LayerKind::Text(def),
            blend: BlendMode::Normal,
            opacity: 1.0,
            visible: true,
            tiles: HashMap::new(),
        }
    }

    pub fn vector(id: LayerId, name: impl Into<String>, def: VectorDef) -> Self {
        Self {
            id,
            name: name.into(),
            kind: LayerKind::Vector(def),
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

    pub fn add_text(&mut self, def: TextDef) -> LayerId {
        let id = self.alloc_id();
        self.layers.push(Layer::text(id, "Text", def));
        id
    }

    pub fn add_vector(&mut self, name: impl Into<String>, def: VectorDef) -> LayerId {
        let id = self.alloc_id();
        self.layers.push(Layer::vector(id, name, def));
        id
    }

    pub fn get(&self, id: LayerId) -> Option<&Layer> {
        self.layers.iter().find(|l| l.id == id)
    }

    pub fn get_mut(&mut self, id: LayerId) -> Option<&mut Layer> {
        self.layers.iter_mut().find(|l| l.id == id)
    }
}
