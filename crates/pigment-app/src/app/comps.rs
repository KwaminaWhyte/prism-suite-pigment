//! Layer comps: named snapshots of per-layer appearance (visibility, opacity,
//! blend mode). A comp records, for every layer present at capture time, that
//! layer's attributes keyed by its stable [`LayerId`]. Restoring a comp re-applies
//! those attributes to the matching layers; layers added since capture are left
//! untouched, and entries for layers since removed are ignored.
//!
//! The capture/restore logic is pure over the layer list (`&[Layer]` /
//! `&mut [Layer]`) so it is unit-testable headlessly — no GPU or app state.
//! Position/transform is intentionally out of scope: Pigment layers carry no
//! persistent position in their model (Move/Transform bakes into pixels), so a
//! comp captures the appearance attributes that *do* live on the layer.

use prism_core::{BlendMode, Layer, LayerId};
use prism_io::document_file::{LayerCompEntry, LayerCompMeta};

/// One layer's snapshotted attributes within a comp.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LayerAttrs {
    pub visible: bool,
    /// 0.0..=1.0
    pub opacity: f32,
    pub blend: BlendMode,
}

impl LayerAttrs {
    /// Snapshot a layer's current appearance attributes.
    fn from_layer(l: &Layer) -> Self {
        Self {
            visible: l.visible,
            opacity: l.opacity,
            blend: l.blend,
        }
    }

    /// Re-apply these attributes onto a layer.
    fn apply_to(&self, l: &mut Layer) {
        l.visible = self.visible;
        l.opacity = self.opacity;
        l.blend = self.blend;
    }
}

/// A named snapshot of every layer's appearance attributes. `entries` are keyed
/// by [`LayerId`] so restore matches layers by stable id and is robust to layers
/// being reordered, added, or removed since capture.
#[derive(Clone, Debug)]
pub(crate) struct LayerComp {
    pub name: String,
    pub entries: Vec<(LayerId, LayerAttrs)>,
}

/// Capture a comp named `name` from the current layer list. Pure.
pub(crate) fn capture_comp(name: impl Into<String>, layers: &[Layer]) -> LayerComp {
    LayerComp {
        name: name.into(),
        entries: layers
            .iter()
            .map(|l| (l.id, LayerAttrs::from_layer(l)))
            .collect(),
    }
}

/// Restore a comp onto the layer list: for each captured entry, find the matching
/// layer by id and re-apply its attributes. Layers with no captured entry, and
/// captured entries with no matching layer, are left as-is / ignored. Pure.
pub(crate) fn apply_comp(comp: &LayerComp, layers: &mut [Layer]) {
    for (id, attrs) in &comp.entries {
        if let Some(layer) = layers.iter_mut().find(|l| l.id == *id) {
            attrs.apply_to(layer);
        }
    }
}

/// Map a runtime comp to its serializable `.pigment` representation. Pure.
pub(crate) fn comp_to_meta(comp: &LayerComp) -> LayerCompMeta {
    LayerCompMeta {
        name: comp.name.clone(),
        entries: comp
            .entries
            .iter()
            .map(|(id, a)| LayerCompEntry {
                id: id.0,
                blend: a.blend.shader_id(),
                opacity: a.opacity,
                visible: a.visible,
            })
            .collect(),
    }
}

/// Inverse of [`comp_to_meta`]: map a serialized comp back to the runtime model,
/// remapping the saved layer ids through `id_map` (old saved id -> freshly
/// allocated load-time id). Entries whose layer is absent from `id_map` (the
/// layer no longer exists in the loaded document) are dropped. Pure.
pub(crate) fn meta_to_comp(
    meta: &LayerCompMeta,
    id_map: &std::collections::HashMap<u64, LayerId>,
) -> LayerComp {
    LayerComp {
        name: meta.name.clone(),
        entries: meta
            .entries
            .iter()
            .filter_map(|e| {
                id_map.get(&e.id).map(|&id| {
                    (
                        id,
                        LayerAttrs {
                            visible: e.visible,
                            opacity: e.opacity,
                            blend: BlendMode::from_shader_id(e.blend),
                        },
                    )
                })
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_core::LayerId;

    fn layer(id: u64, visible: bool, opacity: f32, blend: BlendMode) -> Layer {
        let mut l = Layer::raster(LayerId(id), format!("L{id}"));
        l.visible = visible;
        l.opacity = opacity;
        l.blend = blend;
        l
    }

    // Capturing then immediately restoring is a no-op: every attribute survives
    // the round-trip exactly.
    #[test]
    fn capture_then_restore_round_trips() {
        let mut layers = vec![
            layer(1, true, 1.0, BlendMode::Normal),
            layer(2, false, 0.5, BlendMode::Multiply),
            layer(3, true, 0.25, BlendMode::Screen),
        ];
        let comp = capture_comp("state", &layers);
        // Mutate everything, then restore.
        for l in &mut layers {
            l.visible = !l.visible;
            l.opacity = 0.9;
            l.blend = BlendMode::Overlay;
        }
        apply_comp(&comp, &mut layers);
        assert!(layers[0].visible && (layers[0].opacity - 1.0).abs() < 1e-9);
        assert_eq!(layers[0].blend, BlendMode::Normal);
        assert!(!layers[1].visible && (layers[1].opacity - 0.5).abs() < 1e-9);
        assert_eq!(layers[1].blend, BlendMode::Multiply);
        assert!(layers[2].visible && (layers[2].opacity - 0.25).abs() < 1e-9);
        assert_eq!(layers[2].blend, BlendMode::Screen);
    }

    // Restoring a comp after edits reverts visibility / opacity / blend back to
    // the captured values.
    #[test]
    fn restore_reverts_edits() {
        let mut layers = vec![layer(7, true, 1.0, BlendMode::Normal)];
        let comp = capture_comp("clean", &layers);
        // User edits the layer.
        layers[0].visible = false;
        layers[0].opacity = 0.1;
        layers[0].blend = BlendMode::Darken;
        // Restoring the earlier comp undoes those edits.
        apply_comp(&comp, &mut layers);
        assert!(layers[0].visible);
        assert!((layers[0].opacity - 1.0).abs() < 1e-9);
        assert_eq!(layers[0].blend, BlendMode::Normal);
    }

    // Restore is robust to a layer being removed since capture (the entry is
    // simply ignored) and to a layer being added since capture (it is left
    // untouched). Matching is by stable id, not position.
    #[test]
    fn restore_robust_to_add_remove_and_reorder() {
        let original = vec![
            layer(1, false, 0.2, BlendMode::Multiply),
            layer(2, false, 0.3, BlendMode::Screen),
        ];
        let comp = capture_comp("c", &original);

        // Since capture: layer 2 was removed, a new layer 9 was added, and the
        // surviving layer 1 was reordered to the end and edited.
        let mut now = vec![
            layer(9, true, 1.0, BlendMode::Overlay), // new — must stay untouched
            layer(1, true, 1.0, BlendMode::Normal),  // edited survivor
        ];
        apply_comp(&comp, &mut now);

        // Survivor restored by id regardless of new position.
        let l1 = now.iter().find(|l| l.id == LayerId(1)).unwrap();
        assert!(!l1.visible);
        assert!((l1.opacity - 0.2).abs() < 1e-9);
        assert_eq!(l1.blend, BlendMode::Multiply);

        // Newly-added layer left exactly as it was (no entry in the comp).
        let l9 = now.iter().find(|l| l.id == LayerId(9)).unwrap();
        assert!(l9.visible);
        assert!((l9.opacity - 1.0).abs() < 1e-9);
        assert_eq!(l9.blend, BlendMode::Overlay);

        // The removed layer's entry caused no panic and no spurious insertion.
        assert_eq!(now.len(), 2);
    }

    // comp_to_meta / meta_to_comp round-trip the attributes, and the id remap on
    // load drops entries whose layer no longer exists.
    #[test]
    fn meta_conversion_round_trips_and_remaps_ids() {
        let layers = vec![
            layer(10, false, 0.4, BlendMode::Multiply),
            layer(20, true, 0.9, BlendMode::Screen),
        ];
        let comp = capture_comp("v", &layers);
        let meta = comp_to_meta(&comp);
        assert_eq!(meta.name, "v");
        assert_eq!(meta.entries.len(), 2);

        // On load the saved ids 10/20 map to fresh ids 100/200; saved id 99 (not
        // in this doc) would be dropped — here both survive.
        let mut id_map = std::collections::HashMap::new();
        id_map.insert(10u64, LayerId(100));
        id_map.insert(20u64, LayerId(200));
        let back = meta_to_comp(&meta, &id_map);
        assert_eq!(back.entries.len(), 2);
        assert_eq!(back.entries[0].0, LayerId(100));
        assert_eq!(back.entries[0].1, LayerAttrs { visible: false, opacity: 0.4, blend: BlendMode::Multiply });
        assert_eq!(back.entries[1].0, LayerId(200));

        // A comp whose layer is missing from the id map drops that entry.
        let mut partial = std::collections::HashMap::new();
        partial.insert(10u64, LayerId(100));
        let dropped = meta_to_comp(&meta, &partial);
        assert_eq!(dropped.entries.len(), 1);
        assert_eq!(dropped.entries[0].0, LayerId(100));
    }
}
