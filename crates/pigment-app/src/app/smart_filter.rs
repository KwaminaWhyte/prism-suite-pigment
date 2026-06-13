//! Smart filters: a per-layer, **non-destructive** stack of filters applied on
//! top of a layer's stored (un-filtered) *source* pixels.
//!
//! Unlike the destructive Filter-menu operations (which bake into the layer and
//! are only reversible via the undo stack), a smart filter stays editable: the
//! layer keeps its source pixels, and the displayed/composited result is the
//! source with the *enabled* smart filters applied in order. Adding, removing,
//! re-ordering, toggling, or editing a filter re-applies the whole stack from
//! the source again (see [`crate::canvas::CanvasGpu::reapply_smart_filters`]) —
//! so nothing is ever baked and any filter can be changed later.
//!
//! This module is the **pure model** of the stack (no GPU, no app state): the
//! kinds, their parameters, the stack with its add/remove/reorder/toggle/edit
//! operations, and the `.pigment` serde mapping. The GPU re-apply lives in the
//! canvas; the stacks themselves are held app-side, keyed by [`LayerId`], next
//! to the layer comps (the layer model in `prism-core` stays GPU-/filter-clean).

use super::{with_gpu, PigmentApp};
use prism_core::{LayerId, LayerKind};
use prism_io::document_file::SmartFilterMeta;

/// One enabled smart filter's GPU pass parameters: `(shader_kind, radius, amount)`,
/// as consumed by `CanvasGpu::reapply_smart_filters`.
pub(crate) type SmartPass = (u32, f32, f32);

/// The kinds of filter that can live in a smart-filter stack. Each carries its
/// own editable parameters. This is deliberately a small, representative set —
/// it proves the non-destructive mechanism end-to-end; more kinds slot in by
/// extending this enum, its `gpu_pass`/`id` dispatch, and the params editor.
///
/// The actual pixel math reuses the **existing** GPU filter passes (the same
/// shader kinds the destructive Filter menu uses), so smart filters never
/// reimplement a filter — they only re-run it from the source.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum SmartFilterKind {
    /// Separable Gaussian blur (GPU filter shader kind 1). `radius` in px.
    GaussianBlur { radius: f32 },
    /// Unsharp-mask sharpen (GPU filter shader kind 2). `amount` 0..4.
    Sharpen { amount: f32 },
    /// Posterize: quantize each channel to N display-space levels (GPU filter
    /// shader kind 28). `levels` 2..=32 in the UI (engine accepts 2..=255).
    Posterize { levels: u32 },
}

impl SmartFilterKind {
    /// Stable serialized discriminant (`SmartFilterMeta.kind`). Distinct from the
    /// GPU shader kind so the file format is decoupled from shader numbering.
    fn id(&self) -> u32 {
        match self {
            SmartFilterKind::GaussianBlur { .. } => 0,
            SmartFilterKind::Sharpen { .. } => 1,
            SmartFilterKind::Posterize { .. } => 2,
        }
    }

    /// Pack this kind's parameters into the serialized `[f32; 4]` slot.
    fn params(&self) -> [f32; 4] {
        match *self {
            SmartFilterKind::GaussianBlur { radius } => [radius, 0.0, 0.0, 0.0],
            SmartFilterKind::Sharpen { amount } => [amount, 0.0, 0.0, 0.0],
            SmartFilterKind::Posterize { levels } => [levels as f32, 0.0, 0.0, 0.0],
        }
    }

    /// Reconstruct a kind from its serialized discriminant + params. Unknown ids
    /// fall back to a no-op-ish Gaussian blur of radius 0 so a forward-compatible
    /// document never fails to load (the filter just does nothing).
    fn from_id(id: u32, p: [f32; 4]) -> Self {
        match id {
            1 => SmartFilterKind::Sharpen { amount: p[0] },
            2 => SmartFilterKind::Posterize {
                levels: (p[0].round() as u32).clamp(2, 255),
            },
            _ => SmartFilterKind::GaussianBlur { radius: p[0] },
        }
    }

    /// Short display name for the stack list in the Properties panel.
    pub(crate) fn label(&self) -> &'static str {
        match self {
            SmartFilterKind::GaussianBlur { .. } => "Gaussian Blur",
            SmartFilterKind::Sharpen { .. } => "Sharpen",
            SmartFilterKind::Posterize { .. } => "Posterize",
        }
    }

    /// The GPU filter shader kind + `(radius, amount)` to drive the existing
    /// filter pass for this entry (see `CanvasGpu::reapply_smart_filters`).
    pub(crate) fn gpu_pass(&self) -> SmartPass {
        match *self {
            SmartFilterKind::GaussianBlur { radius } => (1, radius, 0.0),
            SmartFilterKind::Sharpen { amount } => (2, 0.0, amount),
            SmartFilterKind::Posterize { levels } => {
                (28, 0.0, (levels.clamp(2, 255)) as f32)
            }
        }
    }
}

/// One entry in a layer's smart-filter stack: a filter kind plus whether it is
/// currently enabled (a disabled entry stays in the stack but is skipped on
/// re-apply, so toggling it off restores the pixels below it).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SmartFilter {
    pub kind: SmartFilterKind,
    pub enabled: bool,
}

impl SmartFilter {
    pub(crate) fn new(kind: SmartFilterKind) -> Self {
        Self {
            kind,
            enabled: true,
        }
    }
}

/// A layer's ordered, non-destructive smart-filter stack. Bottom-to-top: index 0
/// is applied to the source first, the next on top of that, and so on. Empty for
/// layers with no smart filters (and then not serialized).
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct SmartFilterStack {
    pub filters: Vec<SmartFilter>,
}

impl SmartFilterStack {
    pub(crate) fn is_empty(&self) -> bool {
        self.filters.is_empty()
    }

    /// Append a filter on top of the stack.
    pub(crate) fn add(&mut self, kind: SmartFilterKind) {
        self.filters.push(SmartFilter::new(kind));
    }

    /// Remove the filter at `i` (no-op if out of range).
    pub(crate) fn remove(&mut self, i: usize) {
        if i < self.filters.len() {
            self.filters.remove(i);
        }
    }

    /// Toggle the enabled flag of the filter at `i`.
    pub(crate) fn toggle(&mut self, i: usize) {
        if let Some(f) = self.filters.get_mut(i) {
            f.enabled = !f.enabled;
        }
    }

    /// Move the filter at `i` one position toward the front of the stack (applied
    /// earlier). No-op at the front or out of range.
    pub(crate) fn move_up(&mut self, i: usize) {
        if i > 0 && i < self.filters.len() {
            self.filters.swap(i, i - 1);
        }
    }

    /// Move the filter at `i` one position toward the back of the stack (applied
    /// later). No-op at the back or out of range.
    pub(crate) fn move_down(&mut self, i: usize) {
        if i + 1 < self.filters.len() {
            self.filters.swap(i, i + 1);
        }
    }

    /// The enabled entries' GPU passes, in apply order. This is what the canvas
    /// runs over the source pixels to produce the displayed layer.
    pub(crate) fn enabled_passes(&self) -> Vec<SmartPass> {
        self.filters
            .iter()
            .filter(|f| f.enabled)
            .map(|f| f.kind.gpu_pass())
            .collect()
    }

    /// Map to the serializable `.pigment` representation. Pure.
    pub(crate) fn to_meta(&self) -> Vec<SmartFilterMeta> {
        self.filters
            .iter()
            .map(|f| SmartFilterMeta {
                kind: f.kind.id(),
                params: f.kind.params(),
                enabled: f.enabled,
            })
            .collect()
    }

    /// Inverse of [`Self::to_meta`]: rebuild a stack from serialized entries.
    /// Pure. Unknown kind ids degrade to a harmless zero-radius blur (see
    /// [`SmartFilterKind::from_id`]) rather than failing the load.
    pub(crate) fn from_meta(meta: &[SmartFilterMeta]) -> Self {
        Self {
            filters: meta
                .iter()
                .map(|m| SmartFilter {
                    kind: SmartFilterKind::from_id(m.kind, m.params),
                    enabled: m.enabled,
                })
                .collect(),
        }
    }
}

/// A deferred Smart-Filters panel action, collected while drawing the stack list
/// and dispatched after the section so the `&mut self` mutators don't conflict
/// with the borrows used to render the UI.
pub(crate) enum SmartFilterUi {
    None,
    Add(SmartFilterKind),
    Remove(usize),
    Toggle(usize),
    Up(usize),
    Down(usize),
    Edit(usize, SmartFilterKind),
}

/// Smart-filter operations on the active layer, wiring the pure stack model to
/// the GPU re-apply. Each mutator re-applies the stack from the source so the
/// canvas reflects the edit immediately.
impl PigmentApp {
    /// The smart-filter stack for `id`, or an empty default if the layer has none.
    pub(crate) fn smart_stack(&self, id: LayerId) -> SmartFilterStack {
        self.smart_filters.get(&id).cloned().unwrap_or_default()
    }

    /// Re-apply the active layer's smart-filter stack on the GPU (reset to source,
    /// run the enabled passes in order), then drop the source + map entry if the
    /// stack ended up empty so the layer becomes plain editable pixels again.
    /// Triggers a recomposite.
    fn reapply_active_smart_filters(&mut self, frame: &mut eframe::Frame) {
        let id = self.active_id();
        let stack = self.smart_filters.get(&id).cloned().unwrap_or_default();
        if stack.is_empty() {
            // No (or no-longer-any) smart filters: restore the source pixels and
            // forget the stack so the layer is destructively editable once more.
            self.smart_filters.remove(&id);
            with_gpu(frame, |gpu, d, q| {
                if gpu.has_smart_source(id) {
                    gpu.clear_smart_source(d, q, id);
                }
            });
        } else {
            let passes = stack.enabled_passes();
            with_gpu(frame, |gpu, d, q| {
                gpu.reapply_smart_filters(d, q, id, &passes);
            });
        }
        self.force_composite = true;
    }

    /// Add a smart filter of `kind` to the active layer's stack and re-apply.
    /// Snapshots the layer's current pixels as the source on the first filter.
    pub(crate) fn add_smart_filter(&mut self, frame: &mut eframe::Frame, kind: SmartFilterKind) {
        let id = self.active_id();
        // Only raster layers carry pixels to filter non-destructively.
        if !matches!(
            self.doc.layers.get(id).map(|l| &l.kind),
            Some(LayerKind::Raster)
        ) {
            return;
        }
        self.smart_filters.entry(id).or_default().add(kind);
        self.reapply_active_smart_filters(frame);
    }

    /// Remove the smart filter at `index` on the active layer and re-apply.
    pub(crate) fn remove_smart_filter(&mut self, frame: &mut eframe::Frame, index: usize) {
        let id = self.active_id();
        if let Some(s) = self.smart_filters.get_mut(&id) {
            s.remove(index);
        }
        self.reapply_active_smart_filters(frame);
    }

    /// Toggle the enabled flag of the smart filter at `index` and re-apply.
    pub(crate) fn toggle_smart_filter(&mut self, frame: &mut eframe::Frame, index: usize) {
        let id = self.active_id();
        if let Some(s) = self.smart_filters.get_mut(&id) {
            s.toggle(index);
        }
        self.reapply_active_smart_filters(frame);
    }

    /// Move the smart filter at `index` one step earlier in the stack, re-apply.
    pub(crate) fn move_smart_filter_up(&mut self, frame: &mut eframe::Frame, index: usize) {
        let id = self.active_id();
        if let Some(s) = self.smart_filters.get_mut(&id) {
            s.move_up(index);
        }
        self.reapply_active_smart_filters(frame);
    }

    /// Move the smart filter at `index` one step later in the stack, re-apply.
    pub(crate) fn move_smart_filter_down(&mut self, frame: &mut eframe::Frame, index: usize) {
        let id = self.active_id();
        if let Some(s) = self.smart_filters.get_mut(&id) {
            s.move_down(index);
        }
        self.reapply_active_smart_filters(frame);
    }

    /// Replace the kind (edited params) of the smart filter at `index`, re-apply.
    pub(crate) fn edit_smart_filter(
        &mut self,
        frame: &mut eframe::Frame,
        index: usize,
        kind: SmartFilterKind,
    ) {
        let id = self.active_id();
        if let Some(f) = self
            .smart_filters
            .get_mut(&id)
            .and_then(|s| s.filters.get_mut(index))
        {
            f.kind = kind;
        }
        self.reapply_active_smart_filters(frame);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blur(r: f32) -> SmartFilterKind {
        SmartFilterKind::GaussianBlur { radius: r }
    }

    // Add appends in order; the stack reports the kinds bottom-to-top.
    #[test]
    fn add_appends_in_order() {
        let mut s = SmartFilterStack::default();
        assert!(s.is_empty());
        s.add(blur(2.0));
        s.add(SmartFilterKind::Sharpen { amount: 1.0 });
        s.add(SmartFilterKind::Posterize { levels: 4 });
        assert_eq!(s.filters.len(), 3);
        assert_eq!(s.filters[0].kind, blur(2.0));
        assert_eq!(s.filters[1].kind, SmartFilterKind::Sharpen { amount: 1.0 });
        assert_eq!(s.filters[2].kind, SmartFilterKind::Posterize { levels: 4 });
    }

    // Remove drops the entry at the index and shifts the rest down.
    #[test]
    fn remove_drops_entry() {
        let mut s = SmartFilterStack::default();
        s.add(blur(1.0));
        s.add(SmartFilterKind::Sharpen { amount: 2.0 });
        s.add(blur(3.0));
        s.remove(1);
        assert_eq!(s.filters.len(), 2);
        assert_eq!(s.filters[0].kind, blur(1.0));
        assert_eq!(s.filters[1].kind, blur(3.0));
        // Out-of-range remove is a no-op.
        s.remove(9);
        assert_eq!(s.filters.len(), 2);
    }

    // Toggle flips only the targeted entry's enabled flag, and disabled entries
    // are excluded from the apply order while staying in the stack.
    #[test]
    fn toggle_excludes_from_apply_order_but_keeps_entry() {
        let mut s = SmartFilterStack::default();
        s.add(blur(2.0)); // kind 1, enabled
        s.add(SmartFilterKind::Posterize { levels: 4 }); // kind 28
        assert_eq!(s.enabled_passes().len(), 2);
        s.toggle(0);
        assert!(!s.filters[0].enabled);
        assert!(s.filters[1].enabled);
        // Only the posterize pass remains; the entry itself is still present.
        let passes = s.enabled_passes();
        assert_eq!(passes.len(), 1);
        assert_eq!(passes[0].0, 28);
        assert_eq!(s.filters.len(), 2);
        // Toggling back restores it.
        s.toggle(0);
        assert_eq!(s.enabled_passes().len(), 2);
    }

    // move_up / move_down reorder the apply order; clamped at the ends.
    #[test]
    fn reorder_changes_apply_order() {
        let mut s = SmartFilterStack::default();
        s.add(blur(1.0)); // 0
        s.add(SmartFilterKind::Sharpen { amount: 1.0 }); // 1
        s.add(SmartFilterKind::Posterize { levels: 4 }); // 2
        s.move_up(2); // posterize -> middle
        assert_eq!(s.filters[1].kind, SmartFilterKind::Posterize { levels: 4 });
        assert_eq!(s.filters[2].kind, SmartFilterKind::Sharpen { amount: 1.0 });
        s.move_down(0); // blur -> middle
        assert_eq!(s.filters[0].kind, SmartFilterKind::Posterize { levels: 4 });
        assert_eq!(s.filters[1].kind, blur(1.0));
        // Clamped at the ends: no panic, no change.
        s.move_up(0);
        s.move_down(2);
        assert_eq!(s.filters.len(), 3);
        // The enabled-pass GPU kinds reflect the new order (posterize=28 first).
        let kinds: Vec<u32> = s.enabled_passes().iter().map(|p| p.0).collect();
        assert_eq!(kinds, vec![28, 1, 2]);
    }

    // to_meta / from_meta round-trip kinds, params, and enabled flags exactly.
    #[test]
    fn meta_round_trips() {
        let mut s = SmartFilterStack::default();
        s.add(blur(3.5));
        s.add(SmartFilterKind::Sharpen { amount: 2.25 });
        s.add(SmartFilterKind::Posterize { levels: 6 });
        s.toggle(1); // disable sharpen
        let meta = s.to_meta();
        assert_eq!(meta.len(), 3);
        let back = SmartFilterStack::from_meta(&meta);
        assert_eq!(back, s);
        assert!(!back.filters[1].enabled);
        match back.filters[0].kind {
            SmartFilterKind::GaussianBlur { radius } => assert!((radius - 3.5).abs() < 1e-6),
            _ => panic!("expected blur"),
        }
        match back.filters[2].kind {
            SmartFilterKind::Posterize { levels } => assert_eq!(levels, 6),
            _ => panic!("expected posterize"),
        }
    }

    // A document with no smart filters (legacy / no-stack) loads as an empty,
    // non-serialized stack — the additive-field contract.
    #[test]
    fn legacy_no_stack_loads_empty() {
        let empty: Vec<SmartFilterMeta> = Vec::new();
        let s = SmartFilterStack::from_meta(&empty);
        assert!(s.is_empty());
        assert!(s.to_meta().is_empty());
    }
}
