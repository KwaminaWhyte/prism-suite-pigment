//! Blend modes. Numeric discriminants are stable — they are written to the
//! `.pigment` file and passed straight to the compositor shader as a uniform
//! (PLAN.md §2, RESEARCH.md §2). Do not renumber existing variants.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u32)]
#[derive(Default)]
pub enum BlendMode {
    #[default]
    Normal = 0,
    Multiply = 1,
    Screen = 2,
    Overlay = 3,
    Darken = 4,
    Lighten = 5,
    ColorDodge = 6,
    ColorBurn = 7,
    HardLight = 8,
    SoftLight = 9,
    Difference = 10,
    Exclusion = 11,
    LinearDodge = 12, // "Add"
    LinearBurn = 13,
    // Non-separable HSL modes land in Phase 3.
    Hue = 20,
    Saturation = 21,
    Color = 22,
    Luminosity = 23,
}

impl BlendMode {
    /// Raw value handed to the compositor shader's `blend_mode` uniform.
    pub fn shader_id(self) -> u32 {
        self as u32
    }

    /// Inverse of [`shader_id`]; unknown ids fall back to `Normal`.
    pub fn from_shader_id(id: u32) -> BlendMode {
        match id {
            1 => BlendMode::Multiply,
            2 => BlendMode::Screen,
            3 => BlendMode::Overlay,
            4 => BlendMode::Darken,
            5 => BlendMode::Lighten,
            6 => BlendMode::ColorDodge,
            7 => BlendMode::ColorBurn,
            8 => BlendMode::HardLight,
            9 => BlendMode::SoftLight,
            10 => BlendMode::Difference,
            11 => BlendMode::Exclusion,
            12 => BlendMode::LinearDodge,
            13 => BlendMode::LinearBurn,
            20 => BlendMode::Hue,
            21 => BlendMode::Saturation,
            22 => BlendMode::Color,
            23 => BlendMode::Luminosity,
            _ => BlendMode::Normal,
        }
    }

    /// True if expressible by fixed-function `wgpu::BlendState` (no backdrop
    /// read needed) — the fast path. Everything else runs the switch shader.
    pub fn is_fixed_function(self) -> bool {
        matches!(
            self,
            BlendMode::Normal | BlendMode::Multiply | BlendMode::LinearDodge
        )
    }

    /// Every blend mode, in menu order (separable then HSL).
    pub const ALL: [BlendMode; 18] = [
        BlendMode::Normal,
        BlendMode::Multiply,
        BlendMode::Screen,
        BlendMode::Overlay,
        BlendMode::Darken,
        BlendMode::Lighten,
        BlendMode::ColorDodge,
        BlendMode::ColorBurn,
        BlendMode::HardLight,
        BlendMode::SoftLight,
        BlendMode::Difference,
        BlendMode::Exclusion,
        BlendMode::LinearDodge,
        BlendMode::LinearBurn,
        BlendMode::Hue,
        BlendMode::Saturation,
        BlendMode::Color,
        BlendMode::Luminosity,
    ];

    pub const ALL_SEPARABLE: [BlendMode; 14] = [
        BlendMode::Normal,
        BlendMode::Multiply,
        BlendMode::Screen,
        BlendMode::Overlay,
        BlendMode::Darken,
        BlendMode::Lighten,
        BlendMode::ColorDodge,
        BlendMode::ColorBurn,
        BlendMode::HardLight,
        BlendMode::SoftLight,
        BlendMode::Difference,
        BlendMode::Exclusion,
        BlendMode::LinearDodge,
        BlendMode::LinearBurn,
    ];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_ids_are_stable() {
        // These are serialized to disk + passed to shaders. Must not drift.
        assert_eq!(BlendMode::Normal.shader_id(), 0);
        assert_eq!(BlendMode::Multiply.shader_id(), 1);
        assert_eq!(BlendMode::LinearDodge.shader_id(), 12);
        assert_eq!(BlendMode::Luminosity.shader_id(), 23);
    }

    #[test]
    fn fixed_function_set() {
        assert!(BlendMode::Normal.is_fixed_function());
        assert!(!BlendMode::Overlay.is_fixed_function());
    }
}
