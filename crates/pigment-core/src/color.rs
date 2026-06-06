//! Color types. The working pipeline is linear-light premultiplied (PLAN.md §2);
//! these helpers make the sRGB <-> linear boundary explicit so we never blend in
//! the wrong space.

use serde::{Deserialize, Serialize};

/// A straight (non-premultiplied) RGBA color, channels in 0..=1.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rgba {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Rgba {
    pub const TRANSPARENT: Rgba = Rgba {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 0.0,
    };
    pub const BLACK: Rgba = Rgba {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
    pub const WHITE: Rgba = Rgba {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };

    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    /// Premultiply alpha into the color channels (linear-space operation).
    pub fn premultiplied(self) -> Rgba {
        Rgba::new(self.r * self.a, self.g * self.a, self.b * self.a, self.a)
    }
}

/// Decode a single sRGB-encoded channel (0..=1) to linear light.
pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Encode a single linear-light channel (0..=1) back to sRGB.
pub fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_linear_roundtrip() {
        for i in 0..=255 {
            let c = i as f32 / 255.0;
            let back = linear_to_srgb(srgb_to_linear(c));
            assert!((back - c).abs() < 1e-4, "roundtrip failed at {c}: {back}");
        }
    }

    #[test]
    fn srgb_endpoints() {
        assert!(srgb_to_linear(0.0).abs() < 1e-6);
        assert!((srgb_to_linear(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn premultiply_scales_rgb() {
        let p = Rgba::new(1.0, 0.5, 0.0, 0.5).premultiplied();
        assert!((p.r - 0.5).abs() < 1e-6 && (p.g - 0.25).abs() < 1e-6 && p.a == 0.5);
    }
}
