//! Non-destructive adjustment descriptors. An adjustment layer reads the
//! composited backdrop below it and transforms it (PLAN.md §4 Phase 3). The
//! `(kind, params)` encoding is passed straight to the compositor shader.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Adjustment {
    BrightnessContrast {
        brightness: f32,
        contrast: f32,
    },
    Levels {
        in_black: f32,
        in_white: f32,
        gamma: f32,
    },
    HueSaturation {
        hue: f32,
        saturation: f32,
        lightness: f32,
    },
    Exposure {
        stops: f32,
    },
    Invert,
    Threshold {
        level: f32,
    },
    BlackWhite,
}

impl Adjustment {
    /// Shader kind id (stable; written to disk) + up to four float params.
    pub fn encode(&self) -> (u32, [f32; 4]) {
        match *self {
            Adjustment::BrightnessContrast {
                brightness,
                contrast,
            } => (1, [brightness, contrast, 0.0, 0.0]),
            Adjustment::Levels {
                in_black,
                in_white,
                gamma,
            } => (2, [in_black, in_white, gamma, 0.0]),
            Adjustment::HueSaturation {
                hue,
                saturation,
                lightness,
            } => (3, [hue, saturation, lightness, 0.0]),
            Adjustment::Invert => (4, [0.0; 4]),
            Adjustment::Exposure { stops } => (5, [stops, 0.0, 0.0, 0.0]),
            Adjustment::Threshold { level } => (6, [level, 0.0, 0.0, 0.0]),
            Adjustment::BlackWhite => (7, [0.0; 4]),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Adjustment::BrightnessContrast { .. } => "Brightness/Contrast",
            Adjustment::Levels { .. } => "Levels",
            Adjustment::HueSaturation { .. } => "Hue/Saturation",
            Adjustment::Exposure { .. } => "Exposure",
            Adjustment::Invert => "Invert",
            Adjustment::Threshold { .. } => "Threshold",
            Adjustment::BlackWhite => "Black & White",
        }
    }

    /// Sensible defaults for each kind (identity-ish where applicable).
    pub const DEFAULTS: [Adjustment; 7] = [
        Adjustment::BrightnessContrast {
            brightness: 0.0,
            contrast: 0.0,
        },
        Adjustment::Levels {
            in_black: 0.0,
            in_white: 1.0,
            gamma: 1.0,
        },
        Adjustment::HueSaturation {
            hue: 0.0,
            saturation: 0.0,
            lightness: 0.0,
        },
        Adjustment::Exposure { stops: 0.0 },
        Adjustment::Invert,
        Adjustment::Threshold { level: 0.5 },
        Adjustment::BlackWhite,
    ];
}
