//! Per-channel and luminance histograms for linear-light RGBA `f32` buffers.

/// Per-channel + luminance histograms.
pub struct Histogram {
    pub r: Vec<u32>,
    pub g: Vec<u32>,
    pub b: Vec<u32>,
    pub luma: Vec<u32>, // Rec.709 luma: 0.2126 R + 0.7152 G + 0.0722 B
}

impl Histogram {
    /// All-zero histograms with `bins` buckets each.
    fn zeros(bins: usize) -> Self {
        Histogram {
            r: vec![0; bins],
            g: vec![0; bins],
            b: vec![0; bins],
            luma: vec![0; bins],
        }
    }
}

#[inline]
fn bucket(v: f32, bins: usize) -> usize {
    let c = v.clamp(0.0, 1.0);
    // bins >= 2 guaranteed by caller.
    (c * (bins - 1) as f32) as usize
}

/// Compute histograms with `bins` buckets (e.g. 256).
///
/// A channel value `v` in `0..=1` maps to bucket `floor(clamp(v,0,1) * (bins-1))`.
/// All pixels are counted for every channel (alpha is ignored).
///
/// Guards: `bins` must be `>= 2`, and `rgba.len()` must be a multiple of 4;
/// otherwise all-zero histograms are returned.
pub fn histogram(rgba: &[f32], bins: usize) -> Histogram {
    if bins < 2 || rgba.len() % 4 != 0 {
        return Histogram::zeros(bins.max(0));
    }

    let mut h = Histogram::zeros(bins);

    for px in rgba.chunks_exact(4) {
        let (r, g, b) = (px[0], px[1], px[2]);
        h.r[bucket(r, bins)] += 1;
        h.g[bucket(g, bins)] += 1;
        h.b[bucket(b, bins)] += 1;
        let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        h.luma[bucket(luma, bins)] += 1;
    }

    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solid_mid_gray() {
        let bins = 256;
        let n = 100; // pixels
        // 0.5 linear: bucket = floor(0.5 * 255) = 127 for r/g/b.
        let mut rgba = Vec::with_capacity(n * 4);
        for _ in 0..n {
            rgba.extend_from_slice(&[0.5, 0.5, 0.5, 1.0]);
        }

        let h = histogram(&rgba, bins);

        let expected = (0.5_f32 * (bins - 1) as f32) as usize;
        assert_eq!(h.r[expected], n as u32);
        assert_eq!(h.g[expected], n as u32);
        assert_eq!(h.b[expected], n as u32);
        // luma of (0.5,0.5,0.5) == 0.5 -> same bucket.
        assert_eq!(h.luma[expected], n as u32);

        // Sum of each channel's buckets equals pixel count.
        assert_eq!(h.r.iter().sum::<u32>(), n as u32);
        assert_eq!(h.g.iter().sum::<u32>(), n as u32);
        assert_eq!(h.b.iter().sum::<u32>(), n as u32);
        assert_eq!(h.luma.iter().sum::<u32>(), n as u32);
    }

    #[test]
    fn hdr_values_clamped() {
        // Values above 1.0 must land in the last bucket, not panic.
        let rgba = [2.0, 1.5, 3.0, 1.0, -1.0, 0.0, 0.0, 1.0];
        let h = histogram(&rgba, 16);
        assert_eq!(h.r[15], 1); // 2.0 clamped to 1.0 -> last bucket
        assert_eq!(h.r[0], 1); // -1.0 clamped to 0.0 -> first bucket
        assert_eq!(h.r.iter().sum::<u32>(), 2);
    }

    #[test]
    fn length_mismatch_returns_zero() {
        let rgba = [0.5, 0.5, 0.5]; // not a multiple of 4
        let h = histogram(&rgba, 256);
        assert_eq!(h.r.len(), 256);
        assert_eq!(h.r.iter().sum::<u32>(), 0);
        assert_eq!(h.g.iter().sum::<u32>(), 0);
        assert_eq!(h.b.iter().sum::<u32>(), 0);
        assert_eq!(h.luma.iter().sum::<u32>(), 0);
    }

    #[test]
    fn bins_too_small_returns_zero() {
        let rgba = [0.5, 0.5, 0.5, 1.0];
        let h = histogram(&rgba, 1);
        assert_eq!(h.r.iter().sum::<u32>(), 0);
    }
}
