//! Tone-curve lookup tables for a Curves adjustment.
//!
//! Interpolation uses monotone cubic Hermite (Fritsch–Carlson) so a sorted set
//! of control points produces a smooth, overshoot-free curve. Outputs and the
//! sampled range are clamped to `[0, 1]`; outside the control-point x range the
//! curve extends flat at the first/last `y`.

/// Clean control points: sort by x, drop non-finite, and de-duplicate / discard
/// non-increasing x (keep first occurrence of each x).
fn clean_points(points: &[(f32, f32)]) -> Vec<(f32, f32)> {
    let mut pts: Vec<(f32, f32)> = points
        .iter()
        .copied()
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .collect();

    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut out: Vec<(f32, f32)> = Vec::with_capacity(pts.len());
    for (x, y) in pts {
        match out.last() {
            Some(&(lx, _)) if x <= lx => {} // duplicate or non-increasing x: skip
            _ => out.push((x, y)),
        }
    }
    out
}

/// Compute Fritsch–Carlson tangents for monotone cubic interpolation.
fn fc_tangents(xs: &[f32], ys: &[f32]) -> Vec<f32> {
    let n = xs.len();
    let mut m = vec![0.0f32; n];
    if n < 2 {
        return m;
    }

    // Secant slopes between consecutive points.
    let mut delta = vec![0.0f32; n - 1];
    for i in 0..n - 1 {
        let h = xs[i + 1] - xs[i];
        delta[i] = (ys[i + 1] - ys[i]) / h;
    }

    // Initial tangents.
    m[0] = delta[0];
    m[n - 1] = delta[n - 2];
    for i in 1..n - 1 {
        if delta[i - 1] * delta[i] <= 0.0 {
            m[i] = 0.0; // local extremum: flatten to preserve monotonicity
        } else {
            m[i] = (delta[i - 1] + delta[i]) / 2.0;
        }
    }

    // Fritsch–Carlson adjustment.
    for i in 0..n - 1 {
        if delta[i] == 0.0 {
            m[i] = 0.0;
            m[i + 1] = 0.0;
            continue;
        }
        let a = m[i] / delta[i];
        let b = m[i + 1] / delta[i];
        let s = a * a + b * b;
        if s > 9.0 {
            let t = 3.0 / s.sqrt();
            m[i] = t * a * delta[i];
            m[i + 1] = t * b * delta[i];
        }
    }

    m
}

/// Evaluate the monotone cubic Hermite spline at `x` given knots and tangents.
fn eval_hermite(xs: &[f32], ys: &[f32], m: &[f32], x: f32) -> f32 {
    let n = xs.len();
    // Flat extension outside the knot range.
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[n - 1] {
        return ys[n - 1];
    }

    // Find the segment [xs[i], xs[i+1]] containing x.
    let mut i = 0;
    while i + 1 < n && x > xs[i + 1] {
        i += 1;
    }

    let h = xs[i + 1] - xs[i];
    let t = (x - xs[i]) / h;
    let t2 = t * t;
    let t3 = t2 * t;

    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;

    h00 * ys[i] + h10 * h * m[i] + h01 * ys[i + 1] + h11 * h * m[i + 1]
}

/// Build a lookup table of `n` samples (`n >= 2`) for input `x` in `[0, 1]`
/// mapped through a smooth monotone-cubic curve defined by control `points`.
///
/// Points are sorted/cleaned; with fewer than 2 usable points the identity ramp
/// is returned. Output values are clamped to `[0, 1]`.
pub fn build_lut(points: &[(f32, f32)], n: usize) -> Vec<f32> {
    let n = n.max(2);
    let pts = clean_points(points);

    if pts.len() < 2 {
        // Identity ramp.
        return (0..n).map(|i| i as f32 / (n - 1) as f32).collect();
    }

    let xs: Vec<f32> = pts.iter().map(|p| p.0).collect();
    let ys: Vec<f32> = pts.iter().map(|p| p.1).collect();
    let m = fc_tangents(&xs, &ys);

    (0..n)
        .map(|i| {
            let x = i as f32 / (n - 1) as f32;
            eval_hermite(&xs, &ys, &m, x).clamp(0.0, 1.0)
        })
        .collect()
}

/// Sample a LUT (len `L`) at `x` in `[0, 1]` with linear interpolation between
/// entries. Out-of-range `x` is clamped to the table endpoints.
pub fn sample_lut(lut: &[f32], x: f32) -> f32 {
    if lut.is_empty() {
        return 0.0;
    }
    if lut.len() == 1 {
        return lut[0];
    }

    let x = x.clamp(0.0, 1.0);
    let last = lut.len() - 1;
    let pos = x * last as f32;
    let i = pos.floor() as usize;
    if i >= last {
        return lut[last];
    }
    let frac = pos - i as f32;
    lut[i] + (lut[i + 1] - lut[i]) * frac
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_points_yield_identity_lut() {
        let n = 256;
        let lut = build_lut(&[(0.0, 0.0), (1.0, 1.0)], n);
        assert_eq!(lut.len(), n);
        for (i, &v) in lut.iter().enumerate() {
            let expected = i as f32 / (n - 1) as f32;
            assert!(
                (v - expected).abs() < 1e-4,
                "lut[{i}] = {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn pulling_midpoint_down_lowers_middle() {
        let n = 256;
        let lut = build_lut(&[(0.0, 0.0), (0.5, 0.25), (1.0, 1.0)], n);
        let mid = sample_lut(&lut, 0.5);
        assert!(mid < 0.5, "midpoint should drop below 0.5, got {mid}");
        // Roughly matches the control point.
        assert!((mid - 0.25).abs() < 0.05, "midpoint = {mid}");
    }

    #[test]
    fn fewer_than_two_points_returns_identity() {
        let n = 32;
        for pts in [&[][..], &[(0.3, 0.7)][..]] {
            let lut = build_lut(pts, n);
            for (i, &v) in lut.iter().enumerate() {
                let expected = i as f32 / (n - 1) as f32;
                assert!((v - expected).abs() < 1e-6, "lut[{i}] = {v}");
            }
        }
    }

    #[test]
    fn duplicate_x_deduped_to_identity() {
        // Two points with the same x collapse to one usable point -> identity.
        let n = 16;
        let lut = build_lut(&[(0.5, 0.2), (0.5, 0.9)], n);
        for (i, &v) in lut.iter().enumerate() {
            let expected = i as f32 / (n - 1) as f32;
            assert!((v - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn sample_lut_endpoints_and_midpoint() {
        let lut = vec![0.0, 0.5, 1.0]; // L = 3
        assert!((sample_lut(&lut, 0.0) - 0.0).abs() < 1e-6);
        assert!((sample_lut(&lut, 1.0) - 1.0).abs() < 1e-6);
        assert!((sample_lut(&lut, 0.5) - 0.5).abs() < 1e-6);
        // Quarter point sits between lut[0] and lut[1]: x=0.25 -> pos=0.5 -> 0.25.
        assert!((sample_lut(&lut, 0.25) - 0.25).abs() < 1e-6);
        // Clamp out-of-range.
        assert!((sample_lut(&lut, -1.0) - 0.0).abs() < 1e-6);
        assert!((sample_lut(&lut, 2.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn outputs_stay_within_unit_range() {
        // An aggressive S-curve that could overshoot without clamping.
        let lut = build_lut(&[(0.0, 0.0), (0.25, 0.05), (0.75, 0.95), (1.0, 1.0)], 512);
        for &v in &lut {
            assert!((0.0..=1.0).contains(&v), "value out of range: {v}");
        }
    }

    #[test]
    fn flat_extension_outside_x_range() {
        // Points cover only [0.3, 0.7]; outside should clamp flat.
        let lut = build_lut(&[(0.3, 0.4), (0.7, 0.6)], 256);
        assert!((sample_lut(&lut, 0.0) - 0.4).abs() < 1e-3);
        assert!((sample_lut(&lut, 1.0) - 0.6).abs() < 1e-3);
    }
}
