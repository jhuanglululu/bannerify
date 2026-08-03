//! Deliberately simple scalar lanczos-3 resize — the correctness oracle.
//!
//! Test-only. No SIMD, no streaming, no weight tables: two passes with a full
//! intermediate, `f64` accumulation, coefficients recomputed inline so a bug in
//! [`super::weights`] cannot hide here.

/// Lanczos-3 kernel.
fn lanczos3(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    if x <= -3.0 || x >= 3.0 {
        return 0.0;
    }
    let px = std::f64::consts::PI * x;
    3.0 * px.sin() * (px / 3.0).sin() / (px * px)
}

/// `(start, weights)` per output coordinate, Pillow convention.
fn coeffs(in_size: usize, out_size: usize) -> Vec<(usize, Vec<f64>)> {
    let scale = in_size as f64 / out_size as f64;
    let filter_scale = scale.max(1.0);
    let support = 3.0 * filter_scale;

    (0..out_size)
        .map(|out| {
            let center = (out as f64 + 0.5) * scale;
            let start = (center - support + 0.5).floor().max(0.0) as usize;
            let end = ((center + support + 0.5).floor() as usize).min(in_size);
            let mut w: Vec<f64> = (start..end)
                .map(|i| lanczos3((i as f64 - center + 0.5) / filter_scale))
                .collect();
            let sum: f64 = w.iter().sum();
            if sum != 0.0 {
                for x in &mut w {
                    *x /= sum;
                }
            }
            (start, w)
        })
        .collect()
}

/// Resize one `u8` plane to `dst_width × dst_height` planar `f32`.
pub fn resize_plane(
    plane: &[u8],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) -> Vec<f32> {
    let hx = coeffs(src_width, dst_width);
    let hy = coeffs(src_height, dst_height);

    // Horizontal pass into a full intermediate (dst_width x src_height).
    let mut mid = vec![0.0f64; dst_width * src_height];
    for y in 0..src_height {
        for (x, (start, w)) in hx.iter().enumerate() {
            let mut acc = 0.0;
            for (k, wk) in w.iter().enumerate() {
                acc += wk * f64::from(plane[y * src_width + start + k]);
            }
            mid[y * dst_width + x] = acc;
        }
    }

    // Vertical pass.
    let mut out = vec![0.0f32; dst_width * dst_height];
    for (y, (start, w)) in hy.iter().enumerate() {
        for x in 0..dst_width {
            let mut acc = 0.0;
            for (k, wk) in w.iter().enumerate() {
                acc += wk * mid[(start + k) * dst_width + x];
            }
            out[y * dst_width + x] = acc as f32;
        }
    }
    out
}
