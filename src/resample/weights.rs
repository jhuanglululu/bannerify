//! Per-axis lanczos-3 weight tables.
//!
//! Built once per axis and shared by every row (or column) of that axis. The
//! coefficient convention matches Pillow's `precompute_coeffs`:
//!
//! - box-centre mapping: output coordinate `i` samples the input at
//!   `(i + 0.5) * scale - 0.5`,
//! - kernel support `a = 3`, widened to `a * max(scale, 1)` when downscaling,
//! - the tap window is *clipped* at the image edges and the surviving weights
//!   are renormalised to sum 1 (so source indices never leave the image; no
//!   edge clamping is needed).
//!
//! The number of taps (`ksize`) is uniform across an axis; clipped taps are
//! present with weight 0, which keeps every inner loop remainder-free.

use crate::simd::{AlignedVec, LANES};

/// Lanczos kernel radius.
const A: f64 = 3.0;

/// Lanczos-3 kernel, `a = 3`.
fn lanczos3(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    if x <= -A || x >= A {
        return 0.0;
    }
    let px = std::f64::consts::PI * x;
    A * px.sin() * (px / A).sin() / (px * px)
}

/// Round `v` up to the next multiple of `m`.
pub(crate) fn round_up(v: usize, m: usize) -> usize {
    v.div_ceil(m) * m
}

/// Raw (unpadded, scalar) coefficients for one axis.
struct Coeffs {
    /// Taps per output coordinate (uniform).
    ksize: usize,
    /// First source index of each output coordinate's window.
    starts: Vec<usize>,
    /// `out_len * ksize` weights, zero-padded where the window was clipped.
    weights: Vec<f32>,
}

/// Compute the lanczos-3 coefficients mapping `in_size` samples to `out_size`.
fn coeffs(in_size: usize, out_size: usize) -> Coeffs {
    assert!(in_size > 0 && out_size > 0, "resample: zero-sized axis");

    let scale = in_size as f64 / out_size as f64;
    let filter_scale = scale.max(1.0);
    let support = A * filter_scale;
    let ksize = (support.ceil() as usize) * 2 + 1;

    let mut starts = Vec::with_capacity(out_size);
    let mut weights = vec![0.0f32; out_size * ksize];

    for out in 0..out_size {
        let center = (out as f64 + 0.5) * scale;
        let xmin = ((center - support + 0.5).floor().max(0.0)) as usize;
        let xmax = (((center + support + 0.5).floor() as usize) - xmin).min(in_size - xmin);
        let xmax = xmax.min(ksize).max(1);

        let mut sum = 0.0f64;
        let mut row = vec![0.0f64; xmax];
        for (k, w) in row.iter_mut().enumerate() {
            *w = lanczos3(((k + xmin) as f64 - center + 0.5) / filter_scale);
            sum += *w;
        }
        let norm = if sum != 0.0 { 1.0 / sum } else { 1.0 };
        for (k, w) in row.iter().enumerate() {
            weights[out * ksize + k] = (w * norm) as f32;
        }
        starts.push(xmin);
    }

    Coeffs {
        ksize,
        starts,
        weights,
    }
}

/// Horizontal weights, laid out for aligned lane loads.
///
/// The facade only hands out lane views at lane-aligned offsets, so each output
/// pixel's window starts at `start_lane * LANES` and the sub-lane offset
/// (`start - start_lane * LANES`) is baked into the table by shifting the
/// weights inside a padded window of `win` floats. The inner loop is then a
/// pure aligned `mul_add` chain plus one `hsum`.
pub struct HWeights {
    /// Number of output pixels.
    pub out_len: usize,
    /// Padded window, in floats (a multiple of `LANES`).
    pub win: usize,
    /// Padded window, in lanes.
    pub win_lanes: usize,
    /// Lane index of each output pixel's (aligned) window start.
    pub start_lane: Vec<usize>,
    /// `out_len * win` weights; the padding entries are 0.
    pub table: AlignedVec,
}

impl HWeights {
    /// Build the table for `in_size` → `out_size`.
    pub fn new(in_size: usize, out_size: usize) -> Self {
        let c = coeffs(in_size, out_size);
        let win = round_up(c.ksize + LANES - 1, LANES);
        // `zeroed`, not `new_uninit`: the padding entries must read as 0 and the
        // table is small (out_len * win floats), not an image-sized buffer.
        let mut table = AlignedVec::zeroed(round_up(out_size * win, 16));
        let mut start_lane = Vec::with_capacity(out_size);

        for out in 0..out_size {
            let start = c.starts[out];
            // Aligned window start plus the sub-lane offset baked into the table
            // (written this way, not `start % LANES`, so the scalar backend's
            // `LANES == 1` does not trip clippy's modulo-one lint).
            let lane = start / LANES;
            let off = start - lane * LANES;
            start_lane.push(lane);
            for k in 0..c.ksize {
                table[out * win + off + k] = c.weights[out * c.ksize + k];
            }
        }

        Self {
            out_len: out_size,
            win,
            win_lanes: win / LANES,
            start_lane,
            table,
        }
    }

    /// Minimum padded length of a source row buffer this table may read from.
    pub fn src_padded_len(&self, in_size: usize) -> usize {
        round_up(in_size + self.win + LANES, 16)
    }
}

/// Vertical weights: plain scalars, broadcast with `F32s::splat` in the V pass.
///
/// No alignment shifting here — the V pass walks whole rows, which are aligned
/// by construction, so the taps enter as splatted scalars and nothing is wasted.
pub struct VWeights {
    /// Number of output rows.
    pub out_len: usize,
    /// Taps per output row (uniform).
    pub ksize: usize,
    /// First source row of each output row's window.
    pub starts: Vec<usize>,
    /// `out_len * ksize` weights, zero where the window was clipped.
    pub weights: Vec<f32>,
}

impl VWeights {
    /// Build the table for `in_size` → `out_size`.
    pub fn new(in_size: usize, out_size: usize) -> Self {
        let c = coeffs(in_size, out_size);
        Self {
            out_len: out_size,
            ksize: c.ksize,
            starts: c.starts,
            weights: c.weights,
        }
    }

    /// `(first source row, taps)` for output row `out`.
    #[inline]
    pub fn row(&self, out: usize) -> (usize, &[f32]) {
        (
            self.starts[out],
            &self.weights[out * self.ksize..(out + 1) * self.ksize],
        )
    }
}
