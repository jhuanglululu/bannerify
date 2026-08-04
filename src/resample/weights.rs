//! Per-axis lanczos-3 weight tables.

use crate::simd::{AlignedVec, LANES};

/// Lanczos kernel radius.
const A: f64 = 3.0;

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

/// Lanczos-3 coefficients mapping the source interval `[in0, in1)` — in
/// fractional source-pixel coordinates — onto `out_size` output samples.
///
/// Convention: output `i` samples the input at `(i + 0.5) * scale - 0.5`, with
/// support `A` widened to `A * max(scale, 1)` when downscaling. The tap window
/// is clipped at the image edges and the surviving weights renormalised to sum
/// 1, so source indices never leave the image and no edge clamping is needed.
/// `ksize` is uniform across the axis — clipped taps are present with weight 0,
/// which keeps every inner loop remainder-free.
fn coeffs(in_size: usize, in0: f64, in1: f64, out_size: usize) -> Coeffs {
    assert!(in_size > 0 && out_size > 0, "resample: zero-sized axis");
    assert!(in1 > in0, "resample: empty source window");

    let scale = (in1 - in0) / out_size as f64;
    let filter_scale = scale.max(1.0);
    let support = A * filter_scale;
    let ksize = (support.ceil() as usize) * 2 + 1;

    let mut starts = Vec::with_capacity(out_size);
    let mut weights = vec![0.0f32; out_size * ksize];

    for out in 0..out_size {
        let center = in0 + (out as f64 + 0.5) * scale;
        let xmin = ((center - support + 0.5).floor().max(0.0) as usize).min(in_size - 1);
        let xmax = (((center + support + 0.5).floor().max(0.0) as usize).min(in_size))
            .saturating_sub(xmin);
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
/// Lane views only start at lane-aligned offsets, so each output pixel's window
/// starts at `start_lane * LANES` and the sub-lane offset is baked into the
/// table by shifting the weights inside a padded window of `win` floats. The
/// inner loop is then a pure aligned `mul_add` chain plus one `hsum`.
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
    pub fn new(in_size: usize, in0: f64, in1: f64, out_size: usize) -> Self {
        let c = coeffs(in_size, in0, in1, out_size);
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
/// No alignment shifting is needed — the V pass walks whole aligned rows.
pub struct VWeights {
    /// Taps per output row (uniform).
    pub ksize: usize,
    /// First source row of each output row's window.
    pub starts: Vec<usize>,
    /// `out_len * ksize` weights, zero where the window was clipped.
    pub weights: Vec<f32>,
}

impl VWeights {
    pub fn new(in_size: usize, in0: f64, in1: f64, out_size: usize) -> Self {
        let c = coeffs(in_size, in0, in1, out_size);
        Self {
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
