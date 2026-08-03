//! The variance pre-pass: how many layers each banner cell gets.
//!
//! Detail costs layers, and flat cells waste them, so the layer budget is
//! spread by how busy each cell's *source* pixels are. Ported from the old
//! Rust build's `solver/complexity.rs` (`sort_banner`), which in turn is the
//! Python build's `_banner_complexity`: per cell, the sum over channels of that
//! channel's variance `E[x²] − E[x]²`; then a global min/max normalisation maps
//! the busiest cell to `layer_range.1` layers and the flattest to
//! `layer_range.0`.
//!
//! Two deliberate departures from the old builds, both from
//! `context/designs/pipeline.md`:
//!
//! - It runs on the **source image**, over each cell's source window, not on
//!   the resampled wall. The pipeline never materialises the resampled wall —
//!   each row band is closure-local — and a pre-pass that needs one would
//!   defeat that. Source pixels are also the honest measure of how much detail
//!   the cell is being asked to stand in for.
//! - It runs **before** the row `par_iter` (it is trivially parallel over
//!   cells) so the whole grid is available to every row item at once, which is
//!   what the global min/max normalisation requires.

use rayon::prelude::*;

use crate::geometry::{BANNER_H, BANNER_W, PAD_TOP, VISIBLE_H, offset_column, offset_row};
use crate::layout::Layout;
use crate::resample::PlanarU8;

/// Per-cell layer budget, row-major, `layout.rows * layout.columns` entries.
pub struct LayerGrid {
    /// Cells per row.
    columns: usize,
    /// `n_layers` per cell.
    layers: Vec<usize>,
}

impl LayerGrid {
    /// Layers for banner cell `(row, col)`.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> usize {
        self.layers[row * self.columns + col]
    }

    /// The largest budget in the grid — how big a solver workspace has to be.
    pub fn max(&self) -> usize {
        self.layers.iter().copied().max().unwrap_or(0)
    }

    /// How many cells got each budget, from `range.0` to `range.1` inclusive.
    pub fn histogram(&self, range: (usize, usize)) -> Vec<usize> {
        let mut out = vec![0; range.1 - range.0 + 1];
        for &n in &self.layers {
            out[n - range.0] += 1;
        }
        out
    }
}

/// Compute the layer budget of every cell of the wall.
///
/// `range` is `(min, max)` layers from `--layer-range`.
pub fn layer_grid(source: &PlanarU8, layout: &Layout, range: (usize, usize)) -> LayerGrid {
    let cells = layout.rows * layout.columns;
    let variances: Vec<f32> = (0..cells)
        .into_par_iter()
        .map(|i| cell_variance(source, layout, i / layout.columns, i % layout.columns))
        .collect();

    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in &variances {
        lo = lo.min(v);
        hi = hi.max(v);
    }

    // A wall of uniformly busy cells (or a single cell) has nothing to spread,
    // so everything lands on the low end of the range — as in the old build.
    let span = (range.1 - range.0) as f32;
    let inv = if hi > lo { 1.0 / (hi - lo) } else { 0.0 };
    let layers = variances
        .into_iter()
        .map(|v| {
            let n = (range.0 as f32 + (v - lo) * span * inv).round() as usize;
            n.clamp(range.0, range.1)
        })
        .collect();

    LayerGrid {
        columns: layout.columns,
        layers,
    }
}

/// Sum of the per-channel variances of the source pixels behind one cell.
fn cell_variance(source: &PlanarU8, layout: &Layout, row: usize, col: usize) -> f32 {
    let (x0, x1) = (offset_column(col), offset_column(col) + BANNER_W);
    // The cell's *solved* rows: the whole banner for row 0, the visible bottom
    // 24 rows below it — the same split the solver uses.
    let (y0, y1) = if row == 0 {
        (PAD_TOP, PAD_TOP + BANNER_H)
    } else {
        (offset_row(row), offset_row(row) + VISIBLE_H)
    };

    let (sx0, sx1) = source_span(
        (x0, x1),
        layout.origin.0,
        layout.target_width,
        (layout.window.x0, layout.window.x1),
        source.width,
    );
    let (sy0, sy1) = source_span(
        (y0, y1),
        layout.origin.1,
        layout.target_height,
        (layout.window.y0, layout.window.y1),
        source.height,
    );

    let count = ((sx1 - sx0) * (sy1 - sy0)) as f32;
    let mut variance = 0.0;
    for plane in &source.planes {
        let (mut sum, mut sum2) = (0.0_f32, 0.0_f32);
        for y in sy0..sy1 {
            for &v in &plane[y * source.width + sx0..y * source.width + sx1] {
                let v = f32::from(v);
                sum += v;
                sum2 = v.mul_add(v, sum2);
            }
        }
        let mean = sum / count;
        variance += sum2 / count - mean * mean;
    }
    variance
}

/// Map a wall-pixel interval on one axis to the source-pixel interval it reads.
///
/// `origin` is where the resampled region sits in the wall (non-zero only under
/// `--fill`), `target` its size on this axis, and `window` the source region
/// feeding the resampler. The result is always at least one pixel wide and
/// inside the image, so a cell that is entirely `--fill` padding still gets a
/// (meaningless but finite) variance rather than a division by zero.
fn source_span(
    wall: (usize, usize),
    origin: usize,
    target: usize,
    window: (f64, f64),
    limit: usize,
) -> (usize, usize) {
    let t0 = wall.0.saturating_sub(origin).min(target) as f64;
    let t1 = wall.1.saturating_sub(origin).min(target) as f64;
    let scale = (window.1 - window.0) / target as f64;
    let s0 = (window.0 + t0 * scale).floor().max(0.0) as usize;
    let s1 = (window.0 + t1 * scale).ceil() as usize;
    let s0 = s0.min(limit - 1);
    (s0, s1.clamp(s0 + 1, limit))
}
