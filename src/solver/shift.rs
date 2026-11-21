//! The per-cell shift (`--shift-range X Y`): each banner may solve against its
//! target patch displaced by up to `X` wall pixels sideways and `Y` up or down,
//! while it is still drawn at its real position.
//!
//! The pattern shapes are fixed, so an image edge that falls a pixel away from
//! the nearest pattern edge costs a strip of full-contrast error, and the
//! solver answers with a semi-transparent mix that averages out instead of a
//! clean banner. Displacing the patch lets the image edge meet the pattern
//! edge.
//!
//! ## Choosing the shift
//!
//! Once per cell, before the solve, and only from the feature-map search
//! ([`super::feature`]): every shifted patch is snapped and searched, and the
//! shift whose best banner of at most `--feature-layers` layers scores highest
//! wins. Scores are comparable across shifts because every patch has the same
//! pixel count. The search is cheap next to a solve, and its score measures
//! exactly what the shift is for: how much of the patch a clean banner can
//! cover. Its cost is paid once per shift tried, so it grows with the shift
//! range and with the layer count.
//!
//! Zero shift is scored first and the others follow in order of increasing
//! distance ([`order`]). A later shift has to score strictly higher, so a tie
//! keeps the shift closer to zero and a flat cell stays where it is. The best
//! score so far is the floor of every later search, so the search's own
//! pruning cuts most of them short.
//!
//! ## Limits
//!
//! `X ≤ 2` and `Y ≤ 4` ([`MAX_SHIFT`]) are the banner's margins inside its
//! block, so a shifted patch never leaves the block column it is gathered
//! from.

use crate::geometry::{PAD_SIDE, PAD_TOP};
use crate::simd::Chunk;

use super::cell::{BandView, patch_rows};
use super::feature;
use super::workspace::Plane;

/// The largest `(X, Y)` `--shift-range` accepts.
pub const MAX_SHIFT: (usize, usize) = (PAD_SIDE, PAD_TOP);

/// A cell whose chosen shift is not zero.
pub struct Shifted {
    /// The chosen `(dx, dy)`, in wall pixels.
    pub shift: (isize, isize),
    /// The patch at zero shift, which the reported error is measured against.
    pub unshifted: Box<[Plane; 3]>,
}

/// Picks each cell's shift and gathers its patch at it. One per worker; the
/// buffers are reused from cell to cell.
pub struct ShiftSearch {
    /// Every non-zero shift of the range, in the order they are tried.
    shifts: Vec<(isize, isize)>,
    /// The feature map's layer count, `--feature-layers`.
    layers: usize,
    /// Scratch for the feature-map search.
    feat: feature::Buffers,
    /// The shifted patch being scored.
    cand: [Plane; 3],
}

impl ShiftSearch {
    /// For `range = (X, Y)`, each at most [`MAX_SHIFT`], scoring patches by
    /// their best banner of at most `layers` layers
    /// ([`feature::LAYER_COUNTS`]). `(0, 0)` makes [`ShiftSearch::gather`] a
    /// plain [`BandView::gather`].
    pub fn new(range: (usize, usize), layers: usize) -> Self {
        Self {
            shifts: order(range),
            layers,
            feat: feature::Buffers::default(),
            cand: [Chunk::zeroed(); 3],
        }
    }

    /// Gather cell `(row, col)`'s patch into `target` at the shift whose
    /// feature-map score is highest, `alphas` being the pattern planes. Returns
    /// the shift and the unshifted patch when the shift is not zero, `None`
    /// when it is (then `target` holds the unshifted patch).
    pub fn gather(
        &mut self,
        view: &BandView<'_>,
        row: usize,
        col: usize,
        alphas: &[Plane],
        target: &mut [Plane; 3],
    ) -> Option<Shifted> {
        view.gather(row, col, target);
        if self.shifts.is_empty() {
            return None;
        }

        // A lower row solves only the tail of its planes; the search scores
        // exactly that view, as the solve will.
        let start = patch_rows(row).2;
        let layers = self.layers;
        let mut best = feature::best_score(
            &mut self.feat,
            target,
            alphas,
            start,
            layers,
            f32::NEG_INFINITY,
        )
        .expect("every banner beats -inf");
        let mut chosen: Option<Shifted> = None;
        for &shift in &self.shifts {
            view.gather_at(row, col, shift, &mut self.cand);
            let Some(score) =
                feature::best_score(&mut self.feat, &self.cand, alphas, start, layers, best)
            else {
                continue;
            };
            best = score;
            match &mut chosen {
                Some(c) => c.shift = shift,
                None => {
                    chosen = Some(Shifted {
                        shift,
                        unshifted: Box::new(*target),
                    })
                }
            }
            std::mem::swap(target, &mut self.cand);
        }
        chosen
    }
}

/// Every non-zero `(dx, dy)` with `|dx| ≤ X`, `|dy| ≤ Y`, by increasing
/// Euclidean distance; among equal distances by `dy`, then `dx`, ascending.
fn order((x, y): (usize, usize)) -> Vec<(isize, isize)> {
    let (x, y) = (x as isize, y as isize);
    let mut shifts: Vec<(isize, isize)> = (-y..=y)
        .flat_map(|dy| (-x..=x).map(move |dx| (dx, dy)))
        .filter(|&s| s != (0, 0))
        .collect();
    shifts.sort_by_key(|&(dx, dy)| (dx * dx + dy * dy, dy, dx));
    shifts
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::color::{COLOR_NAMES, COLORS_RGB};
    use crate::geometry::{BANNER_W, BLOCK_SIDE, wall_height};
    use crate::pattern::load;
    use crate::resample::{Plan, PlanarU8};

    fn dye(name: &str) -> [u8; 3] {
        COLORS_RGB[COLOR_NAMES.iter().position(|n| *n == name).expect("a dye")]
    }

    /// The shift chosen for every banner row of a one-column wall of `rows`
    /// banner rows whose pixels in wall column `x` are all `pixel(x)`, scored
    /// by the feature-map search at `layers` layers.
    fn chosen(
        rows: usize,
        range: (usize, usize),
        layers: usize,
        pixel: impl Fn(usize) -> [u8; 3],
    ) -> Vec<(isize, isize)> {
        let (w, h) = (BLOCK_SIDE, wall_height(rows));
        let data: Vec<u8> = (0..w * h).flat_map(|i| pixel(i % w)).collect();
        let image = PlanarU8::from_interleaved(&data, w, h, 3);
        // Same size in and out, so the resampled band is the image itself.
        let band = Plan::new(w, h, w, h).columns(0..w).resample(&image);
        let view = BandView::new(Some(&band), 0, 0, [0; 3]);

        let patterns = load(&HashSet::new());
        let mut search = ShiftSearch::new(range, layers);
        let mut target = [Chunk::zeroed(); 3];
        (0..rows)
            .map(|row| {
                search
                    .gather(&view, row, 0, &patterns.top, &mut target)
                    .map_or((0, 0), |s| s.shift)
            })
            .collect()
    }

    /// A red banner with a white `stripe_right`, composited in sRGB as the game
    /// does and drawn one pixel right of where the banner hangs: wall column
    /// `x` shows banner column `x - 3` (the banner starts at wall column 2).
    /// The stripe does not change down the banner, so one row of its alpha
    /// stands for all of them.
    ///
    /// At `dx = 1` the patch is that banner exactly, on the top row's full
    /// patch and a lower row's visible tail alike. Every `dy` scores the same,
    /// so the tie goes to `dy = 0`.
    #[test]
    fn a_banner_one_pixel_off_its_pattern_edge_picks_the_matching_shift() {
        let patterns = load(&HashSet::new());
        let stripe = &patterns.top[patterns
            .names
            .iter()
            .position(|n| n == "stripe_right")
            .expect("a pattern")];
        let (red, white) = (dye("red"), dye("white"));
        let pixel = |x: usize| {
            let a = stripe[20 * BANNER_W + x.saturating_sub(3).min(BANNER_W - 1)];
            std::array::from_fn(|ch| {
                (f32::from(red[ch]) * (1.0 - a) + f32::from(white[ch]) * a).round() as u8
            })
        };
        for layers in feature::LAYER_COUNTS {
            assert_eq!(
                chosen(2, (2, 4), layers, pixel),
                [(1, 0), (1, 0)],
                "{layers} layers"
            );
        }
    }

    /// Every shift of a flat patch scores the same, and only a strictly higher
    /// score displaces zero shift.
    #[test]
    fn a_flat_cell_keeps_zero_shift() {
        for layers in feature::LAYER_COUNTS {
            assert_eq!(
                chosen(2, (2, 4), layers, |_| dye("cyan")),
                [(0, 0), (0, 0)],
                "{layers} layers"
            );
        }
    }
}
