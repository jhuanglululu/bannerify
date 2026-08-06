//! Perturbation rounds: kick the refined solution out of its local minimum and
//! let the refinement machinery walk back down from somewhere else.
//!
//! Randomness comes from [`Rng`], one xorshift64\* stream per column work item,
//! seeded from `(seed, column)`. The banner row never enters the seeding, so a
//! column's draws are the same however the wall is cut into items, and a run is
//! reproducible from `--seed`. Only this module touches the stream, so a run
//! without `--perturbations` is independent of the seed.

use crate::cli::config::RefinementConfig;
use crate::color::NUM_COLORS;

use super::refine;
use super::workspace::{Plane, Solution, Workspace};

/// A xorshift64\* stream: one per column work item. The state is `u64`
/// arithmetic only, so every machine and backend draws the same sequence.
pub struct Rng(u64);

impl Rng {
    /// The stream for column `column` of a run seeded with `seed`.
    ///
    /// splitmix64 mixes the pair so that neighbouring columns (and seed 0,
    /// column 0) start far apart in the state space; the `| 1` makes the
    /// xorshift fixed point unreachable by construction.
    pub fn new(seed: u64, column: u64) -> Self {
        let mut z = seed.wrapping_add(column.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Self((z ^ (z >> 31)) | 1)
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A value in `0..n` (Lemire's multiply-shift; `n == 0` yields `0`).
    #[inline]
    fn below(&mut self, n: usize) -> usize {
        ((u128::from(self.next_u64()) * n as u128) >> 64) as usize
    }
}

/// Run the perturbation rounds, leaving `solution` the best found and the
/// prefix chain consistent with it.
// Everything here is forwarded to `refine`, which needs all of it; bundling the
// arguments into a struct would only move the same list one level up.
#[allow(clippy::too_many_arguments)]
pub(super) fn rounds(
    ws: &mut Workspace,
    solution: &mut Solution,
    (top_n, duplicates, rounds): (usize, usize, usize),
    cfg: &RefinementConfig,
    lambda: f32,
    alphas: &[Plane],
    n: usize,
    rng: &mut Rng,
) {
    if n == 0 || top_n == 0 || duplicates == 0 || rounds == 0 {
        return;
    }

    // The incumbent pool, best first. Every trial is scored the same way the
    // incumbent was, by the same refinement.
    ws.pool.clear();
    ws.pool.push((solution.error, solution.clone()));

    for _ in 0..rounds {
        // Taken out of the workspace so the re-refine below can borrow the
        // workspace mutably; the buffer goes back at the end.
        let mut trials = std::mem::take(&mut ws.trials);
        trials.clear();
        for (_, cand) in &ws.pool {
            for _ in 0..duplicates {
                let mut trial = cand.clone();
                reroll(&mut trial.layers, alphas.len(), rng);
                trials.push(trial);
            }
        }

        // The re-refine is the whole point — the kick alone almost never
        // improves anything.
        for trial in &mut trials {
            refine::refine(ws, trial, cfg, lambda, alphas, n);
        }

        for trial in trials.drain(..) {
            ws.pool.push((trial.error, trial));
        }
        ws.trials = trials;
        ws.pool
            .sort_by(|a, b| a.0.partial_cmp(&b.0).expect("no NaN errors"));
        ws.pool.truncate(top_n);
    }

    let (error, best) = ws.pool.swap_remove(0);
    *solution = best;
    solution.error = error;
    // The last thing the loop refined was some other trial, so the prefix chain
    // belongs to it, not to the winner.
    super::workspace::rebuild_prefixes(&mut ws.prefixes, ws.off, solution, alphas, n);
}

/// Re-roll one or two distinct layers to a random `(pattern, dye)`.
fn reroll(layers: &mut [(usize, usize)], n_patterns: usize, rng: &mut Rng) {
    let n = layers.len();
    let count = if n > 1 { 1 + rng.below(2) } else { 1 };

    let first = rng.below(n);
    layers[first] = (rng.below(n_patterns), rng.below(NUM_COLORS));
    if count == 2 {
        // Draw from the other `n - 1` slots so the two are distinct.
        let mut second = rng.below(n - 1);
        if second >= first {
            second += 1;
        }
        layers[second] = (rng.below(n_patterns), rng.below(NUM_COLORS));
    }
}
