//! Perturbation rounds: kick the refined solution out of its local minimum and
//! let the refinement machinery walk back down from somewhere else.
//!
//! Randomness comes from [`Rng`], one xorshift64\* stream per block column,
//! seeded from `(seed, column)` and advanced down the column's cells. The
//! banner row never enters the seeding, so a column's draws are the same
//! however the wall is cut into batches ([`skip`] positions each cell's
//! stream before it is solved), and a run is reproducible from `--seed`. Only
//! this module touches the stream, so a run without `--perturbations` is
//! independent of the seed.

use crate::cli::config::RefinementConfig;
use crate::color::NUM_COLORS;

use super::refine;
use super::workspace::{Plane, Solution, SolveCfg, Workspace};

/// A xorshift64\* stream: one per block column. The state is `u64`
/// arithmetic only, so every machine and backend draws the same sequence.
#[derive(Clone)]
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
///
/// [`skip`] replays this function's draws without solving; a change to how
/// many draws a round makes has to be made there too.
// Everything here is forwarded to `refine`, which needs all of it; bundling the
// arguments into a struct would only move the same list one level up.
#[allow(clippy::too_many_arguments)]
pub(super) fn rounds(
    ws: &mut Workspace,
    solution: &mut Solution,
    (top_n, duplicates, rounds): (usize, usize, usize),
    cfg: &RefinementConfig,
    lambda: f32,
    nu: f32,
    mu: f32,
    alphas: &[Plane],
    n: usize,
    rng: &mut Rng,
) {
    if draws_nothing(n, (top_n, duplicates, rounds)) {
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
            refine::refine(ws, trial, cfg, lambda, nu, mu, alphas, n);
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

/// Whether [`rounds`] returns before its first draw for a cell of `n` layers.
fn draws_nothing(n: usize, (top_n, duplicates, rounds): (usize, usize, usize)) -> bool {
    n == 0 || top_n == 0 || duplicates == 0 || rounds == 0
}

/// Advance `rng` past every draw [`rounds`] makes for a cell of `n_layers`
/// layers, without solving the cell.
///
/// This works because the number and order of draws never depend on solver
/// results: a round makes one [`reroll`] per (pool entry, duplicate), and the
/// pool size after a round is `min(top_n, pool + trials)` whatever the trials
/// scored. The draws themselves go through [`reroll`] on a scratch layer list,
/// so the replay and the real rounds share one draw sequence and cannot drift
/// apart. Lets the pipeline position each cell's stream before solving, so a
/// cell's result does not depend on which batch it lands in.
pub fn skip(rng: &mut Rng, cfg: &SolveCfg<'_>, n_layers: usize) {
    // `Workspace::solve` runs the rounds only when perturbation is enabled.
    let Some(perturbations @ (top_n, duplicates, rounds)) = cfg.perturbations else {
        return;
    };
    if draws_nothing(n_layers, perturbations) {
        return;
    }
    let n_patterns = cfg.patterns.top.len();
    let mut layers = vec![(0, 0); n_layers];
    // The pool starts as the incumbent alone.
    let mut pool = 1;
    for _ in 0..rounds {
        let trials = pool * duplicates;
        for _ in 0..trials {
            reroll(&mut layers, n_patterns, rng);
        }
        pool = top_n.min(pool + trials);
    }
}

/// Re-roll one or two distinct layers to a random `(pattern, dye)`.
pub(crate) fn reroll(layers: &mut [(usize, usize)], n_patterns: usize, rng: &mut Rng) {
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

#[cfg(test)]
mod tests {

    use super::*;
    use crate::geometry::TOP_HW;
    use crate::pattern::load;

    /// [`skip`] must leave the stream exactly where a real solve leaves it, or
    /// every cell after the first in a column is solved with the wrong draws.
    /// The configurations cover the pool hitting its `top_n` cap, several
    /// rounds, a single layer (no count draw in [`reroll`]), no layers and no
    /// perturbation at all (no draws), on both patch sizes.
    #[test]
    fn skip_leaves_the_stream_where_a_solve_leaves_it() {
        let patterns = load(&crate::pattern::ids());
        let refinement = RefinementConfig {
            refinement_pass: 1,
            window_size: 2,
            error_threshold: 0.7,
            refinement_candidate: 3,
            exact_candidates: 10,
        };
        let cases = [
            (Some((2, 2, 1)), 4),
            (Some((3, 2, 3)), 5),
            (Some((1, 3, 2)), 1),
            (Some((4, 1, 3)), 0),
            (Some((0, 2, 2)), 4),
            (None, 4),
        ];

        for (perturbations, n_layers) in cases {
            for row in [0, 1] {
                let cfg = SolveCfg {
                    patterns: &patterns,
                    refinement: &refinement,
                    perturbations,
                    feature_weight: 0.0,
                    purity_weight: 0.0,
                    gradient_weight: 0.0,
                };
                let mut ws = Workspace::new(5, refinement.refinement_candidate);
                ws.begin(row);
                for (ch, plane) in ws.target_mut().iter_mut().enumerate() {
                    for i in 0..TOP_HW {
                        plane[i] = ((i * (ch + 3)) % 251) as f32;
                    }
                }

                let mut solved = Rng::new(7, 3);
                ws.solve(n_layers, &cfg, &mut solved);
                let mut skipped = Rng::new(7, 3);
                skip(&mut skipped, &cfg, n_layers);

                assert_eq!(
                    solved.next_u64(),
                    skipped.next_u64(),
                    "{perturbations:?}, {n_layers} layers, row {row}"
                );
            }
        }
    }
}
