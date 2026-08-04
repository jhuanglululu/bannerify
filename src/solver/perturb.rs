//! Perturbation rounds: kick the refined solution out of its local minimum and
//! let the refinement machinery walk back down from somewhere else.
//!
//! New to the Rust build — the old Rust solver stopped after refinement; this
//! is Python's `solver.py` phase 3 (`fit_chunk`, the beam loop), with the
//! process the plan fixes (`context/plans/2-solver.md`):
//!
//! 1. keep the `TOP_N` best candidates found so far (initially just the
//!    refined solution);
//! 2. make `DUPLICATES` copies of each;
//! 3. every copy re-rolls 1–2 random layers to a random `(pattern, dye)`;
//! 4. every copy is re-refined by the same windowed beam — this is where the
//!    quality comes from, the random kick only chooses where to look;
//! 5. pool the copies with the incumbents, keep the best `TOP_N`;
//! 6. repeat `ROUNDS` times; the best of the final pool wins.
//!
//! Any of the three numbers being `0` disables the stage (the old sentinel,
//! handled in [`crate::cli::config`], so `perturbations` arrives as `None`).
//!
//! ## What a kick may touch (measured, 2026-08-04)
//!
//! Full `(pattern, dye)` on any layer, top included — Python's rule. A
//! restricted variant (pattern-only, never the top layer) was tried on the
//! theory that random dyes teach the layers above to paint over the kick;
//! an A/B across two images × three seeds (`tmp/p5/reroll-ab.txt`) measured
//! it consistently ~0.5% *worse* at equal wall time, so the unrestricted
//! rule stands. Recorded in `context/plans/2-solver.md`.
//!
//! ## Other differences from Python
//!
//! Python distributes a fixed *total* trial budget `P` across the beam
//! (`P // n_cands` each, remainder to the front); the plan specifies
//! `DUPLICATES` copies of **each** incumbent instead, so the trial count grows
//! with the beam rather than being divided by it. Python also re-uses the
//! parent's base colour for a trial; so do we — but our refinement
//! re-optimises the base anyway, so it is not sticky.
//!
//! ## Randomness
//!
//! No `rand` dependency: [`Rng`] is xorshift64\*, seeded by splitmix64 from
//! `(seed, column)`. **One stream per column work item**, created when the item
//! starts and advanced sequentially down the column's cells — the banner row
//! never enters the seeding, so a column's draws are the same however the wall
//! is cut into items, and a run is fully reproducible from `--seed`. The stream
//! is touched only by this module, so a run without `--perturbations` is
//! independent of the seed.

use crate::cli::config::RefinementConfig;
use crate::color::NUM_COLORS;

use super::refine;
use super::workspace::{Plane, Solution, Workspace};

/// A xorshift64\* stream: one per column work item.
///
/// Enough randomness for "pick a layer and a pattern" and nothing more —
/// this is a search heuristic, not a simulation. Deterministic and portable:
/// the state is `u64` arithmetic only, so every machine and every backend draws
/// the same sequence.
pub struct Rng(u64);

impl Rng {
    /// The stream for column `column` of a run seeded with `seed`.
    ///
    /// splitmix64 mixes the pair so that neighbouring columns (and seed 0,
    /// column 0) start far apart in the state space; its output is never zero
    /// for our inputs, but the `| 1` makes the xorshift fixed point
    /// unreachable by construction rather than by argument.
    pub fn new(seed: u64, column: u64) -> Self {
        let mut z = seed.wrapping_add(column.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Self((z ^ (z >> 31)) | 1)
    }

    /// The next 64 bits.
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
pub(super) fn rounds(
    ws: &mut Workspace,
    solution: &mut Solution,
    (top_n, duplicates, rounds): (usize, usize, usize),
    cfg: &RefinementConfig,
    alphas: &[Plane],
    n: usize,
    rng: &mut Rng,
) {
    if n == 0 || top_n == 0 || duplicates == 0 || rounds == 0 {
        return;
    }

    // The incumbent pool, best first. `solution.error` is the refined fit's
    // exact OKLab SSE; every trial is scored the same way, by the same
    // refinement.
    ws.pool.clear();
    ws.pool.push((solution.error, solution.clone()));

    for _ in 0..rounds {
        // (a)+(b)+(c): copies of every incumbent, each with 1-2 layers
        // re-rolled. Taken out of the workspace so the re-refine below can
        // borrow the workspace mutably; the buffer goes back at the end.
        let mut trials = std::mem::take(&mut ws.trials);
        trials.clear();
        for (_, cand) in &ws.pool {
            for _ in 0..duplicates {
                let mut trial = cand.clone();
                reroll(&mut trial.layers, alphas.len(), rng);
                trials.push(trial);
            }
        }

        // (d): every copy re-refined. This is the expensive part and the whole
        // point — the kick alone almost never improves anything.
        for trial in &mut trials {
            refine::refine(ws, trial, cfg, alphas, n);
        }

        // (e): pool and keep the best TOP_N.
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

/// Re-roll one or two distinct layers to a random `(pattern, dye)`. See the
/// module docs for why the rule is unrestricted.
fn reroll(layers: &mut [(usize, usize)], n_patterns: usize, rng: &mut Rng) {
    let n = layers.len();
    let count = if n > 1 { 1 + rng.below(2) } else { 1 };

    let first = rng.below(n);
    layers[first] = (rng.below(n_patterns), rng.below(NUM_COLORS));
    if count == 2 {
        // Draw from the other `n - 1` slots so the two are distinct, as
        // Python's `random.sample` guarantees.
        let mut second = rng.below(n - 1);
        if second >= first {
            second += 1;
        }
        layers[second] = (rng.below(n_patterns), rng.below(NUM_COLORS));
    }
}
