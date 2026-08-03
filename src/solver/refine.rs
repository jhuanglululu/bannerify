//! Windowed beam refinement: walk the layer stack backwards and re-choose a
//! sliding window of layers at a time, scoring each candidate against the
//! *whole* banner rather than the partial composite the greedy fill saw.
//!
//! Ported from `../bannerify-old/src/solver/{refine.rs,build.rs}` onto the
//! [`simd`](crate::simd) facade.
//!
//! ## Why a suffix cache
//!
//! The greedy fill can only see the layers already laid: when it picks layer
//! `i` it does not know what layers `i+1..` will paint over it. Refinement
//! fixes that by caching, for each `i`, the affine map the layers *above* `i`
//! apply to whatever is underneath:
//!
//! ```text
//! S_i(x) = x · mul_i + add_i      where   S_n = identity  (mul = 1, add = 0)
//!          S_i = S_{i+1} ∘ (lay (p_i, c_i))
//!          mul_i = (1 − α_i) · mul_{i+1}
//!          add_i = c_i · α_i · mul_{i+1} + add_{i+1}
//! ```
//!
//! `mul` is a product of `1 − α` terms, so it is the same for all three
//! channels — one plane, not three. With prefix `P_i` (the composite below
//! layer `i`) the full banner for a trial `(p, c)` at layer `i` is
//! `P_i·(1−α)·mul + c·(α·mul) + add`, which is again *linear in `c`*: one pass
//! over the patch reduces `Σ res²`, `2·Σ res·(α·mul)` and `Σ (α·mul)²`, and the
//! 16 dyes are scored in closed form from those. Same trick as the greedy fill,
//! now with the rest of the banner accounted for.
//!
//! ## The walk
//!
//! `start_layer` runs from `n−1` down to `0`. Each window re-chooses layers
//! `start_layer, start_layer−1, …` (up to `window_size` of them) as a beam:
//! after each step the best `refinement_candidate` `(candidate, pattern, dye)`
//! triples survive, pruned to those within `best / error_threshold`, and each
//! survivor carries its own suffix cache into the next step. Up to
//! `refinement_pass` passes, stopping early when a pass changes nothing.
//!
//! ## Deviations from the old build
//!
//! 1. **The beam owns its suffix buffers.** The old code held
//!    `Cow<SuffixPatternCache>` and cloned on write; here [`Beam`] preallocates
//!    `refinement_candidate` suffixes per generation and the window seeds
//!    generation 0 with one copy of `suffixes[start_layer+1]`. Same algebra, no
//!    per-window allocation — a `Vec` header swap replaces the generation
//!    flip.
//! 2. **Candidates with infinite error are never selected.** With
//!    `error_threshold == 0` the old code took all `refinement_candidate`
//!    slots, including the `(∞, 0, 0, 0)` sentinels that survive when fewer
//!    real candidates were scored. The count is clamped to the finite ones.
//! 3. **The base dye is re-optimised** at the end of every pass, from
//!    `suffixes[0]` — the same closed form with `α ≡ 1`. The old Rust build
//!    left the greedy base alone; Python's `_refine_batch` re-optimises it, and
//!    the plan makes Python the parity target where the two disagree. It costs
//!    one pass over the patch per refinement pass.
//! 4. **The final composite is rebuilt.** The old build stopped its end-of-pass
//!    prefix rebuild one layer short (it never needed `prefixes[n]`); here that
//!    plane is the answer, so the chain is rebuilt in full and the reported
//!    error is measured on it.

use crate::cli::config::RefinementConfig;
use crate::color::{COLORS_F32, COLORS_WSQ_SUM, NUM_COLORS, W_PERCEPTUAL};
use crate::simd::F32s;
use crate::zip;

use super::greedy::build_prefix;
use super::workspace::{
    Beam, Plane, Solution, Suffix, Workspace, rebuild_prefixes, total_error, view, view_mut,
};

/// Everything a window needs that does not change inside one.
struct Ctx<'a> {
    /// Lane offset of the cell's patch.
    off: usize,
    /// The cell's target planes.
    target: &'a [Plane; 3],
    /// The pattern table.
    alphas: &'a [Plane],
    /// Beam width, window length, pruning threshold, pass count.
    cfg: &'a RefinementConfig,
}

/// Refine `solution` in place, leaving the prefix chain and its error
/// consistent with the result.
///
/// The prefix chain is rebuilt from `solution` first, so this works equally on
/// the greedy fill's output and on a perturbed copy of it.
pub(super) fn refine(
    ws: &mut Workspace,
    solution: &mut Solution,
    cfg: &RefinementConfig,
    alphas: &[Plane],
    n: usize,
) {
    debug_assert_eq!(solution.layers.len(), n);

    let Workspace {
        off,
        target,
        prefixes,
        suffixes,
        beam,
        scratch,
        ..
    } = ws;
    let off = *off;
    let ctx = Ctx {
        off,
        target,
        alphas,
        cfg,
    };

    rebuild_prefixes(prefixes, off, solution, alphas, n);
    empty_suffix(&mut suffixes[n], off);

    for _ in 0..cfg.refinement_pass {
        scratch.clear();
        scratch.extend_from_slice(&solution.layers);
        let previous_base = solution.base;

        for start in (0..n).rev() {
            refine_window(&ctx, prefixes, suffixes, beam, &mut solution.layers, start);
        }

        // `suffixes[0]` now maps a base colour through the whole layer stack,
        // so choosing the base is the same closed form with `α ≡ 1`.
        reopt_base(&ctx, &suffixes[0], solution);
        rebuild_prefixes(prefixes, off, solution, alphas, n);

        if solution.base == previous_base && solution.layers == *scratch {
            break;
        }
    }

    solution.error = total_error(&prefixes[n], target, off);
}

/// Re-choose up to `window_size` layers ending at `start_layer`, as a beam.
fn refine_window(
    ctx: &Ctx<'_>,
    prefixes: &mut [[Plane; 3]],
    suffixes: &mut [Suffix],
    beam: &mut Beam,
    layers: &mut [(usize, usize)],
    start_layer: usize,
) {
    let off = ctx.off;
    let cand_size = beam.best.len();

    // Generation 0: one candidate, the layers above the window unchanged.
    beam.prev[0].copy_from(&suffixes[start_layer + 1], off);
    beam.prev_layers[0].clear();
    let mut n_cand = 1;

    'sliding: for k in 0..ctx.cfg.window_size {
        let layer_idx = start_layer - k;
        beam.best.fill((f32::INFINITY, 0, 0, 0));

        for cand_idx in 0..n_cand {
            let sfx = &beam.prev[cand_idx];
            let prefix = &prefixes[layer_idx];

            for (p_idx, alpha) in ctx.alphas.iter().enumerate() {
                // Channel 0 also reduces `Σ (α·mul)²`, which the other two
                // channels share (`mul` and `α` do not depend on the channel).
                let (res2_0, res_2a_0, eff_a2) = moments_eff(
                    view(&prefix[0], off),
                    view(&ctx.target[0], off),
                    view(alpha, off),
                    view(&sfx.mul, off),
                    view(&sfx.add[0], off),
                );
                let (res2_1, res_2a_1) = moments(
                    view(&prefix[1], off),
                    view(&ctx.target[1], off),
                    view(alpha, off),
                    view(&sfx.mul, off),
                    view(&sfx.add[1], off),
                );
                let (res2_2, res_2a_2) = moments(
                    view(&prefix[2], off),
                    view(&ctx.target[2], off),
                    view(alpha, off),
                    view(&sfx.mul, off),
                    view(&sfx.add[2], off),
                );

                for c_idx in 0..NUM_COLORS {
                    let c = COLORS_F32[c_idx];
                    let err = W_PERCEPTUAL[0] * (res2_0 + res_2a_0 * c[0])
                        + W_PERCEPTUAL[1] * (res2_1 + res_2a_1 * c[1])
                        + W_PERCEPTUAL[2] * (res2_2 + res_2a_2 * c[2])
                        + eff_a2 * COLORS_WSQ_SUM[c_idx];

                    if err < beam.best[cand_size - 1].0 {
                        beam.best[cand_size - 1] = (err, cand_idx, p_idx, c_idx);
                        beam.best
                            .sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).expect("no NaN errors"));
                    }
                }
            }
        }

        // Prune: keep everything within `best / threshold`, never a sentinel.
        let finite = beam.best.partition_point(|c| c.0.is_finite()).max(1);
        n_cand = if ctx.cfg.error_threshold > 0.0 {
            let bound = beam.best[0].0 / ctx.cfg.error_threshold;
            beam.best.partition_point(|c| c.0 <= bound)
        } else {
            cand_size
        }
        .clamp(1, finite);

        for cand_idx in 0..n_cand {
            let (_, from, p_idx, c_idx) = beam.best[cand_idx];

            beam.curr_layers[cand_idx].clear();
            let (curr, prev) = (&mut beam.curr_layers[cand_idx], &beam.prev_layers[from]);
            curr.extend_from_slice(prev);
            curr.push((p_idx, c_idx));

            // Layer 0 has nothing below it, so no further step can follow and
            // the suffix would never be read.
            if layer_idx == 0 {
                break 'sliding;
            }
            build_suffix(
                &mut beam.curr[cand_idx],
                &beam.prev[from],
                &ctx.alphas[p_idx],
                c_idx,
                off,
            );
        }

        if k < ctx.cfg.window_size - 1 {
            beam.swap();
        }
    }

    // The winning chain, deepest layer first.
    for (i, pc) in beam.curr_layers[0].iter().enumerate() {
        layers[start_layer - i] = *pc;
    }

    // This window's top layer is now fixed, so its suffix is too.
    let (left, right) = suffixes.split_at_mut(start_layer + 1);
    let (p, c) = layers[start_layer];
    build_suffix(&mut left[start_layer], &right[0], &ctx.alphas[p], c, off);

    // Prefixes the *next* window (one layer higher up the stack) will read.
    // Prefixes at and above `start_layer` are left stale on purpose: no later
    // window reads them, and the end-of-pass rebuild fixes them all.
    let update_start = (start_layer + 1).saturating_sub(ctx.cfg.window_size);
    let update_end = start_layer.saturating_sub(1);
    for layer in update_start..update_end {
        let (left, right) = prefixes.split_at_mut(layer + 1);
        let (p, c) = layers[layer];
        build_prefix(&left[layer], &mut right[0], &ctx.alphas[p], c, off);
    }
}

/// The five-stream refinement kernel: residual moments of laying *any* dye
/// through `alp` at this layer, with `mul`/`add` folding in the layers above.
///
/// Returns `(Σ res², 2·Σ res·α_eff)` where `α_eff = α·mul` and
/// `res = prefix·(1−α)·mul + add − target`. Live `F32s` at width 4: five
/// streams, two temporaries, two accumulators — nine, well inside NEON's 32.
#[inline]
pub(super) fn moments(
    pre: &[F32s],
    tar: &[F32s],
    alp: &[F32s],
    mul: &[F32s],
    add: &[F32s],
) -> (f32, f32) {
    let mut acc2 = F32s::ZERO;
    let mut acc_a = F32s::ZERO;
    for (pre, tar, alp, mul, add) in zip!(pre, tar, alp, mul, add) {
        let eff = alp * mul;
        let res = (pre * (F32s::ONE - alp)).mul_add(mul, add) - tar;
        acc2 = res.mul_add(res, acc2);
        acc_a = res.mul_add(eff, acc_a);
    }
    (acc2.hsum(), 2.0 * acc_a.hsum())
}

/// [`moments`] plus `Σ α_eff²`, which is channel-independent — reduced once, on
/// channel 0. Ten live `F32s` at width 4.
#[inline]
pub(super) fn moments_eff(
    pre: &[F32s],
    tar: &[F32s],
    alp: &[F32s],
    mul: &[F32s],
    add: &[F32s],
) -> (f32, f32, f32) {
    let mut acc2 = F32s::ZERO;
    let mut acc_a = F32s::ZERO;
    let mut acc_e = F32s::ZERO;
    for (pre, tar, alp, mul, add) in zip!(pre, tar, alp, mul, add) {
        let eff = alp * mul;
        let res = (pre * (F32s::ONE - alp)).mul_add(mul, add) - tar;
        acc2 = res.mul_add(res, acc2);
        acc_a = res.mul_add(eff, acc_a);
        acc_e = eff.mul_add(eff, acc_e);
    }
    (acc2.hsum(), 2.0 * acc_a.hsum(), acc_e.hsum())
}

/// `layer = suffix ∘ (lay `color` through `alpha`)`, ported from the old
/// build's `build_suffix`.
pub(super) fn build_suffix(
    layer: &mut Suffix,
    suffix: &Suffix,
    alpha: &Plane,
    color: usize,
    off: usize,
) {
    for (ch, plane) in layer.add.iter_mut().enumerate() {
        let c = F32s::splat(COLORS_F32[color][ch]);
        for (out, sfx_mul, sfx_add, alp) in zip!(
            mut view_mut(plane, off),
            view(&suffix.mul, off),
            view(&suffix.add[ch], off),
            view(alpha, off)
        ) {
            *out = (c * alp).mul_add(sfx_mul, sfx_add);
        }
    }
    for (out, sfx_mul, alp) in zip!(
        mut view_mut(&mut layer.mul, off),
        view(&suffix.mul, off),
        view(alpha, off)
    ) {
        *out = (F32s::ONE - alp) * sfx_mul;
    }
}

/// The identity suffix: nothing is painted over these layers.
pub(super) fn empty_suffix(suffix: &mut Suffix, off: usize) {
    for ch in 0..3 {
        for out in zip!(mut view_mut(&mut suffix.add[ch], off)) {
            *out = F32s::ZERO;
        }
    }
    for out in zip!(mut view_mut(&mut suffix.mul, off)) {
        *out = F32s::ONE;
    }
}

/// Re-choose the base dye given the affine map of the whole layer stack.
///
/// With `d = add − target` the banner is `c·mul + add`, so the weighted SSE is
/// `Σ w·d² + c·(2·Σ w·mul·d) + c²·Σ w·mul²` — the greedy base sweep with
/// `Σ mul²` in place of `HW`.
fn reopt_base(ctx: &Ctx<'_>, suffix: &Suffix, solution: &mut Solution) {
    let off = ctx.off;

    let mut acc = F32s::ZERO;
    for m in zip!(view(&suffix.mul, off)) {
        acc = m.mul_add(m, acc);
    }
    let mul2 = acc.hsum();

    let mut d2 = [0.0_f32; 3];
    let mut d_2m = [0.0_f32; 3];
    for ch in 0..3 {
        let mut acc2 = F32s::ZERO;
        let mut acc_m = F32s::ZERO;
        for (m, a, t) in zip!(
            view(&suffix.mul, off),
            view(&suffix.add[ch], off),
            view(&ctx.target[ch], off)
        ) {
            let d = a - t;
            acc2 = d.mul_add(d, acc2);
            acc_m = d.mul_add(m, acc_m);
        }
        d2[ch] = acc2.hsum();
        d_2m[ch] = 2.0 * acc_m.hsum();
    }

    let mut best = solution.base;
    let mut min_err = f32::INFINITY;
    for c_idx in 0..NUM_COLORS {
        let c = COLORS_F32[c_idx];
        let err = W_PERCEPTUAL[0] * (d2[0] + d_2m[0] * c[0])
            + W_PERCEPTUAL[1] * (d2[1] + d_2m[1] * c[1])
            + W_PERCEPTUAL[2] * (d2[2] + d_2m[2] * c[2])
            + mul2 * COLORS_WSQ_SUM[c_idx];
        if err < min_err {
            min_err = err;
            best = c_idx;
        }
    }
    solution.base = best;
}
