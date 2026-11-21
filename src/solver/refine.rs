//! Windowed beam refinement: walk the layer stack backwards and re-choose a
//! sliding window of layers at a time, scoring each candidate against the
//! *whole* banner rather than the partial composite the greedy fill saw.
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
//! ## The exact rung (`--exact-candidates N`)
//!
//! The closed form above minimises a *weighted sRGB SSE*, which is not what the
//! eye measures. Solving natively in OKLab does not help — compositing has to
//! happen in sRGB because that is what the game blends in — but *exactly*
//! scoring a shortlist does. So each window step is two rungs:
//!
//! 1. score the whole `patterns × 16 × candidates` grid with the closed form —
//!    one pass over the patch per `(candidate, pattern)`, as before;
//! 2. shortlist `N` of them, build each one's full composite from the
//!    prefix/suffix caches (`pre·(1−α)·mul + c·α·mul + add`, one `O(HW)` pass),
//!    convert it to OKLab and score [`lab_error`] against the cell's OKLab
//!    target.
//!
//! The beam's ranking, its `refinement_candidate` survivors and the
//! `error_threshold` prune all read the **exact** errors, and so does
//! [`Solution::error`] — the closed form is demoted to a shortlisting heuristic.
//! `N = 0` turns rung 2 off and restores the pure closed-form behaviour.
//!
//! `error_threshold` is a ratio (`best / threshold`), so it is scale-free and
//! needs no retuning even though OKLab errors are ~10⁴× smaller than sRGB ones.
//!
//! ### Which `N` candidates get shortlisted
//!
//! Not the flat top `N` of the grid: **one entry per `(candidate, pattern)`
//! pair — that pair's best dye — and the top `N` of those.** The closed form is
//! two different qualities of estimate rolled into one number. Choosing the dye
//! *given* a pattern is a well-conditioned 1-D fit and it gets that essentially
//! right; ranking whole patterns against each other is where the sRGB/OKLab
//! mismatch shows, and that is exactly what the exact rung is for. A flat top-N
//! spends most of its budget on runner-up dyes of the same few patterns, which
//! is the one axis it did not need to check — measured, the per-pair shortlist
//! at `N = 8` beats a flat one at `N = 40` for a fifth of the exact
//! evaluations.
//!
//! ## The feature term
//!
//! With `--feature-weight λ` above zero both rungs also score the cell's feature
//! map ([`crate::solver::feature`]), and they do it in different places:
//!
//! - Rung 1 sees nothing at all. `Σ w·(c−t)² + λ·Σ w·(c−f)²` is, up to a
//!   dye-independent constant, `(1+λ)·Σ w·(c−b)²` with the blended target
//!   `b = (t + λ·f)/(1+λ)`, so the feature term is folded into
//!   [`Ctx::target`] before refinement starts and the moments machinery is
//!   untouched.
//! - Rung 2 adds the term explicitly, in [`ranking_error`]: OKLab distances do
//!   not expand like that.
//!
//! `error_threshold` still compares like with like — a `best` list is filled by
//! exactly one of the two rungs, never a mix — and the error the *solve* finally
//! reports is re-measured against the pure target ([`lab_error`]) so quality
//! numbers stay comparable across λ.
//!
//! ## The purity term
//!
//! With `--purity-weight ν` above zero, rung 2 also adds `ν · Σ ΔE(composite
//! pixel, nearest dye)`, nearest by exact OKLab distance over the 16 dyes
//! ([`crate::color::colors_oklab`]). It does not look at the target: every
//! pixel that is a mix of two dyes is charged, so of two candidates with similar
//! target error the one drawn in cleaner dye colours ranks first. Rung 1 and
//! the closed-form base re-optimisation do not see it, and like the feature
//! term it is left out of the reported error.
//!
//! ## The gradient term
//!
//! With `--gradient-weight μ` above zero, rung 2 also adds
//! `μ · Σ ‖(c_a − c_b) − (t_a − t_b)‖` over every horizontally and every
//! vertically adjacent pixel pair `(a, b)` of the solved view, `c` the
//! composite and `t` the pure target, both in OKLab. It charges every step
//! between neighbouring pixels that the target does not take, and every step
//! the target takes that the composite does not, so an edge one pixel off is
//! charged twice; `--shift-range` lets the target meet the pattern edge.
//!
//! The norm is the same for `(c_a − t_a) − (c_b − t_b)`, so the term needs no
//! target steps: [`ranking_error`] writes each pixel's residual `c − t` to
//! [`Workspace::lab_residual`] as it measures the target ΔE, and
//! [`gradient`] differences each residual lane against the lane one pixel to
//! the right and the lane one pixel row down. A pair-mask plane per direction
//! zeroes the pairs that leave the patch: the last pixel of a row with the
//! first of the next, and the last row with the padding past the plane. Pairs
//! start at the first pixel of the view and only reach right and down, so a
//! lower banner row never reads the hidden head of its planes.
//!
//! The base dye is re-optimised at the end of every pass from `suffixes[0]`
//! (with `α ≡ 1`), and the whole prefix chain is then rebuilt, so `prefixes[n]`
//! is the composite the reported error is measured on.

use crate::cli::config::RefinementConfig;
use std::sync::OnceLock;

use crate::color::{COLORS_F32, COLORS_WSQ_SUM, NUM_COLORS, W_PERCEPTUAL, colors_oklab};
use crate::geometry::{BANNER_W, TOP_HW};
use crate::oklab::srgb_to_oklab;
use crate::simd::{Chunk, F32s, LANES};
use crate::zip;

use super::greedy::build_prefix;
use super::workspace::{
    Beam, PaddedPlane, Plane, Solution, Suffix, Workspace, rebuild_prefixes, view, view_mut,
};

/// A scored candidate: `(error, beam candidate, pattern, dye)`.
type Scored = (f32, usize, usize, usize);

/// The sentinel a top-list slot holds until a real candidate displaces it.
const NO_CAND: Scored = (f32::INFINITY, 0, 0, 0);

/// Everything a window needs that does not change inside one.
struct Ctx<'a> {
    /// Lane offset of the cell's patch.
    off: usize,
    /// The cell's target planes — *blended* with the feature map when the
    /// feature weight is non-zero ([`crate::solver::feature`]), which is what
    /// makes the closed form below score the feature term for free.
    target: &'a [Plane; 3],
    /// The cell's target planes, in OKLab. Never blended: the exact rung scores
    /// the two terms separately.
    lab_target: &'a [Plane; 3],
    /// The cell's feature map, in OKLab. Meaningful only when `lambda > 0`.
    lab_feature: &'a [Plane; 3],
    /// Weight of the feature term in the ranking error; `0` disables it.
    lambda: f32,
    /// Weight of the purity term in the ranking error; `0` disables it.
    nu: f32,
    /// Weight of the gradient term in the ranking error; `0` disables it.
    mu: f32,
    /// The pattern table.
    alphas: &'a [Plane],
    /// Beam width, window length, pruning threshold, pass count, exact top-N.
    cfg: &'a RefinementConfig,
}

/// Insert `entry` into an ascending top-list of fixed length, dropping its
/// current worst.
///
/// Ties keep the earlier insertion first, so the surviving order does not
/// depend on a sort's tie-breaking — one less thing that could differ between
/// SIMD backends.
#[inline]
fn push_top(list: &mut [Scored], entry: Scored) {
    let n = list.len();
    if entry.0 >= list[n - 1].0 {
        return;
    }
    let mut i = n - 1;
    while i > 0 && list[i - 1].0 > entry.0 {
        list[i] = list[i - 1];
        i -= 1;
    }
    list[i] = entry;
}

/// Refine `solution` in place, leaving the prefix chain and its error
/// consistent with the result.
///
/// The prefix chain is rebuilt from `solution` first, so this works equally on
/// the greedy fill's output and on a perturbed copy of it.
#[allow(clippy::too_many_arguments)]
pub(super) fn refine(
    ws: &mut Workspace,
    solution: &mut Solution,
    cfg: &RefinementConfig,
    lambda: f32,
    nu: f32,
    mu: f32,
    alphas: &[Plane],
    n: usize,
) {
    let Workspace {
        off,
        target,
        lab_target,
        lab_feature,
        lab_residual,
        prefixes,
        suffixes,
        beam,
        cand,
        shortlist,
        scratch,
        ..
    } = ws;
    let off = *off;
    let ctx = Ctx {
        off,
        target,
        lab_target,
        lab_feature,
        lambda,
        nu,
        mu,
        alphas,
        cfg,
    };

    // Rung 1 offers at most one entry per (beam candidate, pattern), so asking
    // for more slots than that only wastes memory.
    let n_exact = cfg
        .exact_candidates
        .min(beam.best.len() * alphas.len().max(1));
    shortlist.resize(n_exact, NO_CAND);

    rebuild_prefixes(prefixes, off, solution, alphas, n);
    empty_suffix(&mut suffixes[n], off);

    for _ in 0..cfg.refinement_pass {
        scratch.clear();
        scratch.extend_from_slice(&solution.layers);
        let previous_base = solution.base;

        for start in (0..n).rev() {
            refine_window(
                &ctx,
                prefixes,
                suffixes,
                beam,
                cand,
                lab_residual,
                shortlist,
                &mut solution.layers,
                start,
            );
        }

        // `suffixes[0]` now maps a base colour through the whole layer stack,
        // so choosing the base is the same problem with `α ≡ 1`.
        reopt_base(&ctx, &suffixes[0], cand, lab_residual, solution);
        rebuild_prefixes(prefixes, off, solution, alphas, n);

        if solution.base == previous_base && solution.layers == *scratch {
            break;
        }
    }

    solution.error = ranking_error(
        &prefixes[n],
        lab_target,
        lab_feature,
        lambda,
        nu,
        mu,
        lab_residual,
        off,
    );
}

/// Re-choose up to `window_size` layers ending at `start_layer`, as a beam.
#[allow(clippy::too_many_arguments)]
fn refine_window(
    ctx: &Ctx<'_>,
    prefixes: &mut [[Plane; 3]],
    suffixes: &mut [Suffix],
    beam: &mut Beam,
    cand: &mut [Plane; 3],
    residual: &mut [PaddedPlane; 3],
    shortlist: &mut [Scored],
    layers: &mut [(usize, usize)],
    start_layer: usize,
) {
    let off = ctx.off;
    // Destructured so the exact rung can read `prev` while writing `best`.
    let Beam {
        prev,
        curr,
        prev_layers,
        curr_layers,
        best,
    } = beam;
    let cand_size = best.len();
    let n_exact = shortlist.len();

    // Generation 0: one candidate, the layers above the window unchanged.
    prev[0].copy_from(&suffixes[start_layer + 1], off);
    prev_layers[0].clear();
    let mut n_cand = 1;

    'sliding: for k in 0..ctx.cfg.window_size {
        let layer_idx = start_layer - k;
        best.fill(NO_CAND);
        shortlist.fill(NO_CAND);

        for (cand_idx, sfx) in prev.iter().enumerate().take(n_cand) {
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

                // Rung 1. With the exact rung off the closed form *is* the
                // ranking; with it on, only this pattern's best dye is
                // shortlisted.
                let mut pattern_best = NO_CAND;
                for (c_idx, c) in COLORS_F32.iter().enumerate() {
                    let err = W_PERCEPTUAL[0] * (res2_0 + res_2a_0 * c[0])
                        + W_PERCEPTUAL[1] * (res2_1 + res_2a_1 * c[1])
                        + W_PERCEPTUAL[2] * (res2_2 + res_2a_2 * c[2])
                        + eff_a2 * COLORS_WSQ_SUM[c_idx];

                    let entry = (err, cand_idx, p_idx, c_idx);
                    if n_exact == 0 {
                        push_top(best, entry);
                    } else if err < pattern_best.0 {
                        pattern_best = entry;
                    }
                }
                if n_exact > 0 {
                    push_top(shortlist, pattern_best);
                }
            }
        }

        // Rung 2: build each shortlisted candidate's whole banner and score it
        // in OKLab. This is the only ranking the beam below ever sees.
        for &(closed, cand_idx, p_idx, c_idx) in shortlist.iter() {
            if !closed.is_finite() {
                break;
            }
            composite(
                cand,
                &prefixes[layer_idx],
                &ctx.alphas[p_idx],
                c_idx,
                &prev[cand_idx],
                off,
            );
            push_top(
                best,
                (
                    ranking_error(
                        cand,
                        ctx.lab_target,
                        ctx.lab_feature,
                        ctx.lambda,
                        ctx.nu,
                        ctx.mu,
                        residual,
                        off,
                    ),
                    cand_idx,
                    p_idx,
                    c_idx,
                ),
            );
        }

        // Prune: keep everything within `best / threshold`, never a sentinel.
        // The threshold is a ratio, so it means the same thing whichever rung
        // produced the numbers.
        let finite = best.partition_point(|c| c.0.is_finite()).max(1);
        n_cand = if ctx.cfg.error_threshold > 0.0 {
            let bound = best[0].0 / ctx.cfg.error_threshold;
            best.partition_point(|c| c.0 <= bound)
        } else {
            cand_size
        }
        .clamp(1, finite);

        for cand_idx in 0..n_cand {
            let (_, from, p_idx, c_idx) = best[cand_idx];

            curr_layers[cand_idx].clear();
            let (to, source) = (&mut curr_layers[cand_idx], &prev_layers[from]);
            to.extend_from_slice(source);
            to.push((p_idx, c_idx));

            // Layer 0 has nothing below it, so no further step can follow and
            // the suffix would never be read.
            if layer_idx == 0 {
                break 'sliding;
            }
            build_suffix(
                &mut curr[cand_idx],
                &prev[from],
                &ctx.alphas[p_idx],
                c_idx,
                off,
            );
        }

        if k < ctx.cfg.window_size - 1 {
            std::mem::swap(prev, curr);
            std::mem::swap(prev_layers, curr_layers);
        }
    }

    // The winning chain, deepest layer first.
    for (i, pc) in curr_layers[0].iter().enumerate() {
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

/// `layer = suffix ∘ (lay `color` through `alpha`)`.
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

/// The full banner for laying `color` through `alpha` on top of `prefix`, with
/// `suffix` applying every layer above it — the composite the exact rung scores.
///
/// This is the *sRGB* composite, deliberately: the game blends dye colours in
/// sRGB, so this is the pixel a player will actually see. Blending in OKLab
/// instead costs 1-3% ΔE on anti-aliased pattern edges.
pub(super) fn composite(
    out: &mut [Plane; 3],
    prefix: &[Plane; 3],
    alpha: &Plane,
    color: usize,
    suffix: &Suffix,
    off: usize,
) {
    for ch in 0..3 {
        let c = F32s::splat(COLORS_F32[color][ch]);
        for (out, pre, alp, mul, add) in zip!(
            mut view_mut(&mut out[ch], off),
            view(&prefix[ch], off),
            view(alpha, off),
            view(&suffix.mul, off),
            view(&suffix.add[ch], off)
        ) {
            *out = pre.mul_add(F32s::ONE - alp, c * alp).mul_add(mul, add);
        }
    }
}

/// The cell's target patch, converted to OKLab once per cell.
pub(crate) fn convert_target(target: &[Plane; 3], out: &mut [Plane; 3], off: usize) {
    let (l, rest) = out.split_at_mut(1);
    let (a, b) = rest.split_at_mut(1);
    for (l, a, b, r, g, bl) in zip!(
        mut view_mut(&mut l[0], off),
        mut view_mut(&mut a[0], off),
        mut view_mut(&mut b[0], off),
        view(&target[0], off),
        view(&target[1], off),
        view(&target[2], off)
    ) {
        (*l, *a, *b) = srgb_to_oklab(r, g, bl);
    }
}

/// The number the beam actually ranks on: `Σ ΔE(composite, target)` plus
/// `λ · Σ ΔE(composite, feature)` plus `ν · Σ ΔE(composite, nearest dye)` plus
/// `μ` times the [`gradient`] term.
///
/// One pass, not several calls to [`lab_error`]: the composite's OKLab
/// conversion is the expensive part (a cube root per channel per lane) and every
/// term needs the same one. Nine lane streams is past [`zip!`](crate::zip)'s
/// arity, hence the indexed loop. The gradient term needs every pixel's
/// neighbours, so that pass writes the residuals to `residual` and the pairs are
/// summed after it.
///
/// `λ = ν = μ = 0` returns [`lab_error`] itself, so a run without any of the
/// terms ranks on bit-identical numbers to one built before they existed; a
/// zero weight adds nothing to the other terms' arithmetic either.
#[allow(clippy::too_many_arguments)]
pub(super) fn ranking_error(
    rgb: &[Plane; 3],
    lab_target: &[Plane; 3],
    lab_feature: &[Plane; 3],
    lambda: f32,
    nu: f32,
    mu: f32,
    residual: &mut [PaddedPlane; 3],
    off: usize,
) -> f32 {
    if lambda == 0.0 && nu == 0.0 && mu == 0.0 {
        return lab_error(rgb, lab_target, off);
    }

    let lam = F32s::splat(lambda);
    let nu_s = F32s::splat(nu);
    let dyes: [(F32s, F32s, F32s); NUM_COLORS] =
        colors_oklab().map(|[l, a, b]| (F32s::splat(l), F32s::splat(a), F32s::splat(b)));
    let rgb = [view(&rgb[0], off), view(&rgb[1], off), view(&rgb[2], off)];
    let tar: [&[F32s]; 3] = std::array::from_fn(|ch| view(&lab_target[ch], off));
    let fea: [&[F32s]; 3] = std::array::from_fn(|ch| view(&lab_feature[ch], off));

    let mut acc = F32s::ZERO;
    for i in 0..rgb[0].len() {
        let lab = srgb_to_oklab(rgb[0][i], rgb[1][i], rgb[2][i]);
        let t = (tar[0][i], tar[1][i], tar[2][i]);
        let mut err = delta_e(lab, t);
        if lambda != 0.0 {
            let feature = delta_e(lab, (fea[0][i], fea[1][i], fea[2][i]));
            err = feature.mul_add(lam, err);
        }
        if nu != 0.0 {
            err = purity(lab, &dyes).mul_add(nu_s, err);
        }
        if mu != 0.0 {
            residual[0].lanes_mut()[off + i] = lab.0 - t.0;
            residual[1].lanes_mut()[off + i] = lab.1 - t.1;
            residual[2].lanes_mut()[off + i] = lab.2 - t.2;
        }
        acc += err;
    }
    if mu != 0.0 {
        acc = gradient(residual, off).mul_add(F32s::splat(mu), acc);
    }
    acc.hsum()
}

/// The gradient term before its weight, per lane: `Σ ‖d_a − d_b‖` over every
/// pixel pair `(a, b)` with `b` the right or lower neighbour of `a`, inside
/// the view starting at lane `off`, for OKLab residuals `d` (composite minus
/// target). Equal to `Σ ‖(c_a − c_b) − (t_a − t_b)‖`.
///
/// `a` is an aligned lane; its neighbours are the unaligned lanes one pixel
/// and one pixel row further on, which is why `residual` is padded. A pair is
/// weighed by [`pair_masks`], which is zero where `b` is not `a`'s neighbour
/// inside the patch.
fn gradient(residual: &[PaddedPlane; 3], off: usize) -> F32s {
    let [right_mask, below_mask] = pair_masks();
    let [l, a, b] = residual;
    let at = |plane: &PaddedPlane, px: usize| {
        F32s::from_array(plane[px..px + LANES].try_into().expect("one lane"))
    };
    let mut acc = F32s::ZERO;
    for k in off..TOP_HW / LANES {
        let px = k * LANES;
        let here = (l.lanes()[k], a.lanes()[k], b.lanes()[k]);
        let right = (at(l, px + 1), at(a, px + 1), at(b, px + 1));
        let below = (
            at(l, px + BANNER_W),
            at(a, px + BANNER_W),
            at(b, px + BANNER_W),
        );
        acc = delta_e(here, right).mul_add(right_mask.lanes()[k], acc);
        acc = delta_e(here, below).mul_add(below_mask.lanes()[k], acc);
    }
    acc
}

/// `[right, below]`: per pixel, `1` where the pixel has a neighbour in that
/// direction inside a full patch and `0` where it does not (the last column
/// for `right`, the last row for `below`). The same for every banner row,
/// since a lower row's view ends where a top row's does.
fn pair_masks() -> &'static [Plane; 2] {
    static MASKS: OnceLock<[Plane; 2]> = OnceLock::new();
    MASKS.get_or_init(|| {
        let mut right = Chunk::zeroed();
        let mut below = Chunk::zeroed();
        for px in 0..TOP_HW {
            right[px] = f32::from(u8::from(px % BANNER_W != BANNER_W - 1));
            below[px] = f32::from(u8::from(px + BANNER_W < TOP_HW));
        }
        [right, below]
    })
}

/// `ΔE²` between two OKLab lanes, in the form [`delta_e`] takes the root of.
#[inline]
fn delta_e2(a: (F32s, F32s, F32s), b: (F32s, F32s, F32s)) -> F32s {
    let (dl, da, db) = (a.0 - b.0, a.1 - b.1, a.2 - b.2);
    dl.mul_add(dl, da.mul_add(da, db * db))
}

/// Euclidean distance between two OKLab lanes — a ΔE.
#[inline]
fn delta_e(a: (F32s, F32s, F32s), b: (F32s, F32s, F32s)) -> F32s {
    delta_e2(a, b).sqrt()
}

/// ΔE from an OKLab lane to its nearest dye: the smallest `ΔE²` over the 16
/// dyes, then one square root. Zero for a pixel painted in a single dye.
#[inline]
fn purity(lab: (F32s, F32s, F32s), dyes: &[(F32s, F32s, F32s); NUM_COLORS]) -> F32s {
    dyes[1..]
        .iter()
        .fold(delta_e2(lab, dyes[0]), |min, &dye| {
            min.min(delta_e2(lab, dye))
        })
        .sqrt()
}

/// `Σ ΔE` between an sRGB composite and the OKLab target planes — the **pure
/// target** error, which is what [`Solution::error`] carries out of a solve and
/// what `--debug` reports, whatever the feature weight is.
///
/// Summed *un-squared*, deliberately. Minimising `Σ ΔE²` is a different problem
/// and measurably the wrong one here: on great_wave at `--row 20` it reaches a
/// lower `Σ ΔE²` (6.56 vs 6.77 per cell) and a **worse** picture (0.0815 vs
/// 0.0797 mean ΔE). Squaring buys uniformity of error at the price of the
/// average, and the average is what a viewer sees — a banner wall is a mosaic
/// of 20×40 patches, not a smooth gradient, so there is no visible seam to
/// protect. `Σ ΔE` is also exactly the quantity `--debug` reports, so the
/// solver optimises the number it is judged on.
pub(super) fn lab_error(rgb: &[Plane; 3], lab_target: &[Plane; 3], off: usize) -> f32 {
    let mut acc = F32s::ZERO;
    for (r, g, b, tl, ta, tb) in zip!(
        view(&rgb[0], off),
        view(&rgb[1], off),
        view(&rgb[2], off),
        view(&lab_target[0], off),
        view(&lab_target[1], off),
        view(&lab_target[2], off)
    ) {
        acc += delta_e(srgb_to_oklab(r, g, b), (tl, ta, tb));
    }
    acc.hsum()
}

/// The identity suffix: nothing is painted over these layers.
pub(super) fn empty_suffix(suffix: &mut Suffix, off: usize) {
    for plane in &mut suffix.add {
        for out in zip!(mut view_mut(plane, off)) {
            *out = F32s::ZERO;
        }
    }
    for out in zip!(mut view_mut(&mut suffix.mul, off)) {
        *out = F32s::ONE;
    }
}

/// Re-choose the base dye at the end of a pass, given the affine map of the
/// whole layer stack above it.
///
/// There is no shortlist here — there are only 16 candidates — so with the
/// exact rung on, all 16 are simply scored exactly, the same ranking the window
/// steps use. `suffixes[0]` maps a flat fill through every layer, so a base dye
/// `c` paints `c·mul + add` and building each candidate is one `O(HW)` pass.
fn reopt_base(
    ctx: &Ctx<'_>,
    suffix: &Suffix,
    out: &mut [Plane; 3],
    residual: &mut [PaddedPlane; 3],
    solution: &mut Solution,
) {
    let off = ctx.off;

    if ctx.cfg.exact_candidates > 0 {
        let mut best = solution.base;
        let mut min_err = f32::INFINITY;
        for (c_idx, color) in COLORS_F32.iter().enumerate() {
            for (ch, &component) in color.iter().enumerate() {
                let c = F32s::splat(component);
                for (o, mul, add) in zip!(
                    mut view_mut(&mut out[ch], off),
                    view(&suffix.mul, off),
                    view(&suffix.add[ch], off)
                ) {
                    *o = c.mul_add(mul, add);
                }
            }
            let err = ranking_error(
                out,
                ctx.lab_target,
                ctx.lab_feature,
                ctx.lambda,
                ctx.nu,
                ctx.mu,
                residual,
                off,
            );
            if err < min_err {
                min_err = err;
                best = c_idx;
            }
        }
        solution.base = best;
        return;
    }

    // Closed form, for `--exact-candidates 0`. With `d = add − target` the
    // banner is `c·mul + add`, so the weighted sRGB SSE is
    // `Σ w·d² + c·(2·Σ w·mul·d) + c²·Σ w·mul²` — the greedy base sweep with
    // `Σ mul²` in place of `HW`.

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{BANNER_H, HIDDEN_H};
    use crate::solver::workspace::lane_offset;

    /// A composite and a target that both change along `x`, so the residual at
    /// the end of a pixel row is far from the one at the start of the next.
    /// Pixels above row `head` get unrelated colours: a pair that reaches into
    /// a lower row's hidden head picks up a large step.
    fn planes(head: usize) -> ([Plane; 3], [Plane; 3]) {
        let mut rgb = [Chunk::zeroed(); 3];
        let mut target = [Chunk::zeroed(); 3];
        for i in 0..TOP_HW {
            let (x, y) = ((i % BANNER_W) as f32, (i / BANNER_W) as f32);
            let (c, t) = if i / BANNER_W < head {
                ([250.0, 10.0, 240.0], [5.0, 230.0, 20.0])
            } else {
                (
                    [
                        10.0 * x + 5.0 * y,
                        200.0 - 7.0 * x,
                        30.0 + 3.0 * y + (x % 3.0) * 20.0,
                    ],
                    [
                        if x < 9.0 { 100.0 + 3.0 * y } else { 220.0 },
                        50.0 + 8.0 * x,
                        120.0 - 2.0 * y,
                    ],
                )
            };
            for ch in 0..3 {
                rgb[ch][i] = c[ch];
                target[ch][i] = t[ch];
            }
        }
        (rgb, target)
    }

    fn lab(px: [f32; 3]) -> [f64; 3] {
        let (l, a, b) = srgb_to_oklab(F32s::splat(px[0]), F32s::splat(px[1]), F32s::splat(px[2]));
        [l, a, b].map(|v| f64::from(v.to_array()[0]))
    }

    /// The gradient term from its definition: every right and lower neighbour
    /// pair of the visible rows `top..`, with each step taken separately on the
    /// composite and the target.
    fn reference(rgb: &[Plane; 3], target: &[Plane; 3], top: usize) -> f64 {
        let at = |planes: &[Plane; 3], x: usize, y: usize| {
            let i = y * BANNER_W + x;
            lab([planes[0][i], planes[1][i], planes[2][i]])
        };
        let step = |a: (usize, usize), b: (usize, usize)| {
            let (ca, cb) = (at(rgb, a.0, a.1), at(rgb, b.0, b.1));
            let (ta, tb) = (at(target, a.0, a.1), at(target, b.0, b.1));
            (0..3)
                .map(|ch| ((ca[ch] - cb[ch]) - (ta[ch] - tb[ch])).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        let mut sum = 0.0;
        for y in top..BANNER_H {
            for x in 0..BANNER_W {
                if x + 1 < BANNER_W {
                    sum += step((x, y), (x + 1, y));
                }
                if y + 1 < BANNER_H {
                    sum += step((x, y), (x, y + 1));
                }
            }
        }
        sum
    }

    /// With only the gradient weight on, the ranking error is the target ΔE
    /// plus μ times the gradient term computed from its definition, on the top
    /// row and on a lower row. The lower row runs second on the same residual
    /// scratch, so its hidden head holds the top row's residuals, and its
    /// planes' head holds unrelated colours.
    #[test]
    fn the_gradient_term_sums_the_neighbour_pairs_of_the_visible_rows() {
        let mu = 0.75;
        let mut residual = [Chunk::zeroed(); 3];
        for (row, top) in [(0, 0), (1, HIDDEN_H)] {
            let off = lane_offset(row);
            let (rgb, target) = planes(top);
            let mut lab_target = [Chunk::zeroed(); 3];
            convert_target(&target, &mut lab_target, 0);

            let ranked = ranking_error(
                &rgb,
                &lab_target,
                &lab_target,
                0.0,
                0.0,
                mu,
                &mut residual,
                off,
            );
            let got = f64::from(ranked - lab_error(&rgb, &lab_target, off)) / f64::from(mu);
            let want = reference(&rgb, &target, top);
            assert!(
                (got - want).abs() < 1e-4 * want,
                "row {row}: gradient term {got}, from its definition {want}"
            );
        }
    }
}
