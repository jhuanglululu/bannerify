//! The OKLab final pass: one coordinate-descent sweep scored by perceptual
//! distance instead of weighted sRGB SSE.
//!
//! Everything before this stage optimises `Σ w·(composite − target)²` in sRGB,
//! because that expression is quadratic in the dye colour and so all 16 dyes
//! can be scored from one pass over the patch ([`crate::color`]). Perceptual
//! distance has no such structure: the composite has to be *built* and
//! *converted* per candidate. So the pass is Python's `lab.py` shape, with
//! OKLab in place of CIELAB + CIEDE2000 (`context/plans/2-solver.md`):
//!
//! 1. walk the layers from the top down, keeping the suffix caches
//!    [`super::refine`] already defines;
//! 2. per layer, score *all* `patterns × 16` candidates with the cheap
//!    closed-form sRGB error — the same kernel refinement uses, so this costs
//!    one pass per pattern, not per candidate;
//! 3. keep the top `K` of those (`--lab-refine K`) and build, convert and score
//!    only those exactly, in OKLab;
//! 4. adopt the best if it beats the current banner's OKLab error — the guard
//!    matters, because the current choice need not be in the top `K`;
//! 5. finally re-choose the base dye the same way, from `suffixes[0]`.
//!
//! One sweep, like Python: this is a nudge on top of a converged fit, not a
//! search.
//!
//! ## Which error is reported
//!
//! [`Solution::error`] stays the weighted sRGB SSE throughout, so the number
//! `--debug` prints is comparable across every configuration. This pass
//! optimises a *different* objective, so it can legitimately raise that number
//! while improving the picture — which is why [`Solution::lab_error`] (mean ΔE
//! per pixel) is reported alongside it whenever `--debug` is on, computed here
//! even when `--lab-refine` is off.

use crate::color::{COLORS_F32, COLORS_WSQ_SUM, NUM_COLORS, W_PERCEPTUAL};
use crate::oklab::srgb_to_oklab;
use crate::simd::F32s;
use crate::zip;

use super::refine::{build_suffix, empty_suffix, moments, moments_eff};
use super::workspace::{
    Plane, Solution, Suffix, Workspace, rebuild_prefixes, total_error, view, view_mut,
};

/// Run the pass (when `k` is `Some`) and record the final ΔE.
pub(super) fn pass(
    ws: &mut Workspace,
    solution: &mut Solution,
    k: Option<usize>,
    alphas: &[Plane],
    n: usize,
) {
    let Workspace {
        off,
        hw,
        target,
        prefixes,
        suffixes,
        cand,
        lab_target,
        lab_cands,
        ..
    } = ws;
    let (off, hw) = (*off, *hw);

    convert_target(target, lab_target, off);
    let mut current = lab_error(&prefixes[n], lab_target, off);

    if let Some(k) = k.filter(|_| n > 0) {
        let k = k.clamp(1, alphas.len() * NUM_COLORS);
        empty_suffix(&mut suffixes[n], off);

        for i in (0..n).rev() {
            // --- 2: every candidate, closed form, one pass per pattern -------
            lab_cands.clear();
            let sfx = &suffixes[i + 1];
            let prefix = &prefixes[i];
            for (p_idx, alpha) in alphas.iter().enumerate() {
                let (res2_0, res_2a_0, eff_a2) = moments_eff(
                    view(&prefix[0], off),
                    view(&target[0], off),
                    view(alpha, off),
                    view(&sfx.mul, off),
                    view(&sfx.add[0], off),
                );
                let (res2_1, res_2a_1) = moments(
                    view(&prefix[1], off),
                    view(&target[1], off),
                    view(alpha, off),
                    view(&sfx.mul, off),
                    view(&sfx.add[1], off),
                );
                let (res2_2, res_2a_2) = moments(
                    view(&prefix[2], off),
                    view(&target[2], off),
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
                    lab_cands.push((err, p_idx, c_idx));
                }
            }

            // --- 3+4: exact OKLab scoring of the shortlist -------------------
            lab_cands.select_nth_unstable_by(k - 1, |a, b| {
                a.0.partial_cmp(&b.0).expect("no NaN errors")
            });
            let mut best: Option<(usize, usize)> = None;
            for &(_, p_idx, c_idx) in &lab_cands[..k] {
                composite(cand, prefix, &alphas[p_idx], c_idx, sfx, off);
                let err = lab_error(cand, lab_target, off);
                if err < current {
                    current = err;
                    best = Some((p_idx, c_idx));
                }
            }
            if let Some(pc) = best {
                solution.layers[i] = pc;
            }

            // --- the suffix for the layer below, with this layer settled -----
            let (left, right) = suffixes.split_at_mut(i + 1);
            let (p, c) = solution.layers[i];
            build_suffix(&mut left[i], &right[0], &alphas[p], c, off);
        }

        // --- 5: the base dye, which is just "the whole stack over a flat fill"
        let sfx = &suffixes[0];
        let mut best = None;
        for c_idx in 0..NUM_COLORS {
            flat_composite(cand, c_idx, sfx, off);
            let err = lab_error(cand, lab_target, off);
            if err < current {
                current = err;
                best = Some(c_idx);
            }
        }
        if let Some(c_idx) = best {
            solution.base = c_idx;
        }

        rebuild_prefixes(prefixes, off, solution, alphas, n);
        solution.error = total_error(&prefixes[n], target, off);
    }

    solution.lab_error = current / hw as f32;
}

/// The target patch, converted once per cell.
fn convert_target(target: &[Plane; 3], out: &mut [Plane; 3], off: usize) {
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

/// `Σ ΔE` between an sRGB composite and the OKLab target planes.
fn lab_error(rgb: &[Plane; 3], lab_target: &[Plane; 3], off: usize) -> f32 {
    let mut acc = F32s::ZERO;
    for (r, g, b, tl, ta, tb) in zip!(
        view(&rgb[0], off),
        view(&rgb[1], off),
        view(&rgb[2], off),
        view(&lab_target[0], off),
        view(&lab_target[1], off),
        view(&lab_target[2], off)
    ) {
        let (l, a, b) = srgb_to_oklab(r, g, b);
        let (dl, da, db) = (l - tl, a - ta, b - tb);
        acc += dl.mul_add(dl, da.mul_add(da, db * db)).sqrt();
    }
    acc.hsum()
}

/// The full banner for laying `color` through `alpha` on top of `prefix`, with
/// `suffix` applying everything above it.
fn composite(
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

/// The full banner for a flat base `color` under `suffix` — the base-dye case,
/// where there is no prefix and `α ≡ 1`.
fn flat_composite(out: &mut [Plane; 3], color: usize, suffix: &Suffix, off: usize) {
    for ch in 0..3 {
        let c = F32s::splat(COLORS_F32[color][ch]);
        for (out, mul, add) in zip!(
            mut view_mut(&mut out[ch], off),
            view(&suffix.mul, off),
            view(&suffix.add[ch], off)
        ) {
            *out = c.mul_add(mul, add);
        }
    }
}
