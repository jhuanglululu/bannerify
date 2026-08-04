//! Greedy fill: pick a base dye, then one (pattern, dye) layer at a time,
//! each time the pair that minimises the weighted SSE against the target patch.
//!
//! The layer error is linear in the dye colour, so each layer costs **one**
//! pass over the patch per pattern — reducing the residual moments `Σ res²` and
//! `2·Σ res·α` per channel — plus a scalar closed-form sweep of the 16 dyes
//! over those six numbers: `O(patterns)` passes, not `O(patterns × dyes)`.

use crate::color::{COLORS_F32, COLORS_WSQ_SUM, NUM_COLORS, W_PERCEPTUAL};
use crate::simd::F32s;
use crate::zip;

use super::workspace::{Plane, Solution, view, view_mut};

/// Greedy-fill the patch in `target`, writing the composite chain into
/// `prefixes[0..=n_layers]`.
///
/// `off`/`hw` are the cell's lane offset and pixel count; `alpha2[p]` is
/// `Σ α²` of pattern `p` over exactly that view.
pub(super) fn fill(
    target: &[Plane; 3],
    prefixes: &mut [[Plane; 3]],
    n_layers: usize,
    alphas: &[Plane],
    alpha2: &[f32],
    off: usize,
    hw: usize,
) -> Solution {
    let (base, base_err) = best_base(target, off, hw);
    for ch in 0..3 {
        let c = F32s::splat(COLORS_F32[base][ch]);
        for out in zip!(mut view_mut(&mut prefixes[0][ch], off)) {
            *out = c;
        }
    }

    let mut layers = Vec::with_capacity(n_layers);
    let mut error = base_err;

    for layer in 0..n_layers {
        let mut best = (0_usize, 0_usize);
        let mut min_err = f32::INFINITY;

        let prefix = &prefixes[layer];
        for (p, alpha) in alphas.iter().enumerate() {
            // The residual moments of laying *any* dye through this pattern on
            // top of the current composite.
            let mut res2 = [0.0_f32; 3];
            let mut res_2a = [0.0_f32; 3];
            for ch in 0..3 {
                let mut acc2 = F32s::ZERO;
                let mut acc_a = F32s::ZERO;
                for (pre, tar, alp) in zip!(
                    view(&prefix[ch], off),
                    view(&target[ch], off),
                    view(alpha, off)
                ) {
                    // res = prefix * (1 - alpha) - target
                    let res = pre.mul_add(F32s::ONE - alp, -tar);
                    acc2 = res.mul_add(res, acc2);
                    acc_a = res.mul_add(alp, acc_a);
                }
                res2[ch] = acc2.hsum();
                res_2a[ch] = 2.0 * acc_a.hsum();
            }

            // The closed form scores all 16 dyes from those six numbers.
            for c in 0..NUM_COLORS {
                let color = COLORS_F32[c];
                let err = W_PERCEPTUAL[0] * (res2[0] + res_2a[0] * color[0])
                    + W_PERCEPTUAL[1] * (res2[1] + res_2a[1] * color[1])
                    + W_PERCEPTUAL[2] * (res2[2] + res_2a[2] * color[2])
                    + alpha2[p] * COLORS_WSQ_SUM[c];
                if err < min_err {
                    best = (p, c);
                    min_err = err;
                }
            }
        }

        layers.push(best);
        error = min_err;

        // Composite the chosen layer forward, the last layer included:
        // `prefixes[n_layers]` is the image every later stage scores.
        let (done, rest) = prefixes.split_at_mut(layer + 1);
        build_prefix(&done[layer], &mut rest[0], &alphas[best.0], best.1, off);
    }

    Solution {
        base,
        layers,
        error,
        lab_error: 0.0,
    }
}

/// The base dye minimising `Σ w·(c − t)²`, and that minimum.
///
/// Expanding gives `Σ w·t² + c·(−2·Σ w·t) + HW·Σ w·c²`, so one pass over the
/// patch reduces `Σ t²` and `−2·Σ t` per channel and the 16 dyes are scored
/// from those — the same shape as the layer sweep, with `α ≡ 1`.
fn best_base(target: &[Plane; 3], off: usize, hw: usize) -> (usize, f32) {
    let mut t2 = [0.0_f32; 3];
    let mut n2t = [0.0_f32; 3];
    for ch in 0..3 {
        let mut acc2 = F32s::ZERO;
        let mut acc = F32s::ZERO;
        for tar in zip!(view(&target[ch], off)) {
            acc2 = tar.mul_add(tar, acc2);
            acc += tar;
        }
        t2[ch] = acc2.hsum();
        n2t[ch] = -2.0 * acc.hsum();
    }

    let mut base = 0;
    let mut min_err = f32::INFINITY;
    for c in 0..NUM_COLORS {
        let color = COLORS_F32[c];
        let err = W_PERCEPTUAL[0] * (t2[0] + n2t[0] * color[0])
            + W_PERCEPTUAL[1] * (t2[1] + n2t[1] * color[1])
            + W_PERCEPTUAL[2] * (t2[2] + n2t[2] * color[2])
            + hw as f32 * COLORS_WSQ_SUM[c];
        if err < min_err {
            base = c;
            min_err = err;
        }
    }
    (base, min_err)
}

/// `next = prefix · (1 − α) + color · α`, the one compositing step.
pub(super) fn build_prefix(
    prefix: &[Plane; 3],
    next: &mut [Plane; 3],
    alpha: &Plane,
    color: usize,
    off: usize,
) {
    for ch in 0..3 {
        let c = F32s::splat(COLORS_F32[color][ch]);
        for (out, pre, alp) in zip!(
            mut view_mut(&mut next[ch], off),
            view(&prefix[ch], off),
            view(alpha, off)
        ) {
            *out = pre.mul_add(F32s::ONE - alp, c * alp);
        }
    }
}
