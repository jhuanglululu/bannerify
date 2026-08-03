//! Greedy fill: pick a base dye, then one (pattern, dye) layer at a time,
//! each time the pair that minimises the weighted SSE against the target patch.
//!
//! Ported from `../bannerify-old/src/solver/fill.rs` (and its `build_prefix`)
//! onto the [`simd`](crate::simd) facade. The algebra is unchanged — see
//! [`crate::color`] for the expansion — so the only cost per layer is
//!
//! - **one** pass over the patch per pattern, reducing the two residual moments
//!   `Σ res²` and `2·Σ res·α` per channel, and
//! - a scalar closed-form sweep of the 16 dyes over those six numbers.
//!
//! i.e. `O(patterns)` passes, not `O(patterns × dyes)`. The dye sweep stays
//! scalar exactly as in the old build: it consumes already-reduced scalars, so
//! there is nothing left to vectorise.
//!
//! ## Differences from the old `fill.rs`
//!
//! 1. **The last prefix is built.** The old code skipped compositing the final
//!    layer (`if layer != n_layers - 1`) and then returned the whole prefix
//!    vector to a caller that ignored it. Here the final composite *is* the
//!    thing the preview render draws, so it is built, and prefixes are never
//!    returned — they stay inside the [`Workspace`], reachable through
//!    [`Workspace::composite`].
//! 2. **Buffers are owned by a reusable workspace** instead of a fresh
//!    `uninit!` allocation per banner: one workspace per row work item, reused
//!    across that row's cells, so the solve does no allocation beyond the
//!    per-cell layer list.
//! 3. **The final error is reported.** The dye sweep's minimum at the last
//!    layer already *is* the composite's weighted SSE, so it costs nothing to
//!    keep — the old build recomputed error elsewhere.

use crate::color::{COLORS_F32, COLORS_WSQ_SUM, NUM_COLORS, W_PERCEPTUAL};
use crate::simd::{Chunk, F32s};
use crate::zip;

/// What the solver decided for one banner cell.
#[derive(Clone, Debug)]
pub struct Solution {
    /// Index into [`crate::color::COLOR_NAMES`] of the banner's base dye.
    pub base: usize,
    /// `(pattern index, dye index)` per layer, in application order.
    pub layers: Vec<(usize, usize)>,
    /// Weighted SSE of the final composite against the target patch, summed
    /// over the solved pixels (`HW` of them) — not normalised.
    pub error: f32,
}

/// Reusable per-work-item solver buffers for one patch size.
///
/// `HW` is the patch's pixel count: [`TOP_HW`](crate::geometry::TOP_HW) (the
/// full 20×40 patch, banner row 0) or
/// [`NTOP_HW`](crate::geometry::NTOP_HW) (the visible bottom 24 rows, every
/// other row). Monomorphised on it exactly like the old build, so the reduction
/// loops have compile-time trip counts.
pub struct Workspace<const HW: usize> {
    /// The cell's target pixels, planar, filled by the caller before `solve`.
    target: [Chunk<HW>; 3],
    /// Composite after `i` layers; `prefixes[0]` is the flat base colour.
    prefixes: Vec<[Chunk<HW>; 3]>,
    /// Layers in the last `solve` — which prefix is the final composite.
    depth: usize,
}

impl<const HW: usize> Workspace<HW> {
    /// Allocate for cells of up to `max_layers` layers.
    pub fn new(max_layers: usize) -> Self {
        Self {
            target: [Chunk::zeroed(); 3],
            prefixes: vec![[Chunk::zeroed(); 3]; max_layers + 1],
            depth: 0,
        }
    }

    /// The target planes, to be overwritten with the next cell's patch.
    pub fn target_mut(&mut self) -> &mut [Chunk<HW>; 3] {
        &mut self.target
    }

    /// The composite the last [`Workspace::solve`] produced — what the preview
    /// draws. Planes are in the same row-major patch layout as the target.
    pub fn composite(&self) -> &[Chunk<HW>; 3] {
        &self.prefixes[self.depth]
    }

    /// Solve the patch currently in [`Workspace::target_mut`].
    ///
    /// `alphas`/`alpha2` are the pattern table for this patch size
    /// ([`crate::pattern`]); `n_layers` comes from the variance pre-pass and
    /// must not exceed the `max_layers` the workspace was built for.
    pub fn solve(&mut self, n_layers: usize, alphas: &[Chunk<HW>], alpha2: &[f32]) -> Solution {
        assert!(
            n_layers < self.prefixes.len(),
            "workspace built for fewer layers than requested"
        );
        debug_assert_eq!(alphas.len(), alpha2.len());

        let Self {
            target,
            prefixes,
            depth,
        } = self;
        *depth = n_layers;

        let (base, base_err) = best_base::<HW>(target);
        for ch in 0..3 {
            prefixes[0][ch].fill(COLORS_F32[base][ch]);
        }

        let mut layers = Vec::with_capacity(n_layers);
        let mut error = base_err;

        for layer in 0..n_layers {
            let mut best = (0_usize, 0_usize);
            let mut min_err = f32::INFINITY;

            let prefix = &prefixes[layer];
            for (p, alpha) in alphas.iter().enumerate() {
                // One pass over the patch: the residual moments of laying *any*
                // dye through this pattern on top of the current composite.
                let mut res2 = [0.0_f32; 3];
                let mut res_2a = [0.0_f32; 3];
                for ch in 0..3 {
                    let mut acc2 = F32s::ZERO;
                    let mut acc_a = F32s::ZERO;
                    for (pre, tar, alp) in zip!(&prefix[ch], &target[ch], alpha) {
                        // res = prefix * (1 - alpha) - target
                        let res = pre.mul_add(F32s::ONE - alp, -tar);
                        acc2 = res.mul_add(res, acc2);
                        acc_a = res.mul_add(alp, acc_a);
                    }
                    res2[ch] = acc2.hsum();
                    res_2a[ch] = 2.0 * acc_a.hsum();
                }

                // ...and the closed form scores all 16 dyes from those six.
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

            // Composite the chosen layer forward. Unlike the old build this
            // also runs for the last layer, because `prefixes[n_layers]` is the
            // image the preview render draws.
            let (done, rest) = prefixes.split_at_mut(layer + 1);
            build_prefix(&done[layer], &mut rest[0], &alphas[best.0], best.1);
        }

        Solution {
            base,
            layers,
            error,
        }
    }
}

/// The base dye minimising `Σ w·(c − t)²`, and that minimum.
///
/// Expanding gives `Σ w·t² + c·(−2·Σ w·t) + HW·Σ w·c²`, so one pass over the
/// patch reduces `Σ t²` and `−2·Σ t` per channel and the 16 dyes are scored
/// from those — the same shape as the layer sweep, with `α ≡ 1`.
fn best_base<const HW: usize>(target: &[Chunk<HW>; 3]) -> (usize, f32) {
    let mut t2 = [0.0_f32; 3];
    let mut n2t = [0.0_f32; 3];
    for ch in 0..3 {
        let mut acc2 = F32s::ZERO;
        let mut acc = F32s::ZERO;
        for tar in zip!(&target[ch]) {
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
            + HW as f32 * COLORS_WSQ_SUM[c];
        if err < min_err {
            base = c;
            min_err = err;
        }
    }
    (base, min_err)
}

/// `next = prefix · (1 − α) + color · α`, the one compositing step, ported from
/// the old build's `build_prefix`.
fn build_prefix<const HW: usize>(
    prefix: &[Chunk<HW>; 3],
    next: &mut [Chunk<HW>; 3],
    alpha: &Chunk<HW>,
    color: usize,
) {
    for ch in 0..3 {
        let c = F32s::splat(COLORS_F32[color][ch]);
        for (out, pre, alp) in zip!(mut next[ch], &prefix[ch], alpha) {
            *out = pre.mul_add(F32s::ONE - alp, c * alp);
        }
    }
}
