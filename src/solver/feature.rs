//! The per-cell **feature map**: the idealised two-layer banner the solver is
//! regularised towards.
//!
//! The solver reconstructs a cell at pixel scale, and at pixel scale a flat red
//! region is served just as well by a gradient laid over an orange as by a flat
//! red — the two average to the same thing. They do not *look* the same: the
//! first is muddy. So a second, deliberately impoverished target is built next
//! to the real one and scored alongside it: every pixel snapped to one of the 16
//! dyes, then the best banner of **at most two layers** fitted to that. Two
//! layers is the regulariser — it cannot represent a mixture, only clean colour
//! regions with crisp pattern edges, which is exactly the bias wanted.
//!
//! ## The search
//!
//! For an ordered pattern pair — `p₁` laid first, `p₂` over it — each pixel's
//! colour is a convex combination of the three slots (base, layer 1, layer 2)
//! with weights
//!
//! ```text
//! w₀ = (1−α₁)(1−α₂)      w₁ = α₁(1−α₂)      w₂ = α₂
//! ```
//!
//! Score a candidate by how much of that weight lands on the pixel's snapped
//! dye. Then the total is a sum over slots of independent terms, so
//!
//! ```text
//! score(p₁, d₁, p₂, d₂) = Σ_slots Σ_pixels w_slot · [snap == d_slot]
//!                       = Σ_slots h_slot[d_slot]
//! ```
//!
//! where `h_slot` is a 16-bin histogram of that slot's weight. **The three dyes
//! are therefore independent argmaxes**, and a pair costs one `O(HW)` pass plus
//! `3 × 16` scalar comparisons — no dye loop at all.
//!
//! ## What the pass actually reduces
//!
//! Expanding the weights, every histogram is a combination of quantities that do
//! not depend on the pair:
//!
//! ```text
//! h₂[d] = A₂[d]                            A_p[d] = Σ_{snap == d} α_p
//! h₁[d] = A₁[d] − X[d]                     N[d]   = Σ_{snap == d} 1
//! h₀[d] = N[d] − A₁[d] − A₂[d] + X[d]      X[d]   = Σ_{snap == d} α₁·α₂
//! ```
//!
//! `N` and the `A_p` are reduced once per cell (`P + 1` passes). Only the cross
//! term `X` is genuinely per-pair, so the inner loop is a single multiply and a
//! scattered add — no compositing, no colour arithmetic. It is scalar because
//! the accumulator is indexed by the pixel's dye; a lane-wise version would need
//! a scatter no backend here has.
//!
//! The degenerate banners (base only, base + one layer) fall out of the same
//! algebra with `α ≡ 0` for the missing pattern, and they are enumerated too, so
//! a flat cell gets a flat map instead of an arbitrary pair fitted to noise.
//!
//! ## Pruning
//!
//! `X ≥ 0` and `X[d] ≤ min(A₁[d], A₂[d])` pointwise, so
//! `max h₁ ≤ max A₁` and `max h₀ ≤ max (N − A₂)`: a pair's score is bounded
//! above by three numbers already known before its pass runs, so a pair that
//! cannot beat the incumbent is skipped without touching a pixel. Both loops
//! run in descending order of their half of that bound, which both finds a
//! strong incumbent immediately and turns the skip into an early exit —
//! measured on the 42-pattern table, 290–1250 of the 1807 candidate banners are
//! actually scored. That is what keeps the `O(P²·HW)` shape affordable.

use crate::color::{COLORS_F32, NUM_COLORS, snap_index, snap_lut};
use crate::geometry::TOP_HW;
use crate::simd::F32s;
use crate::zip;

use super::refine::convert_target;
use super::workspace::{Plane, Workspace, view, view_mut};

/// The winning idealised banner: a base dye and up to two `(pattern, dye)`
/// layers, in application order (`layers[0]` is laid first).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Map {
    /// Index into [`crate::color::COLOR_NAMES`].
    pub base: usize,
    /// `Some((pattern, dye))` per slot; `None` where the winner is degenerate.
    pub layers: [Option<(usize, usize)>; 2],
}

/// Per-cell scratch for the search. Sized on first use and reused down the
/// column, like every other workspace buffer.
#[derive(Default)]
pub(super) struct Buffers {
    /// Nearest dye per pixel; only the cell's active view is meaningful.
    snap: Vec<u8>,
    /// `A_p[d]`, row-major by pattern.
    mass: Vec<f32>,
    /// `(argmax_d A_p[d], that maximum)` per pattern — the layer-2 term, and
    /// the pruning bound on the layer-1 term.
    top: Vec<(usize, f32)>,
    /// `max_d (N[d] − A_p[d])` per pattern — the pruning bound on the base term
    /// when `p` is the upper pattern.
    base_bound: Vec<f32>,
    /// Pattern indices, descending by their bound as the upper pattern.
    upper_order: Vec<usize>,
    /// Pattern indices, descending by their bound as the lower pattern.
    lower_order: Vec<usize>,
    /// `N[d]`.
    count: [f32; NUM_COLORS],
}

/// Build the feature map for the patch currently in the workspace's target,
/// leaving it composited in [`Workspace::feature`] (sRGB) and
/// [`Workspace::lab_feature`] (OKLab), and **blending the sRGB target planes
/// towards it** ([`blend_target`]). Returns the banner it chose.
///
/// Only called with `lambda > 0`: at zero the whole pass is skipped, which is
/// what makes the feature map a true no-op rather than a cheap one.
pub(super) fn build(ws: &mut Workspace, alphas: &[Plane], lambda: f32) -> Map {
    let Workspace {
        off,
        hw,
        target,
        feature,
        lab_feature,
        feat,
        ..
    } = ws;
    let off = *off;
    // The active view starts exactly where the lane view does.
    let start = TOP_HW - *hw;

    snap(feat, target, start);
    reduce(feat, alphas, start);

    let map = search(feat, alphas, start);

    paint(feature, &map, alphas, off);
    convert_target(feature, lab_feature, off);
    blend_target(target, feature, lambda, off);

    map
}

/// Fold the feature term into the sRGB target, **in place**.
///
/// The closed-form rungs minimise a weighted SSE, and
/// `Σ w·(c−t)² + λ·Σ w·(c−f)²` differs from `(1+λ)·Σ w·(c−b)²` — where
/// `b = (t + λ·f)/(1+λ)` — only by a term with no `c` in it. So the whole
/// feature term reaches the greedy fill, refinement's rung 1 and the closed-form
/// base sweep by changing three planes, and none of the moment kernels need to
/// know it exists.
///
/// In place rather than in a fourth buffer: by the time this runs the target has
/// already been converted to OKLab for the exact rung, compositing never reads
/// the target at all, and the caller overwrites the planes for every cell — so
/// the pure sRGB target has no reader left. A separate buffer would cost another
/// 9.6 KB per worker and force every closed-form call site to choose between two
/// planes at runtime.
fn blend_target(target: &mut [Plane; 3], feature: &[Plane; 3], lambda: f32, off: usize) {
    let lam = F32s::splat(lambda);
    let inv = F32s::splat(1.0 / (1.0 + lambda));
    for (ch, plane) in target.iter_mut().enumerate() {
        for (t, f) in zip!(mut view_mut(plane, off), view(&feature[ch], off)) {
            *t = f.mul_add(lam, *t) * inv;
        }
    }
}

/// Quantise every pixel of the active view to its nearest dye.
fn snap(feat: &mut Buffers, target: &[Plane; 3], start: usize) {
    let lut = snap_lut();
    feat.snap.resize(TOP_HW, 0);
    for (i, out) in feat.snap.iter_mut().enumerate().skip(start) {
        *out = lut[snap_index(target[0][i], target[1][i], target[2][i])];
    }
}

/// Reduce the pair-independent tables: `N`, every `A_p`, and the two bounds
/// derived from them.
fn reduce(feat: &mut Buffers, alphas: &[Plane], start: usize) {
    let snap = &feat.snap[start..];

    feat.count = [0.0; NUM_COLORS];
    for &d in snap {
        feat.count[d as usize] += 1.0;
    }

    feat.mass.clear();
    feat.mass.resize(alphas.len() * NUM_COLORS, 0.0);
    feat.top.clear();
    feat.base_bound.clear();

    for (p, alpha) in alphas.iter().enumerate() {
        let mass = &mut feat.mass[p * NUM_COLORS..][..NUM_COLORS];
        for (&d, &a) in snap.iter().zip(&alpha[start..]) {
            mass[d as usize] += a;
        }
        feat.top.push(argmax(mass));
        feat.base_bound.push(
            (0..NUM_COLORS)
                .map(|d| feat.count[d] - mass[d])
                .fold(f32::NEG_INFINITY, f32::max),
        );
    }

    // Descending by each pattern's share of the bound, so the pair loop meets
    // its strongest candidates first and its prune turns into an early exit.
    // Ties break on the pattern index: the enumeration order must not depend on
    // the sort's tie-breaking.
    let order = |feat: &Buffers, key: fn(&Buffers, usize) -> f32| {
        let mut order: Vec<usize> = (0..alphas.len()).collect();
        order.sort_unstable_by(|&x, &y| key(feat, y).total_cmp(&key(feat, x)).then(x.cmp(&y)));
        order
    };
    feat.upper_order = order(feat, |f, p| f.base_bound[p] + f.top[p].1);
    feat.lower_order = order(feat, |f, p| f.top[p].1);
}

/// Exhaustively score every ordered pattern pair, the degenerate banners
/// included, and return the best.
fn search(feat: &Buffers, alphas: &[Plane], start: usize) -> Map {
    let snap = &feat.snap[start..];

    // Base only: every pixel votes with weight 1 for its own dye.
    let (base, score) = argmax(&feat.count);
    let mut best = (
        score,
        Map {
            base,
            layers: [None, None],
        },
    );

    // One layer: `h₀ = N − A`, `h₁ = A`.
    for p in 0..alphas.len() {
        let mass = &feat.mass[p * NUM_COLORS..][..NUM_COLORS];
        let (base, base_score) = argmax_sub(&feat.count, mass);
        let (dye, dye_score) = feat.top[p];
        if base_score + dye_score > best.0 {
            best = (
                base_score + dye_score,
                Map {
                    base,
                    layers: [Some((p, dye)), None],
                },
            );
        }
    }

    // Two layers. The upper pattern is the outer loop: its `A₂` term, its dye
    // and its share of the bound are then hoisted out of the inner one. Both
    // loops run in descending bound order, so the first pair that cannot beat
    // the incumbent ends the loop rather than skipping one pair — a pruned
    // *tie* can change which of two equal-scoring maps is returned, nothing
    // else.
    let best_lower = feat.lower_order.first().map_or(0.0, |&p| feat.top[p].1);
    for &upper in &feat.upper_order {
        let mass2 = &feat.mass[upper * NUM_COLORS..][..NUM_COLORS];
        let (dye2, score2) = feat.top[upper];
        let bound2 = feat.base_bound[upper] + score2;
        if bound2 + best_lower <= best.0 {
            break;
        }
        let alpha2 = &alphas[upper][start..];

        for &lower in &feat.lower_order {
            if bound2 + feat.top[lower].1 <= best.0 {
                break;
            }
            let cross = cross(snap, &alphas[lower][start..], alpha2);

            let mass1 = &feat.mass[lower * NUM_COLORS..][..NUM_COLORS];
            let mut best_base = (0, f32::NEG_INFINITY);
            let mut best_dye1 = (0, f32::NEG_INFINITY);
            for d in 0..NUM_COLORS {
                let h1 = mass1[d] - cross[d];
                let h0 = feat.count[d] - mass1[d] - mass2[d] + cross[d];
                if h0 > best_base.1 {
                    best_base = (d, h0);
                }
                if h1 > best_dye1.1 {
                    best_dye1 = (d, h1);
                }
            }

            let score = best_base.1 + best_dye1.1 + score2;
            if score > best.0 {
                best = (
                    score,
                    Map {
                        base: best_base.0,
                        layers: [Some((lower, best_dye1.0)), Some((upper, dye2))],
                    },
                );
            }
        }
    }

    best.1
}

/// Interleaved accumulator sets of the cross term. Neighbouring pixels usually
/// snap to the *same* dye, which would serialise a single accumulator array on
/// the store-to-load latency of one bin. Independent sets, summed at the end,
/// give the dependency chains room to overlap. Measured on the search overall:
/// 4 ways 3.0×, 8 ways 3.4×, 16 ways no further gain.
const CROSS_WAYS: usize = 8;

/// `X[d] = Σ_{snap == d} α₁·α₂` — the one genuinely per-pair reduction, and the
/// search's entire hot loop.
fn cross(snap: &[u8], alpha1: &[f32], alpha2: &[f32]) -> [f32; NUM_COLORS] {
    // Both patch sizes are multiples of 16, so `chunks_exact` never drops a
    // remainder and the tail path this function does not have is unreachable.
    debug_assert!(snap.len().is_multiple_of(CROSS_WAYS));

    let mut ways = [[0.0_f32; NUM_COLORS]; CROSS_WAYS];
    let chunks = snap
        .chunks_exact(CROSS_WAYS)
        .zip(alpha1.chunks_exact(CROSS_WAYS))
        .zip(alpha2.chunks_exact(CROSS_WAYS));
    for ((snap, a1), a2) in chunks {
        for (way, ((&d, &a1), &a2)) in ways.iter_mut().zip(snap.iter().zip(a1).zip(a2)) {
            way[d as usize] += a1 * a2;
        }
    }

    let mut out = ways[0];
    for way in &ways[1..] {
        for (out, w) in out.iter_mut().zip(way) {
            *out += w;
        }
    }
    out
}

/// Composite `map` into `out`, in sRGB — the same blend the game does, so the
/// feature map is a banner a player could actually craft.
fn paint(out: &mut [Plane; 3], map: &Map, alphas: &[Plane], off: usize) {
    for (ch, plane) in out.iter_mut().enumerate() {
        let c = F32s::splat(COLORS_F32[map.base][ch]);
        for o in zip!(mut view_mut(plane, off)) {
            *o = c;
        }
    }
    for &(p, dye) in map.layers.iter().flatten() {
        for (ch, plane) in out.iter_mut().enumerate() {
            let c = F32s::splat(COLORS_F32[dye][ch]);
            for (o, alp) in zip!(mut view_mut(plane, off), view(&alphas[p], off)) {
                *o = o.mul_add(F32s::ONE - alp, c * alp);
            }
        }
    }
}

/// `(argmax, max)` of a 16-bin histogram, ties going to the lower dye index.
fn argmax(bins: &[f32]) -> (usize, f32) {
    let mut best = (0, f32::NEG_INFINITY);
    for (d, &v) in bins.iter().enumerate() {
        if v > best.1 {
            best = (d, v);
        }
    }
    best
}

/// [`argmax`] of `a − b`, without materialising the difference.
fn argmax_sub(a: &[f32], b: &[f32]) -> (usize, f32) {
    let mut best = (0, f32::NEG_INFINITY);
    for (d, (&x, &y)) in a.iter().zip(b).enumerate() {
        if x - y > best.1 {
            best = (d, x - y);
        }
    }
    best
}


#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::color::COLOR_NAMES;
    use crate::geometry::{BANNER_W, NTOP_HW};
    use crate::pattern::{Patterns, load};

    /// Any non-zero weight: the tests below assert on the map and the feature
    /// planes, neither of which the weight touches — it only scales the blend
    /// written back into the target.
    const LAMBDA: f32 = 0.5;

    fn patterns() -> Patterns {
        load(&HashSet::new())
    }

    fn index_of(pats: &Patterns, name: &str) -> usize {
        pats.names.iter().position(|n| n == name).unwrap()
    }

    fn dye(name: &str) -> usize {
        COLOR_NAMES.iter().position(|n| *n == name).unwrap()
    }

    /// A workspace on banner row `row` whose target holds `fill(pixel)`.
    fn workspace(row: usize, fill: impl Fn(usize) -> [f32; 3]) -> Workspace {
        let mut ws = Workspace::new(2, 4);
        ws.begin(row);
        let target = ws.target_mut();
        for i in 0..TOP_HW {
            let px = fill(i);
            for (ch, plane) in target.iter_mut().enumerate() {
                plane[i] = px[ch];
            }
        }
        ws
    }

    /// Composite a map the way the game would — the synthetic targets below are
    /// built with this.
    fn composite(map: &Map, alphas: &[Plane], i: usize) -> [f32; 3] {
        let mut px = COLORS_F32[map.base];
        for &(p, dye) in map.layers.iter().flatten() {
            let a = alphas[p][i];
            for (ch, v) in px.iter_mut().enumerate() {
                *v = *v * (1.0 - a) + COLORS_F32[dye][ch] * a;
            }
        }
        px
    }

    /// The objective straight from its definition: per-pixel slot weights, each
    /// added to its slot's dye when the pixel snapped to that dye.
    ///
    /// [`search`] computes the same number through a decomposition that never
    /// touches a slot weight; this is the independent implementation the tests
    /// below compare against.
    fn naive_score(map: &Map, snap: &[u8], alphas: &[Plane], start: usize) -> f32 {
        let alpha = |slot: usize, i: usize| map.layers[slot].map_or(0.0, |(p, _)| alphas[p][i]);
        let mut total = 0.0;
        for (i, &d) in snap.iter().enumerate().skip(start) {
            let d = usize::from(d);
            let (a1, a2) = (alpha(0, i), alpha(1, i));
            if map.base == d {
                total += (1.0 - a1) * (1.0 - a2);
            }
            if map.layers[0].is_some_and(|(_, dye)| dye == d) {
                total += a1 * (1.0 - a2);
            }
            if map.layers[1].is_some_and(|(_, dye)| dye == d) {
                total += a2;
            }
        }
        total
    }

    /// Every banner the search may return, for a small pattern table.
    fn all_maps(n_patterns: usize) -> Vec<Map> {
        let slots: Vec<Option<usize>> = std::iter::once(None)
            .chain((0..n_patterns).map(Some))
            .collect();
        let mut out = Vec::new();
        for base in 0..NUM_COLORS {
            for &p1 in &slots {
                for &p2 in &slots {
                    for d1 in 0..if p1.is_some() { NUM_COLORS } else { 1 } {
                        for d2 in 0..if p2.is_some() { NUM_COLORS } else { 1 } {
                            out.push(Map {
                                base,
                                layers: [p1.map(|p| (p, d1)), p2.map(|p| (p, d2))],
                            });
                        }
                    }
                }
            }
        }
        out
    }

    /// The decomposed `O(P²·HW)` search finds the true optimum: on a small
    /// pattern table its winner scores what an exhaustive sweep over every
    /// `(base, p₁, d₁, p₂, d₂)` scores, both measured by [`naive_score`].
    ///
    /// The target is deliberately off-palette and non-flat, so the snapped image
    /// is a mix of dyes and no candidate is trivially perfect.
    #[test]
    fn the_search_finds_the_brute_force_optimum() {
        let pats = patterns();
        let alphas = [
            pats.top[index_of(&pats, "half_vertical")],
            pats.top[index_of(&pats, "stripe_bottom")],
            pats.top[index_of(&pats, "gradient")],
        ];

        // A diagonal ramp through three off-palette colours.
        let mut ws = workspace(1, |i| {
            let (x, y) = ((i % BANNER_W) as f32, (i / BANNER_W) as f32);
            let t = (x / 19.0 + y / 39.0) * 0.5;
            [30.0 + 200.0 * t, 90.0 + 60.0 * t, 200.0 - 150.0 * t]
        });

        let map = build(&mut ws, &alphas, LAMBDA);

        let start = TOP_HW - NTOP_HW;
        let snap = ws.feat.snap.clone();
        let got = naive_score(&map, &snap, &alphas, start);
        let best = all_maps(alphas.len())
            .iter()
            .map(|m| naive_score(m, &snap, &alphas, start))
            .fold(f32::NEG_INFINITY, f32::max);

        assert!(
            (got - best).abs() < 0.05,
            "search scored {got}, brute force {best}; map = {map:?}"
        );
    }

    /// A patch that *is* a two-layer banner: the search must not do worse than
    /// the banner it was drawn from. Asserted on the score and not on the layer
    /// list, because a tie is perfectly possible — two patterns can cover the
    /// same pixels — and a tie is not a failure.
    #[test]
    fn a_planted_two_layer_banner_is_not_beaten_by_the_search() {
        let pats = patterns();
        let planted = Map {
            base: dye("white"),
            layers: [
                Some((index_of(&pats, "half_vertical"), dye("blue"))),
                Some((index_of(&pats, "stripe_bottom"), dye("red"))),
            ],
        };

        let mut ws = workspace(0, |i| composite(&planted, &pats.top, i));
        let map = build(&mut ws, &pats.top, LAMBDA);

        let snap = ws.feat.snap.clone();
        let got = naive_score(&map, &snap, &pats.top, 0);
        let planted_score = naive_score(&planted, &snap, &pats.top, 0);
        assert!(
            got >= planted_score - 0.05,
            "search scored {got}, the planted banner {planted_score}; map = {map:?}"
        );
        // The planted banner is drawn in dye colours, so all but its blended
        // edge pixels snap back to themselves: the optimum is near-perfect.
        assert!(
            got > 0.95 * TOP_HW as f32,
            "score {got} of {TOP_HW}; map = {map:?}"
        );
    }

    /// A lower row solves the tail of its planes, so the head — which holds
    /// whatever the previous cell left there — must not reach the result.
    #[test]
    fn a_lower_rows_hidden_head_does_not_reach_the_map() {
        let pats = patterns();
        let visible = Map {
            base: dye("black"),
            layers: [Some((index_of(&pats, "half_vertical"), dye("yellow"))), None],
        };
        let start = TOP_HW - NTOP_HW;

        let of_head = |head: [f32; 3]| {
            let mut ws = workspace(1, |i| {
                if i < start {
                    head
                } else {
                    composite(&visible, &pats.top, i)
                }
            });
            let map = build(&mut ws, &pats.top, LAMBDA);
            (map, ws.feature[0][start..].to_vec())
        };

        let (map, pixels) = of_head(COLORS_F32[dye("lime")]);
        let (other_map, other_pixels) = of_head(COLORS_F32[dye("magenta")]);

        assert_eq!(map, other_map);
        assert_eq!(pixels, other_pixels);
    }

    /// A flat patch gets a flat map. The winner is not unique — any banner whose
    /// slots all carry the same dye composites to the same flat colour — so the
    /// assertion is on the pixels, not on the layer list.
    #[test]
    fn a_flat_patch_yields_a_flat_map() {
        let pats = patterns();
        let red = dye("red");
        let mut ws = workspace(0, |_| COLORS_F32[red]);

        let map = build(&mut ws, &pats.top, LAMBDA);

        assert_eq!(map.base, red, "map = {map:?}");
        for (ch, plane) in ws.feature.iter().enumerate() {
            assert!(
                plane.iter().all(|v| *v == COLORS_F32[red][ch]),
                "channel {ch} is not flat red; map = {map:?}"
            );
        }
    }

    /// The same for a colour nowhere near a dye: a flat mid grey must still come
    /// out flat, in whichever grey the OKLab snap picked.
    #[test]
    fn an_off_palette_flat_patch_snaps_to_one_dye() {
        let pats = patterns();
        let mut ws = workspace(0, |_| [120.0, 122.0, 118.0]);

        let map = build(&mut ws, &pats.top, LAMBDA);

        assert!(
            map.base == dye("gray") || map.base == dye("light_gray"),
            "mid grey mapped to {}",
            COLOR_NAMES[map.base]
        );
        for plane in &ws.feature {
            assert!(plane.iter().all(|v| *v == plane[0]));
        }
    }
}
