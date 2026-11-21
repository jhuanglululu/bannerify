//! The per-cell **feature map**: the idealised banner of a few layers the
//! solver is regularised towards.
//!
//! The solver reconstructs a cell at pixel scale, and at pixel scale a flat red
//! region is served just as well by a gradient laid over an orange as by a flat
//! red — the two average to the same thing. They do not *look* the same: the
//! first is muddy. So a second, deliberately impoverished target is built next
//! to the real one and scored alongside it: every pixel snapped to one of the 16
//! dyes, then the best banner of **at most `N` layers** fitted to that, `N`
//! being `--feature-layers` (2, 3 or 4; default 3). The low layer count is the
//! regulariser — such a banner cannot represent a mixture, only clean colour
//! regions with crisp pattern edges, which is exactly the bias wanted. Each
//! extra layer fits more of the patch and makes the search cost more.
//!
//! The search is exact for every `N`: it returns a banner with the highest
//! score (below), and its pruning only skips candidates that provably cannot
//! beat the best one found so far.
//!
//! ## The objective
//!
//! For patterns `p₁ … pₙ`, `p₁` laid first, each pixel's colour is a convex
//! combination of the `n + 1` slots (base, layer 1, …, layer `n`) with weights
//!
//! ```text
//! w₀ = Π_j (1−α_j)          w_k = α_k · Π_{j>k} (1−α_j)
//! ```
//!
//! A candidate's score is the weight landing on each pixel's snapped dye,
//! `Σ_slots h_slot[d_slot]`, `h_slot` being the 16-bin histogram of that slot's
//! weight, so every slot's dye is an independent argmax. Two facts carry all
//! the searches below:
//!
//! - A layer's weight depends on its own pattern and on the patterns *above*
//!   it, never on those below. The base weight depends on the set of patterns
//!   only, not on their order.
//! - Adding a pattern to a set only lowers, pixel by pixel, the weight of every
//!   slot already there (it multiplies it by some `1 − α`). So every histogram
//!   of a subset bounds the same histogram of the larger set from above, bin by
//!   bin, and so does its maximum.
//!
//! ## Two layers
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
//!
//! ## Three layers
//!
//! Patterns `i` under `j` under `k`, with `X_ab = Σ α_a·α_b` and
//! `T = Σ α_i·α_j·α_k` per dye:
//!
//! ```text
//! h₃ = A_k
//! h₂ = A_j − X_jk
//! h₁ = A_i − X_ij − X_ik + T
//! h₀ = N − A_i − A_j − A_k + X_ij + X_ik + X_jk − T
//! ```
//!
//! First one pass per unordered pair, the diagonal included, tabulates every
//! `X` and the pair's best two-layer score in either order; that also scores
//! every banner of up to two layers exactly. `h₀` does not depend on the
//! order, `h₁` only on which pattern is at the bottom, and the upper two are a
//! pair already tabulated, so one pass computing `T` scores all six orders of
//! an unordered triple.
//!
//! A triple is bounded before its pass in two steps. From the pair tables
//! alone: its base term is at most the smallest of its three pairs' base
//! terms, and its bottom layer's term at most the smallest of that pattern's
//! terms under each of the other two. Every triple beating the incumbent on
//! that bound is sorted by it, descending, and the loop ends at the first one
//! that does not. Then per dye: `T ≤ min X` and `T ≥ max(0, X_ab + X_ac − A_a)`
//! bound `h₀` and `h₁` bin by bin, which skips most of the remaining passes.
//!
//! ## Four layers
//!
//! A four-layer banner is a bottom pattern `q` under an ordered upper triple
//! `S`. The three upper slots are scored exactly as the three layers of `S`
//! alone, and the base and bottom histograms split `S`'s base histogram:
//! with `r = Π_{s∈S} (1−α_s)` per pixel,
//!
//! ```text
//! h_q = Σ r·α_q          h₀ = h₀(S) − h_q
//! ```
//!
//! so once `S` has had its triple pass, each bottom costs one pass of the same
//! shape as the pair pass. Two bounds keep that from running for every
//! `(q, S)`:
//!
//! - Per triple: `max h₀ + max h_q` is at most the sum of the two largest bins
//!   of `h₀(S)` (both histograms lie under `h₀(S)`; on one dye they share it,
//!   on two they take at most its two largest bins). The triple candidates are
//!   bounded, sorted and cut as for three layers with that sum in place of the
//!   largest base bin, so a triple that cannot top a winning four-layer banner
//!   is never passed; one that is passed is scored as a three-layer banner
//!   too.
//! - Per bottom, first from the scalar tables (`q`'s term under any one
//!   pattern of `S`, the base term of any pair `{q, s}`), then per dye:
//!
//!   ```text
//!   h_q ≤ min(h₀(S), A_q − X_qs)                         for each s ∈ S
//!   h_q ≥ max(0, A_q − Σ_s X_qs, A_q + h₀(S) − N)
//!   h₀  ≤ min(h₀(S) − that lower bound, N − A_q − A_s + X_qs)
//!   ```
//!
//!   The two maxima then share one bin of `h₀(S)` if they fall on the same
//!   dye, and take one upper bound each if not. On `docs/example.png` at 40
//!   rows this leaves about 8900 of the up to `42 × 7000` bottom passes per
//!   cell, against 7400 when the final optimum is known from the start.
//!
//! The search then never needs a table over pattern quadruples.

use std::ops::RangeInclusive;

use crate::color::{COLORS_F32, NUM_COLORS, snap_index, snap_lut};
use crate::geometry::TOP_HW;
use crate::simd::F32s;
use crate::zip;

use super::refine::convert_target;
use super::workspace::{Plane, Workspace, view, view_mut};

/// The most layers a feature map can have.
pub const MAX_LAYERS: usize = 4;

/// The layer counts `--feature-layers` accepts.
pub const LAYER_COUNTS: RangeInclusive<usize> = 2..=MAX_LAYERS;

/// The winning idealised banner: a base dye and up to [`MAX_LAYERS`]
/// `(pattern, dye)` layers, in application order (`layers[0]` is laid first).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Map {
    /// Index into [`crate::color::COLOR_NAMES`].
    pub base: usize,
    /// `Some((pattern, dye))` per slot, the used slots first; `None` past the
    /// layer count asked for and where the winner has fewer layers.
    pub layers: [Option<(usize, usize)>; MAX_LAYERS],
}

/// One 16-bin histogram.
type Bins = [f32; NUM_COLORS];

/// A banner of three or four layers' search by its patterns only, bottom
/// first: its dyes are the slot argmaxes, found once for the winner
/// ([`dyes`]).
#[derive(Clone, Copy, Debug)]
struct Stack {
    patterns: [usize; MAX_LAYERS],
    /// Layers used, `0..=MAX_LAYERS`.
    n: usize,
}

impl Stack {
    fn new(patterns: &[usize]) -> Self {
        let mut out = [0; MAX_LAYERS];
        out[..patterns.len()].copy_from_slice(patterns);
        Self {
            patterns: out,
            n: patterns.len(),
        }
    }
}

/// An unordered triple after its pixel pass.
struct Triple {
    /// The base histogram, `h₀`.
    base: Bins,
    /// The three layers' terms in their best order.
    upper: f32,
    /// That order, bottom first.
    order: [usize; 3],
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
    /// Three and four layers: `X_ab` per unordered pair `a ≤ b`, indexed by
    /// [`tri`].
    pair_x: Vec<Bins>,
    /// `P × P`, symmetric: `max_d (N − A_a − A_b + X_ab)`, the pair's base
    /// term.
    pair_base: Vec<f32>,
    /// `P × P`, symmetric: the sum of the two largest bins of that base
    /// histogram.
    pair_base2: Vec<f32>,
    /// `P × P`: `[a·P + b] = max_d (A_a − X_ab)`, the term of `a` laid under
    /// `b`.
    pair_lower: Vec<f32>,
    /// `P × P`, symmetric: the pair's two layer terms, in its better order.
    pair_upper: Vec<f32>,
    /// `(bound, code)` of every triple still in the running, descending by
    /// bound ([`triples`]).
    order: Vec<(f32, u32)>,
    /// Four layers: `Π (1−α)` over the upper triple, per pixel of the view.
    rest: Vec<f32>,
}

/// Build the feature map for the patch currently in the workspace's target,
/// leaving it composited in [`Workspace::feature`] (sRGB) and
/// [`Workspace::lab_feature`] (OKLab), and **blending the sRGB target planes
/// towards it** ([`blend_target`]). Returns the banner it chose, of at most
/// `layers` layers (one of [`LAYER_COUNTS`]).
///
/// Only called with `lambda > 0`: at zero the whole pass is skipped, which is
/// what makes the feature map a true no-op rather than a cheap one.
pub(crate) fn build(ws: &mut Workspace, alphas: &[Plane], lambda: f32, layers: usize) -> Map {
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

    let map = if layers == 2 {
        search2(feat, alphas, start, f32::NEG_INFINITY)
            .expect("every banner beats -inf")
            .1
    } else {
        let (_, stack) = search_deep(feat, alphas, start, layers, f32::NEG_INFINITY)
            .expect("every banner beats -inf");
        dyes(feat, alphas, start, &stack)
    };

    paint(feature, &map, alphas, off);
    convert_target(feature, lab_feature, off);
    blend_target(target, feature, lambda, off);

    map
}

/// The search for banners of at most `layers` layers (one of
/// [`LAYER_COUNTS`]) on `target`'s active view (from element `start`), started
/// from an incumbent score of `floor` carried over from another patch of the
/// same size: the best banner's score if one beats `floor` strictly (at three
/// and four layers, by more than the rounding allowance [`TIE`]), else `None`.
///
/// The floor takes part in the pruning, so a patch that cannot beat it skips
/// most of the pixel passes. At two layers that leaves little more than the
/// snap and the pair-independent tables; at three and four the pair tables are
/// always built. Used by the per-cell shift ([`super::shift`]), which compares
/// scores across patches.
pub(super) fn best_score(
    feat: &mut Buffers,
    target: &[Plane; 3],
    alphas: &[Plane],
    start: usize,
    layers: usize,
    floor: f32,
) -> Option<f32> {
    snap(feat, target, start);
    reduce(feat, alphas, start);
    if layers == 2 {
        search2(feat, alphas, start, floor).map(|(score, _)| score)
    } else {
        search_deep(feat, alphas, start, layers, floor).map(|(score, _)| score)
    }
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

/// The two-layer search: exhaustively score every ordered pattern pair, the
/// degenerate banners included, and return the best with its score — if it
/// scores strictly above `floor`, the incumbent to beat; `None` otherwise.
fn search2(feat: &Buffers, alphas: &[Plane], start: usize, floor: f32) -> Option<(f32, Map)> {
    let snap = &feat.snap[start..];
    let mut best = (floor, None);

    // Base only: every pixel votes with weight 1 for its own dye.
    let (base, score) = argmax(&feat.count);
    if score > best.0 {
        best = (
            score,
            Some(Map {
                base,
                layers: [None; MAX_LAYERS],
            }),
        );
    }

    // One layer: `h₀ = N − A`, `h₁ = A`.
    for p in 0..alphas.len() {
        let mass = &feat.mass[p * NUM_COLORS..][..NUM_COLORS];
        let (base, base_score) = argmax_sub(&feat.count, mass);
        let (dye, dye_score) = feat.top[p];
        if base_score + dye_score > best.0 {
            best = (
                base_score + dye_score,
                Some(Map {
                    base,
                    layers: [Some((p, dye)), None, None, None],
                }),
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
                    Some(Map {
                        base: best_base.0,
                        layers: [Some((lower, best_dye1.0)), Some((upper, dye2)), None, None],
                    }),
                );
            }
        }
    }

    best.1.map(|map| (best.0, map))
}

/// The three- or four-layer search ([`MAX_LAYERS`] at most): the best banner
/// of at most `layers` layers by its patterns, with its score, if it scores
/// more than [`TIE`] above `floor`; `None` otherwise. [`reduce`] must have run.
fn search_deep(
    feat: &mut Buffers,
    alphas: &[Plane],
    start: usize,
    layers: usize,
    floor: f32,
) -> Option<(f32, Stack)> {
    debug_assert!(matches!(layers, 3 | 4));
    let four = layers == 4;
    let p = alphas.len();
    pairs(feat, alphas, start);

    let mut best = Best {
        score: floor,
        stack: None,
    };
    shallow(feat, p, &mut best);
    triples(feat, p, best.bar(), four);

    // `order` is only read below; the passes write other fields.
    let order = std::mem::take(&mut feat.order);
    for &(bound, code) in &order {
        if bound <= best.bar() {
            break;
        }
        let ijk = decode(code, p);
        let (upper, top1, top2) = triple_bound(feat, p, ijk);
        let bound = upper + top1 + if four { top2 } else { 0.0 };
        if bound <= best.bar() {
            continue;
        }

        let t = triple(feat, alphas, start, p, ijk);
        let (base, second) = top_two(&t.base);
        best.offer(base + t.upper, &t.order);
        if four && base + second + t.upper > best.bar() {
            bottoms(feat, alphas, start, p, &t, &mut best);
        }
    }
    feat.order = order;

    best.stack.map(|stack| (best.score, stack))
}

/// Rounding allowance of the three- and four-layer searches, in pixels of
/// weight: a candidate replaces the incumbent only if it scores more than this
/// above it. Their scores are sums of up to a dozen tabulated terms, so an exact
/// tie — a flat patch scored by its base alone and by layers repeating the
/// base dye — can come out a few ten-thousandths apart either way; the
/// allowance keeps the candidate met first (fewer layers, lower indices), and
/// costs at most a hundredth of a pixel of the optimum.
const TIE: f32 = 1e-2;

/// The incumbent of the three- and four-layer searches.
struct Best {
    score: f32,
    stack: Option<Stack>,
}

impl Best {
    /// What a candidate has to score strictly above to replace it.
    fn bar(&self) -> f32 {
        self.score + TIE
    }

    fn offer(&mut self, score: f32, patterns: &[usize]) {
        if score > self.bar() {
            *self = Self {
                score,
                stack: Some(Stack::new(patterns)),
            };
        }
    }
}

/// Index of the unordered pair `{a, b}` in [`Buffers::pair_x`].
fn tri(a: usize, b: usize) -> usize {
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    hi * (hi + 1) / 2 + lo
}

/// `(i, j, k)` of a triple code from [`triples`].
fn decode(code: u32, p: usize) -> [usize; 3] {
    let c = code as usize;
    [c / (p * p), c / p % p, c % p]
}

/// The largest and second-largest bin.
fn top_two(bins: &Bins) -> (f32, f32) {
    let mut top = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &v in bins {
        if v > top.0 {
            top = (v, top.0);
        } else if v > top.1 {
            top.1 = v;
        }
    }
    top
}

/// The pair tables: one pass per unordered pair, the diagonal included.
fn pairs(feat: &mut Buffers, alphas: &[Plane], start: usize) {
    let p = alphas.len();
    let snap = &feat.snap[start..];
    feat.pair_x.clear();
    for table in [
        &mut feat.pair_base,
        &mut feat.pair_base2,
        &mut feat.pair_lower,
        &mut feat.pair_upper,
    ] {
        table.clear();
        table.resize(p * p, 0.0);
    }

    for b in 0..p {
        for a in 0..=b {
            let x = cross(snap, &alphas[a][start..], &alphas[b][start..]);
            let ma = &feat.mass[a * NUM_COLORS..][..NUM_COLORS];
            let mb = &feat.mass[b * NUM_COLORS..][..NUM_COLORS];
            let mut base = [0.0; NUM_COLORS];
            let mut lower_a = f32::NEG_INFINITY;
            let mut lower_b = f32::NEG_INFINITY;
            for d in 0..NUM_COLORS {
                base[d] = feat.count[d] - ma[d] - mb[d] + x[d];
                lower_a = lower_a.max(ma[d] - x[d]);
                lower_b = lower_b.max(mb[d] - x[d]);
            }
            let (top1, top2) = top_two(&base);
            let upper = (lower_a + feat.top[b].1).max(lower_b + feat.top[a].1);
            let (ab, ba) = (a * p + b, b * p + a);
            feat.pair_base[ab] = top1;
            feat.pair_base[ba] = top1;
            feat.pair_base2[ab] = top1 + top2;
            feat.pair_base2[ba] = top1 + top2;
            feat.pair_lower[ab] = lower_a;
            feat.pair_lower[ba] = lower_b;
            feat.pair_upper[ab] = upper;
            feat.pair_upper[ba] = upper;
            debug_assert_eq!(feat.pair_x.len(), tri(a, b));
            feat.pair_x.push(x);
        }
    }
}

/// `a` and `b` in their better order, bottom first; a tie keeps `a` first.
fn pair_order(feat: &Buffers, p: usize, a: usize, b: usize) -> [usize; 2] {
    if feat.pair_lower[a * p + b] + feat.top[b].1 >= feat.pair_lower[b * p + a] + feat.top[a].1 {
        [a, b]
    } else {
        [b, a]
    }
}

/// Score every banner of up to two layers from the tables.
fn shallow(feat: &Buffers, p: usize, best: &mut Best) {
    best.offer(argmax(&feat.count).1, &[]);
    for q in 0..p {
        best.offer(feat.base_bound[q] + feat.top[q].1, &[q]);
    }
    for b in 0..p {
        for a in 0..=b {
            let score = feat.pair_base[a * p + b] + feat.pair_upper[a * p + b];
            if score > best.bar() {
                best.offer(score, &pair_order(feat, p, a, b));
            }
        }
    }
}

/// Fill [`Buffers::order`] with every triple `i ≤ j ≤ k` whose bound from the
/// pair tables beats `best`, descending by that bound, ties by code. With
/// `four` the bound also covers every four-layer banner with the triple on
/// top: the two largest base bins in place of the largest.
fn triples(feat: &mut Buffers, p: usize, best: f32, four: bool) {
    let base = if four {
        &feat.pair_base2
    } else {
        &feat.pair_base
    };
    let (lower, upper) = (&feat.pair_lower, &feat.pair_upper);
    feat.order.clear();
    for j in 0..p {
        for i in 0..=j {
            for k in j..p {
                let (ij, ik, jk) = (i * p + j, i * p + k, j * p + k);
                let (ji, ki, kj) = (j * p + i, k * p + i, k * p + j);
                let h0 = base[ij].min(base[ik]).min(base[jk]);
                let si = lower[ij].min(lower[ik]) + upper[jk];
                let sj = lower[ji].min(lower[jk]) + upper[ik];
                let sk = lower[ki].min(lower[kj]) + upper[ij];
                let bound = h0 + si.max(sj).max(sk);
                if bound > best {
                    feat.order.push((bound, ((i * p + j) * p + k) as u32));
                }
            }
        }
    }
    feat.order
        .sort_unstable_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)));
}

/// Per-dye bounds on a triple's histograms from the pair tables, without a
/// pixel pass: `T ≤ min X` and `T ≥ max(0, X_ab + X_ac − A_a)`. Returns a
/// bound on the three layers' terms in their best order, and the largest
/// and second-largest bin of the bound on `h₀`.
fn triple_bound(feat: &Buffers, p: usize, [i, j, k]: [usize; 3]) -> (f32, f32, f32) {
    let (xij, xik, xjk) = (
        &feat.pair_x[tri(i, j)],
        &feat.pair_x[tri(i, k)],
        &feat.pair_x[tri(j, k)],
    );
    let mi = &feat.mass[i * NUM_COLORS..][..NUM_COLORS];
    let mj = &feat.mass[j * NUM_COLORS..][..NUM_COLORS];
    let mk = &feat.mass[k * NUM_COLORS..][..NUM_COLORS];
    let mut h0 = [0.0; NUM_COLORS];
    let mut h1 = [f32::NEG_INFINITY; 3];
    for d in 0..NUM_COLORS {
        let hi = xij[d].min(xik[d]).min(xjk[d]);
        let lo = (xij[d] + xik[d] - mi[d])
            .max(xij[d] + xjk[d] - mj[d])
            .max(xik[d] + xjk[d] - mk[d])
            .max(0.0);
        h0[d] = feat.count[d] - mi[d] - mj[d] - mk[d] + xij[d] + xik[d] + xjk[d] - lo;
        h1[0] = h1[0].max(mi[d] - xij[d] - xik[d] + hi);
        h1[1] = h1[1].max(mj[d] - xij[d] - xjk[d] + hi);
        h1[2] = h1[2].max(mk[d] - xik[d] - xjk[d] + hi);
    }
    let upper = (h1[0] + feat.pair_upper[j * p + k])
        .max(h1[1] + feat.pair_upper[i * p + k])
        .max(h1[2] + feat.pair_upper[i * p + j]);
    let (top1, top2) = top_two(&h0);
    (upper, top1, top2)
}

/// Score an unordered triple exactly: one pass for `T`, the rest from the pair
/// tables. All six orders are covered; the best is returned bottom first, the
/// first of equal ones in the order `i`, `j`, `k` at the bottom.
fn triple(
    feat: &Buffers,
    alphas: &[Plane],
    start: usize,
    p: usize,
    [i, j, k]: [usize; 3],
) -> Triple {
    let t = cross3(
        &feat.snap[start..],
        &alphas[i][start..],
        &alphas[j][start..],
        &alphas[k][start..],
    );
    let (xij, xik, xjk) = (
        &feat.pair_x[tri(i, j)],
        &feat.pair_x[tri(i, k)],
        &feat.pair_x[tri(j, k)],
    );
    let mi = &feat.mass[i * NUM_COLORS..][..NUM_COLORS];
    let mj = &feat.mass[j * NUM_COLORS..][..NUM_COLORS];
    let mk = &feat.mass[k * NUM_COLORS..][..NUM_COLORS];
    let mut base = [0.0; NUM_COLORS];
    let mut h1 = [f32::NEG_INFINITY; 3];
    for d in 0..NUM_COLORS {
        base[d] = feat.count[d] - mi[d] - mj[d] - mk[d] + xij[d] + xik[d] + xjk[d] - t[d];
        h1[0] = h1[0].max(mi[d] - xij[d] - xik[d] + t[d]);
        h1[1] = h1[1].max(mj[d] - xij[d] - xjk[d] + t[d]);
        h1[2] = h1[2].max(mk[d] - xik[d] - xjk[d] + t[d]);
    }

    let mut best = Triple {
        base,
        upper: f32::NEG_INFINITY,
        order: [i, j, k],
    };
    for (h1, first, a, b) in [(h1[0], i, j, k), (h1[1], j, i, k), (h1[2], k, i, j)] {
        let upper = h1 + feat.pair_upper[a * p + b];
        if upper > best.upper {
            let [a, b] = pair_order(feat, p, a, b);
            best.upper = upper;
            best.order = [first, a, b];
        }
    }
    best
}

/// Score every bottom pattern under the triple `t`, which has had its pass,
/// as a four-layer banner, skipping those the bounds rule out (module docs,
/// "Four layers").
fn bottoms(
    feat: &mut Buffers,
    alphas: &[Plane],
    start: usize,
    p: usize,
    t: &Triple,
    best: &mut Best,
) {
    let [i, j, k] = t.order;
    feat.rest.clear();
    feat.rest.extend(
        alphas[i][start..]
            .iter()
            .zip(&alphas[j][start..])
            .zip(&alphas[k][start..])
            .map(|((&a, &b), &c)| (1.0 - a) * (1.0 - b) * (1.0 - c)),
    );
    let base = argmax(&t.base).1;

    for q in 0..p {
        // From the scalar tables first: `h_q` is at most `q`'s term under any
        // one of the triple, the new base at most any pair's base term, and
        // both at most the triple's base term.
        let cap = |table: &[f32]| {
            base.min(table[q * p + i])
                .min(table[q * p + j])
                .min(table[q * p + k])
        };
        let gain = |bottom: f32, below: f32| (bottom + below).min(base + bottom.min(below));
        if t.upper + gain(cap(&feat.pair_lower), cap(&feat.pair_base)) <= best.bar() {
            continue;
        }

        // Then per dye.
        let mq = &feat.mass[q * NUM_COLORS..][..NUM_COLORS];
        let (xi, xj, xk) = (
            &feat.pair_x[tri(q, i)],
            &feat.pair_x[tri(q, j)],
            &feat.pair_x[tri(q, k)],
        );
        let (mi, mj, mk) = (
            &feat.mass[i * NUM_COLORS..][..NUM_COLORS],
            &feat.mass[j * NUM_COLORS..][..NUM_COLORS],
            &feat.mass[k * NUM_COLORS..][..NUM_COLORS],
        );
        let mut bottom = [0.0; NUM_COLORS];
        let mut below = [0.0; NUM_COLORS];
        for d in 0..NUM_COLORS {
            let lo = (mq[d] - xi[d] - xj[d] - xk[d])
                .max(mq[d] + t.base[d] - feat.count[d])
                .max(0.0);
            bottom[d] = t.base[d].min(mq[d] - xi[d].max(xj[d]).max(xk[d]));
            below[d] = (t.base[d] - lo)
                .min(feat.count[d] - mq[d] - mi[d] + xi[d])
                .min(feat.count[d] - mq[d] - mj[d] + xj[d])
                .min(feat.count[d] - mq[d] - mk[d] + xk[d]);
        }
        if t.upper + joint(&below, &bottom, base) <= best.bar() {
            continue;
        }

        let h = cross(&feat.snap[start..], &feat.rest, &alphas[q][start..]);
        let below = argmax_sub(&t.base, &h).1;
        let score = t.upper + below + argmax(&h).1;
        best.offer(score, &[q, i, j, k]);
    }
}

/// A bound on `max h₀ + max h_q` from per-dye bounds `below ≥ h₀` and
/// `bottom ≥ h_q`, where `h₀ + h_q` is a histogram whose largest bin is
/// `base`: on one dye the two share that bin, on two they take at most their
/// bounds.
fn joint(below: &Bins, bottom: &Bins, base: f32) -> f32 {
    let mut b = [(f32::NEG_INFINITY, 0); 2];
    let mut h = [(f32::NEG_INFINITY, 0); 2];
    for d in 0..NUM_COLORS {
        for (top, v) in [(&mut b, below[d]), (&mut h, bottom[d])] {
            if v > top[0].0 {
                top[1] = top[0];
                top[0] = (v, d);
            } else if v > top[1].0 {
                top[1] = (v, d);
            }
        }
    }
    let apart = if b[0].1 != h[0].1 {
        b[0].0 + h[0].0
    } else {
        (b[0].0 + h[1].0).max(b[1].0 + h[0].0)
    };
    apart.max(base).min(base + b[0].0.min(h[0].0))
}

/// The winner's dyes: every slot's weight binned by snapped dye, straight from
/// the definition, and each slot's argmax (ties to the lower dye index).
fn dyes(feat: &Buffers, alphas: &[Plane], start: usize, stack: &Stack) -> Map {
    let patterns = &stack.patterns[..stack.n];
    let mut bins = [[0.0_f32; NUM_COLORS]; MAX_LAYERS + 1];
    for (q, &d) in feat.snap.iter().enumerate().skip(start) {
        let d = usize::from(d);
        let mut rest = 1.0;
        for (slot, &pattern) in patterns.iter().enumerate().rev() {
            let a = alphas[pattern][q];
            bins[slot + 1][d] += rest * a;
            rest *= 1.0 - a;
        }
        bins[0][d] += rest;
    }

    let mut layers = [None; MAX_LAYERS];
    for (slot, &pattern) in patterns.iter().enumerate() {
        layers[slot] = Some((pattern, argmax(&bins[slot + 1]).0));
    }
    Map {
        base: argmax(&bins[0]).0,
        layers,
    }
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
    // Both patch sizes are multiples of 16, so `as_chunks` never leaves a
    // remainder and the tail path this function does not have is unreachable.
    debug_assert!(snap.len().is_multiple_of(CROSS_WAYS));

    let mut ways = [[0.0_f32; NUM_COLORS]; CROSS_WAYS];
    let chunks = snap
        .as_chunks::<CROSS_WAYS>()
        .0
        .iter()
        .zip(alpha1.as_chunks::<CROSS_WAYS>().0)
        .zip(alpha2.as_chunks::<CROSS_WAYS>().0);
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

/// `T[d] = Σ_{snap == d} α₁·α₂·α₃` — the triple's one reduction, the same
/// shape as [`cross`].
fn cross3(snap: &[u8], alpha1: &[f32], alpha2: &[f32], alpha3: &[f32]) -> Bins {
    debug_assert!(snap.len().is_multiple_of(CROSS_WAYS));

    let mut ways = [[0.0_f32; NUM_COLORS]; CROSS_WAYS];
    let chunks = snap
        .as_chunks::<CROSS_WAYS>()
        .0
        .iter()
        .zip(alpha1.as_chunks::<CROSS_WAYS>().0)
        .zip(alpha2.as_chunks::<CROSS_WAYS>().0)
        .zip(alpha3.as_chunks::<CROSS_WAYS>().0);
    for (((snap, a1), a2), a3) in chunks {
        for (way, (((&d, &a1), &a2), &a3)) in
            ways.iter_mut().zip(snap.iter().zip(a1).zip(a2).zip(a3))
        {
            way[d as usize] += a1 * a2 * a3;
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
    /// The searches compute the same number through decompositions that never
    /// touch a slot weight; this is the independent implementation the tests
    /// below compare against.
    fn naive_score(map: &Map, snap: &[u8], alphas: &[Plane], start: usize) -> f32 {
        let layers: Vec<(usize, usize)> = map.layers.iter().flatten().copied().collect();
        let mut total = 0.0;
        for (i, &d) in snap.iter().enumerate().skip(start) {
            let d = usize::from(d);
            // Top layer first: what is left over is the weight of the slots
            // below.
            let mut rest = 1.0;
            for &(p, dye) in layers.iter().rev() {
                let a = alphas[p][i];
                if dye == d {
                    total += rest * a;
                }
                rest *= 1.0 - a;
            }
            if map.base == d {
                total += rest;
            }
        }
        total
    }

    /// The best [`naive_score`] of every banner of at most `layers` layers
    /// drawn from `alphas`, patterns repeated or not. The slots' dyes are
    /// independent (the score is a sum of one term per slot), so each slot
    /// takes its best dye rather than every dye combination being enumerated;
    /// every pattern sequence is.
    fn brute_force(snap: &[u8], alphas: &[Plane], start: usize, layers: usize) -> f32 {
        fn walk(
            snap: &[u8],
            alphas: &[Plane],
            start: usize,
            left: usize,
            sequence: &mut Vec<usize>,
            best: &mut f32,
        ) {
            // Per slot (base first), the best of the 16 single-dye maps.
            let score: f32 = (0..=sequence.len())
                .map(|slot| {
                    (0..NUM_COLORS)
                        .map(|dye| {
                            let mut layers = [None; MAX_LAYERS];
                            for (s, &p) in sequence.iter().enumerate() {
                                // Only `slot` carries `dye`; a dye no pixel
                                // can snap to keeps the other slots out.
                                let d = if s + 1 == slot { dye } else { NUM_COLORS };
                                layers[s] = Some((p, d));
                            }
                            let base = if slot == 0 { dye } else { NUM_COLORS };
                            naive_score(&Map { base, layers }, snap, alphas, start)
                        })
                        .fold(f32::NEG_INFINITY, f32::max)
                })
                .sum();
            *best = best.max(score);
            if left > 0 {
                for p in 0..alphas.len() {
                    sequence.push(p);
                    walk(snap, alphas, start, left - 1, sequence, best);
                    sequence.pop();
                }
            }
        }
        let mut best = f32::NEG_INFINITY;
        walk(snap, alphas, start, layers, &mut Vec::new(), &mut best);
        best
    }

    /// A small pattern table with one soft-alpha pattern (`gradient`).
    fn small_table(pats: &Patterns) -> Vec<Plane> {
        [
            "half_vertical",
            "stripe_bottom",
            "gradient",
            "diagonal_left",
            "circle",
        ]
        .iter()
        .map(|name| pats.top[index_of(pats, name)])
        .collect()
    }

    /// A lower row's workspace holding a diagonal ramp through three
    /// off-palette colours: the snapped image is a mix of dyes and no
    /// candidate is trivially perfect.
    fn ramp() -> Workspace {
        workspace(1, |i| {
            let (x, y) = ((i % BANNER_W) as f32, (i / BANNER_W) as f32);
            let t = (x / 19.0 + y / 39.0) * 0.5;
            [30.0 + 200.0 * t, 90.0 + 60.0 * t, 200.0 - 150.0 * t]
        })
    }

    /// The decomposed, pruned search finds the true optimum at every layer
    /// count: on a small pattern table its winner, dyes included, scores what
    /// an exhaustive sweep over every pattern sequence scores, both measured by
    /// [`naive_score`].
    #[test]
    fn the_search_finds_the_brute_force_optimum() {
        let pats = patterns();
        let alphas = small_table(&pats);
        let start = TOP_HW - NTOP_HW;

        for layers in LAYER_COUNTS {
            let mut ws = ramp();
            let map = build(&mut ws, &alphas, LAMBDA, layers);

            let snap = ws.feat.snap.clone();
            let got = naive_score(&map, &snap, &alphas, start);
            let best = brute_force(&snap, &alphas, start, layers);
            assert!(
                (got - best).abs() < 0.05,
                "{layers} layers: search scored {got}, brute force {best}; map = {map:?}"
            );
            assert!(map.layers[layers..].iter().all(Option::is_none), "{map:?}");
        }
    }

    /// The incumbent floor of the per-cell shift: a floor at or above the
    /// optimum is not beaten, one just below it is, by the optimum itself.
    #[test]
    fn a_floor_is_beaten_only_by_a_strictly_better_banner() {
        let pats = patterns();
        let alphas = small_table(&pats);
        let start = TOP_HW - NTOP_HW;

        for layers in [3, 4] {
            let mut ws = ramp();
            let score = |ws: &mut Workspace, floor: f32| {
                best_score(&mut ws.feat, &ws.target, &alphas, start, layers, floor)
            };
            let optimum = score(&mut ws, f32::NEG_INFINITY).expect("every banner beats -inf");
            let best = brute_force(&ws.feat.snap.clone(), &alphas, start, layers);
            assert!(
                (optimum - best).abs() < 0.05,
                "{layers} layers: {optimum} vs {best}"
            );

            assert_eq!(score(&mut ws, optimum), None, "{layers} layers");
            assert_eq!(score(&mut ws, best + 0.05), None, "{layers} layers");
            let below = score(&mut ws, best - 0.05).expect("the optimum beats the floor");
            assert!(
                (below - best).abs() < 0.05,
                "{layers} layers: {below} vs {best}"
            );
        }
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
                None,
                None,
            ],
        };

        for layers in LAYER_COUNTS {
            let mut ws = workspace(0, |i| composite(&planted, &pats.top, i));
            let map = build(&mut ws, &pats.top, LAMBDA, layers);

            let snap = ws.feat.snap.clone();
            let got = naive_score(&map, &snap, &pats.top, 0);
            let planted_score = naive_score(&planted, &snap, &pats.top, 0);
            assert!(
                got >= planted_score - 0.05,
                "{layers} layers: search scored {got}, the planted banner {planted_score}; \
                 map = {map:?}"
            );
            // The planted banner is drawn in dye colours, so all but its
            // blended edge pixels snap back to themselves: the optimum is
            // near-perfect.
            assert!(
                got > 0.95 * TOP_HW as f32,
                "{layers} layers: score {got} of {TOP_HW}; map = {map:?}"
            );
        }
    }

    /// A lower row solves the tail of its planes, so the head — which holds
    /// whatever the previous cell left there — must not reach the result.
    #[test]
    fn a_lower_rows_hidden_head_does_not_reach_the_map() {
        let pats = patterns();
        let visible = Map {
            base: dye("black"),
            layers: [
                Some((index_of(&pats, "half_vertical"), dye("yellow"))),
                None,
                None,
                None,
            ],
        };
        let start = TOP_HW - NTOP_HW;

        let of_head = |head: [f32; 3], layers: usize| {
            let mut ws = workspace(1, |i| {
                if i < start {
                    head
                } else {
                    composite(&visible, &pats.top, i)
                }
            });
            let map = build(&mut ws, &pats.top, LAMBDA, layers);
            (map, ws.feature[0][start..].to_vec())
        };

        for layers in LAYER_COUNTS {
            let (map, pixels) = of_head(COLORS_F32[dye("lime")], layers);
            let (other_map, other_pixels) = of_head(COLORS_F32[dye("magenta")], layers);

            assert_eq!(map, other_map, "{layers} layers");
            assert_eq!(pixels, other_pixels, "{layers} layers");
        }
    }

    /// A flat patch gets a flat map. The winner is not unique — any banner whose
    /// slots all carry the same dye composites to the same flat colour — so the
    /// assertion is on the pixels, not on the layer list.
    #[test]
    fn a_flat_patch_yields_a_flat_map() {
        let pats = patterns();
        let red = dye("red");
        for layers in LAYER_COUNTS {
            let mut ws = workspace(0, |_| COLORS_F32[red]);

            let map = build(&mut ws, &pats.top, LAMBDA, layers);

            assert_eq!(map.base, red, "{layers} layers: map = {map:?}");
            for (ch, plane) in ws.feature.iter().enumerate() {
                assert!(
                    plane.iter().all(|v| *v == COLORS_F32[red][ch]),
                    "{layers} layers: channel {ch} is not flat red; map = {map:?}"
                );
            }
        }
    }

    /// The same for a colour nowhere near a dye: a flat mid grey must still come
    /// out flat, in whichever grey the OKLab snap picked.
    #[test]
    fn an_off_palette_flat_patch_snaps_to_one_dye() {
        let pats = patterns();
        for layers in LAYER_COUNTS {
            let mut ws = workspace(0, |_| [120.0, 122.0, 118.0]);

            let map = build(&mut ws, &pats.top, LAMBDA, layers);

            assert!(
                map.base == dye("gray") || map.base == dye("light_gray"),
                "{layers} layers: mid grey mapped to {}",
                COLOR_NAMES[map.base]
            );
            for plane in &ws.feature {
                assert!(plane.iter().all(|v| *v == plane[0]), "{layers} layers");
            }
        }
    }
}
