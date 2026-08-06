//! The per-work-item solver workspace: every buffer a cell solve touches, in
//! one allocation, reused down the column.
//!
//! Every plane is a full top-row [`Plane`], and a lower banner row solves on
//! the lane-aligned tail view of it (elements `TOP_HW - NTOP_HW ..`), so the
//! kernels take runtime-length lane views instead of being generic over the
//! patch size. Only that active view is ever written, so the head of a lower
//! row's planes holds whatever the last top-row cell left there — nothing may
//! read outside the view. Nothing is zeroed between cells for the same reason:
//! every field is fully overwritten before it is read.

use std::time::Duration;

use crate::cli::config::RefinementConfig;
use crate::geometry::{NTOP_HW, TOP_HW};
use crate::pattern::Patterns;
use crate::simd::{Chunk, F32s, LANES};
use crate::zip;

use super::perturb::Rng;
use super::{feature, greedy, perturb, refine};

/// One patch plane: always the full top-row patch, whatever the row.
pub type Plane = Chunk<TOP_HW>;

/// Lane offset at which a lower row's patch starts inside a [`Plane`].
///
/// `TOP_HW - NTOP_HW` is `HIDDEN_H · BANNER_W = 320`, a multiple of 16, so this
/// division is exact for every backend width and the view it names is aligned.
pub const LOWER_OFF: usize = (TOP_HW - NTOP_HW) / LANES;

const _: () = assert!((TOP_HW - NTOP_HW).is_multiple_of(16));

/// The active lane view of a plane for a cell whose patch starts at lane `off`.
#[inline]
pub(super) fn view(plane: &Plane, off: usize) -> &[F32s] {
    &plane.lanes()[off..]
}

/// [`view`], exclusively.
#[inline]
pub(super) fn view_mut(plane: &mut Plane, off: usize) -> &mut [F32s] {
    &mut plane.lanes_mut()[off..]
}

/// What the solver decided for one banner cell.
#[derive(Clone, Debug)]
pub struct Solution {
    /// Index into [`crate::color::COLOR_NAMES`] of the banner's base dye.
    pub base: usize,
    /// `(pattern index, dye index)` per layer, in application order.
    pub layers: Vec<(usize, usize)>,
    /// OKLab `Σ ΔE` of the final composite against the target patch, summed
    /// over the solved pixels — not normalised
    /// ([`super::refine::lab_error`] explains why it is not squared).
    ///
    /// Always the **pure target** error as it leaves [`Workspace::solve`], even
    /// when the solver ranked on a feature-regularised one
    /// ([`super::refine::ranking_error`]): the readout has to mean the same
    /// thing at every `--feature-weight` for the numbers to be comparable.
    /// Inside the solve it carries whichever error that stage ranks on.
    pub error: f32,
    /// [`Solution::error`] per pixel — the number that reads directly as a
    /// colour difference, and what `--debug` reports.
    pub lab_error: f32,
}

/// The affine map a run of layers applies to whatever is underneath it:
/// `x ↦ x · mul + add`, per channel.
///
/// `mul` is channel-independent (it is a product of `1 − α`), which is what
/// lets the refinement kernel reduce `Σ (α·mul)²` once for all three channels.
pub(super) struct Suffix {
    /// Additive term per channel.
    pub add: [Plane; 3],
    /// Multiplicative term, shared by the channels.
    pub mul: Plane,
}

impl Suffix {
    fn new() -> Self {
        Self {
            add: [Chunk::zeroed(); 3],
            mul: Chunk::zeroed(),
        }
    }

    /// Overwrite this suffix's active view with `src`'s.
    pub(super) fn copy_from(&mut self, src: &Self, off: usize) {
        for (out, m) in zip!(mut view_mut(&mut self.mul, off), view(&src.mul, off)) {
            *out = m;
        }
        for ch in 0..3 {
            for (out, a) in zip!(mut view_mut(&mut self.add[ch], off), view(&src.add[ch], off)) {
                *out = a;
            }
        }
    }
}

/// The refinement window's beam: two generations of candidate suffixes and the
/// layer choices that produced them, plus the scored shortlist.
pub(super) struct Beam {
    /// Candidates entering the current window step.
    pub prev: Vec<Suffix>,
    /// Candidates leaving it.
    pub curr: Vec<Suffix>,
    /// Layer choices behind [`Beam::prev`], deepest layer first.
    pub prev_layers: Vec<Vec<(usize, usize)>>,
    /// Layer choices behind [`Beam::curr`].
    pub curr_layers: Vec<Vec<(usize, usize)>>,
    /// `(error, candidate, pattern, dye)` survivors, kept sorted ascending.
    pub best: Vec<(f32, usize, usize, usize)>,
}

impl Beam {
    fn new(cand: usize) -> Self {
        Self {
            prev: (0..cand).map(|_| Suffix::new()).collect(),
            curr: (0..cand).map(|_| Suffix::new()).collect(),
            prev_layers: vec![Vec::new(); cand],
            curr_layers: vec![Vec::new(); cand],
            best: vec![(f32::INFINITY, 0, 0, 0); cand],
        }
    }
}

/// CPU time a work item spent in each solver stage, summed over its cells.
#[derive(Clone, Copy, Default)]
pub struct Stages {
    /// Greedy fill.
    pub greedy: Duration,
    /// Windowed beam refinement.
    pub refine: Duration,
    /// Perturbation rounds (including the re-refinement inside them).
    pub perturb: Duration,
    /// Per-cell OKLab bookkeeping: converting the target and the final ΔE
    /// readout. The exact scoring itself is counted under refinement.
    pub oklab: Duration,
    /// Feature-map construction: the snap, the pattern-pair search and the
    /// blend. Zero when `--feature-weight` is `0`.
    pub feature: Duration,
}

impl Stages {
    pub fn add(&mut self, other: &Stages) {
        self.greedy += other.greedy;
        self.refine += other.refine;
        self.perturb += other.perturb;
        self.oklab += other.oklab;
        self.feature += other.feature;
    }
}

/// Everything a cell solve needs that is shared across the whole run.
pub struct SolveCfg<'a> {
    /// The pattern tables.
    pub patterns: &'a Patterns,
    /// Windowed-refinement settings; `refinement_pass == 0` disables it.
    pub refinement: &'a RefinementConfig,
    /// `(top_n, duplicates, rounds)`, or `None` when disabled.
    pub perturbations: Option<(usize, usize, usize)>,
    /// Weight of the feature-map term, `--feature-weight`; `0` skips the whole
    /// pass ([`super::feature`]).
    pub feature_weight: f32,
}

/// Reusable solver buffers for one work item.
pub struct Workspace {
    /// Lane offset of the current cell's patch: `0` or [`LOWER_OFF`].
    pub(super) off: usize,
    /// Pixels the current cell solves.
    pub(super) hw: usize,
    /// Layers in the last [`Workspace::solve`] — which prefix is the composite.
    depth: usize,
    /// The cell's target pixels, planar, filled by the caller before `solve`.
    ///
    /// **Not pure after `solve` with a non-zero feature weight**: the closed-form
    /// rungs score against a blend of the target and the feature map, and that
    /// blend is written here, over the gathered pixels
    /// ([`super::feature::build`] explains why in place). Read the OKLab
    /// [`Workspace::lab_target`] for the real target — it is converted first.
    pub(super) target: [Plane; 3],
    /// Composite after `i` layers; `prefixes[0]` is the flat base colour.
    pub(super) prefixes: Vec<[Plane; 3]>,
    /// Affine map of layers `i..`; `suffixes[n]` is the identity.
    pub(super) suffixes: Vec<Suffix>,
    /// The refinement window's beam.
    pub(super) beam: Beam,
    /// Scratch composite for refinement's exact rung.
    pub(super) cand: [Plane; 3],
    /// The target converted to OKLab, once per cell.
    pub(super) lab_target: [Plane; 3],
    /// The cell's feature map — the idealised two-layer banner every rung is
    /// pulled towards ([`super::feature`]) — composited in sRGB.
    pub(super) feature: [Plane; 3],
    /// [`Workspace::feature`] in OKLab, for the exact rung.
    pub(super) lab_feature: [Plane; 3],
    /// Scratch for building the feature map.
    pub(super) feat: feature::Buffers,
    /// Perturbation beam: `(error, solution)`, kept sorted ascending.
    pub(super) pool: Vec<(f32, Solution)>,
    /// Refinement's exact-scoring shortlist: the `exact_candidates` best
    /// `(closed-form error, candidate, pattern, dye)` of one window step. Empty
    /// when the exact rung is off.
    pub(super) shortlist: Vec<(f32, usize, usize, usize)>,
    /// Trials a perturbation round is about to score.
    pub(super) trials: Vec<Solution>,
    /// Scratch layer list: the refinement's "did this pass change anything"
    /// snapshot.
    pub(super) scratch: Vec<(usize, usize)>,
    /// Per-stage CPU time, accumulated over the item's cells.
    pub stages: Stages,
}

impl Workspace {
    /// Allocate for cells of up to `max_layers` layers and a beam of
    /// `candidates` (the `--refinement-candidate` setting).
    pub fn new(max_layers: usize, candidates: usize) -> Self {
        Self {
            off: 0,
            hw: TOP_HW,
            depth: 0,
            target: [Chunk::zeroed(); 3],
            prefixes: vec![[Chunk::zeroed(); 3]; max_layers + 1],
            suffixes: (0..max_layers + 1).map(|_| Suffix::new()).collect(),
            beam: Beam::new(candidates.max(1)),
            cand: [Chunk::zeroed(); 3],
            lab_target: [Chunk::zeroed(); 3],
            feature: [Chunk::zeroed(); 3],
            lab_feature: [Chunk::zeroed(); 3],
            feat: feature::Buffers::default(),
            shortlist: Vec::new(),
            pool: Vec::new(),
            trials: Vec::new(),
            scratch: Vec::with_capacity(max_layers),
            stages: Stages::default(),
        }
    }

    /// Point the workspace at banner row `row`'s patch size. Call before
    /// gathering the cell's pixels.
    pub fn begin(&mut self, row: usize) {
        self.off = if row == 0 { 0 } else { LOWER_OFF };
        self.hw = TOP_HW - self.off * LANES;
    }

    /// The target planes, to be overwritten with the next cell's patch. Only
    /// the active view is meaningful.
    pub fn target_mut(&mut self) -> &mut [Plane; 3] {
        &mut self.target
    }

    /// The composite the last [`Workspace::solve`] produced. Only the active
    /// view holds this cell's pixels.
    pub fn composite(&self) -> &[Plane; 3] {
        &self.prefixes[self.depth]
    }

    /// Solve the patch currently in [`Workspace::target_mut`] in two stages:
    /// stage 1 is greedy fill + windowed refinement (the initial solve);
    /// stage 2 is perturbation rounds, each trial re-refined by the same
    /// windowed refinement. Either stage's re-refine/perturb sub-step is
    /// skipped when its configuration disables it.
    ///
    /// The target is converted to OKLab first, because from refinement on every
    /// comparison is an exact OKLab one. The greedy fill is the exception on
    /// purpose: it is a coarse initialiser, so it stays closed-form sRGB and
    /// its `error` is overwritten below.
    ///
    /// With a non-zero feature weight the feature map is built next — before the
    /// solve, after the OKLab conversion — and it *blends the sRGB target planes
    /// in place*, so everything downstream of it that reads
    /// [`Workspace::target`] is scoring the regularised target on purpose. The
    /// error returned at the end is re-measured against the pure OKLab target,
    /// so [`Solution::error`] means the same thing at every λ.
    pub fn solve(&mut self, n_layers: usize, cfg: &SolveCfg<'_>, rng: &mut Rng) -> Solution {
        self.depth = n_layers;

        let alphas = &cfg.patterns.top;
        let alpha2 = if self.off == 0 {
            &cfg.patterns.top_alpha2
        } else {
            &cfg.patterns.lower_alpha2
        };

        let t = std::time::Instant::now();
        refine::convert_target(&self.target, &mut self.lab_target, self.off);
        self.stages.oklab += t.elapsed();

        if cfg.feature_weight > 0.0 {
            let t = std::time::Instant::now();
            feature::build(self, alphas, cfg.feature_weight);
            self.stages.feature += t.elapsed();
        }

        let t = std::time::Instant::now();
        let mut solution = greedy::fill(
            &self.target,
            &mut self.prefixes,
            n_layers,
            alphas,
            alpha2,
            self.off,
            self.hw,
        );
        self.stages.greedy += t.elapsed();

        if cfg.refinement.refinement_pass > 0 && n_layers > 0 {
            let t = std::time::Instant::now();
            refine::refine(
                self,
                &mut solution,
                cfg.refinement,
                cfg.feature_weight,
                alphas,
                n_layers,
            );
            self.stages.refine += t.elapsed();
        }

        if let Some((top_n, duplicates, rounds)) = cfg.perturbations {
            let t = std::time::Instant::now();
            perturb::rounds(
                self,
                &mut solution,
                (top_n, duplicates, rounds),
                cfg.refinement,
                cfg.feature_weight,
                alphas,
                n_layers,
                rng,
            );
            self.stages.perturb += t.elapsed();
        }

        let t = std::time::Instant::now();
        solution.error = refine::lab_error(&self.prefixes[n_layers], &self.lab_target, self.off);
        solution.lab_error = solution.error / self.hw as f32;
        self.stages.oklab += t.elapsed();

        solution
    }
}

/// Rebuild `prefixes[0..=n]` from `solution`, so a stage that changed layer
/// choices leaves the prefix chain consistent with them.
///
/// A free function rather than a method: the stages hold the workspace
/// destructured, so a `&mut self` method would not be callable from inside one.
pub(super) fn rebuild_prefixes(
    prefixes: &mut [[Plane; 3]],
    off: usize,
    solution: &Solution,
    alphas: &[Plane],
    n: usize,
) {
    for (ch, plane) in prefixes[0].iter_mut().enumerate() {
        let base = F32s::splat(crate::color::COLORS_F32[solution.base][ch]);
        for out in zip!(mut view_mut(plane, off)) {
            *out = base;
        }
    }
    for layer in 0..n {
        let (done, rest) = prefixes.split_at_mut(layer + 1);
        let (p, c) = solution.layers[layer];
        greedy::build_prefix(&done[layer], &mut rest[0], &alphas[p], c, off);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::oklab::srgb_to_oklab;
    use crate::pattern::{Patterns, load};
    use crate::simd::F32s;

    const LAYERS: usize = 4;

    fn config<'a>(patterns: &'a Patterns, refinement: &'a RefinementConfig) -> SolveCfg<'a> {
        SolveCfg {
            patterns,
            refinement,
            perturbations: Some((2, 2, 1)),
            feature_weight: 0.0,
        }
    }

    fn refinement() -> RefinementConfig {
        RefinementConfig {
            refinement_pass: 2,
            window_size: 2,
            error_threshold: 0.7,
            refinement_candidate: 5,
            exact_candidates: 20,
        }
    }

    /// A patch with structure in it: two colour regions plus a ramp, so the
    /// solver has something to actually choose patterns for.
    fn target(i: usize) -> [f32; 3] {
        let (x, y) = ((i % 20) as f32, (i / 20) as f32);
        if x < 7.0 {
            [200.0 - 2.0 * y, 40.0 + 3.0 * x, 60.0]
        } else {
            [20.0 + x, 90.0 + 2.0 * y, 220.0 - 4.0 * x]
        }
    }

    fn workspace(row: usize) -> Workspace {
        let mut ws = Workspace::new(LAYERS, 5);
        ws.begin(row);
        for (i, px) in (0..TOP_HW).map(|i| (i, target(i))) {
            for (ch, plane) in ws.target_mut().iter_mut().enumerate() {
                plane[i] = px[ch];
            }
        }
        ws
    }

    /// With `--feature-weight 0` the solve must be the one that existed before
    /// the feature map did. The reference here is that solve, spelled out: OKLab
    /// conversion, greedy fill, refinement, perturbation, pure-target readout —
    /// and no feature map anywhere. Anything that leaks the new path into λ = 0
    /// (a blend applied, a feature term in the ranking, the pass running at all)
    /// moves one of the two and not the other.
    #[test]
    fn a_zero_feature_weight_solves_exactly_as_the_pre_feature_map_solver() {
        let patterns = load(&HashSet::new());
        let refinement = refinement();
        let cfg = config(&patterns, &refinement);

        let mut ws = workspace(0);
        let solution = ws.solve(LAYERS, &cfg, &mut Rng::new(0, 0));

        let mut reference = workspace(0);
        let mut rng = Rng::new(0, 0);
        let expected = {
            let ws = &mut reference;
            refine::convert_target(&ws.target, &mut ws.lab_target, ws.off);
            let mut solution = greedy::fill(
                &ws.target,
                &mut ws.prefixes,
                LAYERS,
                &patterns.top,
                &patterns.top_alpha2,
                ws.off,
                ws.hw,
            );
            refine::refine(ws, &mut solution, &refinement, 0.0, &patterns.top, LAYERS);
            perturb::rounds(
                ws,
                &mut solution,
                (2, 2, 1),
                &refinement,
                0.0,
                &patterns.top,
                LAYERS,
                &mut rng,
            );
            solution.error = refine::lab_error(&ws.prefixes[LAYERS], &ws.lab_target, ws.off);
            solution
        };

        assert_eq!(solution.base, expected.base);
        assert_eq!(solution.layers, expected.layers);
        assert_eq!(solution.error.to_bits(), expected.error.to_bits());

        // The target planes are the second observable: at λ = 0 nothing blends
        // them, so they still hold what the caller gathered.
        for (ch, plane) in ws.target.iter().enumerate() {
            for i in 0..TOP_HW {
                assert_eq!(plane[i], target(i)[ch], "pixel {i}, channel {ch}");
            }
        }
    }

    /// The reported error stays comparable across feature weights: whatever the
    /// solver ranked on, what comes back is `Σ ΔE` of the final composite
    /// against the *pure* target — recomputed here pixel by pixel, from the
    /// patch as it was handed in, without touching the solver's kernels.
    #[test]
    fn the_reported_error_is_pure_target_delta_e_at_a_non_zero_weight() {
        let patterns = load(&HashSet::new());
        let refinement = refinement();
        let cfg = SolveCfg {
            feature_weight: 0.75,
            ..config(&patterns, &refinement)
        };

        let mut ws = workspace(1);
        let solution = ws.solve(LAYERS, &cfg, &mut Rng::new(0, 0));

        let lab = |px: [f32; 3]| {
            let (l, a, b) = srgb_to_oklab(
                F32s::splat(px[0]),
                F32s::splat(px[1]),
                F32s::splat(px[2]),
            );
            [l.to_array()[0], a.to_array()[0], b.to_array()[0]]
        };

        let composite = ws.composite();
        let mut sum = 0.0_f64;
        for (i, _) in composite[0].iter().enumerate().skip(TOP_HW - ws.hw) {
            let got = lab([composite[0][i], composite[1][i], composite[2][i]]);
            let want = lab(target(i));
            let d: f32 = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt();
            sum += f64::from(d);
        }

        let reported = f64::from(solution.error);
        assert!(
            (reported - sum).abs() < 1e-3 * sum,
            "reported {reported}, pure-target {sum}"
        );
        assert!(
            (f64::from(solution.lab_error) - sum / f64::from(ws.hw as f32)).abs() < 1e-6,
            "lab_error is not the per-pixel mean of error"
        );
    }
}
