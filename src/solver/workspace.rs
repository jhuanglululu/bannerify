//! The per-work-item solver workspace: every buffer a cell solve touches, in
//! one allocation, reused down the column.
//!
//! ## One workspace, two patch sizes
//!
//! Stage 2a carried two monomorphised workspaces, `Workspace<TOP_HW>` for
//! banner row 0 and `Workspace<NTOP_HW>` for every other row. Stage 2b merges
//! them (`context/plans/2-solver.md`, "Stage-2b addition"): every plane is a
//! [`Plane`] — a `Chunk<TOP_HW>`, the top-row size — and a lower row solves on
//! the **lane-aligned tail view** of each plane, elements `TOP_HW - NTOP_HW ..
//! TOP_HW` (`320..800`). `320 % 16 == 0`, so that view starts on a lane
//! boundary — and on a 64-byte boundary — at every backend width, from scalar
//! to AVX-512.
//!
//! The kernels therefore take `&[F32s]` / `&mut [F32s]` lane views with a
//! runtime length instead of being generic over `HW`. That is one
//! monomorphisation instead of two, half the instruction cache, and the loops
//! are unchanged: they were already lane loops over a `zip!` of equal-length
//! streams, and a runtime trip count costs an induction variable, not a bounds
//! check (`zip!` iterates, it does not index).
//!
//! Reuse across cells stays free: every field a cell reads is fully overwritten
//! before it is read, so nothing is zeroed between cells. Only the *active
//! view* of each plane is ever written, so the head of a lower row's planes
//! holds whatever the last top-row cell left there — which is exactly why
//! nothing may read outside the view.
//!
//! ## Contents
//!
//! - `target` — the cell's pixels, planar ([`crate::solver::cell`] fills it).
//! - `prefixes[i]` — the composite after `i` layers; `prefixes[0]` is the flat
//!   base, `prefixes[n_layers]` is the composite the solve is scored on.
//! - `suffixes[i]` — the affine map "apply layers `i..` to a colour":
//!   `x ↦ x·mul + add`, per channel. Built walking backwards, the mirror of the
//!   prefixes ([`super::refine`]).
//! - `beam` — the refinement window's candidate suffixes and layer lists.
//! - `rgb` — scratch for [`Workspace::render_rgb`], which replays the finished
//!   solution in sRGB for the preview.
//!
//! ## Which space the planes are in
//!
//! Since phase 4 (`context/plans/4-oklab-native.md`) `target`, `prefixes` and
//! `suffixes` all hold **OKLab** components: the column band is converted once,
//! right after resampling, and every cell borrows Lab pixels out of it. Only
//! `rgb` is sRGB, and only because painting is: the preview composites dye RGB
//! over the pattern alphas exactly as the game does, so the picture is never an
//! inverse transform of a Lab composite.

use std::time::Duration;

use crate::cli::config::RefinementConfig;
use crate::color::{COLORS_F32, lab};
use crate::geometry::{NTOP_HW, TOP_HW};
use crate::oklab::srgb_to_oklab;
use crate::simd::{Chunk, F32s, LANES};
use crate::zip;

use super::perturb::Rng;
use super::{greedy, perturb, refine};

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
    /// OKLab SSE of the final composite against the target patch, summed over
    /// the solved pixels — not normalised. This is the solver's objective.
    pub error: f32,
    /// Mean OKLab ΔE per pixel of the final composite, or `0.0` when it was not
    /// asked for (`--debug`). The same quantity as [`Solution::error`] but per
    /// pixel and un-squared, which is the number that reads as a colour
    /// difference.
    pub lab_error: f32,
}

/// The affine map a run of layers applies to whatever is underneath it:
/// `x ↦ x · mul + add`, per channel.
///
/// `mul` is channel-independent (it is a product of `1 − α`), which is what
/// lets the refinement kernel reduce `Σ (α·mul)²` once for all three channels.
/// Ported from the old build's `SuffixPatternCache`, which packed the same four
/// planes into one array.
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
    /// `(error, candidate, pattern, dye)`, kept sorted ascending.
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

    /// Make this step's output the next step's input. Vectors swap by header,
    /// so no plane is copied.
    pub(super) fn swap(&mut self) {
        std::mem::swap(&mut self.prev, &mut self.curr);
        std::mem::swap(&mut self.prev_layers, &mut self.curr_layers);
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
}

impl Stages {
    /// Fold another item's totals into these.
    pub fn add(&mut self, other: &Stages) {
        self.greedy += other.greedy;
        self.refine += other.refine;
        self.perturb += other.perturb;
    }
}

/// Everything a cell solve needs that is shared across the whole run.
pub struct SolveCfg<'a> {
    /// The pattern tables.
    pub patterns: &'a crate::pattern::Patterns,
    /// Windowed-refinement settings; `refinement_pass == 0` disables it.
    pub refinement: &'a RefinementConfig,
    /// `(top_n, duplicates, rounds)`, or `None` when disabled.
    pub perturbations: Option<(usize, usize, usize)>,
    /// Report each cell's mean OKLab ΔE (`--debug`). Off by default because it
    /// is a per-pixel square root the solve itself has no use for.
    pub report_lab: bool,
}

/// Reusable solver buffers for one work item.
pub struct Workspace {
    /// Lane offset of the current cell's patch: `0` or [`LOWER_OFF`].
    pub(super) off: usize,
    /// Pixels the current cell solves.
    pub(super) hw: usize,
    /// The cell's target pixels, planar, filled by the caller before `solve`.
    pub(super) target: [Plane; 3],
    /// Composite after `i` layers; `prefixes[0]` is the flat base colour.
    pub(super) prefixes: Vec<[Plane; 3]>,
    /// Affine map of layers `i..`; `suffixes[n]` is the identity.
    pub(super) suffixes: Vec<Suffix>,
    /// The refinement window's beam.
    pub(super) beam: Beam,
    /// The solved cell replayed in sRGB, for the preview render.
    rgb: [Plane; 3],
    /// Perturbation beam: `(error, solution)`, kept sorted ascending.
    pub(super) pool: Vec<(f32, Solution)>,
    /// Trials a perturbation round is about to score.
    pub(super) trials: Vec<Solution>,
    /// Scratch layer list (the refinement's "did this pass change anything"
    /// snapshot), kept in the workspace so a pass allocates nothing.
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
            target: [Chunk::zeroed(); 3],
            prefixes: vec![[Chunk::zeroed(); 3]; max_layers + 1],
            suffixes: (0..max_layers + 1).map(|_| Suffix::new()).collect(),
            beam: Beam::new(candidates.max(1)),
            rgb: [Chunk::zeroed(); 3],
            pool: Vec::new(),
            trials: Vec::new(),
            scratch: Vec::with_capacity(max_layers),
            stages: Stages::default(),
        }
    }

    /// Point the workspace at banner row `row`'s patch size. Call before
    /// gathering the cell's pixels — [`Workspace::target_mut`] hands out the
    /// whole plane, but only the active view is meaningful.
    pub fn begin(&mut self, row: usize) {
        self.off = if row == 0 { 0 } else { LOWER_OFF };
        self.hw = TOP_HW - self.off * LANES;
    }

    /// The target planes, to be overwritten with the next cell's patch.
    pub fn target_mut(&mut self) -> &mut [Plane; 3] {
        &mut self.target
    }

    /// Replay `solution` in sRGB and return the composite the preview draws,
    /// recording its ΔE when [`SolveCfg::report_lab`] asks for it.
    ///
    /// The solver's own composite (`prefixes[n]`) is in OKLab, and inverting
    /// that would be both extra work and *wrong*: OKLab compositing is the
    /// solver's approximation on anti-aliased pattern edges, while the game
    /// blends dye RGB over the mask. So the preview re-runs the chosen layer
    /// chain in sRGB — one compositing pass per layer over one patch, a rounding
    /// error next to the solve that produced it — and paints that.
    ///
    /// Only the active view holds this cell's pixels.
    pub fn render_rgb(&mut self, solution: &mut Solution, cfg: &SolveCfg<'_>) -> &[Plane; 3] {
        let off = self.off;
        let patterns = cfg.patterns;
        for (ch, plane) in self.rgb.iter_mut().enumerate() {
            let c = F32s::splat(COLORS_F32[solution.base][ch]);
            for out in zip!(mut view_mut(plane, off)) {
                *out = c;
            }
        }
        // In place: the composite reads and writes the same element, so the
        // chain needs one buffer, not a ping-pong pair.
        for &(p, c) in &solution.layers {
            let alpha = &patterns.top[p];
            for (ch, plane) in self.rgb.iter_mut().enumerate() {
                let col = F32s::splat(COLORS_F32[c][ch]);
                for (out, alp) in zip!(mut view_mut(plane, off), view(alpha, off)) {
                    *out = (*out).mul_add(F32s::ONE - alp, col * alp);
                }
            }
        }

        if cfg.report_lab {
            solution.lab_error = self.delta_e() / self.hw as f32;
        }
        &self.rgb
    }

    /// `Σ ΔE` between the sRGB composite [`Workspace::render_rgb`] just built
    /// and the (OKLab) target.
    ///
    /// Deliberately scored on the *painted* pixels rather than on the solver's
    /// own Lab prefix chain: the two differ on anti-aliased pattern edges, where
    /// the solver blends linearly in Lab and the game blends dye RGB, and the
    /// number worth reporting is the error of the wall a player will build. It
    /// is also what makes this readout comparable with the pre-phase-4 builds,
    /// which measured exactly the same thing.
    fn delta_e(&self) -> f32 {
        let off = self.off;
        let mut acc = F32s::ZERO;
        for (r, g, b, tl, ta, tb) in zip!(
            view(&self.rgb[0], off),
            view(&self.rgb[1], off),
            view(&self.rgb[2], off),
            view(&self.target[0], off),
            view(&self.target[1], off),
            view(&self.target[2], off)
        ) {
            let (l, a, b) = srgb_to_oklab(r, g, b);
            let (dl, da, db) = (l - tl, a - ta, b - tb);
            acc += dl.mul_add(dl, da.mul_add(da, db * db)).sqrt();
        }
        acc.hsum()
    }

    /// Solve the patch currently in [`Workspace::target_mut`]: greedy fill,
    /// windowed refinement, perturbation rounds — each stage skipped when its
    /// configuration disables it. The patch is in OKLab and so is every number
    /// the stages produce.
    ///
    /// `rng` is the column's stream; it is advanced only by the perturbation
    /// stage, so a run without `--perturbations` never touches it and the seed
    /// cannot change the result.
    pub fn solve(&mut self, n_layers: usize, cfg: &SolveCfg<'_>, rng: &mut Rng) -> Solution {
        assert!(
            n_layers < self.prefixes.len(),
            "workspace built for fewer layers than requested"
        );
        let alphas = &cfg.patterns.top;
        let alpha2 = if self.off == 0 {
            &cfg.patterns.top_alpha2
        } else {
            &cfg.patterns.lower_alpha2
        };

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
            refine::refine(self, &mut solution, cfg.refinement, alphas, n_layers);
            self.stages.refine += t.elapsed();
        }

        if let Some((top_n, duplicates, rounds)) = cfg.perturbations {
            let t = std::time::Instant::now();
            perturb::rounds(
                self,
                &mut solution,
                (top_n, duplicates, rounds),
                cfg.refinement,
                alphas,
                n_layers,
                rng,
            );
            self.stages.perturb += t.elapsed();
        }

        solution
    }
}

/// Rebuild `prefixes[0..=n]` from `solution`, so a stage that changed layer
/// choices leaves the prefix chain consistent with them.
///
/// A free function rather than a method: the stages hold the workspace
/// destructured (they need `prefixes`, `suffixes` and `beam` borrowed at once),
/// so a `&mut self` method would not be callable from inside one.
pub(super) fn rebuild_prefixes(
    prefixes: &mut [[Plane; 3]],
    off: usize,
    solution: &Solution,
    alphas: &[Plane],
    n: usize,
) {
    let lab = lab();
    for (ch, plane) in prefixes[0].iter_mut().enumerate() {
        let base = F32s::splat(lab.color[solution.base][ch]);
        for out in zip!(mut view_mut(plane, off)) {
            *out = base;
        }
    }
    for layer in 0..n {
        let (done, rest) = prefixes.split_at_mut(layer + 1);
        let (p, c) = solution.layers[layer];
        greedy::build_prefix(&done[layer], &mut rest[0], &alphas[p], lab.color[c], off);
    }
}

/// OKLab SSE of `prefix` against `target` — the number [`Solution::error`]
/// carries. No channel weights: OKLab is uniform, which is the whole point of
/// solving in it.
pub(super) fn total_error(prefix: &[Plane; 3], target: &[Plane; 3], off: usize) -> f32 {
    let mut err = 0.0;
    for ch in 0..3 {
        let mut acc = F32s::ZERO;
        for (p, t) in zip!(view(&prefix[ch], off), view(&target[ch], off)) {
            let d = p - t;
            acc = d.mul_add(d, acc);
        }
        err += acc.hsum();
    }
    err
}
