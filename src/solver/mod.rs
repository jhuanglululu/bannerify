//! The banner solver: which dyes and patterns approximate each cell.
//!
//! See `context/plans/2-solver.md` and `context/plans/5-exact-refine.md`. A
//! cell is solved in three stages, each skipped when its configuration disables
//! it, all sharing one reusable [`Workspace`]:
//!
//! - [`variance`] — the pre-pass that hands each cell a layer budget.
//! - [`greedy`] — the fill: base dye, then one (pattern, dye) layer at a time.
//! - [`refine`] — windowed beam refinement over prefix/suffix caches, ranking
//!   its candidates by exact OKLab ΔE (`--exact-candidates`).
//! - [`perturb`] — random re-rolls, re-refined, kept if better.
//!
//! Phase 5 folded the old separate OKLab pass (`--lab-refine`) into [`refine`];
//! there is no final pass any more.
//!
//! Support:
//!
//! - [`workspace`] — every buffer the stages touch, one allocation per work
//!   item, sized for the top row and viewed as a tail for the others.
//! - [`cell`] — getting a cell's pixels out of a column band and its composite
//!   back into the column's preview strip.
//! - [`block`] — the background block behind each block cell, matched on the
//!   pixels the banners leave uncovered (phase 3).

pub mod block;
pub mod cell;
pub mod greedy;
pub mod perturb;
pub mod refine;
pub mod variance;
pub mod workspace;

pub use perturb::Rng;
pub use workspace::{Plane, Solution, SolveCfg, Stages, Workspace};
