//! The solver backend boundary: prepared cells in, solutions out.
//!
//! The pipeline prepares cells (resample, gather), hands them to a [`Backend`]
//! in batches, and paints the solutions it gets back. Everything a backend
//! needs about a cell is in its [`PreparedCell`]; everything shared across the
//! run is in [`SolveCfg`].

use super::perturb::Rng;
use super::workspace::{Plane, Solution, SolveCfg, Stages};

/// One banner cell, ready to solve.
pub struct PreparedCell {
    /// Banner row. Row 0 solves the whole patch; every other row solves only
    /// the lane-aligned tail of it ([`super::workspace::LOWER_OFF`]), and the
    /// head of its planes is unspecified.
    pub row: usize,
    /// Pattern layers this cell gets.
    pub n_layers: usize,
    /// The cell's sRGB target as gathered from the resampled image: not
    /// converted, not blended with a feature map.
    pub target: [Plane; 3],
    /// The perturbation stream positioned where this cell's draws start, so
    /// the cell's result does not depend on which batch it lands in.
    pub rng: Rng,
}

/// Something that solves batches of cells.
pub trait Backend {
    /// Solve every cell of `cells`, returning one [`Solution`] per cell in the
    /// same order.
    fn solve(&mut self, cells: &[PreparedCell], cfg: &SolveCfg<'_>) -> Vec<Solution>;

    /// Time spent per solver stage, accumulated over every call so far.
    fn stages(&self) -> Stages;
}
