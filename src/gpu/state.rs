//! The GPU-side data: the structs shared with `kernels.metal`, the buffer set
//! a batch runs in, and how many cells and slots fit a memory budget.

use crate::geometry::TOP_HW;

use super::GpuError;
use super::metal::{Bind, Buffer, Gpu, Plain};

/// Floats in one plane.
pub(super) const PLANE: usize = TOP_HW;

/// Planes of per-cell input: blended sRGB target, OKLab target, OKLab feature.
pub(super) const CELL_PLANES: usize = 9;

/// Per-dispatch parameters (`Params` in `kernels.metal`).
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub(super) struct Params {
    pub n_slots: u32,
    pub nl: u32,
    pub n_pat: u32,
    pub cand: u32,
    pub n_exact: u32,
    pub window: u32,
    pub layer: u32,
    pub step: u32,
    pub generation: u32,
    pub start: u32,
    pub flag: u32,
    pub pad0: u32,
    pub lambda: f32,
    pub threshold: f32,
    pub pad1: f32,
    pub pad2: f32,
}

// SAFETY: `#[repr(C)]`, sixteen 4-byte fields, no padding, any bit pattern.
unsafe impl Plain for Params {}

/// Which cell a slot solves (`Slot` in `kernels.metal`).
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub(super) struct Slot {
    /// Index into the cell buffer.
    pub cell: u32,
    /// First solved plane element: 0 for banner row 0, 320 otherwise.
    pub start: u32,
}

// SAFETY: `#[repr(C)]`, two `u32`s.
unsafe impl Plain for Slot {}

/// A slot's scalar state (`Sol` in `kernels.metal`).
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub(super) struct Sol {
    pub base: u32,
    pub n_cand: u32,
    pub greedy_err: f32,
    pub rank_err: f32,
    pub lab_err: f32,
    pub pad: [u32; 3],
}

// SAFETY: `#[repr(C)]`, eight 4-byte fields, no padding, any bit pattern.
unsafe impl Plain for Sol {}

/// What a refine run is shaped by; fixed for a whole [`super::MetalBackend::solve`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Shape {
    /// Patterns.
    pub n_pat: usize,
    /// Beam width (`refinement_candidate`, at least 1).
    pub cand: usize,
    /// Shortlist length (`exact_candidates`, capped at `cand · n_pat`).
    pub n_exact: usize,
    /// `window_size`.
    pub window: usize,
    /// Largest layer count of any cell.
    pub max_layers: usize,
}

impl Shape {
    /// Whether buffers built for `self` can run work of shape `other`: the
    /// layer count only sets strides per batch, so fewer layers fit.
    pub(super) fn covers(&self, other: &Shape) -> bool {
        Shape {
            max_layers: other.max_layers,
            ..*self
        } == *other
            && self.max_layers >= other.max_layers
    }

    /// Bytes of GPU state one slot needs.
    pub(super) fn slot_bytes(&self) -> usize {
        let planes = (self.max_layers + 1) * 3 // prefixes
            + (self.max_layers + 1) * 4 // suffixes
            + 2 * self.cand * 4; // beam suffixes
        let floats = planes * PLANE
            + self.rung1_floats()
            + self.n_exact.max(1) // rung 2
            + 16 // base re-optimisation
            + 2 * self.max_layers.max(1) // layers
            + 2 * 2 * self.cand * self.window // chains
            + 4 * self.cand // best
            + 4 * self.n_exact.max(1) // shortlist
            + 2 // slot
            + 8; // sol
        floats * 4
    }

    /// Rung-1 output per slot: `(error, dye)` per (candidate, pattern), or all
    /// 16 dye errors with the exact rung off. The greedy fill's per-pattern
    /// output fits in either.
    fn rung1_floats(&self) -> usize {
        let per = if self.n_exact > 0 { 2 } else { 16 };
        self.cand * self.n_pat * per
    }
}

/// Bytes of input one cell needs.
pub(super) const CELL_BYTES: usize = CELL_PLANES * PLANE * 4;

/// Every buffer a batch runs in, sized for `cells` cells and `slots` slots of
/// `shape`.
pub(super) struct Buffers {
    pub shape: Shape,
    pub cells_cap: usize,
    pub slots_cap: usize,
    pub cells: Buffer,
    pub alpha: Buffer,
    pub alpha2: Buffer,
    pub slots: Buffer,
    pub sols: Buffer,
    pub layers: Buffer,
    pub prefixes: Buffer,
    pub suffixes: Buffer,
    pub beam: Buffer,
    pub chains: Buffer,
    pub best: Buffer,
    pub shortlist: Buffer,
    pub rung1: Buffer,
    pub rung2: Buffer,
    pub reopt: Buffer,
}

impl Buffers {
    pub(super) fn new(
        gpu: &Gpu,
        shape: Shape,
        cells_cap: usize,
        slots_cap: usize,
    ) -> Result<Self, GpuError> {
        let f = |n: usize| gpu.buffer(n * 4);
        let s = slots_cap;
        let l = shape.max_layers;
        Ok(Self {
            shape,
            cells_cap,
            slots_cap,
            cells: f(cells_cap * CELL_PLANES * PLANE)?,
            alpha: f(shape.n_pat * PLANE)?,
            alpha2: f(2 * shape.n_pat)?,
            slots: f(s * 2)?,
            sols: f(s * 8)?,
            layers: f(s * 2 * l.max(1))?,
            prefixes: f(s * (l + 1) * 3 * PLANE)?,
            suffixes: f(s * (l + 1) * 4 * PLANE)?,
            beam: f(s * 2 * shape.cand * 4 * PLANE)?,
            chains: f(s * 2 * 2 * shape.cand * shape.window)?,
            best: f(s * 4 * shape.cand)?,
            shortlist: f(s * 4 * shape.n_exact.max(1))?,
            rung1: f(s * shape.rung1_floats())?,
            rung2: f(s * shape.n_exact.max(1))?,
            reopt: f(s * 16)?,
        })
    }

    /// The binding table every encoder of this buffer set uses.
    pub(super) fn bindings(&self) -> [(Bind, &Buffer); 15] {
        [
            (Bind::Cells, &self.cells),
            (Bind::Alpha, &self.alpha),
            (Bind::Alpha2, &self.alpha2),
            (Bind::Slots, &self.slots),
            (Bind::Sols, &self.sols),
            (Bind::Layers, &self.layers),
            (Bind::Prefixes, &self.prefixes),
            (Bind::Suffixes, &self.suffixes),
            (Bind::Beam, &self.beam),
            (Bind::Chains, &self.chains),
            (Bind::Best, &self.best),
            (Bind::Shortlist, &self.shortlist),
            (Bind::Rung1, &self.rung1),
            (Bind::Rung2, &self.rung2),
            (Bind::Reopt, &self.reopt),
        ]
    }
}

/// How many cells and slots to size the buffers for, given `budget` bytes,
/// `cells` cells to solve and at most `slots_per_cell` slots per cell in any
/// one refine.
///
/// Cell inputs take at most a quarter of the budget, and never so much that
/// one slot no longer fits; slot state gets the rest. Both are capped at what
/// the work can use, so a small call allocates little. Succeeds whenever
/// `budget` is at least [`super::min_memory_budget`] for this shape.
pub(super) fn capacity(
    shape: &Shape,
    budget: usize,
    cells: usize,
    slots_per_cell: usize,
) -> Result<(usize, usize), GpuError> {
    let slot_bytes = shape.slot_bytes();
    let cells_cap = cells
        .min(budget / 4 / CELL_BYTES)
        .min(budget.saturating_sub(slot_bytes) / CELL_BYTES)
        .max(1);
    let rest = budget.saturating_sub(cells_cap * CELL_BYTES);
    let slots_cap = (rest / slot_bytes).min(cells_cap * slots_per_cell);
    if slots_cap == 0 {
        return Err(GpuError::OutOfMemory(slot_bytes + CELL_BYTES));
    }
    Ok((cells_cap, slots_cap))
}
