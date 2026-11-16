//! The GPU solver backend: greedy fill, windowed refinement and perturbation
//! rounds on the GPU, behind [`crate::solver::Backend`].
//!
//! ## Slots and lockstep
//!
//! A **slot** is one refine of one cell: its initial solve, or one
//! perturbation trial. A cell is 800 pixels, far too small to occupy a GPU on
//! its own, so the parallelism is across slots: every slot of a batch has the
//! same layer count, hence the same step schedule
//! `pass -> start layer -> window step`, and each step is a handful of
//! dispatches over the whole batch ([`schedule`]). Every per-slot decision is
//! made on the GPU; results are read back once per refine.
//!
//! ## What runs where
//!
//! - CPU, per cell, in parallel: the OKLab target and, with a non-zero feature
//!   weight, the feature map and the blended target, with the CPU solver's own
//!   code ([`crate::solver::refine::convert_target`],
//!   [`crate::solver::feature::build`]), written into the cell buffer.
//! - GPU: greedy fill, every refinement pass, the final error readout.
//! - CPU, between perturbation rounds: each cell's pool, kept exactly as
//!   [`crate::solver::perturb`] keeps it, and the next round's trials, drawn
//!   from the cell's [`crate::solver::Rng`] in the CPU solver's order. Trials
//!   of a round are one more batch of slots.
//!
//! ## Device
//!
//! The solver drives the GPU through wgpu ([`wgpu_device`]): Vulkan or DX12
//! on Linux and Windows, Metal on macOS. The kernels are `kernels.wgsl`.
//!
//! ## Agreement with the CPU solver
//!
//! The kernels evaluate the CPU kernels' expressions with the same structure
//! and break ties the same way (insertion order, first minimum), but the
//! driver compiles them in fast math: the compiler may fuse a multiply and an
//! add, reorder a sum or approximate a square root. Errors therefore agree
//! with the CPU solver to a few parts in 10^7, not bit for bit, and where two
//! candidates are that close the GPU may pick the other one. Measured on a
//! 50-row wall, 12-24% of banners differ from the CPU's while the mean error
//! differs by under 0.1%. Results are deterministic run to run on one device:
//! no kernel uses atomics or depends on scheduling order.
//!
//! ## Memory
//!
//! Slot state (prefix chain, suffix chain, two beam generations) is about
//! 90 planes of 800 `f32`. The backend sizes its buffers from a memory budget
//! ([`GpuBackend::with_memory_budget`]) and from what the device allows in
//! one buffer and one dispatch, and splits any call that exceeds them into
//! several batches, so callers may pass any number of cells.

mod kernel;
mod schedule;
mod source;
mod state;
mod wgpu_device;

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use rayon::prelude::*;

use crate::cli::config::RefinementConfig;
use crate::geometry::{NTOP_HW, TOP_HW};
use crate::pattern::Patterns;
use crate::solver::{Backend, PreparedCell, Rng, Solution, SolveCfg, Stages, Workspace};
use crate::solver::{feature, perturb, refine};

use kernel::Kernel;
use state::{Buffers, CELL_BYTES, CELL_PLANES, PLANE, Shape, Slot, Sol};
use wgpu_device::Gpu;

/// Why the backend could not start (or allocate).
#[derive(Debug)]
pub enum GpuError {
    /// No GPU this backend can use on this machine.
    NoDevice,
    /// The kernels failed to compile, or a pipeline could not be built.
    Compile(String),
    /// Another setup step failed.
    Setup(String),
    /// A buffer of this many bytes could not be allocated, or the memory
    /// budget is too small for even one slot.
    OutOfMemory(usize),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::NoDevice => write!(f, "no usable GPU found"),
            GpuError::Compile(e) => write!(f, "GPU kernels failed to compile: {e}"),
            GpuError::Setup(e) => write!(f, "GPU setup failed: {e}"),
            GpuError::OutOfMemory(n) => write!(f, "could not allocate {n} bytes of GPU memory"),
        }
    }
}

impl std::error::Error for GpuError {}

/// Default memory budget for solver buffers: 2 GiB, enough for a 40-row
/// wall's initial solves in one batch.
const DEFAULT_BUDGET: usize = 2 << 30;

/// The smallest memory budget ([`GpuBackend::with_memory_budget`]) the
/// backend can solve cells of up to `max_layers` layers in, with `n_patterns`
/// patterns and these refinement settings: the input of one cell plus the
/// state of one slot. Any budget at least this large works; a larger one only
/// means fewer, larger batches.
pub fn min_memory_budget(
    n_patterns: usize,
    refinement: &RefinementConfig,
    max_layers: usize,
) -> usize {
    CELL_BYTES + shape(n_patterns, refinement, max_layers).slot_bytes()
}

/// The buffer shape a solve with these settings runs in.
fn shape(n_pat: usize, refinement: &RefinementConfig, max_layers: usize) -> Shape {
    let cand = refinement.refinement_candidate.max(1);
    Shape {
        n_pat,
        cand,
        n_exact: refinement.exact_candidates.min(cand * n_pat),
        window: refinement.window_size.max(1),
        max_layers,
    }
}

/// The GPU implementation of [`Backend`].
///
/// [`Backend::solve`] has no error channel, so it panics if a buffer cannot be
/// allocated or the device fails mid-run; device and compile failures are
/// reported by [`GpuBackend::new`] instead.
pub struct GpuBackend {
    gpu: Gpu,
    /// Kept across calls and reused while the shape and capacity fit.
    buffers: Option<Buffers>,
    budget: usize,
    stages: Stages,
    /// Per-kernel GPU time, when profiling is on.
    profile: Option<BTreeMap<Kernel, Duration>>,
    /// Largest device allocation seen, in bytes.
    peak: usize,
    /// Batches (refines of a set of slots) run so far.
    batches: usize,
    /// Solved-banner counter behind the progress bar, if any.
    progress: Option<Arc<AtomicUsize>>,
}

/// One slot's result as read back: the refined solution, the error it is
/// ranked on, and its pure-target error.
#[derive(Clone)]
struct Scored {
    solution: Solution,
    rank: f32,
    pure: f32,
}

impl GpuBackend {
    /// Open the best real GPU on this machine and compile the kernels.
    pub fn new() -> Result<Self, GpuError> {
        Ok(Self {
            gpu: Gpu::new()?,
            buffers: None,
            budget: DEFAULT_BUDGET,
            stages: Stages::default(),
            profile: None,
            peak: 0,
            batches: 0,
            progress: None,
        })
    }

    /// Count solved cells into `counter` as the work proceeds: a share of a
    /// chunk's cells after its initial solve and after each perturbation round,
    /// so the count reaches the chunk's size once the chunk is done.
    pub fn with_progress(mut self, counter: Arc<AtomicUsize>) -> Self {
        self.progress = Some(counter);
        self
    }

    /// Cap the solver's GPU buffers at `bytes`; work beyond it is split into
    /// batches.
    pub fn with_memory_budget(mut self, bytes: usize) -> Self {
        self.budget = bytes;
        self
    }

    /// The device's name.
    pub fn device_name(&self) -> &str {
        self.gpu.name()
    }

    /// Measure time per kernel from now on. Every dispatch then waits for the
    /// previous one, so wall times measured while profiling are not
    /// representative.
    pub fn set_profiling(&mut self, on: bool) {
        self.profile = on.then(BTreeMap::new);
    }

    /// Time per kernel since profiling was switched on, longest first.
    pub fn kernel_times(&self) -> Vec<(&'static str, Duration)> {
        let mut out: Vec<_> = self
            .profile
            .iter()
            .flatten()
            .map(|(k, t)| (k.name(), *t))
            .collect();
        out.sort_by_key(|t| std::cmp::Reverse(t.1));
        out
    }

    /// Largest amount of device memory the backend has had allocated.
    pub fn peak_allocated(&self) -> usize {
        self.peak
    }

    /// Batches run so far: one per (refine or greedy fill, set of slots).
    pub fn batches(&self) -> usize {
        self.batches
    }

    /// Advance the progress counter to `part` of `parts` equal shares of a
    /// chunk of `cells` cells, from `part - 1` shares.
    fn report(&self, cells: usize, part: usize, parts: usize) {
        if let Some(counter) = &self.progress {
            let share = cells * part / parts - cells * (part - 1) / parts;
            counter.fetch_add(share, Ordering::Relaxed);
        }
    }

    /// Make sure the buffers fit `shape` with the given capacities.
    fn ensure_buffers(&mut self, shape: Shape, cells: usize, slots: usize) -> Result<(), GpuError> {
        let fits = self.buffers.as_ref().is_some_and(|b| {
            b.shape.covers(&shape) && b.cells_cap >= cells && b.slots_cap >= slots
        });
        if !fits {
            // Drop the old set first so both never exist at once.
            self.buffers = None;
            self.buffers = Some(Buffers::new(&self.gpu, shape, cells, slots)?);
            self.peak = self.peak.max(self.gpu.allocated());
        }
        Ok(())
    }
}

/// The perturbation settings, if `perturb::rounds` runs at all for a cell of
/// `n_layers` layers: it returns at once for zero layers or any zero setting.
fn perturbs(cfg: &SolveCfg<'_>, n_layers: usize) -> Option<(usize, usize, usize)> {
    let (top_n, dup, rounds) = cfg.perturbations?;
    (n_layers > 0 && top_n > 0 && dup > 0 && rounds > 0).then_some((top_n, dup, rounds))
}

impl Backend for GpuBackend {
    fn solve(&mut self, cells: &[PreparedCell], cfg: &SolveCfg<'_>) -> Vec<Solution> {
        if cells.is_empty() {
            return Vec::new();
        }
        let shape = shape(
            cfg.patterns.len(),
            cfg.refinement,
            cells.iter().map(|c| c.n_layers).max().unwrap_or(0),
        );
        let slots_per_cell = cfg
            .perturbations
            .map_or(1, |(top_n, dup, _)| (top_n * dup).max(1));
        let (cells_cap, slots_cap) = state::capacity(
            &shape,
            self.budget,
            &self.gpu.limits(),
            cells.len(),
            slots_per_cell,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        self.ensure_buffers(shape, cells_cap, slots_cap)
            .unwrap_or_else(|e| panic!("{e}"));
        upload_patterns(self.buffers.as_mut().expect("just ensured"), cfg.patterns);

        // Cells grouped by layer count, each group in input order.
        let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (i, cell) in cells.iter().enumerate() {
            groups.entry(cell.n_layers).or_default().push(i);
        }

        let mut out: Vec<Option<Solution>> = vec![None; cells.len()];
        for (&nl, members) in &groups {
            for chunk in members.chunks(cells_cap) {
                let chunk_cells: Vec<&PreparedCell> = chunk.iter().map(|&i| &cells[i]).collect();
                let solved = self.solve_chunk(&chunk_cells, nl, cfg);
                for (&i, solution) in chunk.iter().zip(solved) {
                    out[i] = Some(solution);
                }
            }
        }
        out.into_iter()
            .map(|s| s.expect("every cell is in exactly one group"))
            .collect()
    }

    fn stages(&self) -> Stages {
        self.stages
    }
}

impl GpuBackend {
    /// Solve cells that all have `nl` layers and fit the cell buffer.
    fn solve_chunk(
        &mut self,
        cells: &[&PreparedCell],
        nl: usize,
        cfg: &SolveCfg<'_>,
    ) -> Vec<Solution> {
        self.prepare(cells, cfg);

        let refining = cfg.refinement.refinement_pass > 0 && nl > 0;
        let initial: Vec<(usize, Option<&Solution>)> =
            (0..cells.len()).map(|i| (i, None)).collect();
        let first = self.run_slots(cells, nl, cfg, &initial, refining);
        // Progress in equal shares: the initial solve, then one per round.
        let parts = 1 + perturbs(cfg, nl).map_or(0, |(_, _, rounds)| rounds);
        self.report(cells.len(), 1, parts);

        let solved = match perturbs(cfg, nl) {
            Some(p) => {
                let t = Instant::now();
                let best = self.perturb(cells, nl, cfg, first, p);
                self.stages.perturb += t.elapsed();
                best
            }
            None => first,
        };

        solved
            .into_iter()
            .zip(cells)
            .map(|(s, cell)| {
                let hw = if cell.row == 0 { TOP_HW } else { NTOP_HW };
                Solution {
                    error: s.pure,
                    lab_error: s.pure / hw as f32,
                    ..s.solution
                }
            })
            .collect()
    }

    /// Write every cell's GPU inputs: blended sRGB target, OKLab target and
    /// OKLab feature map, computed by the CPU solver's own code.
    fn prepare(&mut self, cells: &[&PreparedCell], cfg: &SolveCfg<'_>) {
        let buffers = self.buffers.as_mut().expect("ensured by solve");
        let lambda = cfg.feature_weight;
        let alphas = &cfg.patterns.top;
        let t = Instant::now();
        let dst = &mut buffers.cells.slice_mut::<f32>()[..cells.len() * CELL_PLANES * PLANE];
        let (oklab, feature) = dst
            .par_chunks_mut(CELL_PLANES * PLANE)
            .zip(cells)
            .map_init(
                // Boxed: a workspace holds several planes inline, too much to keep on
                // the stack of every nested rayon job.
                || Box::new(Workspace::new(0, 1)),
                |ws, (dst, cell)| {
                    ws.begin(cell.row);
                    *ws.target_mut() = cell.target;
                    let t = Instant::now();
                    refine::convert_target(&ws.target, &mut ws.lab_target, ws.off);
                    let oklab = t.elapsed();
                    let t = Instant::now();
                    if lambda > 0.0 {
                        feature::build(ws, alphas, lambda);
                    }
                    let feature = t.elapsed();
                    let planes = ws
                        .target
                        .iter()
                        .chain(&ws.lab_target)
                        .chain(&ws.lab_feature);
                    for (out, plane) in dst.chunks_exact_mut(PLANE).zip(planes) {
                        out.copy_from_slice(&plane[..]);
                    }
                    (oklab, feature)
                },
            )
            .reduce(
                || (Duration::ZERO, Duration::ZERO),
                |a, b| (a.0 + b.0, a.1 + b.1),
            );
        // The two run interleaved in one parallel pass; split its wall time
        // between them in proportion to their CPU time.
        let wall = t.elapsed();
        let total = (oklab + feature).as_secs_f64();
        let share = if total > 0.0 {
            oklab.as_secs_f64() / total
        } else {
            1.0
        };
        self.stages.oklab += wall.mul_f64(share);
        self.stages.feature += wall.mul_f64(1.0 - share);
    }

    /// Run one refine over `slots` (cell index, starting solution), in
    /// batches of the slot capacity. A slot without a starting solution is an
    /// initial solve: greedy fill first, then refinement if `refining`.
    fn run_slots(
        &mut self,
        cells: &[&PreparedCell],
        nl: usize,
        cfg: &SolveCfg<'_>,
        slots: &[(usize, Option<&Solution>)],
        refining: bool,
    ) -> Vec<Scored> {
        let mut out = Vec::with_capacity(slots.len());
        let cap = self.buffers.as_ref().expect("ensured by solve").slots_cap;
        for batch in slots.chunks(cap) {
            out.extend(self.run_batch(cells, nl, cfg, batch, refining));
        }
        out
    }

    fn run_batch(
        &mut self,
        cells: &[&PreparedCell],
        nl: usize,
        cfg: &SolveCfg<'_>,
        batch: &[(usize, Option<&Solution>)],
        refining: bool,
    ) -> Vec<Scored> {
        let buffers = self.buffers.as_mut().expect("ensured by solve");
        let n = batch.len();
        let greedy = batch[0].1.is_none();
        debug_assert!(batch.iter().all(|s| s.1.is_none() == greedy));

        // Slot inputs.
        for (dst, &(cell, _)) in buffers.slots.slice_mut::<Slot>().iter_mut().zip(batch) {
            *dst = Slot {
                cell: cell as u32,
                start: if cells[cell].row == 0 {
                    0
                } else {
                    (TOP_HW - NTOP_HW) as u32
                },
            };
        }
        if !greedy {
            let sols = buffers.sols.slice_mut::<Sol>();
            for (dst, (_, s)) in sols.iter_mut().zip(batch) {
                dst.base = s.expect("trial slots carry a solution").base as u32;
            }
            let layers = buffers.layers.slice_mut::<u32>();
            for (i, (_, s)) in batch.iter().enumerate() {
                for (l, &(p, c)) in s.expect("checked above").layers.iter().enumerate() {
                    layers[2 * (i * nl + l)] = p as u32;
                    layers[2 * (i * nl + l) + 1] = c as u32;
                }
            }
        }

        let shape = buffers.shape;
        let p = schedule::base_params(&shape, cfg.refinement, cfg.feature_weight, n, nl);
        let bound = buffers.bindings();

        if greedy {
            let t = Instant::now();
            let mut enc = self.gpu.encoder(&bound, self.profile.as_mut());
            schedule::greedy(&mut enc, &p);
            enc.finish();
            self.stages.greedy += t.elapsed();
            self.batches += 1;
        }

        let t = Instant::now();
        let mut enc = self.gpu.encoder(&bound, self.profile.as_mut());
        if refining {
            schedule::refine(&mut enc, &p, cfg.refinement);
        }
        schedule::finish(&mut enc, &p);
        enc.finish();
        // Trial refines are timed by the perturbation stage around them.
        if greedy {
            if refining {
                self.stages.refine += t.elapsed();
            } else {
                self.stages.oklab += t.elapsed();
            }
        }
        self.batches += 1;

        buffers.sols.fetch(&self.gpu);
        buffers.layers.fetch(&self.gpu);
        let sols = buffers.sols.slice::<Sol>();
        let layers = buffers.layers.slice::<u32>();
        (0..n)
            .map(|i| {
                let s = sols[i];
                let solution = Solution {
                    base: s.base as usize,
                    layers: (0..nl)
                        .map(|l| {
                            let at = 2 * (i * nl + l);
                            (layers[at] as usize, layers[at + 1] as usize)
                        })
                        .collect(),
                    error: 0.0,
                    lab_error: 0.0,
                };
                // An initial solve that skipped refinement carries the greedy
                // fill's error into the perturbation pool, as on the CPU.
                let rank = if greedy && !refining {
                    s.greedy_err
                } else {
                    s.rank_err
                };
                Scored {
                    solution,
                    rank,
                    pure: s.lab_err,
                }
            })
            .collect()
    }

    /// `perturb::rounds` for every cell of the chunk at once: one refine over
    /// all trials per round.
    fn perturb(
        &mut self,
        cells: &[&PreparedCell],
        nl: usize,
        cfg: &SolveCfg<'_>,
        first: Vec<Scored>,
        (top_n, duplicates, rounds): (usize, usize, usize),
    ) -> Vec<Scored> {
        let n_pat = cfg.patterns.len();
        let mut rngs: Vec<Rng> = cells.iter().map(|c| c.rng.clone()).collect();
        let mut pools: Vec<Vec<Scored>> = first.into_iter().map(|s| vec![s]).collect();

        for round in 0..rounds {
            let mut trials: Vec<(usize, Solution)> = Vec::new();
            for (cell, (pool, rng)) in pools.iter().zip(&mut rngs).enumerate() {
                for entry in pool {
                    for _ in 0..duplicates {
                        let mut trial = entry.solution.clone();
                        perturb::reroll(&mut trial.layers, n_pat, rng);
                        trials.push((cell, trial));
                    }
                }
            }

            let slots: Vec<(usize, Option<&Solution>)> =
                trials.iter().map(|(cell, t)| (*cell, Some(t))).collect();
            let refined = self.run_slots(cells, nl, cfg, &slots, true);

            for ((cell, _), result) in trials.iter().zip(refined) {
                pools[*cell].push(result);
            }
            for pool in &mut pools {
                pool.sort_by(|a, b| a.rank.partial_cmp(&b.rank).expect("no NaN errors"));
                pool.truncate(top_n);
            }
            self.report(cells.len(), round + 2, rounds + 1);
        }

        pools
            .into_iter()
            .map(|mut pool| pool.swap_remove(0))
            .collect()
    }
}

/// Upload the pattern alphas and both `Σ α²` tables.
fn upload_patterns(buffers: &mut Buffers, patterns: &Patterns) {
    let alpha = buffers.alpha.slice_mut::<f32>();
    for (dst, src) in alpha.chunks_exact_mut(PLANE).zip(&patterns.top) {
        dst.copy_from_slice(&src[..]);
    }
    let alpha2 = buffers.alpha2.slice_mut::<f32>();
    let n = patterns.len();
    alpha2[..n].copy_from_slice(&patterns.top_alpha2);
    alpha2[n..2 * n].copy_from_slice(&patterns.lower_alpha2);
}
