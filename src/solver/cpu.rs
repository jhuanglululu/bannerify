//! The CPU backend: [`Workspace::solve`] per cell, in parallel over the cells
//! of a batch.
//!
//! Each rayon worker solves through its own [`Workspace`], kept for the life of
//! the backend, so buffers are allocated once per worker rather than once per
//! cell or per batch. A workspace's result does not depend on which cells it
//! solved before (every field is overwritten before it is read,
//! [`super::workspace`]), so the assignment of cells to workers does not change
//! the output.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use rayon::prelude::*;

use super::batch::{Backend, PreparedCell};
use super::workspace::{Solution, SolveCfg, Stages, Workspace};

/// The existing CPU solver behind the [`Backend`] boundary.
pub struct CpuBackend {
    /// One workspace per rayon worker, indexed by
    /// [`rayon::current_thread_index`]. The lock is only ever taken by its own
    /// worker during a solve, and by [`Backend::stages`] between solves, so it
    /// is never contended. A worker holds it only inside a cell solve, which
    /// never calls back into rayon, so the worker cannot pick up a second cell
    /// while holding it.
    workspaces: Vec<Mutex<Workspace>>,
    /// Solved-banner counter behind the progress bar; bumped once per cell.
    progress: Arc<AtomicUsize>,
}

impl CpuBackend {
    /// A backend for cells of up to `max_layers` layers and a beam of
    /// `candidates` (`--refinement-candidate`), with one workspace per worker
    /// of the current rayon pool.
    pub fn new(max_layers: usize, candidates: usize, progress: Arc<AtomicUsize>) -> Self {
        Self {
            workspaces: (0..rayon::current_num_threads())
                .map(|_| Mutex::new(Workspace::new(max_layers, candidates)))
                .collect(),
            progress,
        }
    }
}

impl Backend for CpuBackend {
    fn solve(&mut self, cells: &[PreparedCell], cfg: &SolveCfg<'_>) -> Vec<Solution> {
        let workspaces = &self.workspaces;
        let progress = &self.progress;
        cells
            .par_iter()
            .map(|cell| {
                let worker = rayon::current_thread_index().unwrap_or(0) % workspaces.len();
                let mut ws = workspaces[worker].lock().expect("no worker panicked");
                ws.begin(cell.row);
                *ws.target_mut() = cell.target;
                let solution = ws.solve(cell.n_layers, cfg, &mut cell.rng.clone());
                progress.fetch_add(1, Ordering::Relaxed);
                solution
            })
            .collect()
    }

    fn stages(&self) -> Stages {
        let mut stages = Stages::default();
        for ws in &self.workspaces {
            stages.add(&ws.lock().expect("no worker panicked").stages);
        }
        stages
    }
}
