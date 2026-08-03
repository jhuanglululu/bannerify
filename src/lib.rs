//! bannerify — Minecraft banner approximation of images.
//!
//! Phase 1: the width-hiding [`simd`] facade, the streamed lanczos-3
//! [`resample`] pipeline, banner-wall [`geometry`] and the [`layout`]
//! arithmetic that turns a requested wall size into a resize job, plus the
//! [`cli`] surface and the [`app`] entry point it drives.
//!
//! Phase 2: the [`color`] and [`pattern`] tables the fit is made of, the
//! [`solver`] that uses them — greedy fill, windowed beam refinement,
//! perturbation rounds and the [`oklab`] final pass — and the composed banner
//! wall it renders.
//!
//! Phase 3: the [`block`] table and the background matcher behind the banners,
//! the [`preview`] downscale both compare panes go through, and [`export`] —
//! the NBT writer, the `.schem` / `.litematic` schematics and the
//! self-contained HTML page that is the tool's output.
//!
//! The binary is a one-line shim over [`app::run_cli`]: everything real lives
//! here, so it is reachable (and reviewable) as library API.
//!
//! Allocation accounting for `--debug` comes from the [`memory`] tracking
//! allocator installed below, which wraps the system allocator process-wide.

/// Allocation tracking for `--debug` memory reporting (see [`memory`]).
#[global_allocator]
static ALLOCATOR: memory::Tracking = memory::Tracking;

pub mod app;
pub mod block;
pub mod cli;
pub mod color;
pub mod export;
pub mod geometry;
pub mod layout;
pub mod logger;
pub mod memory;
pub mod oklab;
pub mod pattern;
pub mod preview;
pub mod resample;
pub mod simd;
pub mod solver;
