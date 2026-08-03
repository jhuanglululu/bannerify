//! bannerify — Minecraft banner approximation of images.
//!
//! Phase 1: the width-hiding [`simd`] facade, the streamed lanczos-3
//! [`resample`] pipeline, banner-wall [`geometry`] and the [`layout`]
//! arithmetic that turns a requested wall size into a resize job, plus the
//! [`cli`] surface and the [`app`] entry point it drives. The solver, block
//! matching and export land in phase 2.
//!
//! The binary is a one-line shim over [`app::run`]: everything real lives here,
//! so it is reachable (and reviewable) as library API.

pub mod app;
pub mod cli;
pub mod geometry;
pub mod layout;
pub mod logger;
pub mod resample;
pub mod simd;
