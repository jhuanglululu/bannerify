//! bannerify — Minecraft banner approximation of images.
//!
//! Phase 1 scaffolding: the width-hiding [`simd`] facade plus a placeholder
//! [`resample`] module for the streamed lanczos-3 pipeline.

pub mod resample;
pub mod simd;
