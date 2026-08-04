//! bannerify — Minecraft banner approximation of images.

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
