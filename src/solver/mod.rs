//! The banner solver: which dyes and patterns approximate each cell.

pub mod block;
pub mod cell;
pub mod greedy;
pub mod perturb;
pub mod refine;
pub mod variance;
pub mod workspace;

pub use perturb::Rng;
pub use workspace::{Plane, Solution, SolveCfg, Stages, Workspace};
