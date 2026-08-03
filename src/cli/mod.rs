//! Command-line surface — one command, no subcommands.
//!
//! Ported from `../bannerify-old/src/cli/mod.rs`; see
//! `context/designs/cli.md`. Resizing is internal to the tool, never a user
//! step, so there is no `resize` subcommand: the user names a wall size in
//! banner rows or columns and the pipeline does the rest.
//!
//! Options under "Generation" and "Refinement" are parsed and validated now but
//! do nothing until the phase-2 solver lands; their help text says so.

use std::path::PathBuf;

pub mod config;

/// `bannerify <input> <output> [options]`
#[derive(clap::Parser)]
#[command(
    name = "bannerify",
    version,
    about = "Approximate an image with a wall of Minecraft banners",
    long_about = "Approximate an image with a wall of Minecraft banners.\n\n\
                  The image is resized to the banner wall internally — you never \
                  need to scale it yourself. Give the wall size with --row or \
                  --columns; the other axis is inferred from the aspect ratio."
)]
pub struct Args {
    /// Input image path
    pub input: PathBuf,
    /// Output html path
    pub output: PathBuf,

    /// Height of output in blocks (number of banner rows + 1)
    #[arg(short, long)]
    pub row: Option<usize>,
    /// Width of output in blocks
    #[arg(short, long)]
    pub columns: Option<usize>,

    /// TOML config file
    #[arg(short = 'f', long = "config", value_name = "CONFIG_FILE")]
    pub config: Option<PathBuf>,

    /// Parallel workers [default: CPU count]
    #[arg(short, long, value_name = "NUMBER_OF_WORKERS")]
    pub workers: Option<usize>,

    /// Print per-stage timings and dump the resized intermediate next to OUTPUT
    #[arg(long)]
    pub debug: bool,

    /// Fit image, preserving aspect ratio [default]
    #[arg(help_heading = "Layout")]
    #[arg(long)]
    pub fit: bool,
    /// Stretch image to fill empty space
    #[arg(help_heading = "Layout")]
    #[arg(long)]
    pub stretch: bool,
    /// Fill empty space with the given color (e.g. '#ff9453', 'rgb(114, 5, 14)', '9,4,87')
    #[arg(help_heading = "Layout")]
    #[arg(long, value_name = "COLOR")]
    pub fill: Option<String>,

    /// Pattern ids to exclude (comma-separated) [inert until phase 2]
    #[arg(help_heading = "Generation")]
    #[arg(short = 'P', long, value_name = "PATTERNS")]
    pub exclude_patterns: Option<String>,

    /// Block ids to exclude (comma-separated) [inert until phase 2]
    #[arg(help_heading = "Generation")]
    #[arg(short = 'B', long, value_name = "BLOCKS")]
    pub exclude_blocks: Option<String>,

    /// Layer Range: [MIN MAX] [default: 4 6] [inert until phase 2]
    #[arg(help_heading = "Generation")]
    #[arg(short = 'L', long, num_args = 2, value_names = ["MIN", "MAX"])]
    pub layer_range: Vec<usize>,

    /// Perturbation search: [TOP_N, DUPLICATES, ROUNDS] [inert until phase 2]
    #[arg(help_heading = "Generation")]
    #[arg(short = 'p', long, num_args = 3, value_names = ["TOP_N", "DUPLICATES", "ROUNDS"])]
    pub perturbations: Vec<usize>,

    /// Enable perceptual (OKLab) refinement pass [inert until phase 2]
    #[arg(help_heading = "Generation")]
    #[arg(short = 'l', long, value_name = "NUMBER_OF_CANDIDATES")]
    pub lab_refine: Option<usize>,

    /// Refinement pass count: [default: 2] [inert until phase 2]
    #[arg(help_heading = "Refinement")]
    #[arg(short = 'R', long, value_names = ["COUNT"])]
    pub refinement_pass: Option<usize>,

    /// Refinement window size: [default: 2] [inert until phase 2]
    #[arg(help_heading = "Refinement")]
    #[arg(short = 'k', long, value_names = ["SIZE"])]
    pub window_size: Option<usize>,

    /// Refinement error threshold for refinement passes (0.0 to 1.0): [default: 0.7] [inert until phase 2]
    #[arg(help_heading = "Refinement")]
    #[arg(short = 'E', long, value_name = "THRESHOLD")]
    pub error_threshold: Option<f32>,

    /// Refinement max candidate: [default: 5] [inert until phase 2]
    #[arg(help_heading = "Refinement")]
    #[arg(short = 'C', long, value_name = "NUMBER_OF_CANDIDATES")]
    pub refinement_candidate: Option<usize>,
}
