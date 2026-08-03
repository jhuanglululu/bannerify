//! Command-line surface — one command, no subcommands.
//!
//! Ported from `../bannerify-old/src/cli/mod.rs`; see
//! `context/designs/cli.md`. Resizing is internal to the tool, never a user
//! step, so there is no `resize` subcommand: the user names a wall size in
//! banner rows or columns and the pipeline does the rest.
//!
//! Every "Generation" and "Refinement" option is live as of solver stage 2b.
//! `--exclude-blocks` is the one that still does nothing: block matching is
//! phase 3, and its help text says so.
//!
//! `--debug` is logging only: per-stage timings and memory, no file dumping.
//! While the pipeline is partial the tool's normal output is the current step's
//! intermediate, so there is nothing extra to dump: today `<OUTPUT>` is the
//! composed banner wall and `<OUTPUT>.jsonl` the per-cell solution.

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
    /// Output path (an HTML export once the pipeline is complete; for now it
    /// receives this stage's intermediates: the composed banner wall as a PNG,
    /// and the per-cell solution as JSONL at the same path with a .jsonl
    /// extension)
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

    /// Log per-stage wall timings and peak memory (no extra files are written)
    #[arg(long)]
    pub debug: bool,

    /// Perturbation RNG seed [default: 0]; runs are fully deterministic per
    /// seed, and different seeds explore different perturbation draws. Only
    /// meaningful together with --perturbations
    #[arg(short = 's', long, value_name = "SEED")]
    pub seed: Option<u64>,

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

    /// Pattern ids to exclude (comma-separated), e.g. 'globe,mojang'; the
    /// solver never lays an excluded pattern
    #[arg(help_heading = "Generation")]
    #[arg(short = 'P', long, value_name = "PATTERNS")]
    pub exclude_patterns: Option<String>,

    /// Block ids to exclude (comma-separated) [inert until phase 3]
    #[arg(help_heading = "Generation")]
    #[arg(short = 'B', long, value_name = "BLOCKS")]
    pub exclude_blocks: Option<String>,

    /// Layer Range: [MIN MAX] [default: 4 6]; flat cells get MIN layers, the
    /// busiest cell of the wall gets MAX
    #[arg(help_heading = "Generation")]
    #[arg(short = 'L', long, num_args = 2, value_names = ["MIN", "MAX"])]
    pub layer_range: Vec<usize>,

    /// Perturbation search: [TOP_N, DUPLICATES, ROUNDS] [default: off]; keep
    /// TOP_N fits, make DUPLICATES copies of each with 1-2 random layers
    /// re-rolled, re-refine them, keep the best TOP_N, ROUNDS times. Any value
    /// 0 disables it
    #[arg(help_heading = "Generation")]
    #[arg(short = 'p', long, num_args = 3, value_names = ["TOP_N", "DUPLICATES", "ROUNDS"])]
    pub perturbations: Vec<usize>,

    /// Final perceptual (OKLab) pass, scoring the N best candidates per layer
    /// exactly instead of by the closed-form RGB error [default: off]
    #[arg(help_heading = "Generation")]
    #[arg(short = 'l', long, value_name = "NUMBER_OF_CANDIDATES")]
    pub lab_refine: Option<usize>,

    /// Refinement pass count: [default: 2]; 0 stops after the greedy fill
    #[arg(help_heading = "Refinement")]
    #[arg(short = 'R', long, value_names = ["COUNT"])]
    pub refinement_pass: Option<usize>,

    /// Refinement window size, in layers re-chosen together: [default: 2]
    #[arg(help_heading = "Refinement")]
    #[arg(short = 'k', long, value_names = ["SIZE"])]
    pub window_size: Option<usize>,

    /// Refinement error threshold for refinement passes (0.0 to 1.0): [default: 0.7]
    #[arg(help_heading = "Refinement")]
    #[arg(short = 'E', long, value_name = "THRESHOLD")]
    pub error_threshold: Option<f32>,

    /// Refinement beam width, in candidates kept per window step: [default: 5]
    #[arg(help_heading = "Refinement")]
    #[arg(short = 'C', long, value_name = "NUMBER_OF_CANDIDATES")]
    pub refinement_candidate: Option<usize>,
}
