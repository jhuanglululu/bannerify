//! Command-line surface — one command, no subcommands.

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
    /// Output path: one self-contained HTML page with the preview, the crafting
    /// guide and the .schem / .litematic downloads embedded in it
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

    /// Largest dimension of the preview images embedded in the HTML
    /// [default: the banner wall's own size, no resize after the solve]; never
    /// larger than the wall itself, since upscaling a banner wall only
    /// inflates the file
    #[arg(long, value_name = "PIXELS")]
    pub preview: Option<usize>,

    /// Also write the full-resolution banner wall to this PNG path; it is NOT
    /// embedded in the HTML, whose preview is the downscaled one
    #[arg(long, value_name = "PATH")]
    pub render: Option<PathBuf>,

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

    /// Block ids to exclude (comma-separated), e.g. 'beacon,ancient_debris';
    /// the background matcher never picks an excluded block
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

    /// Weight of the feature-map term: [default: 0.5]. Every cell is also
    /// fitted by an idealised 2-layer banner in pure dye colors, and the solver
    /// is pulled towards it by this weight -- cleaner flat areas and crisper
    /// edges, at the cost of some pixel-level accuracy. 0 disables it
    #[arg(help_heading = "Generation")]
    #[arg(short = 'F', long, value_name = "WEIGHT")]
    pub feature_weight: Option<f32>,

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

    /// Candidates per window step re-scored exactly in OKLab: [default: 20].
    /// 0 disables exact scoring (faster, worse); otherwise keep it well above
    /// the beam width (--refinement-candidate)
    #[arg(help_heading = "Refinement")]
    #[arg(short = 'x', long, value_name = "NUMBER_OF_CANDIDATES")]
    pub exact_candidates: Option<usize>,
}
