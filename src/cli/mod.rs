use std::path::PathBuf;

use crate::cli::dimension::Dimension;
use crate::cli::resizing::ResizingMethod;

mod color;
pub mod dimension;
pub mod resizing;

#[derive(clap::Parser)]
pub struct Args {
    /// Input image path
    #[arg(value_parser = validate_input)]
    pub input: PathBuf,
    /// Output html path
    pub output: String,

    #[command(flatten)]
    pub dimension: Dimension,

    #[command(flatten)]
    pub resizing_method: ResizingMethod,

    /// TOML config file
    #[arg(short = 'f', long = "config", value_name = "CONFIG_FILE")]
    pub config: Option<String>,

    /// Parallel workers
    #[arg(short, long, value_name = "NUMBER_OF_WORKERS", default_value_t = 1)]
    pub workers: usize,

    /// Max banners per batch chunk
    #[arg(long)]
    pub batch_size: Option<usize>,

    /// Pattern names to exclude (comma-separated)
    #[arg(help_heading = "Generation")]
    #[arg(long, value_name = "PATTERNS")]
    pub exclude_patterns: Option<String>,

    /// Layer range: MIN MAX
    #[arg(help_heading = "Generation")]
    #[arg(short, long, num_args = 2, value_names = ["MIN", "MAX"], default_values_t = [4, 6])]
    pub layers: Vec<usize>,

    /// Color candidates: MIN MAX
    #[arg(help_heading = "Generation")]
    #[arg(long, num_args = 2, value_names = ["MIN", "MAX"], default_values_t = [6, 8])]
    pub color_candidate: Vec<usize>,

    /// Perturbation search: COUNT TOP_N ROUNDS
    #[arg(help_heading = "Generation")]
    #[arg(short, long, num_args = 3, value_names = ["COUNT", "TOP_N", "ROUNDS"], default_values_t = [3, 2, 2])]
    pub perturbations: Vec<usize>,

    /// Enable CIELAB refinement pass
    #[arg(help_heading = "Generation")]
    #[arg(long, value_name = "NUMBER_OF_CANDIDATES")]
    pub lab_refine: Option<usize>,
}

fn validate_input(s: &str) -> Result<PathBuf, String> {
    let p = PathBuf::from(s);
    if !p.exists() {
        Err(format!("\n  file not found"))
    } else {
        Ok(p)
    }
}
