use crate::cli::Args;
use crate::cli::complexity::ComplexityOptions;
use crate::image::load_image;
use crate::image::resize::resize_image;
use crate::image::split::split_image;
use crate::logger::info;
use crate::logger::profiler::{finish_profiling, init_profiler, timed};
use crate::pattern::load_patterns;
use crate::solver::greedy::complexity::sort_banner_greedy;
use crate::solver::optimal::complexity::sort_banner_optimal;
use clap::Parser;

#[cfg(feature = "profiling")]
mod allocator;
mod banner;
mod cli;
mod color;
mod geometry;
mod image;
mod logger;
mod pattern;
mod solver;

fn main() {
    init_profiler();
    let args = Args::parse();

    let patterns = load_patterns(
        args.exclude_patterns
            .map(|x| x.rsplit(',').map(|p| p.to_string()).collect())
            .unwrap_or_default(),
    );
    timed!("patterns loaded");

    let image = load_image(&args.input);
    timed!("image loaded");

    let (row, column, resized_image) = resize_image(&image, args.dimension, args.resizing_method);
    timed!("image resized");

    info!(
        "grid: {}x{} blocks ({} banners)",
        column,
        row + 1,
        column * row
    );

    let (top_banners, ntop_banners) = split_image(&resized_image, row, column);
    timed!("image splitted");

    match &args.complexity {
        ComplexityOptions::Optimal(complexity) => {
            info!("using optimal-color algoirthm");

            let mut layer_dist = vec![0; complexity.layers.1 - complexity.layers.0 + 1];
            let top_tasks = sort_banner_optimal(top_banners, complexity, &mut layer_dist);
            let ntop_tasks = sort_banner_optimal(ntop_banners, complexity, &mut layer_dist);
            timed!("banners sorted");
            info!(
                "layer distribution: {}",
                format_distrbution(complexity.layers.0, layer_dist)
            );
        }
        ComplexityOptions::Greedy(complexity) => {
            info!("using greedy algoirthm");

            let mut layer_dist = vec![0; complexity.layers.1 - complexity.layers.0 + 1];
            let mut color_dist = vec![0; complexity.colors.1 - complexity.colors.0 + 1];
            let top_tasks =
                sort_banner_greedy(top_banners, complexity, &mut layer_dist, &mut color_dist);
            let ntop_tasks =
                sort_banner_greedy(ntop_banners, complexity, &mut layer_dist, &mut color_dist);
            timed!("banners sorted");
            info!(
                "layer distribution: {}",
                format_distrbution(complexity.layers.0, layer_dist)
            );
            info!(
                "color distribution: {}",
                format_distrbution(complexity.colors.0, color_dist)
            );
        }
    }

    finish_profiling!();
}

fn format_distrbution(offset: usize, dist: Vec<usize>) -> String {
    format!(
        "{{ {} }}",
        dist.iter()
            .enumerate()
            .map(|(i, count)| format!("{}: {}", i + offset, count))
            .collect::<Vec<_>>()
            .join(", ")
    )
}
