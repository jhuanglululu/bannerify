#![allow(dead_code, unused_variables)]
use crate::block::load_blocks;
use crate::cli::Args;
use crate::cli::config::Config;
use crate::export::export_to_html;
use crate::image::load_image;
use crate::image::resize::resize_image;
use crate::image::split::split_image;
use crate::logger::info;
use crate::logger::profiler::{finish_profiling, init_profiler, timed};
use crate::pattern::load_patterns;
use crate::solver::block::match_blocks;
use crate::solver::pipeline::process_banners;
use clap::Parser;

#[cfg(feature = "profiling")]
mod allocator;
mod banner;
mod block;
mod cli;
mod color;
mod export;
mod geometry;
mod image;
mod lab;
mod logger;
mod macros;
mod math;
mod pattern;
mod solver;

fn main() {
    init_profiler();
    let args = Args::parse();
    let mut config = Config::from(args);

    if let Some(workers) = config.workers {
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build_global()
            .unwrap();
    }

    info!("using {} workers", rayon::current_num_threads());

    let patterns = load_patterns(&mut config.exclude_patterns);
    info!("loaded {} patterns", patterns.pattern_ids.len());
    timed!("patterns loaded");

    let image = load_image(&config.input);
    timed!("image loaded");

    let (row, col, resized_image) = resize_image(&image, config.dimension, config.resizing_method);
    drop(image);
    timed!("image resized");

    info!("grid: {}x{} blocks ({} banners)", col, row + 1, col * row);

    let (top_banners, ntop_banners) = split_image(&resized_image, row, col);
    timed!("image splitted");

    let (banners, top_cache, ntop_cache) =
        process_banners(&config, patterns, top_banners, ntop_banners);
    timed!("banners processed");

    let blocks = load_blocks(&mut config.exclude_blocks);
    info!("loaded {} blocks", blocks.img_ids.len());
    timed!("blocks loaded");

    let matched_block = match_blocks(&resized_image, (row, col), &blocks);
    timed!("block matched");

    export_to_html(
        &config.output,
        (row, col),
        &blocks.pixels,
        &matched_block,
        top_cache,
        ntop_cache,
    );
    timed!("file exported");

    finish_profiling!();
}
