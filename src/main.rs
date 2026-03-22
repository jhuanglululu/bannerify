use std::fs::File;
use std::io::BufWriter;

use crate::cli::Args;
use crate::geometry::{wall_height, wall_width};
use crate::image::load_image;
use crate::image::resize::resize_image;
use crate::image::split::split_image;
use crate::logger::{elapsed, info, init_clock, since_last};
use crate::pattern::load_patterns;
use ::image::ImageEncoder;
use ::image::codecs::png::PngEncoder;
use clap::Parser;
use peak_alloc::PeakAlloc;

mod banner;
mod cli;
mod color;
mod geometry;
mod image;
mod logger;
mod pattern;

#[global_allocator]
static PEAK_ALLOC: PeakAlloc = PeakAlloc;

fn main() {
    init_clock();
    let args = Args::parse();

    let patterns = load_patterns(
        args.exclude_patterns
            .map(|x| x.rsplit(',').map(|p| p.to_string()).collect())
            .unwrap_or_default(),
    );
    info!("patterns loaded in {} ms", since_last().as_millis());

    let image = load_image(&args.input);
    info!("image loaded in {} ms", since_last().as_millis());

    let (row, column, resized_image) = resize_image(&image, args.dimension, args.resizing_method);
    info!("image resized in {} ms", since_last().as_millis());

    let mut file = File::create("output/test.png").unwrap();
    PngEncoder::new(&mut file)
        .write_image(
            &resized_image,
            wall_width(column) as u32,
            wall_height(row) as u32,
            ::image::ExtendedColorType::Rgb8,
        )
        .unwrap();

    let (top_banners, ntop_banners) = split_image(&resized_image, row, column);
    info!("image splitted in {} ms", since_last().as_millis());

    info!("program finished in {:.3} s", elapsed().as_secs_f64());
    info!("peak memory usage: {:.3} mb", PEAK_ALLOC.peak_usage_as_mb());
}
