use std::path::Path;

use colored::Colorize;
use image::{ImageReader, RgbImage};

use crate::logger::error;

pub mod resize;
pub mod split;

pub fn load_image(path: &Path) -> RgbImage {
    ImageReader::open(path)
        .unwrap_or_else(|_| {
            error!(
                "image '{}' does not exists",
                path.display().to_string().yellow()
            );
            std::process::exit(1);
        })
        .decode()
        .unwrap_or_else(|_| {
            error!(
                "image '{}' can not be decoded",
                path.display().to_string().yellow()
            );
            std::process::exit(1);
        })
        .to_rgb8()
}
