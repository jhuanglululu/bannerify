use std::path::Path;

use colored::Colorize;
use image::{ImageReader, RgbImage};

use crate::logger::error_out;

pub mod resize;
pub mod split;

pub fn load_image(path: &Path) -> RgbImage {
    ImageReader::open(path)
        .unwrap_or_else(|_| {
            error_out!(
                "image '{}' does not exists",
                path.display().to_string().yellow()
            );
        })
        .decode()
        .unwrap_or_else(|_| {
            error_out!(
                "image '{}' can not be decoded",
                path.display().to_string().yellow()
            );
        })
        .to_rgb8()
}
