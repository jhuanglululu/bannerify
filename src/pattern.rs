use std::collections::HashSet;

use image::{GenericImageView, load_from_memory_with_format};
use ndarray::Array3;
use rust_embed::Embed;

use crate::logger::LOGGER;
use crate::logger::{error, info};

#[derive(Embed)]
#[folder = "assets/banners/"]
#[exclude = "*.DS_Store"]
#[exclude = "*.thumbs.db"]
#[exclude = ".*"]
struct PatternImages;

pub struct Patterns {
    pub ids: Vec<String>,
    pub alphas: Array3<f32>, // (n, 40, 20)
}

pub fn load_patterns(mut excludes: HashSet<String>) -> Patterns {
    let ids: Vec<_> = PatternImages::iter()
        .filter_map(|p| {
            p.rsplit_once('.')
                .map(|(n, e)| (n.to_string(), e.to_string()))
        })
        .filter(|(name, extension)| extension == "png" && !excludes.remove(name))
        .map(|(name, _)| name)
        .collect();

    if !excludes.is_empty() {
        for pattern in excludes {
            LOGGER.error(format!("unknown pattern: {}", pattern));
        }
    }

    let length = ids.len();

    let flatten: Vec<f32> = ids
        .iter()
        .flat_map(|id| {
            let image = PatternImages::get(&format!("{}.png", id)).unwrap();
            let img = load_from_memory_with_format(&image.data, image::ImageFormat::Png)
                .unwrap_or_else(|_| {
                    error!("failed to load {}.png", id);
                    std::process::exit(1);
                });

            if img.dimensions() != (20, 40) {
                error!("{}.png does not have dimension (20, 40)", id);
                std::process::exit(1);
            }

            let rgba = img.into_rgba8();
            rgba.as_raw()
                .chunks_exact(4)
                .map(|rgba| rgba[3] as f32 / 255.0)
                .collect::<Vec<_>>()
        })
        .collect();

    let alphas = Array3::from_shape_vec((length, 40, 20), flatten).unwrap();

    info!("loaded {} patterns", ids.len());

    Patterns { ids, alphas }
}
