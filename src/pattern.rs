use std::collections::HashSet;

use image::{GenericImageView, load_from_memory_with_format};
use rust_embed::Embed;

use crate::geometry::{BANNER_H, BANNER_W, NTOP_HW, TOP_HW};
use crate::logger::{error, info};

#[derive(Embed)]
#[folder = "assets/banners/"]
#[exclude = "*.DS_Store"]
#[exclude = "*.thumbs.db"]
#[exclude = ".*"]
struct PatternImages;

pub struct Patterns {
    pub ids: Vec<String>,
    pub alphas: Vec<[f32; TOP_HW]>,       // (n, 800)
    pub ntop_alphas: Vec<[f32; NTOP_HW]>, // (n, 480)
}

const INVERSE_255: f32 = 1.0 / 255.0;

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
            error!("unknown pattern: {}", pattern);
        }
    }

    let length = ids.len();
    let mut alphas: Vec<[f32; TOP_HW]> = Vec::with_capacity(length);
    let mut ntop_alphas: Vec<[f32; NTOP_HW]> = Vec::with_capacity(length);

    unsafe {
        alphas.set_len(length);
        ntop_alphas.set_len(length);
    }

    for (idx, id) in ids.iter().enumerate() {
        let image = PatternImages::get(&format!("{}.png", id)).unwrap();
        let img = load_from_memory_with_format(&image.data, image::ImageFormat::Png)
            .unwrap_or_else(|_| {
                error!("failed to load {}.png", id);
                std::process::exit(1);
            });

        if img.dimensions() != (BANNER_W as u32, BANNER_H as u32) {
            error!("{}.png does not have dimension (20, 40)", id);
            std::process::exit(1);
        }

        let pattern_alphas = &mut alphas[idx];

        pattern_alphas
            .iter_mut()
            .zip(img.into_rgba8().as_raw().chunks_exact(4))
            .for_each(|(dst, rgba)| {
                *dst = rgba[3] as f32 * INVERSE_255;
            });

        ntop_alphas[idx].copy_from_slice(&pattern_alphas[TOP_HW - NTOP_HW..TOP_HW]);
    }

    info!("loaded {} patterns", length);

    Patterns {
        ids,
        alphas,
        ntop_alphas,
    }
}
