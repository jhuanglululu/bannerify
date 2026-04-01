use std::collections::HashSet;

use image::GenericImageView;
use rayon::prelude::*;
use rust_embed::Embed;
use wide::f32x8;

use crate::geometry::*;
use crate::logger::{error, error_out};
use crate::macros::uninit;

#[derive(Embed)]
#[folder = "assets/banners/"]
#[exclude = "*.DS_Store"]
#[exclude = "*.thumbs.db"]
#[exclude = ".*"]
struct PatternImages;

pub struct Patterns {
    pub pattern_ids: Vec<String>,
    pub alphas: Vec<[f32; TOP_HW]>,       // (n, 800)
    pub ntop_alphas: Vec<[f32; NTOP_HW]>, // (n, 480)

    pub alpha_square: Vec<f32>,
    pub ntop_alpha_square: Vec<f32>,
}

const INVERSE_255: f32 = 1.0 / 255.0;

pub fn load_patterns(excludes: &mut HashSet<String>) -> Patterns {
    let pattern_ids: Vec<_> = PatternImages::iter()
        .filter_map(|p| {
            p.rsplit_once('.')
                .map(|(n, e)| (n.to_string(), e.to_string()))
        })
        .filter(|(name, extension)| extension == "png" && !excludes.remove(name))
        .map(|(name, _)| name)
        .collect();

    let excludes: &HashSet<String> = excludes;
    if !excludes.is_empty() {
        for pattern in excludes {
            error!("unknown pattern: {}", pattern);
        }
        std::process::exit(1);
    }

    let length = pattern_ids.len();
    let mut alphas: Vec<[f32; TOP_HW]> = uninit!(length);
    let mut ntop_alphas: Vec<[f32; NTOP_HW]> = uninit!(length);
    let mut alpha_square = uninit!(length);
    let mut ntop_alpha_square = uninit!(length);

    (
        pattern_ids.par_iter(),
        alphas.par_iter_mut(),
        ntop_alphas.par_iter_mut(),
        alpha_square.par_iter_mut(),
        ntop_alpha_square.par_iter_mut(),
    )
        .into_par_iter()
        .for_each(|(p_id, alp, ntop_alp, alp2, ntop_alp2)| {
            let image = PatternImages::get(&format!("{}.png", p_id)).unwrap();
            let img = image::load_from_memory_with_format(&image.data, image::ImageFormat::Png)
                .unwrap_or_else(|_| {
                    error_out!("failed to load {}.png", p_id);
                });

            if img.dimensions() != (BANNER_W as u32, BANNER_H as u32) {
                error_out!("{}.png does not have dimension (20, 40)", p_id);
            }

            alp.iter_mut()
                .zip(img.into_rgba8().as_raw().chunks_exact(4))
                .for_each(|(alp, rgba)| {
                    *alp = rgba[3] as f32 * INVERSE_255;
                });

            ntop_alp.copy_from_slice(&alp[TOP_HW - NTOP_HW..TOP_HW]);

            let mut hidden_alp2 = f32x8::ZERO;
            let mut visible_alp2 = f32x8::ZERO;
            for px in alp[0..HIDDEN_H * BANNER_W].chunks_exact(8).map(f32x8::from) {
                hidden_alp2 += px * px;
            }

            for px in alp[HIDDEN_H * BANNER_W..TOP_HW]
                .chunks_exact(8)
                .map(f32x8::from)
            {
                visible_alp2 += px * px;
            }

            let hidden_alp2 = hidden_alp2.reduce_add();
            let visible_alp2 = visible_alp2.reduce_add();

            *alp2 = hidden_alp2 + visible_alp2;
            *ntop_alp2 = visible_alp2;
        });

    Patterns {
        pattern_ids,
        alphas,
        ntop_alphas,
        alpha_square,
        ntop_alpha_square,
    }
}
