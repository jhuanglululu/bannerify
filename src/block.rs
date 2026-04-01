use std::collections::HashSet;

use std::ops::Range;

use fast_image_resize::{ResizeOptions, Resizer};
use image::GenericImageView;
use rayon::prelude::*;
use rust_embed::Embed;
use wide::f32x8;

use crate::geometry::*;
use crate::lab::rgb_to_lab;
use crate::logger::{error, error_out};
use crate::macros::uninit;

const HOLLOW_X8_LEN: usize = HOLLOW_BLOCK_PIXELS / 8;

pub type HollowBlock = [[f32x8; HOLLOW_X8_LEN]; 3];

const LEFT_MASK: [f32; 8] = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
const RIGHT_MASK: [f32; 8] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];

pub const TOP_MASK: [f32x8; HOLLOW_X8_LEN] = {
    let mut out = [f32x8::ONE; HOLLOW_X8_LEN];
    let mut idx = HOLLOW_X8_LEN - PAD_BOTTOM * 3;
    while idx < HOLLOW_X8_LEN {
        out[idx] = f32x8::new(LEFT_MASK);
        out[idx + 1] = f32x8::ZERO;
        out[idx + 2] = f32x8::new(RIGHT_MASK);
        idx += 3;
    }
    out
};

pub const MIDDLE_MASK: [f32x8; HOLLOW_X8_LEN] = {
    let mut out = [f32x8::ONE; HOLLOW_X8_LEN];

    let mut idx = 0;
    while idx < 3 * PAD_TOP {
        out[idx] = f32x8::new(LEFT_MASK);
        out[idx + 1] = f32x8::ZERO;
        out[idx + 2] = f32x8::new(RIGHT_MASK);
        idx += 3;
    }

    let mut idx = HOLLOW_X8_LEN - PAD_BOTTOM * 3;
    while idx < HOLLOW_X8_LEN {
        out[idx] = f32x8::new(LEFT_MASK);
        out[idx + 1] = f32x8::ZERO;
        out[idx + 2] = f32x8::new(RIGHT_MASK);
        idx += 3;
    }
    out
};

pub const BOTTOM_MASK: [f32x8; HOLLOW_X8_LEN] = {
    let mut out = [f32x8::ONE; HOLLOW_X8_LEN];
    let mut idx = 0;
    while idx < 3 * PAD_TOP {
        out[idx] = f32x8::new(LEFT_MASK);
        out[idx + 1] = f32x8::ZERO;
        out[idx + 2] = f32x8::new(RIGHT_MASK);
        idx += 3;
    }
    out
};

#[derive(Embed)]
#[folder = "assets/blocks/"]
#[exclude = "*.DS_Store"]
#[exclude = "*.thumbs.db"]
#[exclude = ".*"]
struct BlockImages;

pub struct Blocks {
    pub img_ids: Vec<String>,
    pub pixels: Vec<[u8; 3 * BLOCK_PIXELS]>,
    pub hollow: Vec<HollowBlock>,
}

pub fn load_blocks(excludes: &mut HashSet<String>) -> Blocks {
    let img_ids: Vec<_> = BlockImages::iter()
        .filter_map(|p| {
            p.rsplit_once('.')
                .map(|(n, e)| (n.to_string(), e.to_string()))
        })
        .filter(|(name, extension)| extension == "jpg" && !excludes.remove(name))
        .map(|(name, _)| name)
        .collect();

    let excludes: &HashSet<String> = excludes;
    if !excludes.is_empty() {
        for pattern in excludes {
            error!("unknown block: {}", pattern);
        }
        std::process::exit(1);
    }

    let length = img_ids.len();
    let mut pixels: Vec<[u8; 3 * BLOCK_PIXELS]> = uninit!(length);
    let mut hollow: Vec<HollowBlock> = uninit!(length);

    (
        img_ids.par_iter(),
        pixels.par_iter_mut(),
        hollow.par_iter_mut(),
    )
        .into_par_iter()
        .for_each(|(img_id, px, hollow)| {
            let image_data = BlockImages::get(&format!("{}.jpg", img_id)).unwrap();
            let image =
                image::load_from_memory_with_format(&image_data.data, image::ImageFormat::Jpeg)
                    .unwrap_or_else(|_| {
                        error_out!("failed to load {}.jpg", img_id);
                    });

            if image.dimensions() != (16, 16) {
                error_out!("{}.png does not have dimension (16, 16)", img_id);
            }

            let mut resized_img = fast_image_resize::images::Image::new(
                BLOCK_SIDE as u32,
                BLOCK_SIDE as u32,
                fast_image_resize::PixelType::U8x3,
            );
            let mut resizer = Resizer::new();
            resizer
                .resize(&image, &mut resized_img, &ResizeOptions::new())
                .unwrap_or_else(|e| error_out!("{}", e));

            let resized_vec = resized_img.into_vec();

            // build a hollow image

            let mut hollow_rgb: [[f32; HOLLOW_BLOCK_PIXELS]; 3] = uninit!();

            const TOP_RANGE: Range<usize> = 0..PAD_TOP * BLOCK_SIDE;
            for px_idx in TOP_RANGE {
                hollow_rgb[0][px_idx] = resized_vec[3 * px_idx] as f32;
                hollow_rgb[1][px_idx] = resized_vec[3 * px_idx + 1] as f32;
                hollow_rgb[2][px_idx] = resized_vec[3 * px_idx + 2] as f32;
            }

            const MID_RANGE: Range<usize> = TOP_RANGE.end..TOP_RANGE.end + PAD_TWO_SIDE * MID_SIDE;
            const MID_PX_LEN: usize = MID_RANGE.end - MID_RANGE.start;
            const MID_PX_ARR: [usize; MID_PX_LEN] = {
                let mut out = [0; MID_PX_LEN];
                let mut r = PAD_TOP;
                let mut i = 0;
                while r < BLOCK_SIDE - PAD_BOTTOM {
                    out[i] = r * BLOCK_SIDE + MIDDLE_OFFSET[0];
                    out[i + 1] = r * BLOCK_SIDE + MIDDLE_OFFSET[1];
                    out[i + 2] = r * BLOCK_SIDE + MIDDLE_OFFSET[2];
                    out[i + 3] = r * BLOCK_SIDE + MIDDLE_OFFSET[3];
                    i += PAD_TWO_SIDE;
                    r += 1;
                }
                out
            };
            for (out_idx, px_idx) in MID_RANGE.zip(MID_PX_ARR.iter()) {
                hollow_rgb[0][out_idx] = resized_vec[3 * px_idx] as f32;
                hollow_rgb[1][out_idx] = resized_vec[3 * px_idx + 1] as f32;
                hollow_rgb[2][out_idx] = resized_vec[3 * px_idx + 2] as f32;
            }

            const BOT_RANGE: Range<usize> = MID_RANGE.end..MID_RANGE.end + PAD_BOTTOM * BLOCK_SIDE;
            const BOT_PX_RANGE: Range<usize> = BLOCK_PIXELS - PAD_BOTTOM * BLOCK_SIDE..BLOCK_PIXELS;
            for (out_idx, px_idx) in BOT_RANGE.zip(BOT_PX_RANGE) {
                hollow_rgb[0][out_idx] = resized_vec[3 * px_idx] as f32;
                hollow_rgb[1][out_idx] = resized_vec[3 * px_idx + 1] as f32;
                hollow_rgb[2][out_idx] = resized_vec[3 * px_idx + 2] as f32;
            }

            // compute lab from srgb

            for px_idx in (0..HOLLOW_BLOCK_PIXELS).step_by(8) {
                let (l, a, b) = rgb_to_lab(
                    &hollow_rgb[0][px_idx..px_idx + 8],
                    &hollow_rgb[1][px_idx..px_idx + 8],
                    &hollow_rgb[2][px_idx..px_idx + 8],
                );

                hollow[0][px_idx / 8] = l;
                hollow[1][px_idx / 8] = a;
                hollow[2][px_idx / 8] = b;
            }

            px.copy_from_slice(
                &<[u8; 3 * BLOCK_PIXELS]>::try_from(resized_vec).unwrap_or_else(|e| {
                    error_out!("shape mismatch for block: 1728 and {}", e.len())
                }),
            );
        });

    Blocks {
        img_ids,
        pixels,
        hollow,
    }
}
