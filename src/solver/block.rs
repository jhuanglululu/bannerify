use std::mem::MaybeUninit;

use rayon::prelude::*;
use wide::f32x8;

use crate::block::{BOTTOM_MASK, Blocks, MIDDLE_MASK, TOP_MASK};
use crate::geometry::*;
use crate::lab::rgb_to_lab;

#[allow(clippy::uninit_assumed_init, invalid_value)]
pub fn match_blocks(
    image: &[Vec<u8>; 3],
    dimension: (usize, usize), // (row, col)
    blocks: &Blocks,
) -> Vec<usize> {
    let rows = dimension.0 + 1;
    let cols = dimension.1;
    let count = rows * cols;

    let img_width = cols * BLOCK_SIDE;

    (0..count)
        .into_par_iter()
        .map(|idx| {
            let row = idx / cols;
            let col = idx % cols;
            let mut best = 0;
            let mut min_err: f32 = f32::INFINITY;

            let mut tar_hollow: [[f32x8; HOLLOW_BLOCK_PIXELS / 8]; 3] =
                unsafe { MaybeUninit::uninit().assume_init() };

            let mask = if row == 0 {
                TOP_MASK
            } else if row == rows - 1 {
                BOTTOM_MASK
            } else {
                MIDDLE_MASK
            };

            extract_hollow_lab(image, &mut tar_hollow, img_width, row, col);
            for (block_idx, block) in blocks.hollow.iter().enumerate() {
                let mut err_x8 = f32x8::ZERO;

                for ch_idx in 0..3 {
                    for px_idx_x8 in 0..HOLLOW_BLOCK_PIXELS / 8 {
                        let tar = tar_hollow[ch_idx][px_idx_x8];
                        let blk = block[ch_idx][px_idx_x8];
                        let diff = tar - blk;
                        err_x8 += diff * diff * mask[px_idx_x8];
                    }
                }

                let err = err_x8.reduce_add();
                if err < min_err {
                    best = block_idx;
                    min_err = err;
                }
            }

            best
        })
        .collect()
}

#[allow(clippy::uninit_assumed_init, invalid_value)]
fn extract_hollow_lab(
    image: &[Vec<u8>; 3],
    tar_buf: &mut [[f32x8; HOLLOW_BLOCK_PIXELS / 8]; 3],
    img_width: usize,
    row: usize,
    col: usize,
) {
    let mut hollow_rgb: [[f32; HOLLOW_BLOCK_PIXELS]; 3] =
        unsafe { MaybeUninit::uninit().assume_init() };

    let col_offset = col * BLOCK_SIDE;

    for ch_idx in 0..3 {
        for r in 0..PAD_TOP {
            let img_y = r + row * BLOCK_SIDE;
            let img_row_start = img_y * img_width + col_offset;
            let tar_row_start = r * BLOCK_SIDE;
            for c in 0..BLOCK_SIDE {
                hollow_rgb[ch_idx][tar_row_start + c] = image[ch_idx][img_row_start + c] as f32;
            }
        }

        const MID_START: usize = PAD_TOP * BLOCK_SIDE;
        for r in 0..MID_SIDE {
            let img_y = r + PAD_TOP + row * BLOCK_SIDE;
            let img_row_start = img_y * img_width + col_offset;
            let tar_row_start = MID_START + r * PAD_TWO_SIDE;
            for (i, &c) in MIDDLE_OFFSET.iter().enumerate() {
                hollow_rgb[ch_idx][tar_row_start + i] = image[ch_idx][img_row_start + c] as f32;
            }
        }

        const BOT_START: usize = MID_START + MID_SIDE * PAD_TWO_SIDE;
        for r in 0..PAD_BOTTOM {
            let img_y = r + PAD_TOP + MID_SIDE + row * BLOCK_SIDE;
            let img_row_start = img_y * img_width + col_offset;
            let tar_row_start = BOT_START + r * BLOCK_SIDE;
            for c in 0..BLOCK_SIDE {
                hollow_rgb[ch_idx][tar_row_start + c] = image[ch_idx][img_row_start + c] as f32;
            }
        }
    }

    for px_idx in (0..HOLLOW_BLOCK_PIXELS).step_by(8) {
        let (l, a, b) = rgb_to_lab(
            &hollow_rgb[0][px_idx..px_idx + 8],
            &hollow_rgb[1][px_idx..px_idx + 8],
            &hollow_rgb[2][px_idx..px_idx + 8],
        );

        tar_buf[0][px_idx / 8] = l;
        tar_buf[1][px_idx / 8] = a;
        tar_buf[2][px_idx / 8] = b;
    }
}
