use std::io::Cursor;

use rayon::prelude::*;

use crate::geometry::*;
use crate::logger::error_out;

#[allow(clippy::uninit_vec)]
pub fn banner_to_buffer(
    block_assets: &[[u8; 3 * BLOCK_PIXELS]],
    dimension: (usize, usize), // (row, col)
    blocks: &[usize],
    top_cache: &[[[f32; TOP_HW]; 3]],
    ntop_cache: &[[[f32; NTOP_HW]; 3]],
) -> Vec<u8> {
    let px_count = BLOCK_PIXELS * (dimension.0 + 1) * dimension.1;
    let last_banner_pixel_row = BLOCK_SIDE * (dimension.0 + 1) - PAD_BOTTOM;
    let mut buffer = Vec::with_capacity(3 * px_count);
    unsafe {
        buffer.set_len(3 * px_count);
    }

    let line_per_row = BLOCK_SIDE * dimension.1;

    buffer
        .par_chunks_mut(3 * BLOCK_SIDE)
        .enumerate()
        .for_each(|(line_idx, pixels)| {
            let scanline = (line_idx / dimension.1) % BLOCK_SIDE;

            let block_row = line_idx / line_per_row;
            let block_col = line_idx % dimension.1;
            let block = blocks[block_row * dimension.1 + block_col];

            let block_start = scanline * BLOCK_SIDE * 3;
            pixels.copy_from_slice(&block_assets[block][block_start..block_start + BLOCK_SIDE * 3]);

            let img_y = line_idx / dimension.1;

            if img_y < PAD_TOP || img_y >= last_banner_pixel_row {
            } else if img_y < PAD_TOP + BANNER_H {
                let banner_start = (img_y - PAD_TOP) * BANNER_W;
                let banner = top_cache[line_idx % dimension.1];

                for ban_col in 0..BANNER_W {
                    let px_col = ban_col + PAD_SIDE;
                    pixels[3 * px_col] = banner[0][banner_start + ban_col] as u8;
                    pixels[3 * px_col + 1] = banner[1][banner_start + ban_col] as u8;
                    pixels[3 * px_col + 2] = banner[2][banner_start + ban_col] as u8;
                }
            } else {
                let ntop_banner_row = (img_y - offset_row(1)) / VISIBLE_H;
                let banner_y = (img_y - offset_row(1)) % VISIBLE_H;
                let banner_start = banner_y * BANNER_W;
                let banner = ntop_cache[ntop_banner_row * dimension.1 + line_idx % dimension.1];

                for ban_col in 0..BANNER_W {
                    let px_col = ban_col + PAD_SIDE;
                    pixels[3 * px_col] = banner[0][banner_start + ban_col] as u8;
                    pixels[3 * px_col + 1] = banner[1][banner_start + ban_col] as u8;
                    pixels[3 * px_col + 2] = banner[2][banner_start + ban_col] as u8;
                }
            }
        });

    buffer
}

pub fn buffer_to_base64(
    dimension: (usize, usize), // (row, col)
    buffer: Vec<u8>,
) -> String {
    let width = dimension.1 * BLOCK_SIDE;
    let height = (dimension.0 + 1) * BLOCK_SIDE;
    let img = image::RgbImage::from_raw(width as u32, height as u32, buffer).unwrap();
    let mut png_bytes = Vec::new();
    img.write_to(&mut Cursor::new(&mut png_bytes), image::ImageFormat::Png)
        .unwrap_or_else(|e| error_out!("{}", e));
    base64_simd::STANDARD.encode_to_string(png_bytes)
}
