use ndarray::Array3;

use crate::banner::Banner;
use crate::color::DyeColor;
use crate::geometry::{
    BANNER_H, BANNER_W, BLOCK_SIDE, PAD_TOP, VISIBLE_H, offset_column, offset_row,
};

/// split image into banner chunks and return (top banner, non-top banners)
pub fn split_image(image: &[u8], row: usize, column: usize) -> (Vec<Banner>, Vec<Banner>) {
    let img_width = column * BLOCK_SIDE;

    let top_banners = {
        let mut pixels = vec![Array3::<f32>::zeros((BANNER_H, BANNER_W, 3)); column];

        for y in 0..BANNER_H {
            let start = img_width * (PAD_TOP + y);
            for (c, pixel) in pixels.iter_mut().enumerate() {
                let offset_x = offset_column(c);
                for x in 0..BANNER_W {
                    let px = 3 * (start + offset_x + x);

                    pixel[[y, x, 0]] = image[px] as f32;
                    pixel[[y, x, 1]] = image[px + 1] as f32;
                    pixel[[y, x, 2]] = image[px + 2] as f32;
                }
            }
        }

        pixels
            .into_iter()
            .enumerate()
            .map(|(c, arr)| Banner {
                is_top: true,
                row: 0,
                column: c,
                target: arr,
                base: DyeColor::White,
                patterns: Vec::new(),
            })
            .collect()
    };

    let ntop_banners = (1..row)
        .flat_map(|r| {
            let mut pixels = vec![Array3::<f32>::zeros((VISIBLE_H, BANNER_W, 3)); column];

            for y in 0..VISIBLE_H {
                let start = img_width * (offset_row(r) + y);
                for (c, pixel) in pixels.iter_mut().enumerate() {
                    let offset_x = offset_column(c);
                    for x in 0..BANNER_W {
                        let px = 3 * (start + offset_x + x);

                        pixel[[y, x, 0]] = image[px] as f32;
                        pixel[[y, x, 1]] = image[px + 1] as f32;
                        pixel[[y, x, 2]] = image[px + 2] as f32;
                    }
                }
            }

            pixels
                .into_iter()
                .enumerate()
                .map(|(c, arr)| Banner {
                    is_top: false,
                    row: r,
                    column: c,
                    target: arr,
                    base: DyeColor::White,
                    patterns: Vec::new(),
                })
                .collect::<Vec<Banner>>()
        })
        .collect();

    (top_banners, ntop_banners)
}
