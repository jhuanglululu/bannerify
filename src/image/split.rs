use crate::banner::{Banner, NTopBanner, TopBanner};
use crate::geometry::*;

/// split image into banner chunks and return (top banner, non-top banners)
pub fn split_image(
    image: &[Vec<u8>; 3],
    row: usize,
    col: usize,
) -> (Vec<TopBanner>, Vec<NTopBanner>) {
    let img_width = col * BLOCK_SIDE;

    let top = extract_banners_row(image, img_width, col, PAD_TOP);

    let ntop = (1..row)
        .flat_map(|r| extract_banners_row(image, img_width, col, offset_row(r)))
        .collect();

    (top, ntop)
}

/// extract one row of banner
fn extract_banners_row<const HW: usize, const HW_DIV_8: usize>(
    image: &[Vec<u8>; 3],
    img_width: usize,
    num_col: usize,
    y_offset: usize,
) -> Vec<Banner<HW, HW_DIV_8>> {
    let mut pixels: Vec<_> = (0..num_col)
        .map(|_| unsafe { Box::<[[f32; HW]; 3]>::new_uninit().assume_init() })
        .collect();

    for y in 0..HW / BANNER_W {
        let row_start = img_width * (y_offset + y);

        for (col, target) in pixels.iter_mut().enumerate() {
            let base_x = row_start + offset_column(col);
            for x in 0..BANNER_W {
                let px = base_x + x;
                let idx = y * BANNER_W + x;
                target[0][idx] = image[0][px] as f32;
                target[1][idx] = image[1][px] as f32;
                target[2][idx] = image[2][px] as f32;
            }
        }
    }

    pixels.into_iter().map(Banner::new).collect()
}
