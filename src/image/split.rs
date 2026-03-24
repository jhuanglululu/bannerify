use crate::banner::{Banner, NTopBanner, TopBanner};
use crate::geometry::{BANNER_W, BLOCK_SIDE, PAD_TOP, offset_column, offset_row};

/// split image into banner chunks and return (top banner, non-top banners)
pub fn split_image(image: &[u8], row: usize, column: usize) -> (Vec<TopBanner>, Vec<NTopBanner>) {
    let img_width = column * BLOCK_SIDE;

    let top = extract_banners_row(image, img_width, 0, column, PAD_TOP);

    let ntop = (1..row)
        .flat_map(|r| extract_banners_row(image, img_width, r, column, offset_row(r)))
        .collect();

    (top, ntop)
}

/// extract one row of banner
fn extract_banners_row<const HW: usize>(
    image: &[u8],
    img_width: usize,
    row: usize,
    num_col: usize,
    y_offset: usize,
) -> Vec<Banner<HW>> {
    let mut pixels: Vec<_> = (0..num_col)
        .map(|_| unsafe { Box::<[[f32; HW]; 3]>::new_uninit().assume_init() })
        .collect();

    for y in 0..HW / BANNER_W {
        let row_start = img_width * (y_offset + y);
        for (col, target) in pixels.iter_mut().enumerate() {
            let base_x = offset_column(col);
            for x in 0..BANNER_W {
                let px = 3 * (row_start + base_x + x);
                let idx = y * BANNER_W + x;
                target[0][idx] = image[px] as f32;
                target[1][idx] = image[px + 1] as f32;
                target[2][idx] = image[px + 2] as f32;
            }
        }
    }

    pixels
        .into_iter()
        .enumerate()
        .map(|(col, arr)| Banner::new(row, col, arr))
        .collect()
}
