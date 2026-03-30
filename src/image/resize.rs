use fast_image_resize::{PixelType, ResizeOptions, Resizer};
use image::RgbImage;

use crate::cli::config::{Dimension, ResizingMethod};
use crate::geometry::*;
use crate::logger::error_out;

pub fn resize_image(
    source: &RgbImage,
    dimension: Dimension,
    resizing_method: ResizingMethod,
) -> (usize, usize, [Vec<u8>; 3]) {
    let (row, col, img_interleaved) = match resizing_method {
        ResizingMethod::Fit => fit_image(source, dimension),
        ResizingMethod::Fill(color) => fill_image(source, dimension, color),
        ResizingMethod::Stretch => stretch_image(source, dimension),
    };

    let ch_len = img_interleaved.len() / 3;
    let mut r = Vec::with_capacity(ch_len);
    let mut g = Vec::with_capacity(ch_len);
    let mut b = Vec::with_capacity(ch_len);

    unsafe {
        r.set_len(ch_len);
        g.set_len(ch_len);
        b.set_len(ch_len);
    }

    for (px_idx, px) in img_interleaved.chunks_exact(3).enumerate() {
        r[px_idx] = px[0];
        g[px_idx] = px[1];
        b[px_idx] = px[2];
    }

    (row, col, [r, g, b])
}

fn fit_image(image: &RgbImage, dimension: Dimension) -> (usize, usize, Vec<u8>) {
    let (img_w, img_h) = image.dimensions();

    let (row, col) = match dimension {
        Dimension::Row(row) => (row, infer_dimension(img_w, img_h, row + 1)),
        Dimension::Column(col) => (infer_dimension(img_h, img_w, col) - 1, col),
    };

    let width = wall_width(col) as u32;
    let height = wall_height(row) as u32;

    // resize + crop
    let scale = f64::max(width as f64 / img_w as f64, height as f64 / img_h as f64);
    let crop_w = width as f64 / scale;
    let crop_h = height as f64 / scale;
    let crop_x = (img_w as f64 - crop_w) / 2.0;
    let crop_y = (img_h as f64 - crop_h) / 2.0;
    let mut resized_img = fast_image_resize::images::Image::new(width, height, PixelType::U8x3);
    let mut resizer = Resizer::new();
    let options = ResizeOptions::new().crop(crop_x, crop_y, crop_w, crop_h);

    resizer
        .resize(image, &mut resized_img, &options)
        .unwrap_or_else(|e| {
            error_out!("{}", e);
        });

    (row, col, resized_img.into_vec())
}

fn fill_image(image: &RgbImage, dimension: Dimension, color: [u8; 3]) -> (usize, usize, Vec<u8>) {
    let (img_w, img_h) = image.dimensions();

    let (row, col) = match dimension {
        Dimension::Row(row) => (row, infer_dimension(img_w, img_h, row + 1)),
        Dimension::Column(col) => (infer_dimension(img_h, img_w, col) - 1, col),
    };

    let width = wall_width(col) as u32;
    let height = wall_height(row) as u32;

    // resize
    let scale = f64::min(width as f64 / img_w as f64, height as f64 / img_h as f64);
    let resize_w = f64::ceil(img_w as f64 * scale) as u32;
    let resize_h = f64::ceil(img_h as f64 * scale) as u32;
    let mut resized_img =
        fast_image_resize::images::Image::new(resize_w, resize_h, PixelType::U8x3);
    let mut resizer = Resizer::new();
    resizer
        .resize(image, &mut resized_img, &ResizeOptions::new())
        .unwrap_or_else(|e| error_out!("{}", e));

    // overlay
    let resized_buf = resized_img.buffer();
    let mut out = [color[0], color[1], color[2]].repeat((width * height) as usize);

    let x_offset = (width - resize_w) / 2;
    let y_offset = (height - resize_h) / 2;

    for y in 0..resize_h {
        let src_off = (y * resize_w * 3) as usize;
        let dst_off = (((y_offset + y) * width + x_offset) * 3) as usize;
        let len = (resize_w * 3) as usize;
        out[dst_off..dst_off + len].copy_from_slice(&resized_buf[src_off..src_off + len]);
    }

    (row, col, out)
}

fn stretch_image(image: &RgbImage, dimension: Dimension) -> (usize, usize, Vec<u8>) {
    let (img_w, img_h) = image.dimensions();

    let (row, col) = match dimension {
        Dimension::Row(row) => (row, infer_dimension(img_w, img_h, row + 1)),
        Dimension::Column(col) => (infer_dimension(img_h, img_w, col) - 1, col),
    };

    let width = wall_width(col) as u32;
    let height = wall_height(row) as u32;

    // reize
    let mut resized_img = fast_image_resize::images::Image::new(width, height, PixelType::U8x3);
    let mut resizer = Resizer::new();
    resizer
        .resize(image, &mut resized_img, &ResizeOptions::new())
        .unwrap_or_else(|e| error_out!("{}", e));

    (row, col, resized_img.into_vec())
}

/// find the best row/column that minimize distortion
#[inline]
fn infer_dimension(x: u32, y: u32, ref_x: usize) -> usize {
    let ratio = x as f64 / y as f64;
    let low = (ref_x as f64 * ratio).floor() as usize;
    let high = low + 1;

    if low < 1 {
        return 1;
    }

    let target = f64::ln(ratio);
    let err_low = target - f64::ln(low as f64 / ref_x as f64);
    let err_high = f64::ln(high as f64 / ref_x as f64) - target;

    if err_low <= err_high { low } else { high }
}
