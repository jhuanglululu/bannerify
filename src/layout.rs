//! Turning a source image + a requested wall size into a resize job: banner
//! grid inference, source window, and placement inside the wall.
//!
//! Cropping is expressed as a source [`Window`], never as a materialised
//! sub-image.

use crate::geometry::{infer_dimension, wall_height, wall_width};
use crate::resample::Window;

/// The wall size the user asked for: one axis in blocks, the other inferred.
#[derive(Clone, Copy, Debug)]
pub enum Dimension {
    /// Banner rows (the wall is `rows + 1` blocks tall).
    Row(usize),
    /// Banner columns (blocks wide).
    Column(usize),
}

/// How the image is mapped onto the wall when the aspect ratios differ.
#[derive(Clone, Copy, Debug)]
pub enum ResizingMethod {
    /// Scale to cover the wall and centre-crop the overflow (default).
    Fit,
    /// Scale to fit inside the wall and pad the remainder with this colour.
    Fill([u8; 3]),
    /// Ignore the aspect ratio and stretch to the wall exactly.
    Stretch,
}

/// A fully resolved resize job.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    /// Banner rows.
    pub rows: usize,
    /// Banner columns.
    pub columns: usize,
    /// Wall width in banner pixels.
    pub wall_width: usize,
    /// Wall height in banner pixels.
    pub wall_height: usize,
    /// Width of the resampled region (`wall_width` unless padding).
    pub target_width: usize,
    /// Height of the resampled region (`wall_height` unless padding).
    pub target_height: usize,
    /// Source region feeding the resampler.
    pub window: Window,
    /// Where the resampled region sits inside the wall, in pixels.
    pub origin: (usize, usize),
    /// Colour of the padding around the resampled region, if any.
    pub pad: Option<[u8; 3]>,
}

impl Layout {
    pub fn is_padded(&self) -> bool {
        self.pad.is_some()
            && (self.target_width != self.wall_width || self.target_height != self.wall_height)
    }

    pub fn compute(
        img_width: u32,
        img_height: u32,
        dimension: Dimension,
        method: ResizingMethod,
    ) -> Self {
        let (rows, columns) = match dimension {
            Dimension::Row(rows) => (rows, infer_dimension(img_width, img_height, rows + 1)),
            Dimension::Column(columns) => (
                infer_dimension(img_height, img_width, columns).saturating_sub(1),
                columns,
            ),
        };
        let rows = rows.max(1);
        let columns = columns.max(1);

        let wall_w = wall_width(columns);
        let wall_h = wall_height(rows);
        let (iw, ih) = (f64::from(img_width), f64::from(img_height));

        match method {
            ResizingMethod::Stretch => Self {
                rows,
                columns,
                wall_width: wall_w,
                wall_height: wall_h,
                target_width: wall_w,
                target_height: wall_h,
                window: Window::full(img_width as usize, img_height as usize),
                origin: (0, 0),
                pad: None,
            },
            ResizingMethod::Fit => {
                // Scale to cover, then centre-crop: the crop is the source
                // window, so the cropped-away pixels are never resampled.
                let scale = f64::max(wall_w as f64 / iw, wall_h as f64 / ih);
                let crop_w = wall_w as f64 / scale;
                let crop_h = wall_h as f64 / scale;
                let x0 = (iw - crop_w) / 2.0;
                let y0 = (ih - crop_h) / 2.0;
                Self {
                    rows,
                    columns,
                    wall_width: wall_w,
                    wall_height: wall_h,
                    target_width: wall_w,
                    target_height: wall_h,
                    window: Window {
                        x0,
                        y0,
                        x1: x0 + crop_w,
                        y1: y0 + crop_h,
                    },
                    origin: (0, 0),
                    pad: None,
                }
            }
            ResizingMethod::Fill(color) => {
                let scale = f64::min(wall_w as f64 / iw, wall_h as f64 / ih);
                let target_w = ((iw * scale).ceil() as usize).clamp(1, wall_w);
                let target_h = ((ih * scale).ceil() as usize).clamp(1, wall_h);
                Self {
                    rows,
                    columns,
                    wall_width: wall_w,
                    wall_height: wall_h,
                    target_width: target_w,
                    target_height: target_h,
                    window: Window::full(img_width as usize, img_height as usize),
                    origin: ((wall_w - target_w) / 2, (wall_h - target_h) / 2),
                    pad: Some(color),
                }
            }
        }
    }
}
