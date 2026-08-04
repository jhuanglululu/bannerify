//! The two images the HTML's compare slider shows, both produced by this
//! crate's own resampler: the rendered wall canvas downscaled, and the source
//! image resampled through the same source window the wall was built from.

use rayon::prelude::*;

use crate::geometry::BLOCK_SIDE;
use crate::resample::{Plan, Source, Window};

const CHANNELS: usize = 3;

/// Output columns one downscale work item owns.
const BAND: usize = BLOCK_SIDE;

pub fn resize<S: Source + ?Sized>(src: &S, window: Window, width: usize, height: usize) -> Vec<u8> {
    let plan = Plan::with_window(src.width(), src.height(), window, width, height);

    let strips: Vec<Vec<u8>> = (0..width.div_ceil(BAND))
        .into_par_iter()
        .map(|b| {
            let cols = b * BAND..((b + 1) * BAND).min(width);
            let w = cols.len();
            let band = plan.columns(cols).resample(src);
            let mut strip = vec![0u8; height * w * CHANNELS];
            for y in 0..height {
                let row = &mut strip[y * w * CHANNELS..(y + 1) * w * CHANNELS];
                for ch in 0..CHANNELS {
                    for (px, &v) in row.chunks_exact_mut(CHANNELS).zip(band.row(ch, y)) {
                        px[ch] = v.clamp(0.0, 255.0).round() as u8;
                    }
                }
            }
            strip
        })
        .collect();

    let mut out = vec![0u8; width * height * CHANNELS];
    out.par_chunks_mut(width * CHANNELS)
        .enumerate()
        .for_each(|(y, row)| {
            let mut x = 0;
            for strip in &strips {
                let w = strip.len() / (height * CHANNELS);
                let run = w * CHANNELS;
                row[x..x + run].copy_from_slice(&strip[y * run..(y + 1) * run]);
                x += run;
            }
        });
    out
}

/// The preview's pixel size: the source's own dimensions by default, else the
/// wall's aspect ratio scaled so its larger side is `max_dim`.
///
/// Either way the result is clamped to the wall canvas: upscaling a banner wall
/// would only spend megabytes of base64 on interpolation of pixels the solver
/// never had.
pub fn dimensions(
    max_dim: Option<usize>,
    source: (usize, usize),
    wall: (usize, usize),
) -> (usize, usize) {
    let (w, h) = match max_dim {
        None => source,
        Some(n) => {
            let n = n.max(1) as f64;
            let scale = n / wall.0.max(wall.1) as f64;
            (
                ((wall.0 as f64 * scale).round() as usize).max(1),
                ((wall.1 as f64 * scale).round() as usize).max(1),
            )
        }
    };
    (w.clamp(1, wall.0), h.clamp(1, wall.1))
}
