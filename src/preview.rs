//! The two images the HTML's compare slider shows, both produced by this
//! crate's own resampler.
//!
//! - the **generated** pane: the rendered wall canvas, downscaled;
//! - the **original** pane: the source image, resampled through the very same
//!   source window the wall was built from ([`crate::layout::Layout::window`]),
//!   so the two panes frame the same picture and the slider lines up.
//!
//! Both go to the same pixel size, which is what makes them stackable panes
//! rather than two images that need CSS to agree.
//!
//! ## Nothing wall-sized in `f32`
//!
//! The wall canvas is interleaved `u8` and stays that way: it is read *in
//! place* through [`InterleavedU8`], which strides the u8 → f32 widening the
//! H pass already does at load ([`crate::resample`]). No planar copy of the
//! wall is ever materialised.
//!
//! Downscaling is parallel over **column bands**, the same work item shape as
//! the wall pipeline: each band resamples its own output columns and converts
//! them to bytes locally, and one interleave pass assembles the image. So the
//! only `f32` buffers alive at once are the in-flight bands
//! (`height × band width`), never the whole preview.

use rayon::prelude::*;

use crate::geometry::BLOCK_SIDE;
use crate::resample::{Plan, Source, Window};

/// Channels the preview images carry.
const CHANNELS: usize = 3;

/// Output columns one downscale work item owns.
///
/// One block wide, like the wall pipeline's items — the number that made the
/// column split natural there is just as good here, and it keeps a band's
/// working set at a few tens of kilobytes.
const BAND: usize = BLOCK_SIDE;

/// Resample `src`'s region `window` to `width × height`, interleaved RGB.
///
/// Parallel over column bands; the returned buffer is exactly what an encoder
/// wants, so nothing is converted again downstream.
pub fn resize<S: Source + ?Sized>(src: &S, window: Window, width: usize, height: usize) -> Vec<u8> {
    debug_assert_eq!(src.channels(), CHANNELS);
    let plan = Plan::with_window(src.width(), src.height(), window, width, height);

    // One item = one band of output columns, resampled and byte-converted
    // locally. The strips together *are* the output, so the assembly below is
    // the only copy.
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

/// The preview's pixel size.
///
/// Default (`max_dim` is `None`): the **source image's own dimensions** — the
/// picture comes back the size it went in. `--preview N` overrides that with a
/// max-dimension: the larger side becomes `N` and the other follows the wall's
/// aspect ratio.
///
/// Either way the result is clamped to the wall canvas: upscaling a banner wall
/// for an HTML preview would only spend megabytes of base64 on interpolation of
/// pixels the solver never had. (The wall's aspect ratio is within one block of
/// the source's by construction — [`crate::geometry::infer_dimension`] picks
/// the grid that least distorts it — so the default is not a stretch.)
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
