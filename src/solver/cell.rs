//! Cell I/O: gathering a cell's target patch out of a row band, and painting
//! the solved composite back into the row's preview strip.
//!
//! ## Patch geometry
//!
//! Ported from the old build's `image/split.rs`. A banner is [`BANNER_W`] wide
//! and sits [`PAD_SIDE`] pixels inside its 24-pixel block column, so the patch
//! is the 20-wide window at [`offset_column`] — the two pixels either side stay
//! background, which is why the preview shows gaps between banners.
//!
//! Vertically, banner `r` hangs at wall rows `r·BLOCK_SIDE + PAD_TOP ..+
//! BANNER_H`, and banner `r+1` covers its bottom... no: it covers its *top*
//! [`HIDDEN_H`] rows. So:
//!
//! - Banner row 0 has nothing in front of it: the patch is all 40 rows,
//!   starting at wall row [`PAD_TOP`] — which is inside row 0's strip, because
//!   [`banner_row_span`](crate::geometry::banner_row_span) gives row 0
//!   everything above it as well (`0 .. offset_row(1)`, i.e. `0..44`).
//! - Every other row solves only its visible bottom [`VISIBLE_H`] rows, at
//!   [`offset_row`]`(r)`, which is exactly where its strip starts.
//!
//! ## Overlap and strip ownership
//!
//! A banner is 40 rows tall but rows are 24 apart, so banners do overlap — and
//! the overlap is exactly the [`HIDDEN_H`] rows that the banner in front covers
//! completely. Concretely, banner `r`'s wall rows are
//! `r·24+4 .. r·24+44` while strip `r` is `r·24+20 .. (r+1)·24+20` (with row 0
//! and the last row additionally owning the wall's top and bottom padding), so
//!
//! - banner `r` ∩ strip `r` = its visible 24 rows (its full 40 for `r = 0`),
//! - banner `r`'s hidden top rows fall in strip `r−1`, where they are entirely
//!   behind banner `r−1`'s own pixels.
//!
//! **Every strip therefore paints its own banner row and nothing else.** No
//! item reads or writes another item's rows, no draw order has to be agreed
//! on, and the result is bit-identical regardless of scheduling. The strips
//! tile the wall, so nothing is left unpainted either.

use crate::geometry::{
    BANNER_H, BANNER_W, BLOCK_SIDE, HIDDEN_H, PAD_SIDE, PAD_TOP, VISIBLE_H, offset_column,
    offset_row,
};
use crate::resample::RowBand;
use crate::simd::Chunk;

/// Channels in the preview canvas.
const CHANNELS: usize = 3;

/// Preview background over the block rows banner row 0 covers.
///
/// Stage 2a only: the two backgrounds make the top-row split visible at a
/// glance — white means "these rows come from the full 20×40 patch", grey means
/// "these come from a 24-row one". Phase 3 replaces both with matched block
/// textures.
pub const BG_TOP: [u8; CHANNELS] = [255, 255, 255];
/// Preview background everywhere else.
pub const BG_REST: [u8; CHANNELS] = [48, 48, 48];

/// Wall rows painted [`BG_TOP`]: the first two block rows, which is the region
/// banner row 0's 40-row patch reaches into.
const BG_TOP_ROWS: usize = 2 * BLOCK_SIDE;

/// Wall rows of banner row `row`'s patch, and its pixel count.
///
/// Row 0 solves the full patch; the rest solve only what shows.
#[inline]
pub fn patch_rows(row: usize) -> (usize, usize) {
    if row == 0 {
        (PAD_TOP, BANNER_H)
    } else {
        (offset_row(row), VISIBLE_H)
    }
}

/// A row band, addressed in wall-canvas coordinates.
///
/// Under `--fill` a strip can extend past the resampled region; those pixels
/// read as `fallback` (the pad colour) so a cell that straddles the edge still
/// solves against something sensible instead of reading out of bounds.
pub struct BandView<'a> {
    /// `None` when the resampled region misses this strip entirely — possible
    /// only under `--fill`, where a whole banner row can be padding.
    band: Option<&'a RowBand>,
    /// Wall column of the band's column 0.
    x: usize,
    /// Wall row of the band's row 0.
    y: usize,
    /// Value for wall pixels the band does not cover.
    fallback: [f32; CHANNELS],
}

impl<'a> BandView<'a> {
    /// Wrap `band`, whose top-left sample is wall pixel `(x, y)`.
    pub fn new(band: Option<&'a RowBand>, x: usize, y: usize, fallback: [u8; CHANNELS]) -> Self {
        debug_assert!(band.is_none_or(|b| b.channels() == CHANNELS));
        Self {
            band,
            x,
            y,
            fallback: [
                f32::from(fallback[0]),
                f32::from(fallback[1]),
                f32::from(fallback[2]),
            ],
        }
    }

    /// Gather the patch of banner cell `(row, col)` into `target`, planar and
    /// row-major — the layout [`crate::pattern`] documents, so target planes and
    /// pattern planes zip lane for lane.
    ///
    /// `HW` must match the row: `TOP_HW` for row 0, `NTOP_HW` otherwise.
    pub fn gather<const HW: usize>(
        &self,
        row: usize,
        col: usize,
        target: &mut [Chunk<HW>; CHANNELS],
    ) {
        let (y0, rows) = patch_rows(row);
        debug_assert_eq!(rows * BANNER_W, HW, "patch size does not match the row");
        let x0 = offset_column(col);

        for y in 0..rows {
            let band_y = self
                .band
                .and_then(|b| (y0 + y).checked_sub(self.y).filter(|y| *y < b.height));
            for (ch, plane) in target.iter_mut().enumerate() {
                let out = &mut plane[y * BANNER_W..(y + 1) * BANNER_W];
                let (Some(band), Some(band_y)) = (self.band, band_y) else {
                    out.fill(self.fallback[ch]);
                    continue;
                };
                let src = band.row(ch, band_y);
                for (x, dst) in out.iter_mut().enumerate() {
                    *dst = (x0 + x)
                        .checked_sub(self.x)
                        .and_then(|x| src.get(x).copied())
                        .unwrap_or(self.fallback[ch]);
                }
            }
        }
    }
}

/// Paint the preview background over a whole strip.
///
/// `strip_y0` is the strip's first wall row; cells are painted over this
/// afterwards, so the parts that stay visible are the block gaps and the
/// padding above and below the banner rows.
pub fn paint_background(strip: &mut [u8], wall_width: usize, strip_y0: usize) {
    for (y, row) in strip.chunks_exact_mut(wall_width * CHANNELS).enumerate() {
        let color = if strip_y0 + y < BG_TOP_ROWS {
            BG_TOP
        } else {
            BG_REST
        };
        for px in row.chunks_exact_mut(CHANNELS) {
            px.copy_from_slice(&color);
        }
    }
}

/// Paint one solved cell's composite into its strip.
///
/// `composite` is the solver's final prefix, in the same patch layout as the
/// target. Values can sit slightly outside `0..255` (nothing in the compositing
/// algebra clamps, and lanczos overshoot rides in through the target), so the
/// clamp happens here, at the byte edge, exactly as it does for the resampled
/// image.
pub fn paint_cell<const HW: usize>(
    strip: &mut [u8],
    wall_width: usize,
    strip_y0: usize,
    row: usize,
    col: usize,
    composite: &[Chunk<HW>; CHANNELS],
) {
    let (y0, rows) = patch_rows(row);
    debug_assert_eq!(rows * BANNER_W, HW, "composite size does not match the row");
    let x0 = offset_column(col);

    for y in 0..rows {
        let start = ((y0 + y - strip_y0) * wall_width + x0) * CHANNELS;
        let out = &mut strip[start..start + BANNER_W * CHANNELS];
        let src = y * BANNER_W;
        for (x, px) in out.chunks_exact_mut(CHANNELS).enumerate() {
            for (ch, byte) in px.iter_mut().enumerate() {
                *byte = to_u8(composite[ch][src + x]);
            }
        }
    }
}

/// Sample → display byte, clamped and rounded.
#[inline]
fn to_u8(v: f32) -> u8 {
    v.clamp(0.0, 255.0).round() as u8
}

// The layout claims above, checked where they are cheapest to check: a banner
// occupies `BLOCK_SIDE` rows of its own strip plus `HIDDEN_H` rows of the one
// before it, and sits `PAD_SIDE` in from its block column.
const _: () = assert!(BANNER_H == HIDDEN_H + VISIBLE_H);
const _: () = assert!(BANNER_W + 2 * PAD_SIDE == BLOCK_SIDE);
