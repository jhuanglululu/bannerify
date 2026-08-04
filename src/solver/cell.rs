//! Cell I/O: gathering a cell's target patch out of a column band, and painting
//! the solved composite back into the column's preview strip.
//!
//! Banner row `r` hangs at wall rows `r·24+4 .. r·24+44`, so every banner
//! bridges a horizontal block seam and the row above draws over it. Row 0 owns
//! its full 40 rows; row `r ≥ 1` owns only its bottom [`VISIBLE_H`] rows, the
//! tail of a full patch. The owned regions are therefore disjoint and tile the
//! column, so painting rows in any order gives the same pixels as compositing
//! lower-first and drawing the upper banner over it — the occlusion is baked
//! into which rows each cell owns, and nothing needs overdrawing.

use crate::block::TEXTURE_BYTES;
use crate::geometry::{
    BANNER_H, BANNER_W, BLOCK_PIXELS, BLOCK_SIDE, HIDDEN_H, NTOP_HW, PAD_SIDE, PAD_TOP, TOP_HW,
    VISIBLE_H, offset_column, offset_row,
};
use crate::resample::ColBand;
use crate::solver::workspace::Plane;

/// Channels in the preview canvas.
const CHANNELS: usize = 3;

/// Bytes in one row of a column strip.
pub const STRIP_PITCH: usize = BLOCK_SIDE * CHANNELS;

/// Wall rows of banner row `row`'s patch, its height in rows, and the element
/// offset at which it starts inside a [`Plane`].
#[inline]
pub fn patch_rows(row: usize) -> (usize, usize, usize) {
    if row == 0 {
        (PAD_TOP, BANNER_H, 0)
    } else {
        (offset_row(row), VISIBLE_H, TOP_HW - NTOP_HW)
    }
}

/// A column band, addressed in wall-canvas coordinates.
///
/// Under `--fill` a column strip can extend past the resampled region — or miss
/// it entirely; those pixels read as `fallback` (the pad colour) so a cell that
/// straddles the edge still solves against something sensible instead of reading
/// out of bounds.
pub struct BandView<'a> {
    /// `None` when the resampled region misses this column entirely.
    band: Option<&'a ColBand>,
    /// Wall column of the band's column 0.
    x: usize,
    /// Wall row of the band's row 0.
    y: usize,
    /// Value for wall pixels the band does not cover.
    fallback: [f32; CHANNELS],
}

impl<'a> BandView<'a> {
    /// Wrap `band`, whose top-left sample is wall pixel `(x, y)`.
    pub fn new(band: Option<&'a ColBand>, x: usize, y: usize, fallback: [u8; CHANNELS]) -> Self {
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

    /// One wall pixel of one channel, or the pad colour where the band does not
    /// reach.
    #[inline]
    pub fn pixel(&self, y: usize, x: usize, ch: usize) -> f32 {
        let Some(band) = self.band else {
            return self.fallback[ch];
        };
        let (Some(by), Some(bx)) = (
            y.checked_sub(self.y).filter(|y| *y < band.height),
            x.checked_sub(self.x).filter(|x| *x < band.width),
        ) else {
            return self.fallback[ch];
        };
        band.row(ch, by)[bx]
    }

    /// Gather the patch of banner cell `(row, col)` into `target`, planar and
    /// row-major, so target planes and pattern planes zip lane for lane.
    ///
    /// Only the row's active view of each plane is written, leaving the head of
    /// a lower row's planes untouched.
    pub fn gather(&self, row: usize, col: usize, target: &mut [Plane; CHANNELS]) {
        let (y0, rows, base) = patch_rows(row);
        let x0 = offset_column(col);

        for y in 0..rows {
            let band_y = self
                .band
                .and_then(|b| (y0 + y).checked_sub(self.y).filter(|y| *y < b.height));
            for (ch, plane) in target.iter_mut().enumerate() {
                let out = &mut plane[base + y * BANNER_W..base + (y + 1) * BANNER_W];
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

/// Paint one matched block's texture into its column strip.
///
/// The strip is [`BLOCK_SIDE`] pixels wide and starts at wall row 0, so block
/// row `row` owns exactly one contiguous [`TEXTURE_BYTES`] run of it.
pub fn paint_block(strip: &mut [u8], row: usize, texture: &[u8; TEXTURE_BYTES]) {
    let start = row * BLOCK_SIDE * STRIP_PITCH;
    strip[start..start + TEXTURE_BYTES].copy_from_slice(texture);
}

const _: () = assert!(TEXTURE_BYTES == BLOCK_SIDE * STRIP_PITCH);
const _: () = assert!(BLOCK_PIXELS == BLOCK_SIDE * BLOCK_SIDE);

/// Paint one solved cell's composite into its column strip.
///
/// Values can sit slightly outside `0..255` — nothing in the compositing
/// algebra clamps, and lanczos overshoot rides in through the target — so the
/// clamp happens here, at the byte edge.
pub fn paint_cell(strip: &mut [u8], row: usize, composite: &[Plane; CHANNELS]) {
    let (y0, rows, base) = patch_rows(row);

    for y in 0..rows {
        let start = (y0 + y) * STRIP_PITCH + PAD_SIDE * CHANNELS;
        let out = &mut strip[start..start + BANNER_W * CHANNELS];
        let src = base + y * BANNER_W;
        for (x, px) in out.chunks_exact_mut(CHANNELS).enumerate() {
            for (ch, byte) in px.iter_mut().enumerate() {
                *byte = to_u8(composite[ch][src + x]);
            }
        }
    }
}

#[inline]
fn to_u8(v: f32) -> u8 {
    v.clamp(0.0, 255.0).round() as u8
}

// Layout invariants the slicing above relies on.
const _: () = assert!(BANNER_H == HIDDEN_H + VISIBLE_H);
const _: () = assert!(BANNER_W + 2 * PAD_SIDE == BLOCK_SIDE);
