//! Cell I/O: gathering a cell's target patch out of a column band, and painting
//! the solved composite back into the column's preview strip.
//!
//! ## Patch geometry
//!
//! Ported from the old build's `image/split.rs`. A banner is [`BANNER_W`] wide
//! and sits [`PAD_SIDE`] pixels inside its 24-pixel block column, so the patch
//! is the 20-wide window at [`offset_column`] — the two pixels either side stay
//! background, which is why the preview shows gaps between banners. **A banner
//! never crosses a block-column boundary**, which is what makes the column a
//! self-contained work item (`context/designs/pipeline.md`).
//!
//! Vertically, banner row `r` hangs at wall rows `r·24+4 .. r·24+44`, so it
//! touches block rows `r` and `r+1` — every banner bridges a horizontal block
//! seam, and the banner above draws over the one below:
//!
//! - Banner row 0 has nothing in front of it: the patch is all 40 rows,
//!   starting at wall row [`PAD_TOP`].
//! - Banner row `r ≥ 1` is hidden on its local rows `0..16` by row `r−1`'s
//!   bottom, and is visible only on its bottom [`VISIBLE_H`] rows — local
//!   `16..40`, i.e. wall rows [`offset_row`]`(r) .. + VISIBLE_H`.
//!
//! ## Overlap, resolved by extraction
//!
//! Because a lower banner's patch *is* only its visible rows (the tail of the
//! full patch, elements `TOP_HW - NTOP_HW ..`), the visible regions of
//! successive banner rows are disjoint and tile the column: row 0 paints wall
//! rows `4..44`, row 1 paints `44..68`, and so on. Painting them top to bottom
//! therefore produces exactly the pixels that compositing lower-first and
//! drawing the upper banner over it would — the occlusion is already baked into
//! which rows each cell owns, so there is nothing left to overdraw. All of it is
//! internal to one column item: no other item reads or writes these bytes, so
//! the result does not depend on scheduling.

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
///
/// Row 0 solves the full patch; the rest solve only what shows, which is the
/// *tail* of a plane — the same split the workspace's lane views use
/// ([`crate::solver::workspace`]).
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

    /// One wall pixel of one channel, or the pad colour where the band does not
    /// reach. The block matcher gathers scattered pixels rather than runs, so
    /// it addresses the band one sample at a time.
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
    /// row-major — the layout [`crate::pattern`] documents, so target planes and
    /// pattern planes zip lane for lane.
    ///
    /// Only the row's active view of each plane is written: rows below the top
    /// fill elements `TOP_HW - NTOP_HW ..`, leaving the head untouched.
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
/// row `row` owns exactly strip rows `row·24 .. row·24+24` — one contiguous
/// [`TEXTURE_BYTES`] run, which is why this is a single `copy_from_slice`.
/// Banners are painted over it afterwards, so what stays visible is precisely
/// the hollow frame the matcher scored ([`crate::block`]).
pub fn paint_block(strip: &mut [u8], row: usize, texture: &[u8; TEXTURE_BYTES]) {
    let start = row * BLOCK_SIDE * STRIP_PITCH;
    strip[start..start + TEXTURE_BYTES].copy_from_slice(texture);
}

const _: () = assert!(TEXTURE_BYTES == BLOCK_SIDE * STRIP_PITCH);
const _: () = assert!(BLOCK_PIXELS == BLOCK_SIDE * BLOCK_SIDE);

/// Paint one solved cell's composite into its column strip.
///
/// `composite` is the solver's final prefix, in the same patch layout as the
/// target. Values can sit slightly outside `0..255` (nothing in the compositing
/// algebra clamps, and lanczos overshoot rides in through the target), so the
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

/// Sample → display byte, clamped and rounded.
#[inline]
fn to_u8(v: f32) -> u8 {
    v.clamp(0.0, 255.0).round() as u8
}

// The layout claims above, checked where they are cheapest to check: a banner
// shows `VISIBLE_H` rows and hides `HIDDEN_H` behind the banner above it, and
// sits `PAD_SIDE` in from each edge of its block column.
const _: () = assert!(BANNER_H == HIDDEN_H + VISIBLE_H);
const _: () = assert!(BANNER_W + 2 * PAD_SIDE == BLOCK_SIDE);
