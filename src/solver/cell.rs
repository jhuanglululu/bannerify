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
//! Because a lower banner's patch *is* only its visible rows (a
//! `Chunk<NTOP_HW>`, the tail of the full patch), the visible regions of
//! successive banner rows are disjoint and tile the column: row 0 paints wall
//! rows `4..44`, row 1 paints `44..68`, and so on. Painting them top to bottom
//! therefore produces exactly the pixels that compositing lower-first and
//! drawing the upper banner over it would — the occlusion is already baked into
//! which rows each cell owns, so there is nothing left to overdraw. All of it is
//! internal to one column item: no other item reads or writes these bytes, so
//! the result does not depend on scheduling.

use crate::geometry::{
    BANNER_H, BANNER_W, BLOCK_SIDE, HIDDEN_H, PAD_SIDE, PAD_TOP, VISIBLE_H, offset_column,
    offset_row,
};
use crate::resample::ColBand;
use crate::simd::Chunk;

/// Channels in the preview canvas.
const CHANNELS: usize = 3;

/// Bytes in one row of a column strip.
pub const STRIP_PITCH: usize = BLOCK_SIDE * CHANNELS;

/// Preview background over the block rows banner row 0 covers.
///
/// Stage 2a only: the two backgrounds make the top-row split visible at a
/// glance — white means "these rows come from the full 20×40 patch", grey means
/// "these come from a 24-row one". Phase 3 replaces both with matched block
/// textures. The rule is about wall `y`, so it is unaffected by how the wall is
/// cut into work items.
pub const BG_TOP: [u8; CHANNELS] = [255, 255, 255];
/// Preview background everywhere else.
pub const BG_REST: [u8; CHANNELS] = [48, 48, 48];

/// Wall rows painted [`BG_TOP`]: the first two block rows, which is the region
/// banner row 0's 40-row patch reaches into.
const BG_TOP_ROWS: usize = 2 * BLOCK_SIDE;

/// Wall rows of banner row `row`'s patch, and its height in rows.
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

/// Paint the preview background over a whole column strip.
///
/// The strip is [`BLOCK_SIDE`] pixels wide and starts at wall row 0, so a strip
/// row index *is* a wall row. Cells are painted over this afterwards, so what
/// stays visible is the 2-pixel gap either side of each banner and the padding
/// above the first and below the last banner row.
pub fn paint_background(strip: &mut [u8]) {
    for (y, row) in strip.chunks_exact_mut(STRIP_PITCH).enumerate() {
        let color = if y < BG_TOP_ROWS { BG_TOP } else { BG_REST };
        for px in row.chunks_exact_mut(CHANNELS) {
            px.copy_from_slice(&color);
        }
    }
}

/// Paint one solved cell's composite into its column strip.
///
/// `composite` is the solver's final prefix, in the same patch layout as the
/// target. Values can sit slightly outside `0..255` (nothing in the compositing
/// algebra clamps, and lanczos overshoot rides in through the target), so the
/// clamp happens here, at the byte edge.
pub fn paint_cell<const HW: usize>(
    strip: &mut [u8],
    row: usize,
    composite: &[Chunk<HW>; CHANNELS],
) {
    let (y0, rows) = patch_rows(row);
    debug_assert_eq!(rows * BANNER_W, HW, "composite size does not match the row");

    for y in 0..rows {
        let start = (y0 + y) * STRIP_PITCH + PAD_SIDE * CHANNELS;
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
// shows `VISIBLE_H` rows and hides `HIDDEN_H` behind the banner above it, and
// sits `PAD_SIDE` in from each edge of its block column.
const _: () = assert!(BANNER_H == HIDDEN_H + VISIBLE_H);
const _: () = assert!(BANNER_W + 2 * PAD_SIDE == BLOCK_SIDE);
