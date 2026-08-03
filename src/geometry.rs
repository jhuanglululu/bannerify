//! Banner-wall geometry: the fixed pixel dimensions of banners and blocks, and
//! the wall-size arithmetic the resize target is derived from.
//!
//! Ported from `../bannerify-old/src/geometry.rs`. Everything here is
//! phase-2-ready: the solver and exporter address banner pixels through these
//! constants, the CLI currently uses only the wall-size and inference helpers.
//!
//! Layout: banners hang on a wall of blocks. A block is [`BLOCK_SIDE`] pixels
//! square; a banner is [`BANNER_W`] × [`BANNER_H`] pixels, and the top
//! [`HIDDEN_H`] pixels of a banner are covered by the banner above it, leaving
//! [`VISIBLE_H`] visible. A wall of `rows` banner rows is therefore
//! `rows + 1` blocks tall.

/// Banner width, in banner pixels.
pub const BANNER_W: usize = 20;
/// Banner height, in banner pixels.
pub const BANNER_H: usize = 40;

/// Height of the part of a banner hidden by the banner above it.
pub const HIDDEN_H: usize = 16;
/// Height of the part of a banner left visible by the banner above it.
pub const VISIBLE_H: usize = BANNER_H - HIDDEN_H;

/// Pixels in a fully visible (topmost) banner.
pub const TOP_HW: usize = BANNER_H * BANNER_W;
/// Pixels in a partially covered (non-topmost) banner.
pub const NTOP_HW: usize = VISIBLE_H * BANNER_W;

/// Gap between a banner and the top edge of its block.
pub const PAD_TOP: usize = 4;
/// Gap between a banner and the bottom edge of its block.
pub const PAD_BOTTOM: usize = 4;
/// Gap between a banner and the left or right edge of its block.
pub const PAD_SIDE: usize = 2;
/// Combined left + right gap.
pub const PAD_TWO_SIDE: usize = 2 * PAD_SIDE;

/// Side of a block, in banner pixels.
pub const BLOCK_SIDE: usize = 24;
/// Pixels in a block.
pub const BLOCK_PIXELS: usize = BLOCK_SIDE * BLOCK_SIDE;

/// Pixels of a block not covered by the banner in front of it.
pub const HOLLOW_BLOCK_PIXELS: usize =
    BLOCK_PIXELS - (BLOCK_SIDE - PAD_TOP - PAD_BOTTOM) * (BLOCK_SIDE - PAD_TWO_SIDE);

/// Height of the block strip a banner spans vertically.
pub const MID_SIDE: usize = BLOCK_SIDE - PAD_TOP - PAD_BOTTOM;
/// Column offsets of the block pixels flanking a banner.
pub const MIDDLE_OFFSET: [usize; 4] = [0, 1, BLOCK_SIDE - 2, BLOCK_SIDE - 1];

/// Pixel offset from the left of the wall for banner column `column`.
#[inline]
pub const fn offset_column(column: usize) -> usize {
    column * BLOCK_SIDE + PAD_SIDE
}

/// Total wall width in pixels for `columns` banner columns.
#[inline]
pub const fn wall_width(columns: usize) -> usize {
    columns * BLOCK_SIDE
}

/// Pixel offset from the top of the wall for banner row `row`.
#[inline]
pub const fn offset_row(row: usize) -> usize {
    row * BLOCK_SIDE + PAD_TOP + HIDDEN_H
}

/// Total wall height in pixels for `rows` banner rows (`rows + 1` blocks).
#[inline]
pub const fn wall_height(rows: usize) -> usize {
    (rows + 1) * BLOCK_SIDE
}

/// Target-pixel rows owned by banner row `row`, of `rows` banner rows.
///
/// This is the pipeline's work-item split (`context/designs/pipeline.md`): one
/// item per banner row, and the spans tile the wall exactly.
///
/// A banner is [`BANNER_H`] pixels tall but only its bottom [`VISIBLE_H`] shows
/// once the next row hangs over it, so row `r > 0` owns exactly the strip its
/// visible part occupies, `offset_row(r) .. offset_row(r) + VISIBLE_H`. Row 0
/// additionally owns everything above it (its own hidden top half plus the top
/// block padding), and the last row owns the wall's bottom padding — so no
/// pixel of the wall belongs to no item.
pub fn banner_row_span(row: usize, rows: usize) -> core::ops::Range<usize> {
    debug_assert!(row < rows, "banner row out of range");
    let start = if row == 0 { 0 } else { offset_row(row) };
    let end = if row + 1 == rows {
        wall_height(rows)
    } else {
        offset_row(row + 1)
    };
    start..end
}

/// Pick the block count along one axis that least distorts the aspect ratio.
///
/// `x`/`y` are the source dimensions along the axis being inferred and its
/// counterpart; `ref_x` is the known block count of the counterpart axis. The
/// candidate below and above the exact ratio are compared in log space, so the
/// choice is symmetric in "how much does this stretch the image".
#[inline]
pub fn infer_dimension(x: u32, y: u32, ref_x: usize) -> usize {
    let ratio = f64::from(x) / f64::from(y);
    let low = (ref_x as f64 * ratio).floor() as usize;
    let high = low + 1;

    if low < 1 {
        return 1;
    }

    let target = ratio.ln();
    let err_low = target - (low as f64 / ref_x as f64).ln();
    let err_high = (high as f64 / ref_x as f64).ln() - target;

    if err_low <= err_high { low } else { high }
}
