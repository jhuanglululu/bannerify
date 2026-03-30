/// Banner width
pub const BANNER_W: usize = 20;
/// Banner height
pub const BANNER_H: usize = 40;

/// Size of top half of a banner hidden by another banner above it
pub const HIDDEN_H: usize = 16;
/// Size of bottom half of a banner visible when another is banner above it
pub const VISIBLE_H: usize = BANNER_H - HIDDEN_H;

pub const TOP_HW: usize = BANNER_H * BANNER_W;
pub const NTOP_HW: usize = VISIBLE_H * BANNER_W;

/// Gap between banner and top edge of block
pub const PAD_TOP: usize = 4;
/// Gap between banner and bottom edge of block
pub const PAD_BOTTOM: usize = 4;
/// Gap between banner and left or right edge of block
pub const PAD_SIDE: usize = 2;
/// Gap between banner and left or right edge of block
pub const PAD_TWO_SIDE: usize = 2 * PAD_SIDE;

/// Size of a block in banner pixel
pub const BLOCK_SIDE: usize = 24;
/// Number pixels in a block in banner pixel
pub const BLOCK_PIXELS: usize = BLOCK_SIDE * BLOCK_SIDE;

pub const HOLLOW_BLOCK_PIXELS: usize =
    BLOCK_PIXELS - (BLOCK_SIDE - PAD_TOP - PAD_BOTTOM) * (BLOCK_SIDE - PAD_TWO_SIDE);

pub const MID_SIDE: usize = BLOCK_SIDE - PAD_TOP - PAD_BOTTOM;
pub const MIDDLE_OFFSET: [usize; 4] = [0, 1, BLOCK_SIDE - 2, BLOCK_SIDE - 1];

#[inline]
/// Pixel offset from the left for `column` banner column
pub const fn offset_column(column: usize) -> usize {
    column * BLOCK_SIDE + PAD_SIDE
}

#[inline]
/// Total wall width in pixels for `columns` banner columns
pub const fn wall_width(columns: usize) -> usize {
    columns * BLOCK_SIDE
}

#[inline]
/// Pixel offset from the top for `row`th banner row
pub const fn offset_row(row: usize) -> usize {
    row * BLOCK_SIDE + PAD_TOP + HIDDEN_H
}

#[inline]
/// Total wall height in pixels for '`rows`' banner rows
pub const fn wall_height(rows: usize) -> usize {
    (rows + 1) * BLOCK_SIDE
}
