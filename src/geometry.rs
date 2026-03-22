/// Banner width
pub const BANNER_W: usize = 20;
/// Banner height
pub const BANNER_H: usize = 40;

/// Gap between banner and top edge of block
pub const PAD_TOP: usize = 4;
/// Gap between banner and bottom edge of block
pub const PAD_BOTTOM: usize = 4;
/// Gap between banner and left edge of block
pub const PAD_LEFT: usize = 2;
/// Gap between banner and right edge of block
pub const PAD_RIGHT: usize = 2;

/// Size of a block in banner pixel
pub const BLOCK_SIDE: usize = 24;
/// Distance between two banner
pub const STRIDE: usize = 24;

/// Size of top half of a banner hidden by another banner above it
pub const HIDDEN_TOP: usize = 16;

/// Size of bottom half of a banner visible when another is banner above it
pub const VISIBLE_H: usize = BANNER_H - HIDDEN_TOP;

#[inline]
/// Pixel offset from the left for '`column`'th banner column
pub const fn offset_column(column: usize) -> usize {
    column * BLOCK_SIDE + PAD_LEFT
}

#[inline]
/// Total wall width in pixels for '`columns`' banner columns
pub const fn wall_width(columns: usize) -> usize {
    columns * BLOCK_SIDE
}

#[inline]
/// Pixel offset from the top for '`rows'`th banner row
pub const fn offset_row(row: usize) -> usize {
    row * BLOCK_SIDE + HIDDEN_TOP
}

#[inline]
/// Total wall height in pixels for '`rows`' banner rows
pub const fn wall_height(rows: usize) -> usize {
    (rows + 1) * BLOCK_SIDE
}
