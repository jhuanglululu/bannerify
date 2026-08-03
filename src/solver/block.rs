//! Background block matching: which block sits behind the banners of one cell.
//!
//! This runs **inside the column work item**, between resampling the column's
//! band and solving its banner cells (`context/designs/pipeline.md` stage 5).
//! It has to: the score is over the pixels of a block cell that *no banner
//! covers*, and a block column is the only cut of the wall that owns complete
//! block cells — which is the whole reason items are columns and never rows.
//!
//! One cell, one argmin:
//!
//! 1. gather the block cell's exposed pixels out of the column's resampled band
//!    — the frame's index list, [`crate::block`], picks them;
//! 2. convert that gather to OKLab, with the same function that converted the
//!    block textures at load;
//! 3. score every block by Euclidean OKLab distance over those pixels and keep
//!    the smallest.
//!
//! No closed form and no shortlist: unlike the banner solver there is nothing
//! to expand — a block is a fixed texture, not a colour laid through a mask, so
//! the distance is just a difference of two known vectors. The cost is
//! `blocks × frame pixels × 3`, and the frame is 96 pixels on every interior
//! row, which is where almost all the cells are.
//!
//! Ties go to the lower index, i.e. to the alphabetically earlier block id —
//! deterministic, and independent of how the wall was cut into items.

use crate::block::{
    BOTTOM_FRAME, Blocks, FRAME_EDGE, FRAME_MID, Frame, FramePlanes, MIDDLE_FRAME, TOP_FRAME,
    to_oklab,
};
use crate::geometry::BLOCK_SIDE;
use crate::simd::{Chunk, F32s};
use crate::zip;

use super::cell::BandView;

/// Channels the match is scored over.
const CHANNELS: usize = 3;

/// Reusable gather buffers, one per work item — the block matcher's half of the
/// "nothing allocates inside the cell loop" rule the solver workspace follows.
pub struct BlockScratch {
    edge: FramePlanes<FRAME_EDGE>,
    mid: FramePlanes<FRAME_MID>,
}

impl Default for BlockScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl BlockScratch {
    /// Allocate the two frame sizes.
    pub fn new() -> Self {
        Self {
            edge: [Chunk::zeroed(); CHANNELS],
            mid: [Chunk::zeroed(); CHANNELS],
        }
    }
}

/// Match block cell `(row, col)` of a wall `block_rows` blocks tall.
///
/// `view` is the column item's own band, addressed in wall coordinates; `row`
/// is a **block** row, not a banner row.
pub fn match_cell(
    view: &BandView<'_>,
    scratch: &mut BlockScratch,
    blocks: &Blocks,
    row: usize,
    col: usize,
    block_rows: usize,
) -> usize {
    match Frame::of(row, block_rows) {
        Frame::Top => {
            gather(view, row, col, &TOP_FRAME, &mut scratch.edge);
            argmin(&scratch.edge, &blocks.top)
        }
        Frame::Middle => {
            gather(view, row, col, &MIDDLE_FRAME, &mut scratch.mid);
            argmin(&scratch.mid, &blocks.middle)
        }
        Frame::Bottom => {
            gather(view, row, col, &BOTTOM_FRAME, &mut scratch.edge);
            argmin(&scratch.edge, &blocks.bottom)
        }
    }
}

/// Gather the frame's pixels of block cell `(row, col)` and convert to OKLab.
fn gather<const N: usize>(
    view: &BandView<'_>,
    row: usize,
    col: usize,
    idx: &[u16; N],
    out: &mut FramePlanes<N>,
) {
    let (y0, x0) = (row * BLOCK_SIDE, col * BLOCK_SIDE);
    for (i, &px) in idx.iter().enumerate() {
        let (y, x) = (usize::from(px) / BLOCK_SIDE, usize::from(px) % BLOCK_SIDE);
        for (ch, plane) in out.iter_mut().enumerate() {
            plane[i] = view.pixel(y0 + y, x0 + x, ch);
        }
    }
    to_oklab(out);
}

/// The block whose frame is closest to `target` in OKLab.
fn argmin<const N: usize>(target: &FramePlanes<N>, blocks: &[FramePlanes<N>]) -> usize {
    let mut best = 0;
    let mut min = f32::INFINITY;
    for (i, block) in blocks.iter().enumerate() {
        let mut acc = F32s::ZERO;
        for ch in 0..CHANNELS {
            for (t, b) in zip!(&target[ch], &block[ch]) {
                let d = t - b;
                acc = d.mul_add(d, acc);
            }
        }
        let err = acc.hsum();
        if err < min {
            min = err;
            best = i;
        }
    }
    best
}
