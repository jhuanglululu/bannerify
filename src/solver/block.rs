//! Background block matching: which block sits behind the banners of one cell.

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

/// Reusable gather buffers, one per work item.
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
    pub fn new() -> Self {
        Self {
            edge: [Chunk::zeroed(); CHANNELS],
            mid: [Chunk::zeroed(); CHANNELS],
        }
    }
}

/// Match block cell `(row, col)` of a wall `block_rows` blocks tall. `row` is a
/// **block** row, not a banner row.
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

/// The block whose frame is closest to `target` in OKLab; ties go to the lower
/// index, so the result does not depend on how the wall was cut into items.
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
