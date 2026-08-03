//! The background block textures, embedded in the binary and decoded into the
//! matcher's frame tables.
//!
//! `assets/blocks/` holds one 16×16 JPEG per Minecraft block the wall may be
//! built from. A wall pixel is a *banner* pixel, and a block is
//! [`BLOCK_SIDE`]×[`BLOCK_SIDE`] of those, so every texture is resized 16 → 24
//! at load — through this crate's own lanczos-3 resampler
//! ([`crate::resample`]), not a second resizing implementation.
//!
//! ## Hollow frames
//!
//! Banners hang in front of the wall, so most of a block is never seen. Which
//! part *is* seen depends only on the block's row (ported from the old build's
//! `TOP_MASK`/`MIDDLE_MASK`/`BOTTOM_MASK`, re-derived here from the geometry in
//! [`crate::solver::cell`]):
//!
//! ```text
//! block row 0        banner row 0 covers wall y 4..24 of it
//!   -> exposed: rows 0..4 whole, rows 4..24 only the 2 px either side
//!
//! block row 0 < R < rows   rows R-1 and R together cover its whole 20 px width
//!   -> exposed: the 2 px either side, all 24 rows
//!
//! block row R == rows      banner row rows-1 covers wall y 0..20 of it
//!   -> exposed: rows 0..20 only the sides, rows 20..24 whole
//! ```
//!
//! The old build spelled this as a fixed 256-pixel "hollow" patch multiplied by
//! a 0/1 mask per row position. Here the mask *is* the gather list: each
//! [`Frame`] carries the exact pixel indices it exposes ([`FRAME_EDGE`] = 176
//! for the top and bottom rows, [`FRAME_MID`] = 96 for the middle ones), so the
//! matcher never loads — let alone multiplies — a pixel that a banner hides.
//! Both counts are multiples of 16, so they are [`Chunk`] sizes as they stand
//! and no remainder path appears anywhere.
//!
//! ## What is stored per block
//!
//! - [`Blocks::texture`] — the 24×24 RGB bytes the preview paints.
//! - [`Blocks::top`] / [`Blocks::middle`] / [`Blocks::bottom`] — the same
//!   texture gathered through each frame and converted to **OKLab**, which is
//!   the space the match is scored in (plain Euclidean distance; see
//!   [`crate::oklab`]).

use std::collections::HashSet;

use image::GenericImageView;
use rayon::prelude::*;
use rust_embed::Embed;

use crate::geometry::{BLOCK_PIXELS, BLOCK_SIDE, MIDDLE_OFFSET, PAD_BOTTOM, PAD_SIDE, PAD_TOP};
use crate::logger::error_out;
use crate::oklab::srgb_to_oklab;
use crate::resample::{Plan, PlanarU8};
use crate::simd::Chunk;
use crate::zip;

/// Channels a block texture carries.
const CHANNELS: usize = 3;

/// Side of the embedded textures, in texture pixels.
const TEXTURE_SIDE: usize = 16;

/// Bytes in one painted block: 24×24 RGB, interleaved.
pub const TEXTURE_BYTES: usize = CHANNELS * BLOCK_PIXELS;

/// Exposed pixels of a block in the top or the bottom row of the wall: four
/// whole rows plus the 2-pixel flanks of the other twenty.
pub const FRAME_EDGE: usize = PAD_TOP * BLOCK_SIDE + (BLOCK_SIDE - PAD_TOP) * 2 * PAD_SIDE;

/// Exposed pixels of a block in any other row: the flanks, and nothing else.
pub const FRAME_MID: usize = BLOCK_SIDE * 2 * PAD_SIDE;

const _: () = assert!(PAD_TOP == PAD_BOTTOM, "the frames assume symmetric padding");
const _: () = assert!(FRAME_EDGE == 176 && FRAME_MID == 96);

/// One frame's OKLab planes.
pub type FramePlanes<const N: usize> = [Chunk<N>; CHANNELS];

/// Which pixels of a block cell no banner covers, as gather indices into the
/// cell's 24×24 pixels, row-major.
///
/// `full` names the rows exposed over their whole width; every other row
/// exposes only [`MIDDLE_OFFSET`], the two pixels either side of the banner.
const fn frame<const N: usize>(full_lo: usize, full_hi: usize) -> [u16; N] {
    let mut out = [0u16; N];
    let mut i = 0;
    let mut y = 0;
    while y < BLOCK_SIDE {
        if y >= full_lo && y < full_hi {
            let mut x = 0;
            while x < BLOCK_SIDE {
                out[i] = (y * BLOCK_SIDE + x) as u16;
                i += 1;
                x += 1;
            }
        } else {
            let mut k = 0;
            while k < 2 * PAD_SIDE {
                out[i] = (y * BLOCK_SIDE + MIDDLE_OFFSET[k]) as u16;
                i += 1;
                k += 1;
            }
        }
        y += 1;
    }
    assert!(i == N, "frame index list has the wrong length");
    out
}

/// Block row 0: nothing hangs in front of its top [`PAD_TOP`] rows.
pub const TOP_FRAME: [u16; FRAME_EDGE] = frame(0, PAD_TOP);
/// Any interior block row: the banner above and the banner below cover the
/// whole width between them.
pub const MIDDLE_FRAME: [u16; FRAME_MID] = frame(0, 0);
/// The last block row: the bottom [`PAD_BOTTOM`] rows hang below every banner.
pub const BOTTOM_FRAME: [u16; FRAME_EDGE] = frame(BLOCK_SIDE - PAD_BOTTOM, BLOCK_SIDE);

/// Which frame block row `row` of a wall with `block_rows` rows wears.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Frame {
    /// Block row 0.
    Top,
    /// An interior block row.
    Middle,
    /// The last block row.
    Bottom,
}

impl Frame {
    /// The frame of block row `row` on a wall `block_rows` blocks tall.
    #[inline]
    pub fn of(row: usize, block_rows: usize) -> Self {
        if row == 0 {
            Frame::Top
        } else if row + 1 == block_rows {
            Frame::Bottom
        } else {
            Frame::Middle
        }
    }
}

/// The embedded `assets/blocks/` tree.
#[derive(Embed)]
#[folder = "assets/blocks/"]
struct BlockAssets;

/// Every block the background matcher may use.
///
/// Field `i` of every vector describes block `names[i]`; the matcher carries
/// only the index, and the exporter turns it back into an id.
pub struct Blocks {
    /// Block ids, sorted, in table order.
    pub names: Vec<String>,
    /// The same ids with the `minecraft:` namespace, built once because every
    /// schematic entry wants one and none of them wants to allocate per cell.
    pub qualified: Vec<String>,
    /// 24×24 RGB bytes, interleaved — what the preview paints.
    pub texture: Vec<[u8; TEXTURE_BYTES]>,
    /// OKLab planes over [`TOP_FRAME`].
    pub top: Vec<FramePlanes<FRAME_EDGE>>,
    /// OKLab planes over [`MIDDLE_FRAME`].
    pub middle: Vec<FramePlanes<FRAME_MID>>,
    /// OKLab planes over [`BOTTOM_FRAME`].
    pub bottom: Vec<FramePlanes<FRAME_EDGE>>,
}

impl Blocks {
    /// Number of blocks in the table.
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Whether the table is empty (only reachable by excluding everything).
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

/// Decode the embedded blocks, dropping every id in `exclude`.
///
/// Decoding, resizing and the OKLab conversion all run `par_iter` over blocks —
/// there are a few hundred of them and the work is a real fraction of startup,
/// unlike the 42 pattern PNGs. An id in `exclude` that names no block is a user
/// mistake, not a silent no-op: it exits with the list.
pub fn load(exclude: &HashSet<String>) -> Blocks {
    // rust-embed's iteration order follows the directory walk; sort so the
    // block indices — and therefore every distance tie-break — are the same on
    // every machine and every run.
    let mut all: Vec<String> = BlockAssets::iter()
        .filter_map(|p| p.strip_suffix(".jpg").map(str::to_string))
        .collect();
    all.sort_unstable();

    let unknown: Vec<&str> = exclude
        .iter()
        .map(String::as_str)
        .filter(|id| !all.iter().any(|name| name == id))
        .collect();
    if !unknown.is_empty() {
        let mut unknown = unknown;
        unknown.sort_unstable();
        error_out!("unknown block id(s): {}", unknown.join(", "));
    }

    let names: Vec<String> = all.into_iter().filter(|id| !exclude.contains(id)).collect();
    if names.is_empty() {
        error_out!("every background block was excluded — nothing left to build the wall from");
    }

    // One shared plan for the 16 → 24 upscale: same source geometry, same
    // target size for every block, so the weight tables are built once.
    let plan = Plan::new(TEXTURE_SIDE, TEXTURE_SIDE, BLOCK_SIDE, BLOCK_SIDE);

    let texture: Vec<[u8; TEXTURE_BYTES]> = names.par_iter().map(|id| decode(id, &plan)).collect();

    let top = texture.par_iter().map(|t| planes(t, &TOP_FRAME)).collect();
    let middle = texture
        .par_iter()
        .map(|t| planes(t, &MIDDLE_FRAME))
        .collect();
    let bottom = texture
        .par_iter()
        .map(|t| planes(t, &BOTTOM_FRAME))
        .collect();

    let qualified = names.iter().map(|n| format!("minecraft:{n}")).collect();

    Blocks {
        names,
        qualified,
        texture,
        top,
        middle,
        bottom,
    }
}

/// Decode one embedded JPEG and resize it 16 → 24 through our own resampler.
fn decode(id: &str, plan: &Plan) -> [u8; TEXTURE_BYTES] {
    let file = BlockAssets::get(&format!("{id}.jpg"))
        .unwrap_or_else(|| error_out!("internal error: block '{id}' vanished from the binary"));
    let img = image::load_from_memory_with_format(&file.data, image::ImageFormat::Jpeg)
        .unwrap_or_else(|e| error_out!("could not decode block '{id}': {e}"));

    if img.dimensions() != (TEXTURE_SIDE as u32, TEXTURE_SIDE as u32) {
        error_out!("block '{id}' is not {TEXTURE_SIDE}x{TEXTURE_SIDE} pixels");
    }

    let rgb = img.to_rgb8();
    let src = PlanarU8::from_interleaved(rgb.as_raw(), TEXTURE_SIDE, TEXTURE_SIDE, CHANNELS);
    let band = plan.columns(0..BLOCK_SIDE).resample(&src);

    let mut out = [0u8; TEXTURE_BYTES];
    for y in 0..BLOCK_SIDE {
        for ch in 0..CHANNELS {
            let row = band.row(ch, y);
            for (x, &v) in row.iter().enumerate() {
                out[(y * BLOCK_SIDE + x) * CHANNELS + ch] = v.clamp(0.0, 255.0).round() as u8;
            }
        }
    }
    out
}

/// Gather one frame out of a texture and convert it to OKLab.
fn planes<const N: usize>(texture: &[u8; TEXTURE_BYTES], idx: &[u16; N]) -> FramePlanes<N> {
    let mut out = [Chunk::<N>::zeroed(); CHANNELS];
    for (i, &px) in idx.iter().enumerate() {
        for (ch, plane) in out.iter_mut().enumerate() {
            plane[i] = f32::from(texture[usize::from(px) * CHANNELS + ch]);
        }
    }
    to_oklab(&mut out);
    out
}

/// Convert three sRGB planes to OKLab in place.
///
/// Shared with the matcher's target gather ([`crate::solver::block`]), which is
/// the point: the two sides of the distance must be produced by exactly the
/// same conversion.
pub fn to_oklab<const N: usize>(planes: &mut FramePlanes<N>) {
    let (r, rest) = planes.split_at_mut(1);
    let (g, b) = rest.split_at_mut(1);
    for (r, g, b) in zip!(mut r[0], mut g[0], mut b[0]) {
        let (l, a, bb) = srgb_to_oklab(*r, *g, *b);
        (*r, *g, *b) = (l, a, bb);
    }
}
