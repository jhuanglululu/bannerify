//! The 16 Minecraft dye colours and the derived tables the solver needs: the
//! moments of its closed-form error expansion, and the nearest-dye snap LUT the
//! feature map quantises a target patch with.
//!
//! ## The expansion these tables serve
//!
//! For a target `t`, a prefix composite `p` and a pattern alpha `α`, laying dye
//! `c` on top gives `p·(1−α) + c·α`. With `res = p·(1−α) − t` the weighted SSE
//! is
//!
//! ```text
//! Σ w·(res + c·α)² = Σ w·res² + c·Σ 2·w·res·α + c²·Σ w·α²
//! ```
//!
//! so one pass over the pixels produces the two moments `Σ res²` and
//! `2·Σ res·α` per channel, and all 16 dyes are then scored in closed form from
//! those plus the pattern's `Σ α²` ([`COLORS_WSQ_SUM`] folds the per-channel
//! weights into the `c²` term).

use std::sync::OnceLock;

use crate::oklab::srgb_to_oklab;
use crate::simd::{F32s, LANES};

pub const NUM_COLORS: usize = 16;

/// Minecraft dye ids, in the canonical order the tables below use.
pub const COLOR_NAMES: [&str; NUM_COLORS] = [
    "white",
    "orange",
    "magenta",
    "light_blue",
    "yellow",
    "lime",
    "pink",
    "gray",
    "light_gray",
    "cyan",
    "purple",
    "blue",
    "brown",
    "green",
    "red",
    "black",
];

/// The dyes as rendered on a banner, sRGB.
pub const COLORS_RGB: [[u8; 3]; NUM_COLORS] = [
    [255, 255, 255], // white
    [249, 128, 29],  // orange
    [199, 78, 189],  // magenta
    [58, 179, 218],  // light_blue
    [254, 216, 61],  // yellow
    [128, 199, 31],  // lime
    [243, 139, 170], // pink
    [71, 79, 82],    // gray
    [157, 157, 151], // light_gray
    [22, 156, 156],  // cyan
    [137, 50, 184],  // purple
    [60, 68, 170],   // blue
    [131, 84, 50],   // brown
    [94, 124, 22],   // green
    [176, 46, 38],   // red
    [29, 29, 33],    // black
];

pub const COLORS_F32: [[f32; 3]; NUM_COLORS] = {
    let mut out = [[0.0_f32; 3]; NUM_COLORS];
    let mut i = 0;
    while i < NUM_COLORS {
        out[i][0] = COLORS_RGB[i][0] as f32;
        out[i][1] = COLORS_RGB[i][1] as f32;
        out[i][2] = COLORS_RGB[i][2] as f32;
        i += 1;
    }
    out
};

/// Per-channel error weights: perceptual luma.
pub const W_PERCEPTUAL: [f32; 3] = [0.299, 0.587, 0.114];

/// `Σ_ch w_ch · c_ch²` per dye — the `c²` coefficient of the expansion above,
/// with the channel weights already folded in.
pub const COLORS_WSQ_SUM: [f32; NUM_COLORS] = {
    let mut out = [0.0_f32; NUM_COLORS];
    let mut i = 0;
    while i < NUM_COLORS {
        let r = COLORS_F32[i][0];
        let g = COLORS_F32[i][1];
        let b = COLORS_F32[i][2];
        out[i] = W_PERCEPTUAL[0] * r * r + W_PERCEPTUAL[1] * g * g + W_PERCEPTUAL[2] * b * b;
        i += 1;
    }
    out
};

/// Bits per channel of the [`snap_lut`] index.
///
/// 5 bits is the point where the table still fits in L1-adjacent memory (32 KiB
/// of `u8`) while the quantisation step — 8 sRGB levels — stays far below the
/// gaps between the 16 dyes: only colours already almost equidistant from two
/// dyes can land on the "wrong" side, and those are exactly the ones where the
/// choice does not matter.
pub const SNAP_BITS: u32 = 5;

/// Quantisation levels per channel.
const SNAP_LEVELS: usize = 1 << SNAP_BITS;

/// Entries in the snap LUT.
pub const SNAP_ENTRIES: usize = SNAP_LEVELS * SNAP_LEVELS * SNAP_LEVELS;

static SNAP_LUT: OnceLock<Box<[u8; SNAP_ENTRIES]>> = OnceLock::new();

/// Quantised sRGB → index of the nearest dye in OKLab.
///
/// Built once per process, on first use. Nearest is Euclidean distance in OKLab,
/// not sRGB: the feature map is a *perceptual* idealisation of the patch, and in
/// the dark blues and greens a banner wall is full of an sRGB nearest-neighbour
/// picks a visibly different dye ([`crate::oklab`] documents the same choice for
/// the solver's exact rung).
///
/// Callers in a hot loop should hoist this call: it is an atomic load per call.
pub fn snap_lut() -> &'static [u8; SNAP_ENTRIES] {
    SNAP_LUT.get_or_init(build_snap_lut)
}

/// Index into [`snap_lut`] of one sRGB pixel.
///
/// Values ride in from a lanczos-resampled target and can overshoot `0..=255`,
/// so they are clamped; `+ 0.5` then truncate is the same round-to-nearest the
/// OKLab linearisation table uses, so a pixel is quantised from the byte a
/// viewer would actually see.
#[inline]
pub fn snap_index(r: f32, g: f32, b: f32) -> usize {
    #[inline]
    fn q(v: f32) -> usize {
        ((v + 0.5).clamp(0.0, 255.0) as usize) >> (8 - SNAP_BITS)
    }
    (q(r) << (2 * SNAP_BITS)) | (q(g) << SNAP_BITS) | q(b)
}

/// The 8-bit sRGB value a quantisation bucket stands for: bit-replication, which
/// maps `0 → 0` and `31 → 255` and spreads the rest evenly.
#[inline]
const fn dequantize(q: usize) -> f32 {
    (((q << (8 - SNAP_BITS)) | (q >> (2 * SNAP_BITS - 8))) & 0xFF) as f32
}

fn build_snap_lut() -> Box<[u8; SNAP_ENTRIES]> {
    let dyes: [[f32; 3]; NUM_COLORS] = std::array::from_fn(|i| {
        let c = COLORS_F32[i];
        let (l, a, b) = srgb_to_oklab(F32s::splat(c[0]), F32s::splat(c[1]), F32s::splat(c[2]));
        [l.to_array()[0], a.to_array()[0], b.to_array()[0]]
    });

    // A `Box<[u8; N]>` built through a Vec: `Box::new([0; N])` would build the
    // 32 KiB array on the stack first.
    let mut out = vec![0_u8; SNAP_ENTRIES].into_boxed_slice();

    // `SNAP_ENTRIES` is a power of two and `LANES` is 1, 4 or 8, so the lane
    // chunks divide the table exactly — no remainder path.
    const _: () = assert!(SNAP_ENTRIES.is_multiple_of(16));
    for (chunk, slot) in out.chunks_exact_mut(LANES).enumerate() {
        let base = chunk * LANES;
        let comp = |shift: u32| {
            F32s::from_array(std::array::from_fn(|i| {
                dequantize(((base + i) >> shift) & (SNAP_LEVELS - 1))
            }))
        };
        let (l, a, b) = srgb_to_oklab(comp(2 * SNAP_BITS), comp(SNAP_BITS), comp(0));
        let (l, a, b) = (l.to_array(), a.to_array(), b.to_array());

        for (i, out) in slot.iter_mut().enumerate() {
            let mut best = 0;
            let mut min = f32::INFINITY;
            for (idx, dye) in dyes.iter().enumerate() {
                let (dl, da, db) = (l[i] - dye[0], a[i] - dye[1], b[i] - dye[2]);
                let d2 = dl * dl + da * da + db * db;
                if d2 < min {
                    min = d2;
                    best = idx as u8;
                }
            }
            *out = best;
        }
    }

    out.try_into().expect("the Vec was allocated at that length")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The property the feature map rests on: a patch already painted in dye
    /// colours quantises back to exactly those dyes.
    #[test]
    fn every_dye_snaps_to_itself() {
        let lut = snap_lut();
        for (i, c) in COLORS_F32.iter().enumerate() {
            let got = lut[snap_index(c[0], c[1], c[2])];
            assert_eq!(
                got as usize, i,
                "{} snapped to {}",
                COLOR_NAMES[i], COLOR_NAMES[got as usize]
            );
        }
    }

    /// Out-of-range inputs (lanczos overshoot) clamp instead of indexing out of
    /// the table.
    #[test]
    fn overshooting_pixels_clamp_into_the_table() {
        assert_eq!(snap_index(-40.0, -1.0, 300.0), snap_index(0.0, 0.0, 255.0));
        assert!(snap_index(1e30, 1e30, 1e30) < SNAP_ENTRIES);
    }
}
