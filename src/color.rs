//! The 16 Minecraft dye colours and the derived tables the solver's closed-form
//! error expansion needs.
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
