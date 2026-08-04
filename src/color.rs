//! The 16 Minecraft dye colours and the derived tables the solver's closed-form
//! error expansion needs.
//!
//! Ported from `../bannerify-old/src/color.rs`. The old build also kept the 16
//! dyes packed into SIMD registers (`COLORS_R/G/B`) for a vectorised colour
//! sweep; the greedy solver never used them — its colour loop is a scalar
//! closed form over 16 candidates, evaluated once per pattern on three
//! already-reduced moments — so only the scalar tables are ported here.
//!
//! ## The expansion these tables serve
//!
//! For a target `t`, a prefix composite `p` and a pattern alpha `α`, laying dye
//! `c` on top gives `p·(1−α) + c·α`. With `res = p·(1−α) − t` the SSE is
//!
//! ```text
//! Σ (res + c·α)² = Σ res² + c·Σ 2·res·α + c²·Σ α²
//! ```
//!
//! so one pass over the pixels produces the two moments `Σ res²` and
//! `2·Σ res·α` per channel, and all 16 dyes are then scored in closed form from
//! those plus the pattern's `Σ α²` (the `c²` coefficient is
//! [`LabTables::sq_sum`]).
//!
//! ## Two spaces, two tables
//!
//! Since phase 4 the solver's working space is **OKLab**, not sRGB
//! (`context/plans/4-oklab-native.md`): the target band is converted once per
//! column item and every composite the solver forms is a linear-in-Lab blend,
//! so the expansion above runs on Lab components with no channel weights at all
//! — OKLab is already perceptually uniform, which is exactly why the old
//! `W_PERCEPTUAL` luma weights are gone. [`lab`] serves that side.
//!
//! [`COLORS_F32`] stays, in sRGB, because *painting* did not move: the preview
//! composites dye RGB over the pattern alphas byte-for-byte as the game does
//! ([`crate::solver::workspace::Workspace::render_rgb`]), and the exporter
//! writes dye ids and swatches.

use std::sync::LazyLock;

use crate::oklab::srgb_to_oklab_one;

/// Number of dyes a banner can use.
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

/// [`COLORS_RGB`] widened once, so the solver never converts in a loop.
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

/// The dye tables the solver actually scores with, in OKLab.
pub struct LabTables {
    /// The 16 dyes as `(L, a, b)`.
    pub color: [[f32; 3]; NUM_COLORS],
    /// `Σ_ch c_ch²` per dye — the `c²` coefficient of the expansion above.
    /// Unweighted: OKLab needs no channel weights.
    pub sq_sum: [f32; NUM_COLORS],
}

/// Built once, on first use, rather than in `const`.
///
/// A `const` table would have to re-derive OKLab in `f64` `const fn` (there is
/// no `const` `cbrt`), which is a *second* conversion implementation: the dye
/// side of every distance would then come from different arithmetic than the
/// target side, which goes through [`srgb_to_oklab_one`] on the SIMD facade.
/// Both sides must be produced by exactly the same conversion — the same rule
/// [`crate::block::to_oklab`] follows for block textures — so the table is
/// computed by calling that very function, once, at first touch. The cost is
/// 16 conversions per process and one relaxed load per table access, hoisted
/// out of every loop that uses it.
static LAB: LazyLock<LabTables> = LazyLock::new(|| {
    let color = COLORS_F32.map(srgb_to_oklab_one);
    let sq_sum = color.map(|[l, a, b]| l * l + a * a + b * b);
    LabTables { color, sq_sum }
});

/// The OKLab dye tables. Bind once outside a loop, not per iteration.
#[inline]
pub fn lab() -> &'static LabTables {
    &LAB
}
