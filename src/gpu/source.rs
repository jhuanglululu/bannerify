//! The MSL source the backend compiles: a generated prelude of constant tables
//! followed by `kernels.metal`.
//!
//! The tables are printed from the CPU solver's own `f32` values (the dye
//! table, its weighted squares, the luma weights), so both sides use the same
//! numbers bit for bit; Rust's `{:?}` prints the shortest decimal that rounds
//! back to the same `f32`, which the MSL compiler then parses to that `f32`.
//!
//! The OKLab constants are private to [`crate::oklab`], so the linearisation
//! table and the two matrices are rebuilt here the same way that module builds
//! them (an exact `f64` Newton iteration, then one rounding to `f32`; the
//! published `f64` coefficients, rounded once).

use std::fmt::Write;

use crate::color::{COLORS_F32, COLORS_WSQ_SUM, NUM_COLORS, W_PERCEPTUAL};
use crate::geometry::TOP_HW;

/// `oklab::LINEARIZE`, rebuilt: sRGB byte to linear light.
fn linearize_table() -> [f32; 256] {
    fn pow_2_4(x: f64) -> f64 {
        let x2 = x * x;
        let x4 = x2 * x2;
        let t = x4 * x4 * x4;
        const A: f64 = 0.4618055522441864;
        let mut y = x2 * (A * x + (1.0 - A));
        for _ in 0..20 {
            let y2 = y * y;
            let y4 = y2 * y2;
            y -= (y4 * y - t) / (5.0 * y4);
        }
        y
    }
    std::array::from_fn(|i| {
        let c = i as f64 / 255.0;
        if c <= 0.04045 {
            (c / 12.92) as f32
        } else {
            pow_2_4((c + 0.055) / 1.055) as f32
        }
    })
}

/// `oklab::M1`: linear sRGB to LMS.
const M1: [[f64; 3]; 3] = [
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
];

/// `oklab::M2`: cube-rooted LMS to OKLab.
const M2: [[f64; 3]; 3] = [
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
];

/// An `f32` as an MSL literal that parses back to the same value.
fn lit(v: f32) -> String {
    let s = format!("{v:?}");
    if s.contains(['.', 'e']) {
        format!("{s}f")
    } else {
        format!("{s}.0f")
    }
}

fn list(values: impl IntoIterator<Item = f32>) -> String {
    values.into_iter().map(lit).collect::<Vec<_>>().join(", ")
}

/// The full MSL source: prelude, then the kernels.
pub(super) fn msl() -> String {
    let mut out = String::new();
    // `writeln!` into a `String` cannot fail.
    let _ = writeln!(out, "#include <metal_stdlib>\nusing namespace metal;");
    let _ = writeln!(out, "#define HW {TOP_HW}u");
    let _ = writeln!(out, "#define PL {}u", TOP_HW / 4);
    let _ = writeln!(out, "#define NC {NUM_COLORS}u");
    let colors = COLORS_F32
        .iter()
        .map(|c| format!("{{{}}}", list(c.iter().copied())))
        .collect::<Vec<_>>()
        .join(", ");
    let _ = writeln!(out, "constant float COLORS[16][3] = {{{colors}}};");
    let _ = writeln!(
        out,
        "constant float WSQ[16] = {{{}}};",
        list(COLORS_WSQ_SUM)
    );
    for (i, w) in W_PERCEPTUAL.iter().enumerate() {
        let _ = writeln!(out, "constant float W{i} = {};", lit(*w));
    }
    let _ = writeln!(
        out,
        "constant float LIN[256] = {{{}}};",
        list(linearize_table())
    );
    for (name, m) in [("M1", M1), ("M2", M2)] {
        for (r, row) in m.iter().enumerate() {
            for (c, v) in row.iter().enumerate() {
                let _ = writeln!(out, "constant float {name}_{r}{c} = {};", lit(*v as f32));
            }
        }
    }
    out.push_str(include_str!("kernels.metal"));
    out
}
