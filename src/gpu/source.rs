//! The MSL source the backend compiles: a generated prelude of constant tables
//! followed by `kernels.metal`.
//!
//! The tables are printed from the CPU solver's own constants (the dye table,
//! its weighted squares, the luma weights, and [`crate::oklab`]'s
//! linearisation table and matrices), so both sides use the same numbers bit
//! for bit; Rust's `{:?}` prints the shortest decimal that rounds back to the
//! same `f32`, which the MSL compiler then parses to that `f32`. The matrices
//! are `f64` and are rounded to `f32` exactly as the CPU kernels round them.

use std::fmt::Write;

use crate::color::{COLORS_F32, COLORS_WSQ_SUM, NUM_COLORS, W_PERCEPTUAL};
use crate::geometry::TOP_HW;
use crate::oklab::{LINEARIZE, M1, M2};

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
    let _ = writeln!(out, "constant float LIN[256] = {{{}}};", list(LINEARIZE));
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
