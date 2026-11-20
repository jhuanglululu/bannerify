//! The WGSL source the device layer compiles: a generated prelude of constant
//! tables followed by `kernels.wgsl`.
//!
//! The tables are printed from the CPU solver's own constants (the dye table,
//! its weighted squares, the dyes in OKLab, the luma weights, and
//! [`crate::oklab`]'s linearisation table and matrices), so both sides use the same numbers bit
//! for bit; Rust's `{:?}` prints the shortest decimal that rounds back to the
//! same `f32`, which the shader compiler then parses to that `f32`. The matrices
//! are `f64` and are rounded to `f32` exactly as the CPU kernels round them.

use std::fmt::Write;

use crate::color::{COLORS_F32, COLORS_WSQ_SUM, NUM_COLORS, W_PERCEPTUAL, colors_oklab};
use crate::geometry::TOP_HW;
use crate::oklab::{LINEARIZE, M1, M2};

/// An `f32` as a WGSL literal that parses back to the same value.
fn wgsl_lit(v: f32) -> String {
    let s = format!("{v:?}");
    if s.contains(['.', 'e']) {
        s
    } else {
        format!("{s}.0")
    }
}

fn wgsl_list(values: impl IntoIterator<Item = f32>) -> String {
    values
        .into_iter()
        .map(wgsl_lit)
        .collect::<Vec<_>>()
        .join(", ")
}

/// The full WGSL source for workgroups of `group` threads: prelude, then the
/// kernels.
pub(super) fn wgsl(group: usize) -> String {
    let mut out = String::new();
    // `writeln!` into a `String` cannot fail.
    let _ = writeln!(out, "const GROUP: u32 = {group}u;");
    let _ = writeln!(out, "const PL: u32 = {}u;", TOP_HW / 4);
    let _ = writeln!(out, "const NC: u32 = {NUM_COLORS}u;");
    let colors = COLORS_F32
        .iter()
        .map(|c| format!("vec3<f32>({})", wgsl_list(c.iter().copied())))
        .collect::<Vec<_>>()
        .join(", ");
    let _ = writeln!(out, "const COLORS = array<vec3<f32>, 16>({colors});");
    let colors_lab = colors_oklab()
        .iter()
        .map(|c| format!("vec3<f32>({})", wgsl_list(c.iter().copied())))
        .collect::<Vec<_>>()
        .join(", ");
    let _ = writeln!(
        out,
        "const COLORS_LAB = array<vec3<f32>, 16>({colors_lab});"
    );
    let _ = writeln!(
        out,
        "const WSQ = array<f32, 16>({});",
        wgsl_list(COLORS_WSQ_SUM)
    );
    for (i, w) in W_PERCEPTUAL.iter().enumerate() {
        let _ = writeln!(out, "const W{i}: f32 = {};", wgsl_lit(*w));
    }
    let _ = writeln!(
        out,
        "const LIN = array<f32, 256>({});",
        wgsl_list(LINEARIZE)
    );
    for (name, m) in [("M1", M1), ("M2", M2)] {
        for (r, row) in m.iter().enumerate() {
            for (c, v) in row.iter().enumerate() {
                let _ = writeln!(out, "const {name}_{r}{c}: f32 = {};", wgsl_lit(*v as f32));
            }
        }
    }
    out.push_str(include_str!("kernels.wgsl"));
    out
}
