//! The shader sources the device layers compile: a generated prelude of
//! constant tables followed by `kernels.metal` (MSL) or `kernels.wgsl` (WGSL).
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
#[cfg(target_os = "macos")]
fn lit(v: f32) -> String {
    let s = format!("{v:?}");
    if s.contains(['.', 'e']) {
        format!("{s}f")
    } else {
        format!("{s}.0f")
    }
}

#[cfg(target_os = "macos")]
fn list(values: impl IntoIterator<Item = f32>) -> String {
    values.into_iter().map(lit).collect::<Vec<_>>().join(", ")
}

/// The full MSL source: prelude, then the kernels.
#[cfg(target_os = "macos")]
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
