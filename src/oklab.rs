//! sRGB → OKLab, on the [`simd`](crate::simd) facade.
//!
//! The solver's closed-form dye sweep is a weighted SSE in sRGB, which is what
//! makes scoring 16 dyes from one pass over the patch possible
//! ([`crate::color`]). That metric does not agree with the eye in the dark
//! blues and greens a banner wall is full of, so refinement demotes it to a
//! shortlisting heuristic and ranks the shortlist by Euclidean distance in
//! OKLab instead ([`crate::solver::refine`]). This module is that conversion —
//! also used by the background block matcher ([`crate::block`]).
//!
//! OKLab (Björn Ottosson, 2020) replaces the Python build's CIELAB + CIEDE2000:
//! plain Euclidean distance in OKLab is already about as good a perceptual
//! metric as CIEDE2000, and it is a matrix, a cube root and another matrix —
//! no hue angles, no `atan2`, no branch-per-term correction terms. That
//! decision is recorded in `context/plans/2-solver.md`.
//!
//! ## The pipeline
//!
//! ```text
//! sRGB byte  --LUT-->  linear light  --M1-->  LMS  --cbrt-->  L'M'S'  --M2-->  OKLab
//! ```
//!
//! ## Accuracy, and what runs per lane
//!
//! Two steps have no vector instruction on any backend here, and both are done
//! per lane through [`F32s::to_array`]/[`F32s::from_array`]:
//!
//! - **Linearisation** is a 256-entry table lookup, i.e. a gather; NEON has
//!   none. Using a table means the input is quantised to integer sRGB — which
//!   is not a loss: the banner is *rendered* as bytes, so the colour the table
//!   scores is exactly the colour the viewer will see. Unlike the old build's
//!   table ([`../bannerify-old/src/lab.rs`], which approximated `x^2.4` by
//!   `x²·(A·x + B)`, up to 5.3e-3 absolute error in linear light), this table is
//!   built by Newton iteration in `const` context and is exact to f32.
//! - **Cube root** is `f32::cbrt` per lane, i.e. correctly rounded libm rather
//!   than a vector Newton iteration off a bit-hack seed. The linearisation LUT
//!   above already forces a lane round-trip inside the same function, so a
//!   vector cube root would have to pay the same store/load; and doing it in
//!   libm is what makes the NEON and scalar backends produce *bit-identical*
//!   OKLab values.
//!
//!   Since phase 5 this is the solver's hot loop — refinement converts a whole
//!   patch per exact candidate — and the cube roots are indeed where the time
//!   goes. It is left alone deliberately (`context/plans/5-exact-refine.md`):
//!   an approximate cube root would trade the one thing this module currently
//!   guarantees, cross-backend identity, for a speedup that
//!   `--exact-candidates` already exposes as a dial.
//!
//! Everything between them — both matrices — is `F32s` arithmetic. There is no
//! data-dependent branch left in the vector part (the sRGB piecewise lives in
//! the `const` table builder), so no [`Mask::select`](crate::simd::Mask::select)
//! appears here.

use crate::simd::F32s;

/// sRGB byte → linear light.
///
/// The exact sRGB EOTF: `c/12.92` below the knee, `((c+0.055)/1.055)^2.4`
/// above. `powf` is not `const`, so the upper branch is evaluated as
/// `x^2.4 = (x^12)^(1/5)` by Newton iteration on `y⁵ − x¹² = 0`, seeded with
/// the old build's cubic approximation and run to convergence in `f64`.
const LINEARIZE: [f32; 256] = {
    let mut out = [0.0_f32; 256];
    let mut i = 0;
    while i < 256 {
        let c = i as f64 / 255.0;
        out[i] = if c <= 0.04045 {
            (c / 12.92) as f32
        } else {
            pow_2_4((c + 0.055) / 1.055) as f32
        };
        i += 1;
    }
    out
};

/// `x^2.4` for `x` in `(0, 1]`, in `const` context.
const fn pow_2_4(x: f64) -> f64 {
    // t = x^12, whose exact fifth root is x^2.4.
    let x2 = x * x;
    let x4 = x2 * x2;
    let t = x4 * x4 * x4;

    // Seed: the cubic approximation the old build shipped as its final answer.
    const A: f64 = 0.4618055522441864;
    let mut y = x2 * (A * x + (1.0 - A));

    // Newton on y^5 - t: y -= (y^5 - t) / (5 y^4). Quadratic convergence from a
    // seed within ~35% relative error; 20 steps is far past f64 convergence.
    let mut step = 0;
    while step < 20 {
        let y2 = y * y;
        let y4 = y2 * y2;
        y -= (y4 * y - t) / (5.0 * y4);
        step += 1;
    }
    y
}

/// Linear sRGB → LMS (Ottosson's `M1`), row-major.
///
/// Held at `f64` so the published coefficients are transcribed exactly; every
/// use narrows to `f32` at compile time.
const M1: [[f64; 3]; 3] = [
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
];

/// Cube-rooted LMS → OKLab (Ottosson's `M2`), row-major. See [`M1`].
const M2: [[f64; 3]; 3] = [
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
];

/// Convert a lane of sRGB (nominally `0..=255`, clamped) to OKLab.
///
/// Returns `(L, a, b)`. `L` is roughly `0..=1`; `a`/`b` are small signed
/// numbers, so a Euclidean distance in this space is directly a ΔE.
#[inline]
pub fn srgb_to_oklab(r: F32s, g: F32s, b: F32s) -> (F32s, F32s, F32s) {
    let (lr, lg, lb) = (linearize(r), linearize(g), linearize(b));

    let l = cbrt(matrix_row(M1[0], lr, lg, lb));
    let m = cbrt(matrix_row(M1[1], lr, lg, lb));
    let s = cbrt(matrix_row(M1[2], lr, lg, lb));

    (
        matrix_row(M2[0], l, m, s),
        matrix_row(M2[1], l, m, s),
        matrix_row(M2[2], l, m, s),
    )
}

/// One row of a 3×3 colour matrix applied to a lane triple.
#[inline]
fn matrix_row(row: [f64; 3], x: F32s, y: F32s, z: F32s) -> F32s {
    x.mul_add(
        F32s::splat(row[0] as f32),
        y.mul_add(F32s::splat(row[1] as f32), z * F32s::splat(row[2] as f32)),
    )
}

/// sRGB values → linear light, per lane through [`LINEARIZE`].
///
/// Out-of-range inputs are clamped: the solver's composites are convex
/// combinations of dye colours and so always in range, but a lanczos-resampled
/// target can overshoot, and it is scored through here too.
#[inline]
fn linearize(v: F32s) -> F32s {
    let lanes = v.to_array();
    F32s::from_array(std::array::from_fn(|i| {
        // `+ 0.5` then truncate is round-to-nearest, matching the rounding the
        // preview render uses when it writes the same composite out as bytes.
        LINEARIZE[(lanes[i] + 0.5).clamp(0.0, 255.0) as usize]
    }))
}

/// Cube root, per lane. See the module docs for why this is not vectorised.
#[inline]
fn cbrt(v: F32s) -> F32s {
    let lanes = v.to_array();
    F32s::from_array(std::array::from_fn(|i| lanes[i].cbrt()))
}
