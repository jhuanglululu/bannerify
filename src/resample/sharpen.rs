//! Unsharp mask over a resampled column band.
//!
//! The solver minimises squared error against the resampled target over a
//! coarse pattern palette, and squared error is happiest averaging: the closest
//! reachable composite of a detailed patch is usually a flatter one. Feeding the
//! solver a pre-sharpened target biases that averaging back towards the edges,
//! so the composite it settles on reads as sharp rather than as a mean.
//!
//! The mask is the textbook `out = t + k * (t - blur(t))` with a separable
//! Gaussian blur — pure convolution, so the pass is deterministic and adds no
//! state to the pipeline.
//!
//! Band-local by construction: a band knows nothing about its neighbours, so the
//! edge rule here is replicate at the band's *own* `[0, width)` extent, never a
//! peek into the row's alignment padding (which is zero, and would darken the
//! rightmost columns). The caller hides the resulting seam by resampling a band
//! wider than the columns it actually needs — see `col_items` in
//! [`crate::app`] — and the padding beyond `width` is left exactly as the
//! resampler wrote it.

use crate::simd::AlignedVec;

/// Gaussian taps for sigma 1.0, radius 2, normalised to sum 1.
///
/// Truncating the discrete sigma-1.0 kernel at two taps each side keeps ≈99.1%
/// of its mass, so the tail costs nothing visible. The caller pays for the width
/// directly — [`RADIUS`] is how many extra columns a band has to resample on
/// each side — which is why the kernel is kept this narrow.
const TAPS: [f32; 5] = [
    0.054_488_68,
    0.244_201_35,
    0.402_619_94,
    0.244_201_35,
    0.054_488_68,
];

/// Kernel radius, in samples.
///
/// Public because a band's output only equals a whole-image sharpen where the
/// kernel had real neighbours to read: callers must resample this many extra
/// columns on each side of the columns they actually want (see `col_items` in
/// [`crate::app`]). Deriving both from this one constant is what stops a wider
/// kernel from silently reintroducing the band seam.
pub(crate) const RADIUS: usize = TAPS.len() / 2;

/// Sharpen every channel of a band in place by `amount`.
///
/// `planes` are the band's planar `f32` channels, `height` rows of `stride`
/// floats each of which only the first `width` are image data. The
/// `amount <= 0.0` early return is the only gate — callers hand the configured
/// amount straight through, including a disabling `0.0`.
pub(super) fn unsharp(
    planes: &mut [AlignedVec],
    width: usize,
    height: usize,
    stride: usize,
    amount: f32,
) {
    if amount <= 0.0 || width == 0 || height == 0 {
        return;
    }

    // Horizontally blurred band, compact (`width` per row) because nothing reads
    // it through a lane view — it exists only to give the vertical pass a plane
    // to walk. One buffer, reused across channels.
    let mut blur = vec![0.0f32; width * height];

    for plane in planes {
        // Pass 1: horizontal, plane -> blur. The plane is left untouched, so the
        // vertical pass can still read the original `t` out of it.
        for y in 0..height {
            let src = &plane[y * stride..y * stride + width];
            let dst = &mut blur[y * width..(y + 1) * width];
            for (x, out) in dst.iter_mut().enumerate() {
                let mut acc = 0.0;
                for (k, &w) in TAPS.iter().enumerate() {
                    acc += w * src[clamp_index(x, k, width)];
                }
                *out = acc;
            }
        }

        // Pass 2: vertical over `blur`, folded straight into the mask so the
        // fully blurred band never has to be materialised.
        for y in 0..height {
            for x in 0..width {
                let mut acc = 0.0;
                for (k, &w) in TAPS.iter().enumerate() {
                    acc += w * blur[clamp_index(y, k, height) * width + x];
                }
                let t = &mut plane[y * stride + x];
                *t += amount * (*t - acc);
            }
        }
    }
}

/// Index `i` offset by tap `k`, replicated at both ends of a `len`-long axis.
#[inline]
fn clamp_index(i: usize, k: usize, len: usize) -> usize {
    (i + k).saturating_sub(RADIUS).min(len - 1)
}
