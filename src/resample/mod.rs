//! Streamed lanczos-3 resampler.
//!
//! Separable lanczos-3 (Pillow-compatible coefficients) that never materialises
//! a full-size intermediate: the output is produced in horizontal bands and
//! handed to a [`BandSink`], so resized data dies as soon as it is consumed and
//! memory stays `O(input) + O(sink)`. See `context/designs/streamed-lanczos.md`.
//!
//! ```
//! use bannerify::resample::{Options, PlanarU8, resize_to_planar_f32};
//!
//! let src = PlanarU8 { width: 4, height: 4, planes: vec![vec![128u8; 16]] };
//! let out = resize_to_planar_f32(&src, 8, 8, Options::default());
//! assert_eq!(out[0].len(), 64);
//! ```

mod pipeline;
mod sink;
mod weights;

#[cfg(test)]
mod naive;
#[cfg(test)]
mod tests;

pub use pipeline::{Options, Plan, PlanarU8, resize_to_planar_f32, run};
pub use sink::{Band, BandSink, ChecksumSink, PlanarF32Sink};
