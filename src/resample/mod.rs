//! Streamed lanczos-3 resampler.
//!
//! Separable lanczos-3 with Pillow-compatible coefficients. Nothing here is
//! wall-sized: a caller builds one shared [`Plan`] (source geometry, target
//! size, horizontal weights) and then asks it for the band of output rows it
//! owns, which it resamples into a local [`RowBand`] and consumes on the spot.
//! That is the pipeline's parallel work item — see
//! `context/designs/streamed-lanczos.md` and `context/designs/pipeline.md`.
//!
//! ```
//! use bannerify::resample::{Plan, PlanarU8};
//!
//! let src = PlanarU8 { width: 4, height: 4, planes: vec![vec![128u8; 16]] };
//! let plan = Plan::new(4, 4, 8, 8);
//!
//! // One work item: output rows 2..6.
//! let band = plan.rows(2..6).resample(&src);
//! assert_eq!(band.height, 4);
//! assert_eq!(band.row(0, 0).len(), 8);
//! ```

mod pipeline;
mod weights;

pub use pipeline::{Plan, PlanarU8, RowBand, RowPlan, Window};
