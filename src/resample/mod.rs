//! Streamed lanczos-3 resampler.
//!
//! Separable lanczos-3 with Pillow-compatible coefficients. Nothing here is
//! wall-sized: a caller builds one shared [`Plan`] (source geometry, target
//! size, vertical weights) and then asks it for the band of output columns it
//! owns, which it resamples into a local [`ColBand`] and consumes on the spot.
//! That is the pipeline's parallel work item — see
//! `context/designs/streamed-lanczos.md` and `context/designs/pipeline.md`.
//!
//! ```
//! use bannerify::resample::{Plan, PlanarU8};
//!
//! let src = PlanarU8 { width: 4, height: 4, planes: vec![vec![128u8; 16]] };
//! let plan = Plan::new(4, 4, 8, 8);
//!
//! // One work item: output columns 2..6, full height.
//! let band = plan.columns(2..6).resample(&src);
//! assert_eq!(band.width, 4);
//! assert_eq!(band.height, 8);
//! assert_eq!(band.row(0, 0).len(), 4);
//! ```

mod pipeline;
mod weights;

pub use pipeline::{ColBand, ColumnPlan, Plan, PlanarU8, Window};
