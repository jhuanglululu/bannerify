//! Width-hiding SIMD facade.
//!
//! Everything else in the crate writes kernels against this module; nothing
//! outside it names the lane count or an architecture type. See
//! `context/designs/simd-interface.md`.
//!
//! Backends: aarch64 NEON (`float32x4_t`, `LANES == 4`) and a scalar `f32`
//! fallback (`LANES == 1`, correctness oracle). The `force-scalar` cargo
//! feature selects the scalar backend on any architecture so the two can be
//! cross-checked on the same machine.
//!
//! Hard rules the API upholds:
//!
//! - **No hidden memory traffic.** Only the constructor you name writes to a
//!   buffer; [`AlignedVec::new_uninit`] writes nothing.
//! - **No remainder paths.** Lengths are multiples of 16 — enforced at compile
//!   time for [`Chunk`], at construction for [`AlignedVec`].
//! - **No auto-splat.** `F32s ⊕ f32` does not exist; use [`F32s::splat`].

mod aligned;
mod chunk;
mod zip;

#[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
mod neon;
#[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
use neon as backend;

#[cfg(not(all(target_arch = "aarch64", not(feature = "force-scalar"))))]
mod scalar;
#[cfg(not(all(target_arch = "aarch64", not(feature = "force-scalar"))))]
use scalar as backend;

pub use aligned::AlignedVec;
pub use backend::{F32s, LANES, Mask};
pub use chunk::Chunk;
pub use zip::{LaneSrc, LaneSrcMut};

#[cfg(test)]
mod tests;
