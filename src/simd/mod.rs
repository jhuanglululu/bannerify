//! Width-hiding SIMD facade.
//!
//! Everything else in the crate writes kernels against this module; nothing
//! outside it names the lane count or an architecture type. See
//! `context/designs/simd-interface.md`.
//!
//! Backends: aarch64 NEON (`float32x4_t`, `LANES == 4`), x86_64 AVX2+FMA
//! (`__m256`, `LANES == 8`), and a scalar `f32` fallback (`LANES == 1`,
//! correctness oracle — also the path for an x86_64 build without AVX2/FMA
//! enabled). The `force-scalar` cargo feature selects the scalar backend on any
//! architecture so they can be cross-checked on the same machine.
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

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma",
    not(feature = "force-scalar")
))]
mod avx2;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma",
    not(feature = "force-scalar")
))]
use avx2 as backend;

// Everything else — x86_64 without those target features included, so a build
// for a pre-AVX2 CPU still works — falls back to the scalar backend.
#[cfg(not(any(
    all(target_arch = "aarch64", not(feature = "force-scalar")),
    all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma",
        not(feature = "force-scalar")
    )
)))]
mod scalar;
#[cfg(not(any(
    all(target_arch = "aarch64", not(feature = "force-scalar")),
    all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma",
        not(feature = "force-scalar")
    )
)))]
use scalar as backend;

pub use aligned::AlignedVec;
pub use backend::{F32s, LANES, Mask};
pub use chunk::Chunk;
pub use zip::{LaneSrc, LaneSrcMut};
