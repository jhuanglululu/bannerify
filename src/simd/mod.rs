//! Width-hiding SIMD facade: lane type, aligned buffers, and the backend
//! selection between NEON, AVX2+FMA and a scalar fallback.

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

// Everything else, including x86_64 built without those target features, falls
// back to the scalar backend.
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
