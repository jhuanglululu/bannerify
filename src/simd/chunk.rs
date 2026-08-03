//! Fixed-size 64-byte-aligned buffer, the compile-time-sized source of lane views.

use core::ops::{Deref, DerefMut};

use super::{F32s, LANES};

/// Owned 64-byte-aligned `[f32; N]`. `N` must be a multiple of 16 (checked at
/// compile time in every constructor), so lane views never need a remainder path.
#[repr(align(64))]
#[derive(Clone, Copy, Debug)]
pub struct Chunk<const N: usize>([f32; N]);

impl<const N: usize> Chunk<N> {
    /// Compile-time `N % 16 == 0` check; instantiate from every entry point.
    const CHECK_N: () = assert!(
        N.is_multiple_of(16),
        "Chunk<N>: N must be a non-zero multiple of 16"
    );

    /// All elements zero.
    #[inline]
    pub fn zeroed() -> Self {
        const { Self::CHECK_N };
        Self([0.0; N])
    }

    /// All elements `x`.
    #[inline]
    pub fn splat(x: f32) -> Self {
        const { Self::CHECK_N };
        Self([x; N])
    }

    /// Overwrite every element with `x`.
    #[inline]
    pub fn fill(&mut self, x: f32) {
        self.0.fill(x);
    }

    /// Lane view; length `N / LANES`.
    #[inline]
    pub fn lanes(&self) -> &[F32s] {
        const { Self::CHECK_N };
        // SAFETY: `Self` is 64-byte aligned and `F32s` is `#[repr(transparent)]`
        // over a register of `LANES` f32s (align <= 64, no invalid bit patterns),
        // so the first `N / LANES * LANES == N` floats form exactly that many
        // valid, aligned, initialized `F32s` inside this borrow.
        unsafe { core::slice::from_raw_parts(self.0.as_ptr().cast::<F32s>(), N / LANES) }
    }

    /// Mutable lane view; length `N / LANES`.
    #[inline]
    pub fn lanes_mut(&mut self) -> &mut [F32s] {
        const { Self::CHECK_N };
        // SAFETY: as `lanes`, and the `&mut self` borrow makes the view exclusive.
        unsafe { core::slice::from_raw_parts_mut(self.0.as_mut_ptr().cast::<F32s>(), N / LANES) }
    }
}

impl<const N: usize> Default for Chunk<N> {
    #[inline]
    fn default() -> Self {
        Self::zeroed()
    }
}

impl<const N: usize> Deref for Chunk<N> {
    type Target = [f32];
    #[inline]
    fn deref(&self) -> &[f32] {
        &self.0
    }
}

impl<const N: usize> DerefMut for Chunk<N> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [f32] {
        &mut self.0
    }
}
