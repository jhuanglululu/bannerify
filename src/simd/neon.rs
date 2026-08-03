//! aarch64 NEON backend: one [`F32s`] == one `float32x4_t` register.
//!
//! NEON (Advanced SIMD) is part of the aarch64 baseline, so every intrinsic
//! used here is unconditionally available on this target; the `unsafe` on each
//! call is a formality the intrinsic signatures impose.

use core::arch::aarch64::*;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Lanes per [`F32s`] for this backend.
pub const LANES: usize = 4;

/// A vector of [`LANES`] `f32`s: exactly one native register.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct F32s(float32x4_t);

/// Result of a lane-wise comparison; consume with [`Mask::select`].
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Mask(uint32x4_t);

impl F32s {
    /// All lanes zero.
    // SAFETY: `float32x4_t` is four `f32`s with no niches or padding, so any
    // `[f32; 4]` bit pattern is a valid register value.
    pub const ZERO: Self = Self(unsafe { core::mem::transmute::<[f32; 4], float32x4_t>([0.0; 4]) });
    /// All lanes one.
    // SAFETY: as above.
    pub const ONE: Self = Self(unsafe { core::mem::transmute::<[f32; 4], float32x4_t>([1.0; 4]) });

    /// Broadcast `x` to every lane.
    #[inline(always)]
    pub fn splat(x: f32) -> Self {
        // SAFETY: `vdupq_n_f32` is a baseline aarch64 NEON intrinsic with no preconditions.
        Self(unsafe { vdupq_n_f32(x) })
    }

    /// `self * a + b`, fused (single rounding) — emits `fmla`.
    #[inline(always)]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        // SAFETY: `vfmaq_f32(acc, x, y)` computes `acc + x * y`; baseline NEON, no preconditions.
        Self(unsafe { vfmaq_f32(b.0, self.0, a.0) })
    }

    /// Sum of all lanes. Keep out of inner loops.
    #[inline(always)]
    pub fn hsum(self) -> f32 {
        // SAFETY: baseline NEON intrinsic, no preconditions.
        unsafe { vaddvq_f32(self.0) }
    }

    /// Lane-wise square root.
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        // SAFETY: baseline NEON intrinsic, no preconditions.
        Self(unsafe { vsqrtq_f32(self.0) })
    }

    /// Lane-wise minimum.
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        // SAFETY: baseline NEON intrinsic, no preconditions.
        Self(unsafe { vminq_f32(self.0, other.0) })
    }

    /// Lane-wise maximum.
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        // SAFETY: baseline NEON intrinsic, no preconditions.
        Self(unsafe { vmaxq_f32(self.0, other.0) })
    }

    /// Lane-wise `self < other`.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Mask {
        // SAFETY: baseline NEON intrinsic, no preconditions.
        Mask(unsafe { vcltq_f32(self.0, other.0) })
    }

    /// Lane-wise `self > other`.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Mask {
        // SAFETY: baseline NEON intrinsic, no preconditions.
        Mask(unsafe { vcgtq_f32(self.0, other.0) })
    }

    /// Build a register from lane values, in order.
    ///
    /// The inverse of [`F32s::to_array`], and the same cost class: a store plus
    /// a load. It exists for the one thing vector code genuinely cannot do —
    /// a per-lane table lookup (NEON has no gather) — which is how
    /// [`crate::oklab`] linearises sRGB. Not for arithmetic.
    #[inline(always)]
    pub fn from_array(v: [f32; LANES]) -> Self {
        // SAFETY: `[f32; 4]` and `float32x4_t` have the same size and layout,
        // and every bit pattern is valid for both.
        Self(unsafe { core::mem::transmute::<[f32; 4], float32x4_t>(v) })
    }

    /// Lane values, in order. Diagnostics and cold paths only.
    #[inline(always)]
    pub fn to_array(self) -> [f32; LANES] {
        // SAFETY: `float32x4_t` and `[f32; 4]` have the same size and layout,
        // and every bit pattern is valid for both.
        unsafe { core::mem::transmute::<float32x4_t, [f32; 4]>(self.0) }
    }
}

impl Mask {
    /// Lane-wise `if mask { a } else { b }`.
    #[inline(always)]
    pub fn select(self, a: F32s, b: F32s) -> F32s {
        // SAFETY: `vbslq_f32` selects bit-wise from `a`/`b` using `self`, whose
        // lanes are all-ones or all-zeros because they came from a comparison.
        F32s(unsafe { vbslq_f32(self.0, a.0, b.0) })
    }
}

impl core::fmt::Debug for F32s {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.to_array()).finish()
    }
}

impl Add for F32s {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        // SAFETY: baseline NEON intrinsic, no preconditions.
        Self(unsafe { vaddq_f32(self.0, rhs.0) })
    }
}

impl Sub for F32s {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        // SAFETY: baseline NEON intrinsic, no preconditions.
        Self(unsafe { vsubq_f32(self.0, rhs.0) })
    }
}

impl Mul for F32s {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        // SAFETY: baseline NEON intrinsic, no preconditions.
        Self(unsafe { vmulq_f32(self.0, rhs.0) })
    }
}

impl Div for F32s {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        // SAFETY: baseline NEON intrinsic, no preconditions.
        Self(unsafe { vdivq_f32(self.0, rhs.0) })
    }
}

impl Neg for F32s {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        // SAFETY: baseline NEON intrinsic, no preconditions.
        Self(unsafe { vnegq_f32(self.0) })
    }
}

impl AddAssign for F32s {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for F32s {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for F32s {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl DivAssign for F32s {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}
