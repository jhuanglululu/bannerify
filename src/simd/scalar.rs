//! Scalar fallback backend: `LANES == 1`, one `f32` per [`F32s`].

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Lanes per [`F32s`] for this backend.
pub const LANES: usize = 1;

/// A vector of [`LANES`] `f32`s.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct F32s(f32);

/// Result of a lane-wise comparison; consume with [`Mask::select`].
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Mask(bool);

impl F32s {
    pub const ZERO: Self = Self(0.0);
    pub const ONE: Self = Self(1.0);

    #[inline(always)]
    pub fn splat(x: f32) -> Self {
        Self(x)
    }

    /// `self * a + b`, fused (single rounding) — matches the vector backends.
    #[inline(always)]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        Self(self.0.mul_add(a.0, b.0))
    }

    /// Sum of all lanes. Keep out of inner loops.
    #[inline(always)]
    pub fn hsum(self) -> f32 {
        self.0
    }

    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self(self.0.sqrt())
    }

    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(self.0.min(other.0))
    }

    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(self.0.max(other.0))
    }

    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Mask {
        Mask(self.0 < other.0)
    }

    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Mask {
        Mask(self.0 > other.0)
    }

    /// Build a register from lane values, in order.
    #[inline(always)]
    pub fn from_array(v: [f32; LANES]) -> Self {
        Self(v[0])
    }

    #[inline(always)]
    pub fn to_array(self) -> [f32; LANES] {
        [self.0]
    }
}

impl Mask {
    #[inline(always)]
    pub fn select(self, a: F32s, b: F32s) -> F32s {
        if self.0 { a } else { b }
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
        Self(self.0 + rhs.0)
    }
}

impl Sub for F32s {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl Mul for F32s {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0)
    }
}

impl Div for F32s {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        Self(self.0 / rhs.0)
    }
}

impl Neg for F32s {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(-self.0)
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
