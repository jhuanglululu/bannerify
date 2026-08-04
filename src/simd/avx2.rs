//! x86_64 AVX2 backend: one [`F32s`] == one `__m256` register.
//!
//! This module is compiled only when `avx2` and `fma` are enabled as *target*
//! features for the whole crate (see the cfg in `super`), so every intrinsic
//! used here is unconditionally available and inlines like any other code; the
//! `unsafe` on each call is a formality the intrinsic signatures impose. The
//! CLI checks the running CPU actually has both at startup
//! ([`crate::app::run_cli`]) — a binary built this way must not be run on a
//! pre-2013 x86 CPU.

use core::arch::x86_64::*;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Lanes per [`F32s`] for this backend.
pub const LANES: usize = 8;

/// A vector of [`LANES`] `f32`s: exactly one native register.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct F32s(__m256);

/// Result of a lane-wise comparison; consume with [`Mask::select`].
///
/// Held as `__m256` rather than an integer vector because that is what
/// `_mm256_cmp_ps` produces and what `_mm256_blendv_ps` consumes.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Mask(__m256);

impl F32s {
    /// All lanes zero.
    // SAFETY: `__m256` is eight `f32`s with no niches or padding, so any
    // `[f32; 8]` bit pattern is a valid register value.
    pub const ZERO: Self = Self(unsafe { core::mem::transmute::<[f32; 8], __m256>([0.0; 8]) });
    /// All lanes one.
    // SAFETY: as above.
    pub const ONE: Self = Self(unsafe { core::mem::transmute::<[f32; 8], __m256>([1.0; 8]) });

    /// Broadcast `x` to every lane.
    #[inline(always)]
    pub fn splat(x: f32) -> Self {
        // SAFETY: `_mm256_set1_ps` requires `avx`, implied by the module's `avx2` cfg.
        Self(unsafe { _mm256_set1_ps(x) })
    }

    /// `self * a + b`, fused (single rounding) — emits `vfmadd*ps`.
    #[inline(always)]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        // SAFETY: `_mm256_fmadd_ps(x, y, z)` computes `x * y + z`; requires `fma`, per the module cfg.
        Self(unsafe { _mm256_fmadd_ps(self.0, a.0, b.0) })
    }

    /// Sum of all lanes. Keep out of inner loops.
    ///
    /// x86 has no single-instruction horizontal add: this folds the upper 128
    /// bits onto the lower, then halves twice within the 128-bit lane — three
    /// adds and three shuffles, an order of magnitude more than NEON's `faddp`.
    #[inline(always)]
    pub fn hsum(self) -> f32 {
        // SAFETY: all of these require at most `avx`/`sse3`, implied by the module's `avx2` cfg;
        // the shuffle immediates are compile-time constants in range.
        unsafe {
            // 8 -> 4: add the high 128-bit half to the low one.
            let lo = _mm256_castps256_ps128(self.0);
            let hi = _mm256_extractf128_ps::<1>(self.0);
            let q = _mm_add_ps(lo, hi);
            // 4 -> 2: add the upper pair to the lower pair.
            let d = _mm_add_ps(q, _mm_movehl_ps(q, q));
            // 2 -> 1: add lane 1 to lane 0.
            let s = _mm_add_ss(d, _mm_shuffle_ps::<0b01>(d, d));
            _mm_cvtss_f32(s)
        }
    }

    /// Lane-wise square root.
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        // SAFETY: `_mm256_sqrt_ps` requires `avx`, implied by the module's `avx2` cfg.
        Self(unsafe { _mm256_sqrt_ps(self.0) })
    }

    /// Lane-wise minimum.
    ///
    /// NaN behaviour is uncontracted, as on NEON: `vminps` returns its second
    /// operand when either input is NaN. Nothing in the crate feeds these NaNs.
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        // SAFETY: `_mm256_min_ps` requires `avx`, implied by the module's `avx2` cfg.
        Self(unsafe { _mm256_min_ps(self.0, other.0) })
    }

    /// Lane-wise maximum. NaN behaviour is uncontracted; see [`F32s::min`].
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        // SAFETY: `_mm256_max_ps` requires `avx`, implied by the module's `avx2` cfg.
        Self(unsafe { _mm256_max_ps(self.0, other.0) })
    }

    /// Lane-wise `self < other`.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Mask {
        // SAFETY: `_mm256_cmp_ps` requires `avx`, implied by the module's `avx2` cfg;
        // `_CMP_LT_OQ` is a valid compile-time predicate.
        Mask(unsafe { _mm256_cmp_ps::<_CMP_LT_OQ>(self.0, other.0) })
    }

    /// Lane-wise `self > other`.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Mask {
        // SAFETY: as above, with the `_CMP_GT_OQ` predicate.
        Mask(unsafe { _mm256_cmp_ps::<_CMP_GT_OQ>(self.0, other.0) })
    }

    /// Build a register from lane values, in order.
    ///
    /// The inverse of [`F32s::to_array`], and the same cost class: a store plus
    /// a load. It exists for the one thing vector code genuinely cannot do
    /// cheaply — a per-lane table lookup — which is how [`crate::oklab`]
    /// linearises sRGB. Not for arithmetic.
    #[inline(always)]
    pub fn from_array(v: [f32; LANES]) -> Self {
        // SAFETY: `[f32; 8]` and `__m256` have the same size and layout, and
        // every bit pattern is valid for both.
        Self(unsafe { core::mem::transmute::<[f32; 8], __m256>(v) })
    }

    /// Lane values, in order. Diagnostics and cold paths only.
    #[inline(always)]
    pub fn to_array(self) -> [f32; LANES] {
        // SAFETY: `__m256` and `[f32; 8]` have the same size and layout, and
        // every bit pattern is valid for both.
        unsafe { core::mem::transmute::<__m256, [f32; 8]>(self.0) }
    }
}

impl Mask {
    /// Lane-wise `if mask { a } else { b }`.
    #[inline(always)]
    pub fn select(self, a: F32s, b: F32s) -> F32s {
        // SAFETY: `_mm256_blendv_ps(x, y, m)` takes `y` where `m`'s sign bit is
        // set and `x` elsewhere, hence the swapped argument order; requires
        // `avx`, implied by the module's `avx2` cfg.
        F32s(unsafe { _mm256_blendv_ps(b.0, a.0, self.0) })
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
        // SAFETY: `_mm256_add_ps` requires `avx`, implied by the module's `avx2` cfg.
        Self(unsafe { _mm256_add_ps(self.0, rhs.0) })
    }
}

impl Sub for F32s {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        // SAFETY: `_mm256_sub_ps` requires `avx`, implied by the module's `avx2` cfg.
        Self(unsafe { _mm256_sub_ps(self.0, rhs.0) })
    }
}

impl Mul for F32s {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        // SAFETY: `_mm256_mul_ps` requires `avx`, implied by the module's `avx2` cfg.
        Self(unsafe { _mm256_mul_ps(self.0, rhs.0) })
    }
}

impl Div for F32s {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        // SAFETY: `_mm256_div_ps` requires `avx`, implied by the module's `avx2` cfg.
        Self(unsafe { _mm256_div_ps(self.0, rhs.0) })
    }
}

impl Neg for F32s {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        // Sign-bit flip rather than `0.0 - x`: it is one `vxorps`, and it
        // matches NEON's `fneg` on every input — `0.0 - x` would turn `-0.0`
        // into `+0.0` and quieten signalling NaNs.
        // SAFETY: `_mm256_xor_ps` requires `avx`, implied by the module's `avx2` cfg.
        Self(unsafe { _mm256_xor_ps(self.0, _mm256_set1_ps(-0.0)) })
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
