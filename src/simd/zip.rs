//! Lane-view sources and the [`zip!`](crate::zip) macro.

use super::F32s;
use super::aligned::AlignedVec;
use super::chunk::Chunk;

/// Anything that can be viewed as a shared slice of lanes.
pub trait LaneSrc {
    fn lanes_ref(&self) -> &[F32s];
}

/// Anything that can be viewed as an exclusive slice of lanes.
pub trait LaneSrcMut {
    fn lanes_mut_ref(&mut self) -> &mut [F32s];
}

impl<const N: usize> LaneSrc for Chunk<N> {
    #[inline(always)]
    fn lanes_ref(&self) -> &[F32s] {
        self.lanes()
    }
}

impl<const N: usize> LaneSrcMut for Chunk<N> {
    #[inline(always)]
    fn lanes_mut_ref(&mut self) -> &mut [F32s] {
        self.lanes_mut()
    }
}

impl LaneSrc for AlignedVec {
    #[inline(always)]
    fn lanes_ref(&self) -> &[F32s] {
        self.lanes()
    }
}

impl LaneSrcMut for AlignedVec {
    #[inline(always)]
    fn lanes_mut_ref(&mut self) -> &mut [F32s] {
        self.lanes_mut()
    }
}

impl LaneSrc for [F32s] {
    #[inline(always)]
    fn lanes_ref(&self) -> &[F32s] {
        self
    }
}

impl LaneSrcMut for [F32s] {
    #[inline(always)]
    fn lanes_mut_ref(&mut self) -> &mut [F32s] {
        self
    }
}

/// Zip 1–6 lane-view sources into one iterator of flat tuples.
///
/// Read streams yield [`F32s`](crate::simd::F32s) by value; streams written
/// `mut expr` yield `&mut F32s`.
///
/// ```
/// use bannerify::simd::{Chunk, F32s};
/// use bannerify::zip;
///
/// let a = Chunk::<16>::splat(2.0);
/// let b = Chunk::<16>::splat(3.0);
/// let mut out = Chunk::<16>::zeroed();
/// for (o, x, y) in zip!(mut out, &a, &b) {
///     *o = x.mul_add(y, F32s::ONE);
/// }
/// assert_eq!(out[0], 7.0);
/// ```
#[macro_export]
macro_rules! zip {
    // Method syntax (not UFCS) so auto-ref/auto-deref accepts owned buffers,
    // `&`/`&mut` references and slices alike; the trait import is block-local.
    (@src mut $e:expr) => {{
        #[allow(unused_imports)]
        use $crate::simd::LaneSrcMut as _;
        $e.lanes_mut_ref().iter_mut()
    }};
    (@src $e:expr) => {{
        #[allow(unused_imports)]
        use $crate::simd::LaneSrc as _;
        $e.lanes_ref().iter().copied()
    }};

    // --- parse the comma-separated stream list -------------------------
    (@parse [$($acc:expr),*] mut $e:expr, $($rest:tt)+) => {
        $crate::zip!(@parse [$($acc,)* $crate::zip!(@src mut $e)] $($rest)+)
    };
    (@parse [$($acc:expr),*] $e:expr, $($rest:tt)+) => {
        $crate::zip!(@parse [$($acc,)* $crate::zip!(@src $e)] $($rest)+)
    };
    (@parse [$($acc:expr),*] mut $e:expr $(,)?) => {
        $crate::zip!(@build [$($acc,)* $crate::zip!(@src mut $e)])
    };
    (@parse [$($acc:expr),*] $e:expr $(,)?) => {
        $crate::zip!(@build [$($acc,)* $crate::zip!(@src $e)])
    };

    // --- build the zipped iterator (arity 1..=6) -----------------------
    (@build [$a:expr]) => {{ $a }};
    (@build [$a:expr, $b:expr]) => {{
        let (a, b) = ($a, $b);
        ::core::debug_assert_eq!(a.len(), b.len(), "zip!: stream lengths differ");
        a.zip(b)
    }};
    (@build [$a:expr, $b:expr, $c:expr]) => {{
        let (a, b, c) = ($a, $b, $c);
        ::core::debug_assert_eq!(a.len(), b.len(), "zip!: stream lengths differ");
        ::core::debug_assert_eq!(a.len(), c.len(), "zip!: stream lengths differ");
        a.zip(b).zip(c).map(|((a, b), c)| (a, b, c))
    }};
    (@build [$a:expr, $b:expr, $c:expr, $d:expr]) => {{
        let (a, b, c, d) = ($a, $b, $c, $d);
        ::core::debug_assert_eq!(a.len(), b.len(), "zip!: stream lengths differ");
        ::core::debug_assert_eq!(a.len(), c.len(), "zip!: stream lengths differ");
        ::core::debug_assert_eq!(a.len(), d.len(), "zip!: stream lengths differ");
        a.zip(b).zip(c).zip(d).map(|(((a, b), c), d)| (a, b, c, d))
    }};
    (@build [$a:expr, $b:expr, $c:expr, $d:expr, $e:expr]) => {{
        let (a, b, c, d, e) = ($a, $b, $c, $d, $e);
        ::core::debug_assert_eq!(a.len(), b.len(), "zip!: stream lengths differ");
        ::core::debug_assert_eq!(a.len(), c.len(), "zip!: stream lengths differ");
        ::core::debug_assert_eq!(a.len(), d.len(), "zip!: stream lengths differ");
        ::core::debug_assert_eq!(a.len(), e.len(), "zip!: stream lengths differ");
        a.zip(b)
            .zip(c)
            .zip(d)
            .zip(e)
            .map(|((((a, b), c), d), e)| (a, b, c, d, e))
    }};
    (@build [$a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr]) => {{
        let (a, b, c, d, e, f) = ($a, $b, $c, $d, $e, $f);
        ::core::debug_assert_eq!(a.len(), b.len(), "zip!: stream lengths differ");
        ::core::debug_assert_eq!(a.len(), c.len(), "zip!: stream lengths differ");
        ::core::debug_assert_eq!(a.len(), d.len(), "zip!: stream lengths differ");
        ::core::debug_assert_eq!(a.len(), e.len(), "zip!: stream lengths differ");
        ::core::debug_assert_eq!(a.len(), f.len(), "zip!: stream lengths differ");
        a.zip(b)
            .zip(c)
            .zip(d)
            .zip(e)
            .zip(f)
            .map(|(((((a, b), c), d), e), f)| (a, b, c, d, e, f))
    }};

    // --- entry point ---------------------------------------------------
    ($($streams:tt)+) => {
        $crate::zip!(@parse [] $($streams)+)
    };
}
