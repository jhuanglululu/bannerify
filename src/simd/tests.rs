//! Tests for the SIMD facade.
//!
//! Every test must pass under both backends: `cargo test` (NEON on aarch64)
//! and `cargo test --features force-scalar`.

use super::{AlignedVec, Chunk, F32s, LANES};
use crate::zip;

const N: usize = 64;

/// Deterministic pseudo-random stream; no NaNs (backend NaN semantics for
/// `min`/`max` differ and are not part of the contract).
struct Rng(u32);

impl Rng {
    fn new(seed: u32) -> Self {
        Self(seed.wrapping_mul(2_654_435_761).wrapping_add(1))
    }

    /// Uniform in `[-0.5, 0.5) * scale`.
    fn next(&mut self, scale: f32) -> f32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((self.0 >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * scale
    }
}

fn data(seed: u32, scale: f32) -> Chunk<N> {
    let mut rng = Rng::new(seed);
    let mut c = Chunk::<N>::zeroed();
    for x in c.iter_mut() {
        *x = rng.next(scale);
    }
    c
}

/// name, vector op, scalar reference op.
type BinCase = (&'static str, fn(F32s, F32s) -> F32s, fn(f32, f32) -> f32);
/// name, vector assign op, scalar reference op.
type AssignCase = (&'static str, fn(&mut F32s, F32s), fn(f32, f32) -> f32);

fn rel_err(got: f64, want: f64) -> f64 {
    (got - want).abs() / want.abs().max(1e-12)
}

// ---------------------------------------------------------------- F32s ops

#[test]
fn binary_ops_match_scalar_math() {
    let a = data(1, 4.0);
    let b = data(2, 3.0);
    let mut out = Chunk::<N>::zeroed();

    let cases: [BinCase; 6] = [
        ("add", |x, y| x + y, |x, y| x + y),
        ("sub", |x, y| x - y, |x, y| x - y),
        ("mul", |x, y| x * y, |x, y| x * y),
        ("div", |x, y| x / y, |x, y| x / y),
        ("min", F32s::min, f32::min),
        ("max", F32s::max, f32::max),
    ];

    for (name, vec_op, scalar_op) in cases {
        for (o, x, y) in zip!(mut out, &a, &b) {
            *o = vec_op(x, y);
        }
        for i in 0..N {
            assert_eq!(out[i], scalar_op(a[i], b[i]), "{name} lane {i}");
        }
    }
}

#[test]
fn assign_ops_match_binary_ops() {
    let a = data(3, 2.0);
    let b = data(4, 5.0);
    let mut out = Chunk::<N>::zeroed();

    let cases: [AssignCase; 4] = [
        ("add_assign", |x: &mut F32s, y| *x += y, |x, y| x + y),
        ("sub_assign", |x: &mut F32s, y| *x -= y, |x, y| x - y),
        ("mul_assign", |x: &mut F32s, y| *x *= y, |x, y| x * y),
        ("div_assign", |x: &mut F32s, y| *x /= y, |x, y| x / y),
    ];

    for (name, vec_op, scalar_op) in cases {
        for (o, x, y) in zip!(mut out, &a, &b) {
            *o = x;
            vec_op(o, y);
        }
        for i in 0..N {
            assert_eq!(out[i], scalar_op(a[i], b[i]), "{name} lane {i}");
        }
    }
}

#[test]
fn neg_matches_scalar() {
    let a = data(5, 7.0);
    let mut out = Chunk::<N>::zeroed();
    for (o, x) in zip!(mut out, &a) {
        *o = -x;
    }
    for i in 0..N {
        assert_eq!(out[i], -a[i], "lane {i}");
    }
}

#[test]
fn mul_add_is_fused() {
    let a = data(6, 1.0);
    let b = data(7, 1.0);
    let c = data(8, 1.0);
    let mut out = Chunk::<N>::zeroed();
    for (o, x, y, z) in zip!(mut out, &a, &b, &c) {
        *o = x.mul_add(y, z);
    }
    for i in 0..N {
        // Bit-exact against the fused scalar op (single rounding).
        assert_eq!(out[i], a[i].mul_add(b[i], c[i]), "lane {i}");
    }
}

#[test]
fn splat_and_consts() {
    let mut out = Chunk::<N>::zeroed();
    for o in zip!(mut out) {
        *o = F32s::splat(3.5);
    }
    assert!(out.iter().all(|&x| x == 3.5));

    for o in zip!(mut out) {
        *o = F32s::ZERO;
    }
    assert!(out.iter().all(|&x| x == 0.0));

    for o in zip!(mut out) {
        *o = F32s::ONE;
    }
    assert!(out.iter().all(|&x| x == 1.0));
}

#[test]
fn hsum_matches_scalar_sum() {
    let a = data(9, 6.0);
    let mut total = 0.0f64;
    for (k, lane) in a.lanes().iter().enumerate() {
        let want: f64 = a[k * LANES..(k + 1) * LANES]
            .iter()
            .map(|&x| x as f64)
            .sum();
        let got = lane.hsum() as f64;
        assert!(rel_err(got, want) < 1e-6, "lane {k}: {got} vs {want}");
        total += got;
    }
    let want: f64 = a.iter().map(|&x| x as f64).sum();
    assert!(rel_err(total, want) < 1e-5, "{total} vs {want}");
}

#[test]
fn compare_and_select() {
    let a = data(10, 2.0);
    let b = data(11, 2.0);
    let mut lt = Chunk::<N>::zeroed();
    let mut gt = Chunk::<N>::zeroed();
    for (l, g, x, y) in zip!(mut lt, mut gt, &a, &b) {
        *l = x.simd_lt(y).select(x, y);
        *g = x.simd_gt(y).select(x, y);
    }
    for i in 0..N {
        assert_eq!(lt[i], if a[i] < b[i] { a[i] } else { b[i] }, "lt lane {i}");
        assert_eq!(gt[i], if a[i] > b[i] { a[i] } else { b[i] }, "gt lane {i}");
    }
}

// ------------------------------------------------------------------ Chunk

#[test]
fn chunk_alignment_and_views() {
    let mut c = Chunk::<N>::splat(2.0);
    assert_eq!(c.as_ptr() as usize % 64, 0, "chunk must be 64-byte aligned");
    assert_eq!(c.lanes().len(), N / LANES);
    assert_eq!(c.lanes_mut().len(), N / LANES);
    assert_eq!(c.len(), N);
    assert!(c.iter().all(|&x| x == 2.0));

    c.fill(-1.5);
    assert!(c.iter().all(|&x| x == -1.5));

    let zeroed = Chunk::<N>::zeroed();
    assert!(zeroed.iter().all(|&x| x == 0.0));
}

#[test]
fn chunk_deref_roundtrip() {
    let mut c = Chunk::<32>::zeroed();
    for (i, x) in c.iter_mut().enumerate() {
        *x = i as f32;
    }
    // Written through `DerefMut`, read back through the lane view.
    let mut sum = F32s::ZERO;
    for x in zip!(&c) {
        sum += x;
    }
    assert_eq!(sum.hsum(), (0..32).map(|i| i as f32).sum::<f32>());

    // ... and written through the lane view, read back through `Deref`.
    for x in zip!(mut c) {
        *x = F32s::splat(9.0);
    }
    assert!(c.iter().all(|&x| x == 9.0));
}

// ------------------------------------------------------------- AlignedVec

#[test]
fn aligned_vec_alignment_and_views() {
    let v = AlignedVec::zeroed(128);
    assert_eq!(v.as_ptr() as usize % 64, 0, "must be 64-byte aligned");
    assert_eq!(v.len(), 128);
    assert!(!v.is_empty());
    assert_eq!(v.lanes().len(), 128 / LANES);
    assert!(v.iter().all(|&x| x == 0.0));

    let empty = AlignedVec::zeroed(0);
    assert!(empty.is_empty());
    assert_eq!(empty.lanes().len(), 0);
}

#[test]
fn aligned_vec_uninit_write_once() {
    // SAFETY: every element is written below before it is read.
    let mut v = unsafe { AlignedVec::new_uninit(64) };
    for (i, lane) in v.lanes_mut().iter_mut().enumerate() {
        *lane = F32s::splat(i as f32);
    }
    for i in 0..64 {
        assert_eq!(v[i], (i / LANES) as f32);
    }
    assert_eq!(v.as_ptr() as usize % 64, 0);
}

#[test]
fn aligned_vec_from_lane_fn_and_fill() {
    let v = AlignedVec::from_lane_fn(48, |i| F32s::splat(i as f32 * 2.0));
    for i in 0..48 {
        assert_eq!(v[i], (i / LANES) as f32 * 2.0);
    }

    let mut s = AlignedVec::splat(16, 4.25);
    assert!(s.iter().all(|&x| x == 4.25));
    s.fill(-2.0);
    assert!(s.iter().all(|&x| x == -2.0));
}

#[test]
#[should_panic(expected = "len must be a multiple of 16")]
fn aligned_vec_rejects_bad_length() {
    let _ = AlignedVec::zeroed(20);
}

// -------------------------------------------------------------------- zip!

#[test]
fn zip_arities() {
    let a = AlignedVec::splat(32, 1.0);
    let b = AlignedVec::splat(32, 2.0);
    let c = AlignedVec::splat(32, 3.0);
    let d = AlignedVec::splat(32, 4.0);
    let e = AlignedVec::splat(32, 5.0);
    let f = AlignedVec::splat(32, 6.0);
    let lanes = 32 / LANES;

    assert_eq!(zip!(&a).count(), lanes);
    assert_eq!(zip!(&a, &b).count(), lanes);

    let mut acc = F32s::ZERO;
    for (x, y, z) in zip!(&a, &b, &c) {
        acc += x + y + z;
    }
    assert_eq!(acc.hsum(), 32.0 * 6.0);

    let mut acc = F32s::ZERO;
    for (w, x, y, z) in zip!(&a, &b, &c, &d) {
        acc += w + x + y + z;
    }
    assert_eq!(acc.hsum(), 32.0 * 10.0);

    let mut acc = F32s::ZERO;
    for (v, w, x, y, z) in zip!(&a, &b, &c, &d, &e) {
        acc += v + w + x + y + z;
    }
    assert_eq!(acc.hsum(), 32.0 * 15.0);

    let mut acc = F32s::ZERO;
    for (u, v, w, x, y, z) in zip!(&a, &b, &c, &d, &e, &f) {
        acc += u + v + w + x + y + z;
    }
    assert_eq!(acc.hsum(), 32.0 * 21.0);
}

#[test]
fn zip_multiple_mut_streams() {
    let src = data(12, 3.0);
    let mut out = Chunk::<N>::zeroed();
    // Two exclusive streams from one buffer via `split_at_mut` — borrow
    // splitting is the caller's job.
    let half = N / LANES / 2;
    let (lo, hi) = out.lanes_mut().split_at_mut(half);
    let (src_lo, src_hi) = src.lanes().split_at(half);
    for (l, h, x, y) in zip!(mut lo, mut hi, src_lo, src_hi) {
        *l = x + y;
        *h = x - y;
    }
    for i in 0..N / 2 {
        assert_eq!(out[i], src[i] + src[N / 2 + i]);
        assert_eq!(out[N / 2 + i], src[i] - src[N / 2 + i]);
    }
}

#[test]
fn zip_accepts_mixed_source_kinds() {
    let chunk = Chunk::<32>::splat(1.5);
    let vec = AlignedVec::splat(32, 2.5);
    let mut out = AlignedVec::zeroed(32);
    let slice: &[F32s] = chunk.lanes();
    for (o, x, y) in zip!(mut out, slice, &vec) {
        *o = x * y;
    }
    assert!(out.iter().all(|&x| x == 3.75));
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "stream lengths differ")]
fn zip_debug_asserts_equal_lengths() {
    let a = AlignedVec::zeroed(32);
    let b = AlignedVec::zeroed(16);
    let _ = zip!(&a, &b).count();
}

// --------------------------------------------------- reference kernel port

/// Greedy residual moments from the design doc, computed through the facade.
fn residual_moments(prefix: &AlignedVec, target: &AlignedVec, pattern: &AlignedVec) -> (f32, f32) {
    let mut res2 = F32s::ZERO;
    let mut res_a = F32s::ZERO;
    for (pre, tar, alp) in zip!(prefix, target, pattern) {
        let res = pre.mul_add(F32s::ONE - alp, -tar);
        res2 = res.mul_add(res, res2);
        res_a = res.mul_add(alp, res_a);
    }
    (res2.hsum(), 2.0 * res_a.hsum())
}

#[test]
fn residual_moments_match_scalar_reference() {
    const LEN: usize = 4096;
    // Write-once fill through `DerefMut`; values in [0, 1) like real
    // colour/alpha planes.
    let mk = |seed: u32| {
        let mut rng = Rng::new(seed);
        // SAFETY: every element is written before any read.
        let mut v = unsafe { AlignedVec::new_uninit(LEN) };
        for x in v.iter_mut() {
            *x = rng.next(1.0) + 0.5;
        }
        v
    };
    let prefix = mk(21);
    let target = mk(22);
    let pattern = mk(23);

    let (res2, res_2a) = residual_moments(&prefix, &target, &pattern);

    let mut want2 = 0.0f64;
    let mut want_a = 0.0f64;
    for i in 0..LEN {
        let res = prefix[i] as f64 * (1.0 - pattern[i] as f64) - target[i] as f64;
        want2 += res * res;
        want_a += res * pattern[i] as f64;
    }
    let want_2a = 2.0 * want_a;

    assert!(rel_err(res2 as f64, want2) < 1e-4, "res2 {res2} vs {want2}");
    assert!(
        rel_err(res_2a as f64, want_2a) < 1e-4,
        "res_2a {res_2a} vs {want_2a}"
    );
}
