//! Tests for the streamed resampler.
//!
//! Every test must pass under both backends: `cargo test` (NEON on aarch64) and
//! `cargo test --features force-scalar`.

use super::naive;
use super::{Band, BandSink, ChecksumSink, Options, Plan, PlanarU8, resize_to_planar_f32, run};

/// Deterministic pseudo-random plane data.
struct Rng(u32);

impl Rng {
    fn next_u8(&mut self) -> u8 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.0 >> 24) as u8
    }
}

/// A `channels`-plane image of pseudo-random bytes.
fn image(width: usize, height: usize, channels: usize, seed: u32) -> PlanarU8 {
    let mut rng = Rng(seed.wrapping_mul(2_654_435_761).wrapping_add(1));
    PlanarU8 {
        width,
        height,
        planes: (0..channels)
            .map(|_| (0..width * height).map(|_| rng.next_u8()).collect())
            .collect(),
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Streamed output must match the naive reference for every geometry below.
fn check_against_naive(sw: usize, sh: usize, dw: usize, dh: usize, band_rows: usize) {
    let src = image(sw, sh, 2, (sw * 7919 + sh * 104_729 + dw * 31 + dh) as u32);
    let got = resize_to_planar_f32(&src, dw, dh, Options { band_rows });

    for (ch, plane) in src.planes.iter().enumerate() {
        let want = naive::resize_plane(plane, sw, sh, dw, dh);
        let diff = max_abs_diff(&got[ch], &want);
        assert!(
            diff < 1e-3,
            "{sw}x{sh} -> {dw}x{dh} (band_rows={band_rows}, ch={ch}): max abs diff {diff}"
        );
    }
}

#[test]
fn matches_naive_upscale() {
    check_against_naive(37, 23, 128, 96, 32);
}

#[test]
fn matches_naive_downscale() {
    check_against_naive(200, 150, 47, 33, 8);
}

#[test]
fn matches_naive_identity() {
    check_against_naive(64, 64, 64, 64, 16);
}

#[test]
fn matches_naive_non_uniform_scale() {
    // Up in x, down in y.
    check_against_naive(80, 120, 301, 29, 7);
}

#[test]
fn matches_naive_tiny() {
    check_against_naive(16, 16, 48, 32, 32);
}

#[test]
fn matches_naive_odd_sizes() {
    for (sw, sh, dw, dh) in [
        (1, 1, 5, 7),
        (3, 17, 19, 3),
        (17, 5, 1, 1),
        (33, 31, 65, 63),
        (100, 1, 7, 9),
    ] {
        check_against_naive(sw, sh, dw, dh, 4);
    }
}

/// Band height must not change the result.
#[test]
fn band_height_is_invisible() {
    let src = image(71, 53, 1, 5);
    let a = resize_to_planar_f32(&src, 143, 111, Options { band_rows: 1 });
    let b = resize_to_planar_f32(&src, 143, 111, Options { band_rows: 1024 });
    assert_eq!(a[0], b[0]);
}

/// A flat image must stay flat (weights sum to 1 everywhere, edges included).
#[test]
fn constant_image_is_preserved() {
    let src = PlanarU8 {
        width: 23,
        height: 19,
        planes: vec![vec![200u8; 23 * 19]],
    };
    let out = resize_to_planar_f32(&src, 91, 7, Options::default());
    let diff = out[0]
        .iter()
        .map(|v| (v - 200.0).abs())
        .fold(0.0f32, f32::max);
    assert!(diff < 1e-3, "constant image drifted by {diff}");
}

/// The checksum sink must see every output sample exactly once.
#[test]
fn checksum_sink_covers_the_whole_output() {
    let src = image(40, 30, 3, 11);
    let plan = Plan::new(40, 30, 97, 61, Options { band_rows: 9 });
    let sink = ChecksumSink::new();
    run(&plan, &src, &sink);
    assert_eq!(sink.samples(), (97 * 61 * 3) as u64);

    // Same checksum as summing the collected planes.
    let planes = resize_to_planar_f32(&src, 97, 61, Options { band_rows: 9 });
    let want: u64 = planes
        .iter()
        .flat_map(|p| p.iter())
        .fold(0u64, |acc, &v| acc.wrapping_add((v * 256.0) as i64 as u64));
    assert_eq!(sink.checksum(), want);
}

/// Bands must tile the output exactly, per channel.
#[test]
fn bands_tile_the_output() {
    use std::sync::Mutex;

    struct Coverage(Mutex<Vec<(usize, std::ops::Range<usize>)>>);
    impl BandSink for Coverage {
        fn band(&self, band: Band<'_>) {
            assert_eq!(band.data.len(), band.rows.len() * band.stride);
            self.0.lock().unwrap().push((band.channel, band.rows));
        }
    }

    let src = image(20, 20, 2, 3);
    let plan = Plan::new(20, 20, 33, 70, Options { band_rows: 16 });
    let cov = Coverage(Mutex::new(Vec::new()));
    run(&plan, &src, &cov);

    let mut seen = cov.0.into_inner().unwrap();
    seen.sort_by_key(|(c, r)| (*c, r.start));
    assert_eq!(seen.len(), 2 * plan.bands());
    for ch in 0..2 {
        let mut next = 0;
        for (c, r) in seen.iter().filter(|(c, _)| *c == ch) {
            assert_eq!(*c, ch);
            assert_eq!(r.start, next);
            next = r.end;
        }
        assert_eq!(next, 70);
    }
}

/// `from_interleaved` is the decode-edge conversion; check it deinterleaves.
#[test]
fn from_interleaved_splits_planes() {
    let data: Vec<u8> = (0..12).collect();
    let img = PlanarU8::from_interleaved(&data, 2, 2, 3);
    assert_eq!(img.planes[0], vec![0, 3, 6, 9]);
    assert_eq!(img.planes[1], vec![1, 4, 7, 10]);
    assert_eq!(img.planes[2], vec![2, 5, 8, 11]);
}
