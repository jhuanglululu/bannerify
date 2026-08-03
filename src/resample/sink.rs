//! Band sinks: where finished bands go.
//!
//! The pipeline never materialises the full resized image itself — it hands
//! each finished band to a [`BandSink`] and immediately reuses the buffer.

use std::ops::Range;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// One finished band of one channel, planar `f32`.
pub struct Band<'a> {
    /// Channel index this band belongs to.
    pub channel: usize,
    /// Output rows covered, `start..end`.
    pub rows: Range<usize>,
    /// Valid pixels per row.
    pub width: usize,
    /// Row pitch inside [`Band::data`] (`>= width`; the tail is padding).
    pub stride: usize,
    /// `rows.len() * stride` floats.
    pub data: &'a [f32],
}

impl Band<'_> {
    /// The `width` valid samples of the `i`-th row of this band.
    #[inline]
    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.stride..i * self.stride + self.width]
    }
}

/// Consumer of finished bands.
///
/// The pipeline runs bands in parallel with rayon, so the sink is shared as
/// `&self` and must be `Sync`; sinks that accumulate use interior mutability.
/// (The alternative — one owned sink per band — was rejected: it cannot express
/// a single collector, which is exactly what the PNG and test paths need.)
pub trait BandSink: Sync {
    /// Called once per finished band. The borrowed buffer is reused for later
    /// bands, so the sink must copy anything it wants to keep.
    fn band(&self, band: Band<'_>);
}

/// Collects every band into full planar `f32` planes.
///
/// This is the correctness / PNG-encode path, not the timing path: it holds the
/// whole output image, and bands copy into it under a per-channel mutex.
pub struct PlanarF32Sink {
    width: usize,
    height: usize,
    planes: Vec<Mutex<Vec<f32>>>,
}

impl PlanarF32Sink {
    /// A collector for a `width * height` image with `channels` planes.
    pub fn new(width: usize, height: usize, channels: usize) -> Self {
        Self {
            width,
            height,
            planes: (0..channels)
                .map(|_| Mutex::new(vec![0.0f32; width * height]))
                .collect(),
        }
    }

    /// The collected planes, row-major, `width * height` each.
    pub fn into_planes(self) -> Vec<Vec<f32>> {
        self.planes
            .into_iter()
            .map(|p| p.into_inner().expect("sink mutex poisoned"))
            .collect()
    }
}

impl BandSink for PlanarF32Sink {
    fn band(&self, band: Band<'_>) {
        debug_assert_eq!(band.width, self.width);
        debug_assert!(band.rows.end <= self.height);
        let mut plane = self.planes[band.channel]
            .lock()
            .expect("sink mutex poisoned");
        for (i, y) in band.rows.clone().enumerate() {
            plane[y * self.width..(y + 1) * self.width].copy_from_slice(band.row(i));
        }
    }
}

/// Discards band data, keeping only an order-independent checksum.
///
/// The pure-timing sink: no output allocation, but the samples are still read
/// so the resample work cannot be optimised away.
#[derive(Default)]
pub struct ChecksumSink {
    sum: AtomicU64,
    samples: AtomicU64,
}

impl ChecksumSink {
    /// A fresh, empty checksum sink.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sum of every emitted sample (fixed point, 1/256 units — integer adds, so
    /// the result does not depend on band completion order).
    pub fn checksum(&self) -> u64 {
        self.sum.load(Ordering::Relaxed)
    }

    /// Number of samples emitted.
    pub fn samples(&self) -> u64 {
        self.samples.load(Ordering::Relaxed)
    }
}

impl BandSink for ChecksumSink {
    fn band(&self, band: Band<'_>) {
        let mut acc: u64 = 0;
        for i in 0..band.rows.len() {
            for &v in band.row(i) {
                acc = acc.wrapping_add((v * 256.0) as i64 as u64);
            }
        }
        self.sum.fetch_add(acc, Ordering::Relaxed);
        self.samples
            .fetch_add((band.rows.len() * band.width) as u64, Ordering::Relaxed);
    }
}
