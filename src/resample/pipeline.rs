//! The streamed band pipeline.
//!
//! Output is produced in horizontal bands. Each `(channel, band)` task owns a
//! ring of `ksize_y` horizontally-resampled rows; per output row it pulls the
//! source rows it still needs through the H pass into the ring (ring indexing —
//! rows are never shifted or copied), then the V pass is a `ksize_y`-tap
//! weighted sum straight into the band buffer. No full-size intermediate and no
//! full-size output ever exist inside the pipeline.

use rayon::prelude::*;

use super::sink::{Band, BandSink};
use super::weights::{HWeights, VWeights, round_up};
use crate::simd::{AlignedVec, F32s, LANES};
use crate::zip;

/// A decoded image as separate `u8` planes, row-major, `width * height` each.
///
/// Planar is the only layout the crate uses past decode; the single
/// interleaved → planar conversion happens here, at the decode edge.
pub struct PlanarU8 {
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    /// One plane per channel.
    pub planes: Vec<Vec<u8>>,
}

impl PlanarU8 {
    /// Split interleaved samples into planes. Decode-edge only.
    pub fn from_interleaved(data: &[u8], width: usize, height: usize, channels: usize) -> Self {
        assert_eq!(data.len(), width * height * channels, "size mismatch");
        let planes = (0..channels)
            .map(|c| data.iter().skip(c).step_by(channels).copied().collect())
            .collect();
        Self {
            width,
            height,
            planes,
        }
    }
}

/// A rectangular source region, in fractional source-pixel coordinates.
///
/// This is Pillow's `box`: it is applied inside the weight tables, so cropping
/// costs nothing and never materialises a sub-image.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Window {
    /// Left edge.
    pub x0: f64,
    /// Top edge.
    pub y0: f64,
    /// Right edge (exclusive).
    pub x1: f64,
    /// Bottom edge (exclusive).
    pub y1: f64,
}

impl Window {
    /// The whole `width × height` image.
    pub fn full(width: usize, height: usize) -> Self {
        Self {
            x0: 0.0,
            y0: 0.0,
            x1: width as f64,
            y1: height as f64,
        }
    }
}

/// Tuning knobs for a resize.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Output rows per band. Bands × channels is the unit of parallelism, and a
    /// band's working set is `band_rows * out_width * 4` bytes per task.
    pub band_rows: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self { band_rows: 32 }
    }
}

/// Everything that depends only on the source/target geometry: the two weight
/// tables and the band layout. Build once, reuse for every channel and image of
/// the same shape.
pub struct Plan {
    src_width: usize,
    src_height: usize,
    h: HWeights,
    v: VWeights,
    band_rows: usize,
    /// Row pitch of band buffers, in floats (multiple of 16).
    out_stride: usize,
    /// Padded length of the H-pass source row scratch buffer.
    src_stride: usize,
}

impl Plan {
    /// Build the plan for the whole `src_width × src_height` source →
    /// `dst_width × dst_height`.
    pub fn new(
        src_width: usize,
        src_height: usize,
        dst_width: usize,
        dst_height: usize,
        opts: Options,
    ) -> Self {
        Self::with_window(
            src_width,
            src_height,
            Window::full(src_width, src_height),
            dst_width,
            dst_height,
            opts,
        )
    }

    /// Build the plan for the source region `window` → `dst_width × dst_height`.
    pub fn with_window(
        src_width: usize,
        src_height: usize,
        window: Window,
        dst_width: usize,
        dst_height: usize,
        opts: Options,
    ) -> Self {
        assert!(opts.band_rows > 0, "band_rows must be positive");
        let h = HWeights::new(src_width, window.x0, window.x1, dst_width);
        let v = VWeights::new(src_height, window.y0, window.y1, dst_height);
        let src_stride = h.src_padded_len(src_width);
        Self {
            src_width,
            src_height,
            h,
            v,
            band_rows: opts.band_rows,
            out_stride: round_up(dst_width, 16),
            src_stride,
        }
    }

    /// Output width in pixels.
    pub fn dst_width(&self) -> usize {
        self.h.out_len
    }

    /// Output height in pixels.
    pub fn dst_height(&self) -> usize {
        self.v.out_len
    }

    /// Number of bands the output is split into.
    pub fn bands(&self) -> usize {
        self.dst_height().div_ceil(self.band_rows)
    }
}

/// Run the streamed resampler, emitting finished bands to `sink`.
///
/// `(channel, band)` pairs run in parallel; source rows on a band boundary are
/// H-passed once per adjacent band, which is the (negligible) cost of not
/// sharing state between tasks.
pub fn run<S: BandSink>(plan: &Plan, src: &PlanarU8, sink: &S) {
    assert_eq!(src.width, plan.src_width, "source width mismatch");
    assert_eq!(src.height, plan.src_height, "source height mismatch");

    let bands = plan.bands();
    let tasks: Vec<(usize, usize)> = (0..src.planes.len())
        .flat_map(|c| (0..bands).map(move |b| (c, b)))
        .collect();

    tasks.par_iter().for_each(|&(channel, band)| {
        run_band(plan, &src.planes[channel], channel, band, sink);
    });
}

/// Resize `src` into full planar `f32` planes (convenience wrapper around the
/// collecting sink; the full image *does* exist here, by request of the caller).
pub fn resize_to_planar_f32(
    src: &PlanarU8,
    dst_width: usize,
    dst_height: usize,
    opts: Options,
) -> Vec<Vec<f32>> {
    let plan = Plan::new(src.width, src.height, dst_width, dst_height, opts);
    let sink = super::PlanarF32Sink::new(dst_width, dst_height, src.planes.len());
    run(&plan, src, &sink);
    sink.into_planes()
}

/// One `(channel, band)` task.
fn run_band<S: BandSink>(plan: &Plan, plane: &[u8], channel: usize, band: usize, sink: &S) {
    let row0 = band * plan.band_rows;
    let row1 = (row0 + plan.band_rows).min(plan.dst_height());
    let stride = plan.out_stride;
    let kv = plan.v.ksize;

    // Source row scratch: `zeroed` so the tail past `src_width` reads as a
    // finite 0.0 for the padded window; the first `src_width` floats are
    // overwritten for every row loaded.
    let mut srow = AlignedVec::zeroed(plan.src_stride);

    // Ring of `kv` H-passed rows. Write-once: `h_pass` writes every element of a
    // slot, and a slot is only read after the row that filled it was loaded.
    let mut ring: Vec<AlignedVec> = (0..kv)
        .map(|_| {
            // SAFETY: each slot is fully written by `h_pass` (which writes all
            // `stride` floats, padding included) before `load` returns, and no
            // slot is read before the corresponding `load`.
            unsafe { AlignedVec::new_uninit(stride) }
        })
        .collect();

    // SAFETY: every element of the band buffer is written by `v_pass` below —
    // one full `stride`-float row per output row of the band — before the sink
    // is handed a borrow of it.
    let mut buf = unsafe { AlignedVec::new_uninit((row1 - row0) * stride) };

    let mut next_src = plan.v.starts[row0];
    for out_row in row0..row1 {
        let (start, taps) = plan.v.row(out_row);
        for y in next_src.max(start)..start + kv {
            let clamped = y.min(plan.src_height - 1);
            load_row(plane, plan.src_width, clamped, &mut srow);
            h_pass(&plan.h, &srow, &mut ring[y % kv]);
        }
        next_src = next_src.max(start + kv);
        v_pass(&ring, kv, start, taps, &mut buf, out_row - row0, stride);
    }

    sink.band(Band {
        channel,
        rows: row0..row1,
        width: plan.dst_width(),
        stride,
        data: &buf,
    });
}

/// Widen source row `y` into `dst[..src_width]` (u8 → f32 at H-pass load).
fn load_row(plane: &[u8], src_width: usize, y: usize, dst: &mut AlignedVec) {
    let row = &plane[y * src_width..(y + 1) * src_width];
    for (d, &s) in dst[..src_width].iter_mut().zip(row) {
        *d = f32::from(s);
    }
}

/// Horizontal pass: one padded source row in, one full output row out.
fn h_pass(w: &HWeights, src: &AlignedVec, dst: &mut AlignedVec) {
    let table = w.table.lanes();
    let src_lanes = src.lanes();
    let out = &mut dst[..];
    for x in 0..w.out_len {
        let s = w.start_lane[x];
        let wl = &table[x * w.win_lanes..(x + 1) * w.win_lanes];
        let sl = &src_lanes[s..s + w.win_lanes];
        let mut acc = F32s::ZERO;
        for (a, b) in zip!(wl, sl) {
            acc = a.mul_add(b, acc);
        }
        out[x] = acc.hsum();
    }
    // Row padding: written (never uninitialised) and zero, so the V pass can
    // treat whole rows uniformly.
    for v in &mut out[w.out_len..] {
        *v = 0.0;
    }
}

/// Vertical pass: `taps`-tap weighted sum of ring rows into band row `dst_row`.
fn v_pass(
    ring: &[AlignedVec],
    kv: usize,
    start: usize,
    taps: &[f32],
    buf: &mut AlignedVec,
    dst_row: usize,
    stride: usize,
) {
    let lanes = stride / LANES;
    let lo = dst_row * lanes;
    let hi = lo + lanes;

    let w0 = F32s::splat(taps[0]);
    let src = &ring[start % kv];
    for (o, s) in zip!(mut buf.lanes_mut()[lo..hi], src) {
        *o = s * w0;
    }
    for (k, &t) in taps.iter().enumerate().skip(1) {
        let wk = F32s::splat(t);
        let src = &ring[(start + k) % kv];
        for (o, s) in zip!(mut buf.lanes_mut()[lo..hi], src) {
            *o = s.mul_add(wk, *o);
        }
    }
}
