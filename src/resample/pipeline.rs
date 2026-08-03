//! Streamed lanczos-3, sliced by output row band.
//!
//! The unit of work is a **band of output rows** ([`RowPlan`]), resampled by
//! its owner into a closure-local [`RowBand`] — that is the pipeline's parallel
//! work item (see `context/designs/pipeline.md`), so this module hands out no
//! parallelism of its own and orchestrates nothing.
//!
//! Inside a band: a ring of `ksize_y` horizontally-resampled rows is filled by
//! the H pass (ring indexing — rows are never shifted or copied), and each
//! output row is a `ksize_y`-tap weighted sum of the ring straight into the band
//! buffer. Nothing wall-sized is ever materialised here; a band's working set is
//! `(ksize_y + band_height) * padded_width * 4` bytes per channel.
//!
//! The horizontal weight table is expensive to build (a lanczos evaluation per
//! output pixel per tap) and identical for every band, so it lives in the shared
//! [`Plan`]; only the small vertical table is per band.

use std::ops::Range;

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

    /// Number of channels.
    pub fn channels(&self) -> usize {
        self.planes.len()
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

/// The shared, immutable half of a resize: source geometry, target size, and the
/// horizontal weight table. Build once, borrow from every work item.
pub struct Plan {
    src_width: usize,
    src_height: usize,
    window: Window,
    dst_height: usize,
    h: HWeights,
    /// Row pitch of band buffers, in floats (multiple of 16).
    out_stride: usize,
    /// Padded length of the H-pass source row scratch buffer.
    src_stride: usize,
}

impl Plan {
    /// Build the plan for the whole source → `dst_width × dst_height`.
    pub fn new(src_width: usize, src_height: usize, dst_width: usize, dst_height: usize) -> Self {
        Self::with_window(
            src_width,
            src_height,
            Window::full(src_width, src_height),
            dst_width,
            dst_height,
        )
    }

    /// Build the plan for the source region `window` → `dst_width × dst_height`.
    pub fn with_window(
        src_width: usize,
        src_height: usize,
        window: Window,
        dst_width: usize,
        dst_height: usize,
    ) -> Self {
        let h = HWeights::new(src_width, window.x0, window.x1, dst_width);
        let src_stride = h.src_padded_len(src_width);
        Self {
            src_width,
            src_height,
            window,
            dst_height,
            h,
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
        self.dst_height
    }

    /// The source rows band `rows` reads from, in fractional source pixels.
    ///
    /// Adjacent bands overlap by the vertical kernel support (~`6 * scale`
    /// rows): every band is computed independently from the source, so the
    /// overlap is re-read (and re-H-passed), never shared.
    pub fn src_rows(&self, rows: &Range<usize>) -> (f64, f64) {
        let scale = (self.window.y1 - self.window.y0) / self.dst_height as f64;
        (
            self.window.y0 + rows.start as f64 * scale,
            self.window.y0 + rows.end as f64 * scale,
        )
    }

    /// The plan for one band of output rows.
    ///
    /// The band's vertical weights are computed for the sub-window `rows` maps
    /// to, so the coefficients are the same ones the whole-image plan would
    /// produce for those rows — banding is invisible in the result.
    pub fn rows(&self, rows: Range<usize>) -> RowPlan<'_> {
        assert!(rows.end <= self.dst_height, "band outside the output");
        self.band(self.src_rows(&rows), rows.len())
    }

    /// The plan for `height` output rows covering the source interval
    /// `src_rows` — the same thing [`Plan::rows`] builds, addressed by the work
    /// item's own source rect instead of by output row indices.
    pub fn band(&self, src_rows: (f64, f64), height: usize) -> RowPlan<'_> {
        assert!(height > 0, "empty band");
        RowPlan {
            plan: self,
            v: VWeights::new(self.src_height, src_rows.0, src_rows.1, height),
            height,
        }
    }
}

/// The plan for one band of output rows: the shared [`Plan`] plus the band's own
/// vertical weights.
pub struct RowPlan<'a> {
    plan: &'a Plan,
    v: VWeights,
    height: usize,
}

impl RowPlan<'_> {
    /// Output rows in this band.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Resample this band out of `src`.
    ///
    /// The returned [`RowBand`] is the caller's local: the phase-2 solver
    /// borrows its cell patches straight out of it, and it dies with the work
    /// item. Note for later: these per-item buffers (band + ring + scratch) are
    /// a natural fit for a per-worker arena handed out without zeroing, which
    /// would remove the per-item allocation traffic entirely.
    pub fn resample(&self, src: &PlanarU8) -> RowBand {
        let plan = self.plan;
        assert_eq!(src.width, plan.src_width, "source width mismatch");
        assert_eq!(src.height, plan.src_height, "source height mismatch");

        let height = self.height;
        let stride = plan.out_stride;
        let kv = self.v.ksize;

        // Source row scratch: `zeroed` so the tail past `src_width` reads as a
        // finite 0.0 for the padded window; the first `src_width` floats are
        // overwritten for every row loaded.
        let mut srow = AlignedVec::zeroed(plan.src_stride);

        // Ring of `kv` H-passed rows, reused in place across the whole band.
        let mut ring: Vec<AlignedVec> = (0..kv)
            .map(|_| {
                // SAFETY: every element of a slot is written by `h_pass` before
                // the slot is read — a slot is only read for a source row that
                // the loop below has already loaded and H-passed.
                unsafe { AlignedVec::new_uninit(stride) }
            })
            .collect();

        let planes = src
            .planes
            .iter()
            .map(|plane| {
                // SAFETY: `v_pass` writes every element of the band buffer —
                // one full `stride`-float row per output row — before it is read.
                let mut buf = unsafe { AlignedVec::new_uninit(height * stride) };
                let mut next_src = self.v.starts[0];
                for local in 0..height {
                    let (start, taps) = self.v.row(local);
                    for y in next_src.max(start)..start + kv {
                        let clamped = y.min(plan.src_height - 1);
                        load_row(plane, plan.src_width, clamped, &mut srow);
                        h_pass(&plan.h, &srow, &mut ring[y % kv]);
                    }
                    next_src = next_src.max(start + kv);
                    v_pass(&ring, kv, start, taps, &mut buf, local, stride);
                }
                buf
            })
            .collect();

        RowBand {
            width: plan.dst_width(),
            height,
            stride,
            planes,
        }
    }
}

/// One band of resampled output, planar `f32`, one plane per channel.
///
/// Rows are padded to a multiple of 16 floats ([`RowBand::stride`]); only the
/// first [`RowBand::width`] samples of each row are image data.
pub struct RowBand {
    /// Valid samples per row.
    pub width: usize,
    /// Rows in this band.
    pub height: usize,
    /// Row pitch in floats.
    pub stride: usize,
    planes: Vec<AlignedVec>,
}

impl RowBand {
    /// Number of channels.
    pub fn channels(&self) -> usize {
        self.planes.len()
    }

    /// Row `y` of channel `channel`, `width` samples (padding excluded).
    #[inline]
    pub fn row(&self, channel: usize, y: usize) -> &[f32] {
        let start = y * self.stride;
        &self.planes[channel][start..start + self.width]
    }
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
