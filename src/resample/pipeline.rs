//! Streamed lanczos-3, sliced by output **column** band.
//!
//! The unit of work is a band of output columns ([`ColumnPlan`]), resampled by
//! its owner into a closure-local [`ColBand`] — that is the pipeline's parallel
//! work item (one block column; see `context/designs/pipeline.md`), so this
//! module hands out no parallelism of its own and orchestrates nothing.
//!
//! Inside a band: a ring of `ksize_y` horizontally-resampled rows is filled by
//! the H pass (ring indexing — rows are never shifted or copied), and each
//! output row is a `ksize_y`-tap weighted sum of the ring straight into the band
//! buffer. Nothing wall-sized is ever materialised here; a band's working set is
//! `(ksize_y + dst_height) * padded_band_width * 4` bytes per channel, plus one
//! source-row scratch.
//!
//! ## Which table is shared
//!
//! A weight table costs one lanczos evaluation per output coordinate per tap,
//! and every band on an axis wants the same coefficients — so the axis that is
//! **not** cut belongs in the shared [`Plan`]. With column items that is the
//! vertical axis: [`Plan`] holds the full-height [`VWeights`], and each column
//! item builds the small [`HWeights`] for its own handful of output columns
//! inside the closure (parallel, and proportional to the band, not the wall).
//! This is the mirror image of the banner-row split this module used to have.
//!
//! Sub-window weights are exact, not approximate: the coefficient for output
//! `j` of a band starting at output `c` is centred at
//! `x0 + (c + j + 0.5) * scale`, which is what the whole-image table would put
//! at output `c + j`. Banding is invisible in the result on either axis.

use std::ops::Range;

use super::weights::{HWeights, VWeights, round_up};
use crate::simd::{AlignedVec, F32s, LANES};
use crate::zip;

/// What the H pass can read from: an image the band asks for one span of one
/// row of one channel at a time.
///
/// Two implementations, and the difference between them is *where the u8 → f32
/// widening reads from*, never an extra buffer: [`PlanarU8`] (the decoded source
/// image) reads a contiguous run, [`InterleavedU8`] (the rendered wall canvas)
/// reads the same run strided by the channel count. That is what lets the
/// preview downscale run straight off the canvas — the canvas is never copied
/// into planes, because the only thing that would want the planar copy is this
/// load, and this load can stride.
pub trait Source: Sync {
    /// Image width in pixels.
    fn width(&self) -> usize;
    /// Image height in pixels.
    fn height(&self) -> usize;
    /// Number of channels.
    fn channels(&self) -> usize;
    /// Widen row `y` of `channel`, columns `span`, into `dst` at the same
    /// indices. Everything outside the span must be left untouched.
    fn load_span(&self, channel: usize, y: usize, span: (usize, usize), dst: &mut AlignedVec);
}

/// A decoded image as separate `u8` planes, row-major, `width * height` each.
///
/// Planar is the layout the crate uses for the decoded source; the single
/// interleaved → planar conversion happens at the decode edge.
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

impl Source for PlanarU8 {
    fn width(&self) -> usize {
        self.width
    }
    fn height(&self) -> usize {
        self.height
    }
    fn channels(&self) -> usize {
        self.planes.len()
    }
    fn load_span(&self, channel: usize, y: usize, span: (usize, usize), dst: &mut AlignedVec) {
        let (x_lo, x_hi) = span;
        let plane = &self.planes[channel];
        let row = &plane[y * self.width + x_lo..y * self.width + x_hi];
        for (d, &s) in dst[x_lo..x_hi].iter_mut().zip(row) {
            *d = f32::from(s);
        }
    }
}

/// An interleaved `u8` image borrowed as a resample source — the rendered wall
/// canvas, which the preview downscale reads *in place*.
pub struct InterleavedU8<'a> {
    /// Row-major, `width * height * channels` samples.
    pub data: &'a [u8],
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    /// Samples per pixel.
    pub channels: usize,
}

impl Source for InterleavedU8<'_> {
    fn width(&self) -> usize {
        self.width
    }
    fn height(&self) -> usize {
        self.height
    }
    fn channels(&self) -> usize {
        self.channels
    }
    fn load_span(&self, channel: usize, y: usize, span: (usize, usize), dst: &mut AlignedVec) {
        let (x_lo, x_hi) = span;
        let c = self.channels;
        let base = (y * self.width + x_lo) * c;
        let row = &self.data[base..base + (x_hi - x_lo) * c];
        for (d, px) in dst[x_lo..x_hi].iter_mut().zip(row.chunks_exact(c)) {
            *d = f32::from(px[channel]);
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

/// The shared, immutable half of a resize: source geometry, target size, and the
/// vertical weight table. Build once, borrow from every work item.
pub struct Plan {
    src_width: usize,
    src_height: usize,
    window: Window,
    dst_width: usize,
    dst_height: usize,
    v: VWeights,
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
        let v = VWeights::new(src_height, window.y0, window.y1, dst_height);
        Self {
            src_width,
            src_height,
            window,
            dst_width,
            dst_height,
            v,
        }
    }

    /// Output width in pixels.
    pub fn dst_width(&self) -> usize {
        self.dst_width
    }

    /// Output height in pixels.
    pub fn dst_height(&self) -> usize {
        self.dst_height
    }

    /// The source columns band `cols` reads from, in fractional source pixels.
    ///
    /// Adjacent bands overlap here by the horizontal kernel support (~`3 *
    /// scale` columns each side): every band is computed independently from the
    /// source, so the overlap is re-read (and re-H-passed), never shared. That
    /// duplicate work is the accepted cost of the column split — it replaces the
    /// vertical tap overlap the banner-row split used to pay.
    pub fn src_cols(&self, cols: &Range<usize>) -> (f64, f64) {
        let scale = (self.window.x1 - self.window.x0) / self.dst_width as f64;
        (
            self.window.x0 + cols.start as f64 * scale,
            self.window.x0 + cols.end as f64 * scale,
        )
    }

    /// The plan for one band of output columns, full output height.
    pub fn columns(&self, cols: Range<usize>) -> ColumnPlan<'_> {
        assert!(cols.end <= self.dst_width, "band outside the output");
        self.band(self.src_cols(&cols), cols.len())
    }

    /// The plan for `width` output columns covering the source interval
    /// `src_cols` — the same thing [`Plan::columns`] builds, addressed by the
    /// work item's own source rect instead of by output column indices.
    pub fn band(&self, src_cols: (f64, f64), width: usize) -> ColumnPlan<'_> {
        assert!(width > 0, "empty band");
        let h = HWeights::new(self.src_width, src_cols.0, src_cols.1, width);
        let src_stride = h.src_padded_len(self.src_width);
        ColumnPlan {
            plan: self,
            h,
            width,
            src_stride,
            out_stride: round_up(width, 16),
        }
    }
}

/// The plan for one band of output columns: the shared [`Plan`] plus the band's
/// own horizontal weights.
pub struct ColumnPlan<'a> {
    plan: &'a Plan,
    h: HWeights,
    width: usize,
    /// Padded length of the H-pass source row scratch buffer.
    src_stride: usize,
    /// Row pitch of the band buffer, in floats (multiple of 16).
    out_stride: usize,
}

impl ColumnPlan<'_> {
    /// Output columns in this band.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Resample this band out of `src`.
    ///
    /// The returned [`ColBand`] is the caller's local: the solver borrows its
    /// cell patches straight out of it, and it dies with the work item. Note for
    /// later: these per-item buffers (band + ring + scratch) are a natural fit
    /// for a per-worker arena handed out without zeroing, which would remove the
    /// per-item allocation traffic entirely.
    pub fn resample<S: Source + ?Sized>(&self, src: &S) -> ColBand {
        let plan = self.plan;
        assert_eq!(src.width(), plan.src_width, "source width mismatch");
        assert_eq!(src.height(), plan.src_height, "source height mismatch");

        let height = plan.dst_height;
        let stride = self.out_stride;
        let kv = plan.v.ksize;

        // The only source columns this band's H pass ever touches. Tap windows
        // start at `start_lane * LANES` and are `win` floats long, and
        // `start_lane` is non-decreasing across outputs, so the whole band reads
        // exactly this span — a sliver of each source row, which is the point of
        // loading a span rather than the row.
        let x_lo = self.h.start_lane[0] * LANES;
        let x_hi = (self.h.start_lane[self.width - 1] * LANES + self.h.win).min(plan.src_width);

        // Source row scratch: `zeroed`, and only `[x_lo, x_hi)` is ever
        // rewritten, so both the alignment padding inside a tap window and the
        // tail past `src_width` read as a finite 0.0 on every row.
        let mut srow = AlignedVec::zeroed(self.src_stride);

        // Ring of `kv` H-passed rows, reused in place across the whole band.
        let mut ring: Vec<AlignedVec> = (0..kv)
            .map(|_| {
                // SAFETY: every element of a slot is written by `h_pass` before
                // the slot is read — a slot is only read for a source row that
                // the loop below has already loaded and H-passed.
                unsafe { AlignedVec::new_uninit(stride) }
            })
            .collect();

        let planes = (0..src.channels())
            .map(|channel| {
                // SAFETY: `v_pass` writes every element of the band buffer —
                // one full `stride`-float row per output row — before it is read.
                let mut buf = unsafe { AlignedVec::new_uninit(height * stride) };
                let mut next_src = plan.v.starts[0];
                for local in 0..height {
                    let (start, taps) = plan.v.row(local);
                    for y in next_src.max(start)..start + kv {
                        let clamped = y.min(plan.src_height - 1);
                        src.load_span(channel, clamped, (x_lo, x_hi), &mut srow);
                        h_pass(&self.h, &srow, &mut ring[y % kv]);
                    }
                    next_src = next_src.max(start + kv);
                    v_pass(&ring, kv, start, taps, &mut buf, local, stride);
                }
                buf
            })
            .collect();

        ColBand {
            width: self.width,
            height,
            stride,
            planes,
        }
    }
}

/// One band of resampled output — a full-height slice of output columns —
/// planar `f32`, one plane per channel.
///
/// Rows are padded to a multiple of 16 floats ([`ColBand::stride`]); only the
/// first [`ColBand::width`] samples of each row are image data.
pub struct ColBand {
    /// Valid samples per row: this band's output columns.
    pub width: usize,
    /// Rows in this band — the full output height.
    pub height: usize,
    /// Row pitch in floats.
    pub stride: usize,
    planes: Vec<AlignedVec>,
}

impl ColBand {
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

    /// Every channel plane whole, as a mutable lane view.
    ///
    /// The one mutation a band exposes, and it exists for exactly one caller:
    /// the pipeline converts the band in place from sRGB to OKLab immediately
    /// after resampling ([`crate::app`]), so the banner solver and the block
    /// matcher both read the perceptual space out of the buffer the resampler
    /// filled — one pass over the band instead of one per cell or per candidate
    /// (`context/plans/4-oklab-native.md`).
    ///
    /// The view covers the row padding as well as the `width` valid samples.
    /// That is deliberate: the padding is zeroed, whole lanes are cheaper than a
    /// remainder path, and nothing ever reads it — [`ColBand::row`] stops at
    /// `width`. A pointwise transform therefore does not need to know where the
    /// rows are.
    pub fn lanes_mut(&mut self) -> impl Iterator<Item = &mut [F32s]> {
        self.planes.iter_mut().map(AlignedVec::lanes_mut)
    }
}

/// Horizontal pass: one padded source row in, one band-wide output row out.
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
