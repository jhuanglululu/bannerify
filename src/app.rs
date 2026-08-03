//! Application entry point: the pipeline as one flat `par_iter` over banner
//! rows.
//!
//! One work item is one banner row (`context/designs/pipeline.md`): the closure
//! borrows the source planes, the shared weights and the layout, resamples its
//! own row band locally through the windowed plan, and writes its strip of the
//! wall canvas — no shared mutable state, no locking, no cross-item handoff.
//!
//! Phase 1 stops after the resample, so the tool's normal output is that
//! intermediate: the resized wall image, written as a PNG to `<OUTPUT>`. When
//! the solver lands it slots into [`render_row`] between the resample and the
//! strip write, and this intermediate-output code is deleted rather than kept
//! behind a flag.

use std::ops::Range;
use std::time::{Duration, Instant};

use clap::Parser;
use colored::Colorize;
use rayon::prelude::*;

use crate::cli::Args;
use crate::cli::config::Config;
use crate::geometry::banner_row_span;
use crate::layout::Layout;
use crate::logger::{error_out, info};
use crate::memory;
use crate::resample::{Plan, PlanarU8, Window};

/// Channels the pipeline works in (RGB; the decoder normalises to this).
const CHANNELS: usize = 3;

/// Parse arguments, validate, and run the pipeline.
pub fn run_cli() {
    let args = Args::parse();
    let config = Config::from(args);

    if let Some(workers) = config.workers {
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build_global()
            .unwrap_or_else(|e| {
                error_out!(
                    "could not start {} workers: {}",
                    workers,
                    e.to_string().red()
                );
            });
    }
    info!("using {} workers", rayon::current_num_threads());

    run(&config);
}

// ---------------------------------------------------------------- work items

/// Where one row item's resampled pixels land in the wall canvas.
#[derive(Clone, Copy, Debug)]
struct Rect {
    /// Left edge, in canvas pixels.
    x: usize,
    /// Top edge, in canvas pixels.
    y: usize,
    /// Width in pixels.
    width: usize,
    /// Height in pixels.
    height: usize,
}

/// One unit of parallel work: one banner row of the wall.
///
/// Carries only rects and indices — never a pixel payload. Everything else the
/// closure needs is a shared borrow in [`RowContext`].
#[derive(Clone, Debug)]
struct RowItem {
    /// Banner row index.
    row: usize,
    /// Source region this row reads, in fractional source pixels. Adjacent rows
    /// overlap here by the vertical kernel support — each row band is computed
    /// independently from the source, so the overlap is re-read, never shared.
    src_rect: Window,
    /// Where this row's resampled pixels land in the canvas.
    dest_rect: Rect,
    /// Rows of the resample target this item covers.
    target_rows: Range<usize>,
    /// Canvas rows this item owns exclusively (its strip). Equal to the
    /// `dest_rect` rows except under `--fill`, where the strip also holds the
    /// padding above and below the image.
    strip_rows: Range<usize>,
}

/// Everything a row item borrows.
///
/// Phase 2 adds the shared solver tables here (patterns, dye tables, the
/// per-cell `n_layers` grid from the variance pre-pass, the config); the item
/// closure signature does not change.
struct RowContext<'a> {
    source: &'a PlanarU8,
    plan: &'a Plan,
    layout: &'a Layout,
}

/// Split the wall into one item per banner row.
fn row_items(layout: &Layout, plan: &Plan) -> Vec<RowItem> {
    let (ox, oy) = layout.origin;
    (0..layout.rows)
        .map(|row| {
            let strip_rows = banner_row_span(row, layout.rows);
            // The part of this strip the resampled image covers: under --fill
            // the image is smaller than the wall, so a strip can hold fewer
            // target rows than it has canvas rows (or none at all).
            let lo = strip_rows.start.clamp(oy, oy + layout.target_height);
            let hi = strip_rows.end.clamp(oy, oy + layout.target_height);
            let target_rows = (lo - oy)..(hi - oy);
            let (y0, y1) = plan.src_rows(&target_rows);
            RowItem {
                row,
                src_rect: Window {
                    x0: layout.window.x0,
                    y0,
                    x1: layout.window.x1,
                    y1,
                },
                dest_rect: Rect {
                    x: ox,
                    y: lo,
                    width: layout.target_width,
                    height: target_rows.len(),
                },
                target_rows,
                strip_rows,
            }
        })
        .collect()
}

/// Produce one banner row of the wall.
///
/// Shared borrows arrive through `ctx`; `strip` is this item's exclusive slice
/// of the wall canvas — exactly `item.strip_rows` full canvas rows of
/// interleaved RGB `u8`, so no two items ever touch the same byte.
///
/// Phase 2 slots in between the resample and the strip write: solve each cell of
/// the row against `band` (cells borrow their patches straight out of it, no
/// copy), match the background block on the uncovered pixels, render the cell
/// into `strip`, and return the row's per-cell results instead of `()`.
fn render_row(ctx: &RowContext<'_>, item: &RowItem, strip: &mut [u8]) {
    let wall_width = ctx.layout.wall_width;
    debug_assert_eq!(
        item.strip_rows,
        banner_row_span(item.row, ctx.layout.rows),
        "an item's strip is exactly its banner row's span"
    );

    // Padding first: only reachable under --fill, and only for the parts of
    // this strip the image does not cover.
    if let Some(color) = ctx.layout.pad {
        fill_padding(strip, wall_width, item, color);
    }

    if item.target_rows.is_empty() {
        return;
    }

    // The row's own band, resampled locally out of the source region
    // `item.src_rect` and dropped at the end of this call.
    let band = ctx
        .plan
        .band((item.src_rect.y0, item.src_rect.y1), item.dest_rect.height)
        .resample(ctx.source);
    debug_assert_eq!(band.channels(), CHANNELS);
    debug_assert_eq!(band.width, item.dest_rect.width);

    let y_offset = item.dest_rect.y - item.strip_rows.start;
    for y in 0..band.height {
        let (r, g, b) = (band.row(0, y), band.row(1, y), band.row(2, y));
        let start = ((y_offset + y) * wall_width + item.dest_rect.x) * CHANNELS;
        let row = &mut strip[start..start + band.width * CHANNELS];
        for (x, px) in row.chunks_exact_mut(CHANNELS).enumerate() {
            px[0] = to_u8(r[x]);
            px[1] = to_u8(g[x]);
            px[2] = to_u8(b[x]);
        }
    }
}

/// Paint the parts of `strip` the resampled image does not cover.
fn fill_padding(strip: &mut [u8], wall_width: usize, item: &RowItem, color: [u8; 3]) {
    let top = item.dest_rect.y - item.strip_rows.start;
    let covered = top..top + item.dest_rect.height;
    for (y, row) in strip.chunks_exact_mut(wall_width * CHANNELS).enumerate() {
        let inner = if covered.contains(&y) {
            item.dest_rect.x..item.dest_rect.x + item.dest_rect.width
        } else {
            0..0
        };
        for (x, px) in row.chunks_exact_mut(CHANNELS).enumerate() {
            if !inner.contains(&x) {
                px.copy_from_slice(&color);
            }
        }
    }
}

/// Resampled sample → display byte. Lanczos ringing is clipped here, at the
/// encode edge, not inside the pipeline: the `f32` band keeps the overshoot.
#[inline]
fn to_u8(v: f32) -> u8 {
    v.clamp(0.0, 255.0).round() as u8
}

// -------------------------------------------------------------------- driver

/// Decode, lay out, run the row items, write this phase's output.
fn run(config: &Config) {
    // ---- decode (and the one interleaved -> planar conversion) -------------
    let t = Instant::now();
    let decoded = image::open(&config.input).unwrap_or_else(|e| {
        error_out!(
            "could not read '{}': {}",
            config.input.display().to_string().yellow(),
            e.to_string().red()
        );
    });
    let rgb = decoded.to_rgb8();
    let (src_w, src_h) = (rgb.width() as usize, rgb.height() as usize);
    let source = PlanarU8::from_interleaved(rgb.as_raw(), src_w, src_h, CHANNELS);
    drop(rgb);
    let t_decode = t.elapsed();

    // ---- layout + shared weights ------------------------------------------
    let t = Instant::now();
    let layout = Layout::compute(
        src_w as u32,
        src_h as u32,
        config.dimension,
        config.resizing_method,
    );
    let plan = Plan::with_window(
        src_w,
        src_h,
        layout.window,
        layout.target_width,
        layout.target_height,
    );
    let items = row_items(&layout, &plan);
    let t_plan = t.elapsed();

    info!(
        "grid: {}x{} blocks ({} banners)",
        layout.columns,
        layout.rows + 1,
        layout.columns * layout.rows
    );
    info!(
        "wall: {}x{} px, resampling {}x{} -> {}x{} in {} banner-row items",
        layout.wall_width,
        layout.wall_height,
        src_w,
        src_h,
        layout.target_width,
        layout.target_height,
        items.len()
    );

    // ---- the pipeline: one flat par_iter over banner rows ------------------
    // The canvas has to exist for the encode either way; it is split into
    // per-item strips up front, so items write straight into it without locking
    // and without a wall-sized f32 buffer anywhere.
    let t = Instant::now();
    let mut canvas = vec![0u8; layout.wall_width * layout.wall_height * CHANNELS];
    let strips = split_strips(&mut canvas, &items, layout.wall_width);
    let ctx = RowContext {
        source: &source,
        plan: &plan,
        layout: &layout,
    };
    strips
        .into_par_iter()
        .zip(items.par_iter())
        .for_each(|(strip, item)| render_row(&ctx, item, strip));
    let t_pipeline = t.elapsed();

    // ---- write this step's output ------------------------------------------
    let t = Instant::now();
    let out =
        image::RgbImage::from_raw(layout.wall_width as u32, layout.wall_height as u32, canvas)
            .unwrap_or_else(|| error_out!("internal error: canvas has the wrong size"));
    out.save(&config.output).unwrap_or_else(|e| {
        error_out!(
            "could not write '{}': {}",
            config.output.display().to_string().yellow(),
            e.to_string().red()
        );
    });
    let t_encode = t.elapsed();

    info!(
        "wrote '{}': the resized banner wall — this phase's intermediate output \
         (banner conversion lands in the next phase)",
        config.output.display().to_string().yellow()
    );

    if config.debug {
        let mpix = (layout.target_width * layout.target_height) as f64 / 1e6;
        debug_line("decode", t_decode, None);
        debug_line("layout+weights", t_plan, None);
        debug_line(
            "pipeline",
            t_pipeline,
            Some(format!(
                "{:.1} MPix/s ({:.1} MPix/s summed over {CHANNELS} channels)",
                mpix / t_pipeline.as_secs_f64(),
                CHANNELS as f64 * mpix / t_pipeline.as_secs_f64()
            )),
        );
        debug_line("encode", t_encode, None);
        debug_line("total", t_decode + t_plan + t_pipeline + t_encode, None);
        println!(
            "{}: {:<16} peak {}, still live {}",
            "debug".blue().bold(),
            "memory",
            memory::format_bytes(memory::peak_bytes()),
            memory::format_bytes(memory::live_bytes())
        );
    }
}

/// Hand each item its own exclusive slice of the canvas.
fn split_strips<'a>(
    canvas: &'a mut [u8],
    items: &[RowItem],
    wall_width: usize,
) -> Vec<&'a mut [u8]> {
    let mut rest = canvas;
    let mut strips = Vec::with_capacity(items.len());
    for item in items {
        let (strip, tail) = rest.split_at_mut(item.strip_rows.len() * wall_width * CHANNELS);
        strips.push(strip);
        rest = tail;
    }
    debug_assert!(rest.is_empty(), "row items must tile the canvas exactly");
    strips
}

/// One `debug: <stage> <time>` line.
fn debug_line(stage: &str, d: Duration, note: Option<String>) {
    let note = note.map(|n| format!("   {n}")).unwrap_or_default();
    println!(
        "{}: {:<16} {:>9.3} s{}",
        "debug".blue().bold(),
        stage,
        d.as_secs_f64(),
        note
    );
}
