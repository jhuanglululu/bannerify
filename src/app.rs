//! Application entry point: the pipeline as one flat `par_iter` over banner
//! rows.
//!
//! One work item is one banner row (`context/designs/pipeline.md`): the closure
//! borrows the source planes, the shared weights, the pattern tables, the layer
//! grid and the layout, resamples its own row band locally through the windowed
//! plan, solves its cells against that band, and writes its strip of the wall
//! canvas — no shared mutable state, no locking, no cross-item handoff.
//!
//! Phase 2 stops after the greedy fill (stage 2a, `context/plans/2-solver.md`),
//! so the tool's normal output is that intermediate: the composed banner wall as
//! a PNG at `<OUTPUT>`, plus the per-cell decisions as JSONL beside it. The
//! resized-image intermediate phase 1 wrote is gone, as planned — earlier
//! intermediates are deleted when a later stage lands, never kept behind a flag.

use std::ops::Range;
use std::path::Path;
use std::time::{Duration, Instant};

use clap::Parser;
use colored::Colorize;
use rayon::prelude::*;

use crate::cli::Args;
use crate::cli::config::Config;
use crate::geometry::{NTOP_HW, TOP_HW, banner_row_span};
use crate::layout::Layout;
use crate::logger::{error_out, info};
use crate::memory;
use crate::pattern::{self, Patterns};
use crate::resample::{Plan, PlanarU8, Window};
use crate::simd::Chunk;
use crate::solver::cell::{BandView, paint_background, paint_cell};
use crate::solver::variance::{LayerGrid, layer_grid};
use crate::solver::{Solution, Workspace, write_jsonl};

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
struct RowContext<'a> {
    source: &'a PlanarU8,
    plan: &'a Plan,
    layout: &'a Layout,
    patterns: &'a Patterns,
    /// Per-cell layer budget from the variance pre-pass.
    layers: &'a LayerGrid,
    /// Largest budget in the grid — the size every workspace is built for.
    max_layers: usize,
}

/// What one row item produces.
struct RowOutcome {
    /// One entry per cell of the row, left to right.
    cells: Vec<Solution>,
    /// CPU time this item spent resampling.
    resample: Duration,
    /// CPU time this item spent solving and painting cells.
    solve: Duration,
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
/// interleaved RGB `u8`, so no two items ever touch the same byte. Which rows a
/// strip paints, and why that needs no coordination between items despite
/// banners overlapping, is [`crate::solver::cell`]'s module docs.
fn render_row(ctx: &RowContext<'_>, item: &RowItem, strip: &mut [u8]) -> RowOutcome {
    let wall_width = ctx.layout.wall_width;
    debug_assert_eq!(
        item.strip_rows,
        banner_row_span(item.row, ctx.layout.rows),
        "an item's strip is exactly its banner row's span"
    );

    paint_background(strip, wall_width, item.strip_rows.start);

    // The row's own band, resampled locally out of the source region
    // `item.src_rect` and dropped at the end of this call. It is absent only
    // when `--fill` leaves this row entirely padding.
    let t = Instant::now();
    let band = (!item.target_rows.is_empty()).then(|| {
        ctx.plan
            .band((item.src_rect.y0, item.src_rect.y1), item.dest_rect.height)
            .resample(ctx.source)
    });
    let resample = t.elapsed();
    debug_assert!(band.as_ref().is_none_or(|b| b.channels() == CHANNELS));
    debug_assert!(
        band.as_ref()
            .is_none_or(|b| b.width == item.dest_rect.width)
    );

    let view = BandView::new(
        band.as_ref(),
        item.dest_rect.x,
        item.dest_rect.y,
        ctx.layout.pad.unwrap_or([0; CHANNELS]),
    );

    // Banner row 0 has nothing hanging in front of it, so it solves the full
    // 20x40 patch; every other row solves only its visible 24 rows. Same code,
    // two monomorphisations.
    let t = Instant::now();
    let cells = if item.row == 0 {
        solve_row::<TOP_HW>(
            ctx,
            item,
            &view,
            strip,
            &ctx.patterns.top,
            &ctx.patterns.top_alpha2,
        )
    } else {
        solve_row::<NTOP_HW>(
            ctx,
            item,
            &view,
            strip,
            &ctx.patterns.lower,
            &ctx.patterns.lower_alpha2,
        )
    };
    let solve = t.elapsed();

    RowOutcome {
        cells,
        resample,
        solve,
    }
}

/// Solve and paint every cell of one banner row.
///
/// The workspace is built once per row and reused across its cells, so the only
/// per-cell allocation is the solution's layer list.
fn solve_row<const HW: usize>(
    ctx: &RowContext<'_>,
    item: &RowItem,
    view: &BandView<'_>,
    strip: &mut [u8],
    alphas: &[Chunk<HW>],
    alpha2: &[f32],
) -> Vec<Solution> {
    let mut workspace = Workspace::<HW>::new(ctx.max_layers);
    (0..ctx.layout.columns)
        .map(|col| {
            view.gather::<HW>(item.row, col, workspace.target_mut());
            let solution = workspace.solve(ctx.layers.get(item.row, col), alphas, alpha2);
            paint_cell::<HW>(
                strip,
                ctx.layout.wall_width,
                item.strip_rows.start,
                item.row,
                col,
                workspace.composite(),
            );
            solution
        })
        .collect()
}

// -------------------------------------------------------------------- driver

/// Decode, lay out, run the row items, write this phase's outputs.
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

    // ---- layout + shared weights + pattern tables --------------------------
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
    let patterns = pattern::load(&config.exclude_patterns);
    let t_plan = t.elapsed();

    info!(
        "grid: {}x{} blocks ({} banners), {} patterns x {} dyes",
        layout.columns,
        layout.rows + 1,
        layout.columns * layout.rows,
        patterns.len(),
        crate::color::NUM_COLORS
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

    // ---- variance pre-pass -------------------------------------------------
    // Runs on the source image, before the row items, because the layer budget
    // is a global min/max normalisation across every cell of the wall.
    let t = Instant::now();
    let layers = layer_grid(&source, &layout, config.n_layers);
    let max_layers = layers.max();
    let t_variance = t.elapsed();

    info!(
        "layers {}-{}: {}",
        config.n_layers.0,
        config.n_layers.1,
        layers
            .histogram(config.n_layers)
            .iter()
            .enumerate()
            .map(|(i, n)| format!("{}x{}", n, i + config.n_layers.0))
            .collect::<Vec<_>>()
            .join(" ")
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
        patterns: &patterns,
        layers: &layers,
        max_layers,
    };
    let outcomes: Vec<RowOutcome> = strips
        .into_par_iter()
        .zip(items.par_iter())
        .map(|(strip, item)| render_row(&ctx, item, strip))
        .collect();
    let t_pipeline = t.elapsed();

    // Reduce the per-item stats now, so the outcomes can be consumed for the
    // JSONL without copying every cell's layer list.
    let (cpu_resample, cpu_solve) = outcomes
        .iter()
        .fold((Duration::ZERO, Duration::ZERO), |(r, s), o| {
            (r + o.resample, s + o.solve)
        });
    let total_err: f64 = outcomes
        .iter()
        .flat_map(|o| o.cells.iter())
        .map(|c| f64::from(c.error))
        .sum();
    let cells: Vec<Vec<Solution>> = outcomes.into_iter().map(|o| o.cells).collect();

    // ---- write this step's outputs -----------------------------------------
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

    let jsonl = jsonl_path(&config.output);
    write_jsonl(&jsonl, &patterns, &cells);
    let t_encode = t.elapsed();

    info!(
        "wrote '{}': the composed banner wall, and '{}': the per-cell solution \
         — this stage's intermediate outputs (refinement lands in the next stage)",
        config.output.display().to_string().yellow(),
        jsonl.display().to_string().yellow()
    );

    if config.debug {
        let n_cells = layout.rows * layout.columns;
        let mean_err = total_err / n_cells as f64;

        debug_line("decode", t_decode, None);
        debug_line("layout+patterns", t_plan, None);
        debug_line("variance", t_variance, None);
        debug_line(
            "pipeline",
            t_pipeline,
            Some(format!(
                "{:.0} cells/s",
                n_cells as f64 / t_pipeline.as_secs_f64()
            )),
        );
        debug_line("  resample (cpu)", cpu_resample, None);
        debug_line("  solve (cpu)", cpu_solve, None);
        debug_line("encode", t_encode, None);
        debug_line(
            "total",
            t_decode + t_plan + t_variance + t_pipeline + t_encode,
            None,
        );
        println!(
            "{}: {:<16} {:.1} weighted SSE per cell",
            "debug".blue().bold(),
            "error",
            mean_err
        );
        println!(
            "{}: {:<16} peak {}, still live {}",
            "debug".blue().bold(),
            "memory",
            memory::format_bytes(memory::peak_bytes()),
            memory::format_bytes(memory::live_bytes())
        );
    }
}

/// `<OUTPUT>` with its extension replaced by `.jsonl`.
fn jsonl_path(output: &Path) -> std::path::PathBuf {
    output.with_extension("jsonl")
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
