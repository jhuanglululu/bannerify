//! Application entry point: the pipeline as one flat `par_iter` over block
//! columns.
//!
//! One work item is one block column (`context/designs/pipeline.md`): a
//! full-height, 24-pixel-wide slice of the wall. The closure borrows the source
//! planes, the shared vertical weights, the pattern tables, the layer grid and
//! the layout; it builds its own horizontal weights, resamples its own column
//! band locally, solves its cells top to bottom, and paints them into its own
//! strip — no shared mutable state, no locking, no cross-item handoff.
//!
//! Columns, never rows: banners bridge every horizontal block seam, so no
//! horizontal cut lets an item own complete block rows, while a vertical cut on
//! a block-column boundary cuts nothing (banners never cross columns). The
//! banner-over-banner overlap — and, in phase 3, block-behind-banner
//! compositing — is then entirely internal to an item.
//!
//! Phase 2 stops after the greedy fill (stage 2a, `context/plans/2-solver.md`),
//! so the tool's normal output is that intermediate: the composed banner wall as
//! a PNG at `<OUTPUT>`, plus the per-cell decisions as JSONL beside it. The
//! resized-image intermediate phase 1 wrote is gone, as planned — earlier
//! intermediates are deleted when a later stage lands, never kept behind a flag.

use std::ops::Range;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use clap::Parser;
use colored::Colorize;
use rayon::prelude::*;

use crate::cli::Args;
use crate::cli::config::Config;
use crate::geometry::BLOCK_SIDE;
use crate::layout::Layout;
use crate::logger::{error_out, info};
use crate::memory;
use crate::pattern;
use crate::resample::{Plan, PlanarU8, Window};
use crate::solver::cell::{BandView, STRIP_PITCH, paint_background, paint_cell};
use crate::solver::variance::{LayerGrid, layer_grid};
use crate::solver::{Rng, Solution, SolveCfg, Stages, Workspace, write_jsonl};

/// Channels the pipeline works in (RGB; the decoder normalises to this).
const CHANNELS: usize = 3;

// ---------------------------------------------------------------- work items

/// Where one column item's resampled pixels land in the wall canvas.
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

/// One unit of parallel work: one block column of the wall.
///
/// Carries only rects and indices — never a pixel payload. Everything else the
/// closure needs is a shared borrow in [`ColContext`].
#[derive(Clone, Debug)]
struct ColItem {
    /// Block (and banner) column index.
    col: usize,
    /// Source region this column reads, in fractional source pixels. Adjacent
    /// columns overlap here by the horizontal kernel support — each band is
    /// computed independently from the source, so the overlap is re-read, never
    /// shared.
    src_rect: Window,
    /// Where this column's resampled pixels land in the canvas.
    dest_rect: Rect,
    /// Columns of the resample target this item covers.
    target_cols: Range<usize>,
    /// Canvas columns this item owns exclusively (its strip). Always exactly one
    /// block wide: block columns tile the wall with nothing left over.
    strip_cols: Range<usize>,
}

/// Everything a column item borrows.
struct ColContext<'a> {
    source: &'a PlanarU8,
    plan: &'a Plan,
    layout: &'a Layout,
    /// Per-cell layer budget from the variance pre-pass.
    layers: &'a LayerGrid,
    /// Largest budget in the grid — the size every workspace is built for.
    max_layers: usize,
    /// The solver stages' shared configuration.
    solve: SolveCfg<'a>,
    /// Perturbation RNG seed; each item derives its stream from it and its
    /// column index.
    seed: u64,
}

/// What one column item produces.
struct ColOutcome {
    /// One entry per cell of the column, top to bottom.
    cells: Vec<Solution>,
    /// The column's rendered preview strip: `wall_height` rows of
    /// [`STRIP_PITCH`] bytes, interleaved into the canvas after the `par_iter`.
    strip: Vec<u8>,
    /// CPU time this item spent resampling.
    resample: Duration,
    /// CPU time this item spent solving and painting cells.
    solve: Duration,
    /// ...split by solver stage.
    stages: Stages,
}

/// Split the wall into one item per block column.
fn col_items(layout: &Layout, plan: &Plan) -> Vec<ColItem> {
    let (ox, oy) = layout.origin;
    (0..layout.columns)
        .map(|col| {
            let strip_cols = col * BLOCK_SIDE..(col + 1) * BLOCK_SIDE;
            // The part of this strip the resampled image covers: under --fill
            // the image is narrower than the wall, so a strip can hold fewer
            // target columns than it has canvas columns (or none at all).
            let lo = strip_cols.start.clamp(ox, ox + layout.target_width);
            let hi = strip_cols.end.clamp(ox, ox + layout.target_width);
            let target_cols = (lo - ox)..(hi - ox);
            let (x0, x1) = plan.src_cols(&target_cols);
            ColItem {
                col,
                src_rect: Window {
                    x0,
                    y0: layout.window.y0,
                    x1,
                    y1: layout.window.y1,
                },
                dest_rect: Rect {
                    x: lo,
                    y: oy,
                    width: target_cols.len(),
                    height: layout.target_height,
                },
                target_cols,
                strip_cols,
            }
        })
        .collect()
}

/// Produce one block column of the wall.
///
/// Shared borrows arrive through `ctx`; the strip the item paints is its own
/// local buffer (`wall_height × 24 × 3` bytes, part of the item's bounded
/// memory), interleaved into the canvas by the driver afterwards. Why local
/// rather than a slice of the canvas: the canvas is row-major, so a column of it
/// is not a contiguous `&mut [u8]` and `split_at_mut` cannot hand one out.
/// Painting locally and interleaving once keeps the whole pipeline in safe code,
/// and the copy lands in a buffer that has to exist for the PNG encode anyway.
fn render_column(ctx: &ColContext<'_>, item: &ColItem) -> ColOutcome {
    debug_assert_eq!(
        item.strip_cols.len(),
        BLOCK_SIDE,
        "one item, one block column"
    );
    let mut strip = vec![0u8; ctx.layout.wall_height * STRIP_PITCH];
    paint_background(&mut strip);

    // The column's own band, resampled locally out of the source region
    // `item.src_rect` and dropped at the end of this call. It is absent only
    // when `--fill` leaves this column entirely padding.
    let t = Instant::now();
    let band = (!item.target_cols.is_empty()).then(|| {
        ctx.plan
            .band((item.src_rect.x0, item.src_rect.x1), item.dest_rect.width)
            .resample(ctx.source)
    });
    let resample = t.elapsed();
    debug_assert!(band.as_ref().is_none_or(|b| b.channels() == CHANNELS));
    debug_assert!(
        band.as_ref()
            .is_none_or(|b| b.height == item.dest_rect.height)
    );

    let view = BandView::new(
        band.as_ref(),
        item.dest_rect.x,
        item.dest_rect.y,
        ctx.layout.pad.unwrap_or([0; CHANNELS]),
    );

    // Cells top to bottom, through one workspace. Banner row 0 has nothing
    // hanging in front of it, so it solves the full 20x40 patch; every other row
    // solves only its visible 24 rows — the lane-aligned tail of the same
    // buffers (`crate::solver::workspace`), which is why one allocation serves
    // both and nothing is re-zeroed between cells.
    let t = Instant::now();
    let mut cells = Vec::with_capacity(ctx.layout.rows);
    let mut workspace = Workspace::new(ctx.max_layers, ctx.solve.refinement.refinement_candidate);
    // One PRNG stream per column item, advanced down the column's cells: the
    // banner row never enters the seeding, so the draws a cell sees depend only
    // on the seed, the column and how many cells precede it.
    let mut rng = Rng::new(ctx.seed, item.col as u64);
    for row in 0..ctx.layout.rows {
        workspace.begin(row);
        view.gather(row, item.col, workspace.target_mut());
        let solution = workspace.solve(ctx.layers.get(row, item.col), &ctx.solve, &mut rng);
        paint_cell(&mut strip, row, workspace.composite());
        cells.push(solution);
    }
    let solve = t.elapsed();

    ColOutcome {
        cells,
        strip,
        resample,
        solve,
        stages: workspace.stages,
    }
}

/// Copy every column strip into the wall canvas.
///
/// Parallel over canvas rows: each canvas row is written once, left to right,
/// gathering one 24-pixel run from each strip. This is the only wall-sized copy
/// in the pipeline, and it is acceptable only because the canvas has to exist
/// for the PNG encode regardless — nothing else wall-sized is materialised.
///
/// Kept as a standalone step on purpose: the full-resolution canvas is an
/// intermediate consumer, and the HTML export will want a *downscaled* preview
/// (through this crate's own resampler). That stage slots in right after this
/// one, taking the assembled canvas as its input.
fn interleave(canvas: &mut [u8], outcomes: &[ColOutcome], wall_width: usize) {
    canvas
        .par_chunks_mut(wall_width * CHANNELS)
        .enumerate()
        .for_each(|(y, row)| {
            for (c, outcome) in outcomes.iter().enumerate() {
                let dst = c * STRIP_PITCH;
                let src = y * STRIP_PITCH;
                row[dst..dst + STRIP_PITCH].copy_from_slice(&outcome.strip[src..src + STRIP_PITCH]);
            }
        });
}

// -------------------------------------------------------------------- driver

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

/// Decode, lay out, run the column items, write this stage's outputs.
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
    let items = col_items(&layout, &plan);
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
        "wall: {}x{} px, resampling {}x{} -> {}x{} in {} block-column items",
        layout.wall_width,
        layout.wall_height,
        src_w,
        src_h,
        layout.target_width,
        layout.target_height,
        items.len()
    );

    // ---- variance pre-pass -------------------------------------------------
    // Runs on the source image, before the column items, because the layer
    // budget is a global min/max normalisation across every cell of the wall.
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

    // ---- the pipeline: one flat par_iter over block columns ----------------
    let t = Instant::now();
    let ctx = ColContext {
        source: &source,
        plan: &plan,
        layout: &layout,
        layers: &layers,
        max_layers,
        solve: SolveCfg {
            patterns: &patterns,
            refinement: &config.refinement,
            perturbations: config.perturbations,
            lab_refine: config.lab_refine,
            report_lab: config.debug,
        },
        seed: config.seed,
    };
    let outcomes: Vec<ColOutcome> = items
        .par_iter()
        .map(|item| render_column(&ctx, item))
        .collect();
    let t_pipeline = t.elapsed();

    // ---- interleave the column strips into the canvas ----------------------
    let t = Instant::now();
    let mut canvas = vec![0u8; layout.wall_width * layout.wall_height * CHANNELS];
    interleave(&mut canvas, &outcomes, layout.wall_width);
    let t_interleave = t.elapsed();

    // Reduce the per-item stats now, so the outcomes can be consumed for the
    // JSONL without copying every cell's layer list.
    let (cpu_resample, cpu_solve) = outcomes
        .iter()
        .fold((Duration::ZERO, Duration::ZERO), |(r, s), o| {
            (r + o.resample, s + o.solve)
        });
    let mut stages = Stages::default();
    for outcome in &outcomes {
        stages.add(&outcome.stages);
    }
    let total_err: f64 = outcomes
        .iter()
        .flat_map(|o| o.cells.iter())
        .map(|c| f64::from(c.error))
        .sum();
    let total_lab: f64 = outcomes
        .iter()
        .flat_map(|o| o.cells.iter())
        .map(|c| f64::from(c.lab_error))
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
         — this stage's intermediate outputs (block background and HTML export land in phase 3)",
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
        debug_line("    greedy (cpu)", stages.greedy, None);
        debug_line("    refine (cpu)", stages.refine, None);
        debug_line("    perturb (cpu)", stages.perturb, None);
        debug_line("    oklab (cpu)", stages.oklab, None);
        debug_line(
            "interleave",
            t_interleave,
            Some(format!("{} column strips -> canvas", layout.columns)),
        );
        debug_line("encode", t_encode, None);
        debug_line(
            "total",
            t_decode + t_plan + t_variance + t_pipeline + t_interleave + t_encode,
            None,
        );
        println!(
            "{}: {:<16} {:.1} weighted SSE per cell, {:.4} mean OKLab dE per pixel",
            "debug".blue().bold(),
            "error",
            mean_err,
            total_lab / n_cells as f64
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
fn jsonl_path(output: &Path) -> PathBuf {
    output.with_extension("jsonl")
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
