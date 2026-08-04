//! Application entry point: the pipeline as one flat `par_iter` over block
//! columns.
//!
//! One work item is one block column (`context/designs/pipeline.md`): a
//! full-height, 24-pixel-wide slice of the wall. The closure borrows the source
//! planes, the shared vertical weights, the pattern and block tables, the layer
//! grid and the layout; it builds its own horizontal weights, resamples its own
//! column band locally, matches a background block for each of its block cells,
//! solves its banner cells top to bottom, and paints all of it into its own
//! strip — no shared mutable state, no locking, no cross-item handoff.
//!
//! Columns, never rows: banners bridge every horizontal block seam, so no
//! horizontal cut lets an item own complete block cells, while a vertical cut on
//! a block-column boundary cuts nothing (banners never cross columns). The
//! banner-over-banner overlap and the block-behind-banner compositing are then
//! entirely internal to an item.
//!
//! Phase 3 completes the pipeline, so the tool's normal output is no longer an
//! intermediate: `<OUTPUT>` is the self-contained HTML export, and the phase-2
//! intermediates (the bare wall PNG, the per-cell JSONL) are gone — earlier
//! intermediates are deleted when a later stage lands, never kept behind a flag.
//! The full-resolution wall is still available, but only when asked for by name:
//! `--render PATH`.

use std::ops::Range;
use std::path::Path;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use clap::Parser;
use colored::Colorize;
use rayon::prelude::*;

use crate::block::{self, Blocks};
use crate::cli::Args;
use crate::cli::config::Config;
use crate::export::{Wall, html, schematic};
use crate::geometry::BLOCK_SIDE;
use crate::layout::Layout;
use crate::logger::{error_out, info};
use crate::memory;
use crate::pattern;
use crate::preview;
use crate::resample::{InterleavedU8, Plan, PlanarU8, Window};
use crate::solver::block::{BlockScratch, match_cell};
use crate::solver::cell::{BandView, STRIP_PITCH, paint_block, paint_cell};
use crate::solver::variance::{LayerGrid, layer_grid};
use crate::solver::{Rng, Solution, SolveCfg, Stages, Workspace};

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
    /// The background block table.
    blocks: &'a Blocks,
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
    /// One entry per banner cell of the column, top to bottom.
    cells: Vec<Solution>,
    /// One matched block index per *block* cell of the column — one more entry
    /// than `cells`, because a wall of `rows` banner rows is `rows + 1` blocks
    /// tall.
    blocks: Vec<usize>,
    /// The column's rendered preview strip: `wall_height` rows of
    /// [`STRIP_PITCH`] bytes, interleaved into the canvas after the `par_iter`.
    strip: Vec<u8>,
    /// CPU time this item spent resampling.
    resample: Duration,
    /// CPU time this item spent matching and painting background blocks.
    blocks_cpu: Duration,
    /// CPU time this item spent solving and painting banner cells.
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
/// and the copy lands in a buffer that has to exist for the downscale anyway.
///
/// Order inside the item is the order the wall is built in: blocks first, over
/// the whole strip, then banners over them. Nothing has to be composited, because
/// the banners' visible regions tile exactly and the block pixels that survive
/// are precisely the hollow frame the matcher scored ([`crate::block`]).
fn render_column(ctx: &ColContext<'_>, item: &ColItem) -> ColOutcome {
    debug_assert_eq!(
        item.strip_cols.len(),
        BLOCK_SIDE,
        "one item, one block column"
    );
    let mut strip = vec![0u8; ctx.layout.wall_height * STRIP_PITCH];

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

    // ---- the wall behind the banners ---------------------------------------
    let t = Instant::now();
    let block_rows = ctx.layout.rows + 1;
    let mut scratch = BlockScratch::new();
    let mut blocks = Vec::with_capacity(block_rows);
    for row in 0..block_rows {
        let id = match_cell(&view, &mut scratch, ctx.blocks, row, item.col, block_rows);
        paint_block(&mut strip, row, &ctx.blocks.texture[id]);
        blocks.push(id);
    }
    let blocks_cpu = t.elapsed();

    // ---- the banners hanging on it -----------------------------------------
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
        blocks,
        strip,
        resample,
        blocks_cpu,
        solve,
        stages: workspace.stages,
    }
}

/// Copy every column strip into the wall canvas.
///
/// Parallel over canvas rows: each canvas row is written once, left to right,
/// gathering one 24-pixel run from each strip. This is the only wall-sized copy
/// in the pipeline, and it is acceptable only because the canvas has to exist
/// for the preview downscale regardless — nothing else wall-sized is
/// materialised, and in particular the canvas is never widened to `f32`
/// ([`crate::preview`] reads it strided, in place).
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
    check_cpu_features();

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

/// Refuse to run an AVX2 build on a CPU that predates AVX2.
///
/// Compiled in only alongside the AVX2 SIMD backend (`src/simd/mod.rs` gates it
/// on the same cfg), so on aarch64 — and on any scalar build — this is an empty
/// function that vanishes.
///
/// The check is raw `cpuid` rather than `is_x86_feature_detected!` on purpose:
/// that macro is `cfg!(target_feature = ..) || runtime_detect(..)`, and this
/// build enables `+avx2,+fma` crate-wide (`.cargo/config.toml`), so the macro
/// would fold to a compile-time `true` and never fire. `cpuid` asks the CPU.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma",
    not(feature = "force-scalar")
))]
fn check_cpu_features() {
    use std::arch::x86_64::{__cpuid, __cpuid_count, _xgetbv};

    // SAFETY: `cpuid` leaves 0 and 1 exist on every x86_64 CPU; leaf 7 is read
    // only after leaf 0 reports it as supported, and `xgetbv` only after the
    // OSXSAVE bit says the OS enabled it.
    let ok = unsafe {
        let leaf1 = __cpuid(1);
        let fma = leaf1.ecx & (1 << 12) != 0;
        let osxsave = leaf1.ecx & (1 << 27) != 0;
        let avx = leaf1.ecx & (1 << 28) != 0;
        // The OS must actually preserve the XMM and YMM register state.
        let os_ymm = osxsave && (_xgetbv(0) & 0b110) == 0b110;
        let avx2 = __cpuid(0).eax >= 7 && __cpuid_count(7, 0).ebx & (1 << 5) != 0;
        fma && avx && os_ymm && avx2
    };

    if !ok {
        error_out!(
            "this build requires {} (any Intel/AMD CPU from ~2013 onwards); \
             rebuild with {} for older CPUs",
            "AVX2+FMA".red(),
            "--features force-scalar".yellow()
        );
    }
}

/// No-op: this build's SIMD backend has no runtime CPU requirements.
#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma",
    not(feature = "force-scalar")
)))]
fn check_cpu_features() {}

/// Decode, lay out, run the column items, write the export.
fn run(config: &Config) {
    let started = Instant::now();

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

    // ---- layout + shared weights + pattern and block tables ----------------
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
    let (patterns, blocks) = rayon::join(
        || pattern::load(&config.exclude_patterns),
        || block::load(&config.exclude_blocks),
    );
    let t_plan = t.elapsed();

    info!(
        "grid: {}x{} blocks ({} banners), {} patterns x {} dyes, {} blocks",
        layout.columns,
        layout.rows + 1,
        layout.columns * layout.rows,
        patterns.len(),
        crate::color::NUM_COLORS,
        blocks.len(),
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
        blocks: &blocks,
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

    // Reduce the per-item stats now, so the outcomes can be consumed without
    // copying every cell's layer list.
    let (cpu_resample, cpu_blocks, cpu_solve) = outcomes.iter().fold(
        (Duration::ZERO, Duration::ZERO, Duration::ZERO),
        |(r, b, s), o| (r + o.resample, b + o.blocks_cpu, s + o.solve),
    );
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
    let (cells, block_ids): (Vec<Vec<Solution>>, Vec<Vec<usize>>) =
        outcomes.into_iter().map(|o| (o.cells, o.blocks)).unzip();

    // ---- the full-resolution wall, only when asked for by name -------------
    let t = Instant::now();
    if let Some(path) = &config.render {
        write_png(path, &canvas, layout.wall_width, layout.wall_height);
        info!(
            "wrote '{}': the full-resolution wall render",
            path.display().to_string().yellow()
        );
    }
    let t_render = t.elapsed();

    // ---- the two preview panes, both through our own resampler -------------
    let t = Instant::now();
    let (pw, ph) = preview::dimensions(
        config.preview,
        (src_w, src_h),
        (layout.wall_width, layout.wall_height),
    );
    let wall_src = InterleavedU8 {
        data: &canvas,
        width: layout.wall_width,
        height: layout.wall_height,
        channels: CHANNELS,
    };
    let (generated, original) = rayon::join(
        || {
            preview::resize(
                &wall_src,
                Window::full(layout.wall_width, layout.wall_height),
                pw,
                ph,
            )
        },
        || preview::resize(&source, layout.window, pw, ph),
    );
    let t_preview = t.elapsed();
    drop(canvas);

    let (preview_png, original_jpeg) = rayon::join(
        || encode(&generated, pw, ph, image::ImageFormat::Png),
        || encode(&original, pw, ph, image::ImageFormat::Jpeg),
    );
    let t_encode = t.elapsed() - t_preview;

    // ---- the exports -------------------------------------------------------
    let t = Instant::now();
    let wall = Wall {
        rows: layout.rows,
        columns: layout.columns,
        patterns: &patterns,
        blocks: &blocks,
        block_ids: &block_ids,
        cells: &cells,
    };
    let stem = config
        .output
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "banners".to_string());
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let (schem_bytes, litematic_bytes) = rayon::join(
        || schematic::schem(&wall),
        || schematic::litematic(&wall, &stem, now.as_millis() as i64),
    );
    let t_schematic = t.elapsed();

    let t = Instant::now();
    let input_name = config
        .input
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| config.input.display().to_string());
    let report = html::Report {
        input: &input_name,
        stem: &stem,
        preview: &preview_png,
        original: &original_jpeg,
        original_mime: "image/jpeg",
        size: (pw, ph),
        litematic: &litematic_bytes,
        schem: &schem_bytes,
    };
    let page = html::page(&wall, &report);
    std::fs::write(&config.output, page.as_bytes()).unwrap_or_else(|e| {
        error_out!(
            "could not write '{}': {}",
            config.output.display().to_string().yellow(),
            e.to_string().red()
        );
    });
    let t_html = t.elapsed();

    info!(
        "wrote '{}': the banner chart ({}), preview {}x{}",
        config.output.display().to_string().yellow(),
        memory::format_bytes(page.len()),
        pw,
        ph,
    );

    if config.debug {
        let n_cells = layout.rows * layout.columns;
        let mean_err = total_err / n_cells as f64;

        debug_line("decode", t_decode, None);
        debug_line("layout+tables", t_plan, None);
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
        debug_line(
            "  blocks (cpu)",
            cpu_blocks,
            Some(format!(
                "{} block cells x {} candidates",
                (layout.rows + 1) * layout.columns,
                blocks.len()
            )),
        );
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
        if config.render.is_some() {
            debug_line("render png", t_render, None);
        }
        debug_line(
            "preview",
            t_preview,
            Some(format!("wall + original -> {pw}x{ph}")),
        );
        debug_line(
            "  encode",
            t_encode,
            Some(format!(
                "{} + {}",
                memory::format_bytes(preview_png.len()),
                memory::format_bytes(original_jpeg.len())
            )),
        );
        debug_line(
            "schematics",
            t_schematic,
            Some(format!(
                "{} .schem + {} .litematic, gzipped",
                memory::format_bytes(schem_bytes.len()),
                memory::format_bytes(litematic_bytes.len())
            )),
        );
        debug_line("html", t_html, Some(memory::format_bytes(page.len())));
        debug_line("total", started.elapsed(), None);
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

/// Encode interleaved RGB in memory.
fn encode(data: &[u8], width: usize, height: usize, format: image::ImageFormat) -> Vec<u8> {
    let img = image::RgbImage::from_raw(width as u32, height as u32, data.to_vec())
        .unwrap_or_else(|| error_out!("internal error: image buffer has the wrong size"));
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, format)
        .unwrap_or_else(|e| error_out!("could not encode {format:?}: {e}"));
    out.into_inner()
}

/// Write interleaved RGB straight to a PNG file (`--render`).
fn write_png(path: &Path, data: &[u8], width: usize, height: usize) {
    let img = image::RgbImage::from_raw(width as u32, height as u32, data.to_vec())
        .unwrap_or_else(|| error_out!("internal error: canvas has the wrong size"));
    img.save(path).unwrap_or_else(|e| {
        error_out!(
            "could not write '{}': {}",
            path.display().to_string().yellow(),
            e.to_string().red()
        );
    });
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
