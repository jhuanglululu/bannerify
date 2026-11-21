//! Application entry point: the pipeline in three phases over groups of block
//! columns.
//!
//! 1. prepare (rayon over columns): resample the column band, match and paint
//!    the background blocks, gather every banner cell's target into a
//!    [`PreparedCell`] (at the cell's chosen shift, under `--shift-range`);
//! 2. solve: the group's cells go to a [`Backend`] as one batch;
//! 3. paint (rayon over columns): each solution's composite is rebuilt on the
//!    CPU and painted into its column strip, and a cell solved against a
//!    shifted patch has its error re-measured against the unshifted one.
//!
//! Columns, never rows: banners bridge every horizontal block seam, so no
//! horizontal cut lets a column item own complete block cells, while a
//! vertical cut on a block-column boundary cuts nothing (banners never cross
//! columns). The banner-over-banner overlap and the block-behind-banner
//! compositing are then entirely internal to a column — no shared mutable
//! state, no locking, no cross-column handoff.

use std::io::{IsTerminal, stderr};
use std::ops::Range;
use std::path::Path;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use clap::Parser;
use clap::error::{ContextKind, ContextValue, ErrorKind};
use colored::Colorize;
use rayon::prelude::*;
use serde_json::{Map, Value, json};

use crate::block::{self, Blocks};
use crate::block_set;
use crate::cli::config::{Config, format_name};
use crate::cli::{Args, BackendChoice, ListKind, OutputFormat};
use crate::color::{COLOR_NAMES, NUM_COLORS};
use crate::export::{Wall, color_label, hex, html, json as json_export, pattern_label, schematic};
use crate::geometry::{BANNER_W, BLOCK_SIDE};
use crate::gpu::GpuBackend;
use crate::layout::Layout;
use crate::logger::{
    self, Diagnostic, Kind, Mode, SCHEMA_VERSION, debug_print, fatal, info, internal,
};
use crate::memory;
use crate::pattern;
use crate::preview;
use crate::progress::Progress;
use crate::resample::{InterleavedU8, Plan, PlanarU8, Window};
use crate::simd::Chunk;
use crate::solver::block::{BlockScratch, match_cell};
use crate::solver::cell::{
    BandView, Compositor, STRIP_PITCH, paint_block, paint_cell, patch_rows, target_error,
};
use crate::solver::shift::ShiftSearch;
use crate::solver::variance::{LayerGrid, layer_grid};
use crate::solver::{Backend, CpuBackend, Plane, PreparedCell, Rng, Solution, SolveCfg, perturb};

const CHANNELS: usize = 3;

// ---------------------------------------------------------------- work items

/// Banner cells prepared, solved and painted per column group: the bound on
/// how many [`PreparedCell`]s (about 10 KB each) and solutions exist at once.
/// Large enough that a group keeps every worker busy, small enough that a
/// 15,000-banner wall never holds every cell's target at once.
const GROUP_CELLS: usize = 4096;

/// Where one column item's resampled pixels land in the wall canvas.
#[derive(Clone, Copy, Debug)]
struct Rect {
    /// Left edge, in canvas pixels.
    x: usize,
    /// Top edge, in canvas pixels.
    y: usize,
    /// Width in pixels.
    width: usize,
}

/// One unit of parallel prepare work: one block column of the wall.
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
    /// The solver stages' shared configuration.
    solve: SolveCfg<'a>,
    /// Perturbation RNG seed; each column derives its stream from it and its
    /// column index.
    seed: u64,
    /// `--shift-range`; `(0, 0)` gathers every cell at its real position.
    shift_range: (usize, usize),
}

/// What preparing one column produces besides its cells.
struct PreparedColumn {
    /// One matched block index per *block* cell of the column — one more entry
    /// than it has banner cells, because a wall of `rows` banner rows is
    /// `rows + 1` blocks tall.
    blocks: Vec<usize>,
    /// The column's preview strip with the background blocks painted:
    /// `wall_height` rows of [`STRIP_PITCH`] bytes.
    strip: Vec<u8>,
    /// CPU time spent resampling.
    resample: Duration,
    /// CPU time spent matching and painting background blocks.
    blocks_cpu: Duration,
    /// Per banner cell, top to bottom: the patch at zero shift when the cell
    /// was gathered at a non-zero one, which its reported error is measured
    /// against. `None` for every cell when the shift range is `(0, 0)`.
    unshifted: Vec<Option<Box<[Plane; CHANNELS]>>>,
    /// CPU time spent gathering the banner cells, choosing their shifts
    /// included.
    gather: Duration,
}

/// One finished column.
struct ColOutcome {
    /// One entry per banner cell of the column, top to bottom.
    cells: Vec<Solution>,
    /// See [`PreparedColumn::blocks`].
    blocks: Vec<usize>,
    /// The column's rendered preview strip, banners over blocks, interleaved
    /// into the canvas once every column is done.
    strip: Vec<u8>,
}

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
                },
                target_cols,
            }
        })
        .collect()
}

/// Prepare one block column: resample its band, match and paint the blocks
/// behind the banners, and write every banner cell, top to bottom, into
/// `cells` (one entry per banner row). Every field of every entry is
/// overwritten except the head of a lower row's target planes, which the
/// solver never reads.
///
/// The strip is the column's own local buffer rather than a slice of the
/// canvas: the canvas is row-major, so a column of it is not a contiguous
/// `&mut [u8]` and `split_at_mut` cannot hand one out. Painting locally and
/// interleaving once keeps the whole pipeline in safe code.
///
/// Blocks are painted first, over the whole strip, and the banners over them
/// later ([`paint_column`]). Nothing has to be composited, because the
/// banners' visible regions tile exactly and the block pixels that survive are
/// precisely the hollow frame the matcher scored ([`crate::block`]).
fn prepare_column(
    ctx: &ColContext<'_>,
    item: &ColItem,
    cells: &mut [PreparedCell],
) -> PreparedColumn {
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

    // ---- the banner cells, ready to solve ----------------------------------
    // Banner row 0 has nothing hanging in front of it, so it solves the full
    // 20x40 patch; every other row solves only its visible 24 rows, the
    // lane-aligned tail of the same planes (`crate::solver::workspace`).
    //
    // One PRNG stream per column, advanced down the column's cells: the banner
    // row never enters the seeding, so the draws a cell sees depend only on the
    // seed, the column and how many cells precede it. Each cell takes a copy of
    // the stream as it stands, and the stream is then replayed past that
    // cell's draws for the next one.
    let mut rng = Rng::new(ctx.seed, item.col as u64);
    let mut shifts = ShiftSearch::new(ctx.shift_range);
    let mut unshifted = Vec::with_capacity(cells.len());
    let mut gather = Duration::ZERO;
    for (row, cell) in cells.iter_mut().enumerate() {
        cell.row = row;
        cell.n_layers = ctx.layers.get(row, item.col);
        let t = Instant::now();
        let shifted = shifts.gather(
            &view,
            row,
            item.col,
            &ctx.solve.patterns.top,
            &mut cell.target,
        );
        gather += t.elapsed();
        unshifted.push(shifted.map(|s| s.unshifted));
        cell.rng = rng.clone();
        perturb::skip(&mut rng, &ctx.solve, cell.n_layers);
    }

    PreparedColumn {
        blocks,
        strip,
        resample,
        blocks_cpu,
        unshifted,
        gather,
    }
}

/// Paint one column's solved cells over its blocks.
///
/// The composites are rebuilt from the solutions here ([`Compositor`]), not
/// taken from the backend, so every backend's results are painted through the
/// same CPU compositing step.
///
/// A cell with an `unshifted` patch was solved against a shifted one, so the
/// error its backend reported is against the wrong target; it is re-measured
/// here, on the CPU for every backend, against the unshifted patch.
fn paint_column(
    strip: &mut [u8],
    cells: &mut [Solution],
    unshifted: &[Option<Box<[Plane; CHANNELS]>>],
    alphas: &[Plane],
) {
    let mut compositor = Compositor::default();
    for (row, (solution, unshifted)) in cells.iter_mut().zip(unshifted).enumerate() {
        let composite = compositor.composite(solution, row, alphas);
        if let Some(target) = unshifted {
            let hw = patch_rows(row).1 * BANNER_W;
            solution.error = target_error(composite, target, row);
            solution.lab_error = solution.error / hw as f32;
        }
        paint_cell(strip, row, composite);
    }
}

/// Run the whole wall through prepare, solve and paint, one group of columns
/// at a time, and return the finished columns left to right together with the
/// prepare phase's resample, block-matching and cell-gathering CPU times.
///
/// Each group is prepared in parallel over its columns, solved as one batch,
/// then painted in parallel over its columns. Only one group's prepared cells
/// exist at a time ([`GROUP_CELLS`]), in one buffer that every group reuses
/// and each column writes its own slice of; the strips are kept for the final
/// interleave.
fn render_wall(
    ctx: &ColContext<'_>,
    items: &[ColItem],
    backend: &mut dyn Backend,
) -> (Vec<ColOutcome>, Duration, Duration, Duration) {
    let rows = ctx.layout.rows;
    let group_columns = (GROUP_CELLS / rows).max(1);
    let mut outcomes = Vec::with_capacity(items.len());
    let (mut resample, mut blocks_cpu) = (Duration::ZERO, Duration::ZERO);
    let mut gather = Duration::ZERO;
    let mut cells = Vec::new();

    for group in items.chunks(group_columns) {
        // ---- prepare -------------------------------------------------------
        cells.resize_with(group.len() * rows, || PreparedCell {
            row: 0,
            n_layers: 0,
            target: [Chunk::zeroed(); CHANNELS],
            rng: Rng::new(0, 0),
        });
        let prepared: Vec<PreparedColumn> = group
            .par_iter()
            .zip(cells.par_chunks_mut(rows))
            .map(|(item, cells)| prepare_column(ctx, item, cells))
            .collect();
        for column in &prepared {
            resample += column.resample;
            blocks_cpu += column.blocks_cpu;
            gather += column.gather;
        }

        // ---- solve ---------------------------------------------------------
        let solutions = backend.solve(&cells, &ctx.solve);

        // ---- paint ---------------------------------------------------------
        // The unshifted patches are dropped right after painting: they are only
        // needed for the error readout, and the wall's solutions outlive them.
        let mut solutions = solutions.into_iter();
        let mut finished: Vec<(ColOutcome, Vec<_>)> = prepared
            .into_iter()
            .map(
                |PreparedColumn {
                     blocks,
                     strip,
                     unshifted,
                     ..
                 }| {
                    let outcome = ColOutcome {
                        cells: solutions.by_ref().take(rows).collect(),
                        blocks,
                        strip,
                    };
                    (outcome, unshifted)
                },
            )
            .collect();
        finished.par_iter_mut().for_each(|(outcome, unshifted)| {
            paint_column(
                &mut outcome.strip,
                &mut outcome.cells,
                unshifted,
                &ctx.solve.patterns.top,
            );
        });
        outcomes.extend(finished.into_iter().map(|(outcome, _)| outcome));
    }

    (outcomes, resample, blocks_cpu, gather)
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

// ------------------------------------------------------------------- backend

/// The backend the solve runs on, as resolved from `--backend`.
enum Solver {
    Cpu(CpuBackend),
    Gpu(Box<GpuBackend>),
}

/// What `--debug` and `--json` report about the GPU backend after a run.
struct GpuReport {
    device: String,
    batches: usize,
    /// Largest device allocation, in bytes.
    peak: usize,
}

impl Solver {
    fn backend(&mut self) -> &mut dyn Backend {
        match self {
            Solver::Cpu(cpu) => cpu,
            Solver::Gpu(gpu) => gpu.as_mut(),
        }
    }

    /// The backend's name as `--backend` spells it.
    fn name(&self) -> &'static str {
        match self {
            Solver::Cpu(_) => "cpu",
            Solver::Gpu(_) => "gpu",
        }
    }

    fn gpu_report(&self) -> Option<GpuReport> {
        match self {
            Solver::Cpu(_) => None,
            Solver::Gpu(gpu) => Some(GpuReport {
                device: gpu.device_name().to_string(),
                batches: gpu.batches(),
                peak: gpu.peak_allocated(),
            }),
        }
    }
}

/// Why `--backend gpu` cannot run, as the error it exits with.
fn gpu_unavailable(reason: impl std::fmt::Display) -> Diagnostic {
    Diagnostic::new(
        Kind::GpuUnavailable,
        format!("the GPU backend is unavailable: {reason}"),
    )
    .hint("use --backend cpu, or --backend auto to fall back to the CPU")
}

/// Open the GPU backend if `--backend` asks for it. `Err` means the CPU
/// backend runs, and carries the reason when `auto` fell back to it (`None`
/// under `--backend cpu`). Under `gpu` a failure is fatal (exit code 6);
/// under `auto` it is a warning.
fn open_gpu(config: &Config) -> Result<GpuBackend, Option<String>> {
    if config.backend == BackendChoice::Cpu {
        return Err(None);
    }
    match GpuBackend::new() {
        Ok(gpu) => Ok(gpu.with_memory_budget(config.gpu_memory << 20)),
        Err(e) if config.backend == BackendChoice::Gpu => fatal(gpu_unavailable(e)),
        Err(e) => {
            logger::warn(Diagnostic::new(
                Kind::GpuUnavailable,
                format!("the GPU backend is unavailable ({e}); solving on the CPU"),
            ));
            Err(Some(e.to_string()))
        }
    }
}

/// The `--list backends` entry for the GPU backend.
fn gpu_entry() -> Value {
    match GpuBackend::new() {
        Ok(gpu) => json!({"id": "gpu", "available": true, "device": gpu.device_name()}),
        Err(e) => json!({"id": "gpu", "available": false, "reason": e.to_string()}),
    }
}

// -------------------------------------------------------------------- driver

/// Parse arguments, validate, and run the pipeline.
pub fn run_cli() {
    // Every human-facing line is on stderr, so colour follows stderr, not the
    // stdout `colored` checks by default.
    if !stderr().is_terminal() && std::env::var_os("CLICOLOR_FORCE").is_none() {
        colored::control::set_override(false);
    }
    // Known before parsing, so a CPU or parse failure is reported as JSON too.
    if std::env::args_os().any(|a| a == "--json") {
        logger::set_mode(Mode::Json);
    }
    check_cpu_features();

    let args = parse_args();
    logger::set_mode(if args.json {
        Mode::Json
    } else if args.quiet {
        Mode::Quiet
    } else {
        Mode::Text
    });
    if let Some(kind) = args.list {
        list(kind);
        return;
    }
    let config = Config::from(args);

    if let Some(workers) = config.workers {
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build_global()
            .unwrap_or_else(|e| internal(format!("could not start {workers} workers: {e}")));
    }
    info!("using {} workers", rayon::current_num_threads());

    run(&config);
}

/// clap's own errors keep clap's output in text mode; under `--json` they
/// become a `usage` diagnostic. Help and version always print normally.
fn parse_args() -> Args {
    Args::try_parse().unwrap_or_else(|e| {
        let informational = matches!(
            e.kind(),
            ErrorKind::DisplayHelp
                | ErrorKind::DisplayVersion
                | ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand
        );
        if informational || logger::mode() != Mode::Json {
            e.exit();
        }
        // The rendered error up to the first blank line, which drops the
        // usage block and the "For more information" footer.
        let rendered = e.render().to_string();
        let first = rendered.split("\n\n").next().unwrap_or_default();
        let message = first
            .strip_prefix("error: ")
            .unwrap_or(first)
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        let mut d = Diagnostic::new(Kind::Usage, message).hint("run `bannerify --help` for usage");
        match e.get(ContextKind::InvalidArg) {
            Some(ContextValue::String(arg)) => {
                d = d.option(arg.split_whitespace().next().unwrap_or(arg));
            }
            Some(ContextValue::Strings(args)) => d = d.option(args.join(" ")),
            _ => {}
        }
        if let Some(ContextValue::String(value)) = e.get(ContextKind::InvalidValue) {
            d = d.value(value);
        }
        fatal(d)
    })
}

/// `--list KIND`: the valid ids, one per line, or one JSON object.
fn list(kind: ListKind) {
    let (key, entries): (&str, Vec<Value>) = match kind {
        ListKind::Patterns => (
            "patterns",
            pattern::ids()
                .into_iter()
                .map(|id| json!({"id": id, "name": pattern_label(&id)}))
                .collect(),
        ),
        ListKind::Blocks => (
            "blocks",
            block::ids()
                .into_iter()
                .map(|id| json!({"id": id}))
                .collect(),
        ),
        ListKind::Sets => (
            "sets",
            block_set::builtin()
                .into_iter()
                .map(|(id, blocks)| json!({"id": id, "blocks": blocks}))
                .collect(),
        ),
        ListKind::Dyes => (
            "dyes",
            (0..NUM_COLORS)
                .map(|i| json!({"id": COLOR_NAMES[i], "name": color_label(i), "hex": hex(i)}))
                .collect(),
        ),
        ListKind::Backends => (
            "backends",
            vec![json!({"id": "cpu", "available": true}), gpu_entry()],
        ),
    };
    if logger::mode() == Mode::Json {
        logger::print_result(&json!({"ok": true, "schema_version": SCHEMA_VERSION, key: entries}));
    } else {
        // Text lists only what can be used here: a backend this machine
        // cannot run is left out.
        for entry in entries.iter().filter(|e| e["available"] != false) {
            println!("{}", entry["id"].as_str().expect("every entry has an id"));
        }
    }
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
        fatal(
            Diagnostic::new(
                Kind::UnsupportedCpu,
                "this build requires AVX2+FMA (any Intel/AMD CPU from ~2013 onwards)",
            )
            .hint("rebuild with `--features force-scalar` for older CPUs"),
        );
    }
}

#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma",
    not(feature = "force-scalar")
)))]
fn check_cpu_features() {}

fn run(config: &Config) {
    let started = Instant::now();

    // ---- the solver backend ------------------------------------------------
    // Opened first, so `--backend gpu` on a machine without a usable GPU fails
    // before any work is done.
    let t = Instant::now();
    let gpu = open_gpu(config);
    let t_backend = t.elapsed();
    // The fallback reason is repeated here, next to the backend's name, so
    // the line answers "why the CPU?" without the warning above it.
    match &gpu {
        Ok(gpu) => info!("backend: gpu ({})", gpu.device_name()),
        Err(Some(reason)) => info!("backend: cpu (GPU unavailable: {reason})"),
        Err(None) => info!("backend: cpu"),
    }
    let gpu = gpu.ok();

    // ---- decode (and the one interleaved -> planar conversion) -------------
    let t = Instant::now();
    let decoded = image::open(&config.input).unwrap_or_else(|e| {
        fatal(
            Diagnostic::new(
                Kind::InputError,
                format!("could not read input '{}': {e}", config.input.display()),
            )
            .path(config.input.display()),
        )
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
        || block::load(&config.blocks),
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
        "layers {}",
        layers
            .histogram(config.n_layers)
            .iter()
            .enumerate()
            .map(|(i, n)| format!("{}: {}", i + config.n_layers.0, n))
            .collect::<Vec<_>>()
            .join(", ")
    );

    // ---- the pipeline: prepare -> solve -> paint, per column group ---------
    let t = Instant::now();
    let progress = Progress::start(layout.rows * layout.columns);
    let ctx = ColContext {
        source: &source,
        plan: &plan,
        layout: &layout,
        blocks: &blocks,
        layers: &layers,
        solve: SolveCfg {
            patterns: &patterns,
            refinement: &config.refinement,
            perturbations: config.perturbations,
            feature_weight: config.feature_weight,
            purity_weight: config.purity_weight,
            gradient_weight: config.gradient_weight,
        },
        seed: config.seed,
        shift_range: config.shift_range,
    };
    let mut solver = match gpu {
        Some(gpu) => Solver::Gpu(Box::new(gpu.with_progress(progress.counter()))),
        None => Solver::Cpu(CpuBackend::new(
            max_layers,
            config.refinement.refinement_candidate,
            progress.counter(),
        )),
    };
    let (outcomes, cpu_resample, cpu_blocks, cpu_gather) =
        render_wall(&ctx, &items, solver.backend());
    let stages = solver.backend().stages();
    let t_pipeline = t.elapsed();
    progress.finish();
    let gpu_report = solver.gpu_report();

    // ---- interleave the column strips into the canvas ----------------------
    let t = Instant::now();
    let needs_canvas = config.format == OutputFormat::Html || config.render.is_some();
    let canvas = needs_canvas.then(|| {
        let mut canvas = vec![0u8; layout.wall_width * layout.wall_height * CHANNELS];
        interleave(&mut canvas, &outcomes, layout.wall_width);
        canvas
    });
    let t_interleave = t.elapsed();

    // The solve time is the sum of the backend's stages: CPU time summed over
    // the workers for the CPU backend, wall time for the GPU backend.
    let t_solve = stages.feature + stages.greedy + stages.refine + stages.perturb + stages.oklab;
    // Reduce the per-cell errors now, so the outcomes can be consumed without
    // copying every cell's layer list.
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
    let render_bytes = config.render.as_ref().map(|path| {
        let bytes = write_png(
            path,
            canvas.as_ref().expect("render needs canvas"),
            layout.wall_width,
            layout.wall_height,
        );
        info!(
            "wrote '{}': the full-resolution wall render",
            path.display().to_string().yellow()
        );
        bytes
    });
    let t_render = t.elapsed();

    // HTML alone needs the comparison previews and their encoded images.
    let previews = if config.format == OutputFormat::Html {
        Some({
            let t = Instant::now();
            let (pw, ph) =
                preview::dimensions(config.preview, (layout.wall_width, layout.wall_height));
            let wall_src = InterleavedU8 {
                data: canvas.as_ref().expect("HTML needs canvas"),
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
            (pw, ph, preview_png, original_jpeg, t_preview, t_encode)
        })
    } else {
        drop(canvas);
        None
    };

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
    let (schem_bytes, litematic_bytes) = match config.format {
        OutputFormat::Html => rayon::join(
            || schematic::schem(&wall),
            || schematic::litematic(&wall, &stem, now.as_millis() as i64),
        ),
        OutputFormat::Schem => (schematic::schem(&wall), Vec::new()),
        OutputFormat::Litematic => (
            Vec::new(),
            schematic::litematic(&wall, &stem, now.as_millis() as i64),
        ),
        OutputFormat::Json => (Vec::new(), Vec::new()),
    };
    let t_schematic = t.elapsed();

    let input_name = config
        .input
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| config.input.display().to_string());
    let t = Instant::now();
    let page = previews
        .as_ref()
        .map(|(pw, ph, preview_png, original_jpeg, _, _)| {
            let report = html::Report {
                input: &input_name,
                stem: &stem,
                preview: preview_png,
                original: original_jpeg,
                original_mime: "image/jpeg",
                size: (*pw, *ph),
                litematic: &litematic_bytes,
                schem: &schem_bytes,
            };
            html::page(&wall, &report)
        });
    let t_html = t.elapsed();
    let document =
        (config.format == OutputFormat::Json).then(|| json_export::document(&wall, &input_name));
    let (bytes, description) = match config.format {
        OutputFormat::Html => (
            page.as_ref().expect("HTML page").as_bytes(),
            "HTML banner chart",
        ),
        OutputFormat::Schem => (schem_bytes.as_slice(), "Sponge schematic"),
        OutputFormat::Litematic => (litematic_bytes.as_slice(), "Litematica schematic"),
        OutputFormat::Json => (
            document.as_deref().expect("JSON document"),
            "banner wall JSON",
        ),
    };
    std::fs::write(&config.output, bytes).unwrap_or_else(|e| write_failed(&config.output, e));
    info!(
        "wrote '{}': {} ({})",
        config.output.display().to_string().yellow(),
        description,
        memory::format_bytes(bytes.len()),
    );

    let n_cells = layout.rows * layout.columns;
    let err_per_cell = total_err / n_cells as f64;
    let lab_per_pixel = total_lab / n_cells as f64;

    // (stage, wall time, note); the leading spaces nest a stage under the one
    // above it in the text form.
    let mut timings: Vec<(&str, Duration, Option<String>)> = Vec::new();
    if config.debug {
        timings.push((
            "backend",
            t_backend,
            Some(match &gpu_report {
                Some(gpu) => format!("gpu, {}", gpu.device),
                None => "cpu".to_string(),
            }),
        ));
        timings.push(("decode", t_decode, None));
        timings.push(("layout+tables", t_plan, None));
        timings.push(("variance", t_variance, None));
        timings.push((
            "pipeline",
            t_pipeline,
            Some(format!(
                "{:.0} cells/s",
                n_cells as f64 / t_pipeline.as_secs_f64()
            )),
        ));
        timings.push(("  resample", cpu_resample, None));
        timings.push((
            "  blocks",
            cpu_blocks,
            Some(format!(
                "{} block cells x {} candidates",
                (layout.rows + 1) * layout.columns,
                blocks.len()
            )),
        ));
        if config.shift_range != (0, 0) {
            timings.push((
                "  shift",
                cpu_gather,
                Some(format!(
                    "gather and shift choice, range {} {}",
                    config.shift_range.0, config.shift_range.1
                )),
            ));
        }
        timings.push((
            "  solve",
            t_solve,
            Some(match &gpu_report {
                Some(gpu) => format!(
                    "GPU backend, wall time; {} batches, peak device memory {}",
                    gpu.batches,
                    memory::format_bytes(gpu.peak)
                ),
                None => format!(
                    "CPU backend, CPU time summed over {} workers",
                    rayon::current_num_threads()
                ),
            }),
        ));
        timings.push(("    feature", stages.feature, None));
        timings.push(("    greedy+refine", stages.greedy + stages.refine, None));
        timings.push(("    perturb+refine", stages.perturb, None));
        timings.push(("    oklab", stages.oklab, None));
        if needs_canvas {
            timings.push((
                "interleave",
                t_interleave,
                Some(format!("{} column strips -> canvas", layout.columns)),
            ));
        }
        if config.render.is_some() {
            timings.push(("render png", t_render, None));
        }
        if let Some((pw, ph, preview_png, original_jpeg, t_preview, t_encode)) = &previews {
            timings.push((
                "preview",
                *t_preview,
                Some(format!("wall + original -> {pw}x{ph}")),
            ));
            timings.push((
                "  encode",
                *t_encode,
                Some(format!(
                    "{} + {}",
                    memory::format_bytes(preview_png.len()),
                    memory::format_bytes(original_jpeg.len())
                )),
            ));
        }
        let schematic_sizes = [(".schem", &schem_bytes), (".litematic", &litematic_bytes)]
            .into_iter()
            .filter(|(_, bytes)| !bytes.is_empty())
            .map(|(format, bytes)| format!("{} {format}", memory::format_bytes(bytes.len())))
            .collect::<Vec<_>>()
            .join(" + ");
        if !schematic_sizes.is_empty() {
            timings.push((
                "schematics",
                t_schematic,
                Some(format!("{schematic_sizes}, gzipped")),
            ));
        }
        if let Some(page) = &page {
            timings.push(("html", t_html, Some(memory::format_bytes(page.len()))));
        }
        timings.push(("total", started.elapsed(), None));

        for (stage, d, note) in &timings {
            let note = note.as_ref().map(|n| format!("   {n}")).unwrap_or_default();
            debug_print(format!("{stage:<18} {:>9.3} s{note}", d.as_secs_f64()));
        }
        debug_print(format!(
            "{:<18} {err_per_cell:.3} OKLab dE per cell, {lab_per_pixel:.4} mean OKLab dE per pixel",
            "error"
        ));
        debug_print(format!(
            "{:<18} peak {}, still live {}",
            "memory",
            memory::format_bytes(memory::peak_bytes()),
            memory::format_bytes(memory::live_bytes())
        ));
    }

    if logger::mode() == Mode::Json {
        let mut summary = json!({
            "ok": true,
            "schema_version": SCHEMA_VERSION,
            "output": {
                "path": config.output.display().to_string(),
                "format": format_name(config.format),
                "bytes": bytes.len(),
            },
            "render": config.render.as_ref().zip(render_bytes).map(|(path, bytes)| json!({
                "path": path.display().to_string(),
                "bytes": bytes,
            })),
            "input": {
                "path": config.input.display().to_string(),
                "width": src_w,
                "height": src_h,
            },
            "wall": {
                "columns": layout.columns,
                "banner_rows": layout.rows,
                "block_rows": layout.rows + 1,
                "banners": n_cells,
                "width_px": layout.wall_width,
                "height_px": layout.wall_height,
            },
            "settings": config.settings_json(rayon::current_num_threads(), solver.name()),
            "materials": json_export::materials(&wall),
            "quality": {
                "delta_e_per_cell": err_per_cell,
                "mean_delta_e_per_pixel": lab_per_pixel,
            },
        });
        if config.debug {
            let seconds: Map<String, Value> = timings
                .iter()
                .map(|(stage, d, _)| (stage.trim().to_string(), json!(d.as_secs_f64())))
                .collect();
            summary["timings"] = Value::Object(seconds);
            summary["memory"] = json!({
                "peak_bytes": memory::peak_bytes(),
                "live_bytes": memory::live_bytes(),
            });
            if let Some(gpu) = &gpu_report {
                summary["gpu"] = json!({
                    "device": gpu.device,
                    "batches": gpu.batches,
                    "peak_device_bytes": gpu.peak,
                });
            }
        }
        summary["warnings"] = json!(logger::warnings());
        logger::print_result(&summary);
    }
}

fn encode(data: &[u8], width: usize, height: usize, format: image::ImageFormat) -> Vec<u8> {
    let img = image::RgbImage::from_raw(width as u32, height as u32, data.to_vec())
        .unwrap_or_else(|| internal("image buffer has the wrong size"));
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, format)
        .unwrap_or_else(|e| internal(format!("could not encode {format:?}: {e}")));
    out.into_inner()
}

/// Write the wall render and return the file's size in bytes.
fn write_png(path: &Path, data: &[u8], width: usize, height: usize) -> u64 {
    let img = image::RgbImage::from_raw(width as u32, height as u32, data.to_vec())
        .unwrap_or_else(|| internal("canvas has the wrong size"));
    img.save(path).unwrap_or_else(|e| write_failed(path, e));
    std::fs::metadata(path).map_or(0, |m| m.len())
}

fn write_failed(path: &Path, e: impl std::fmt::Display) -> ! {
    fatal(
        Diagnostic::new(
            Kind::WriteError,
            format!("could not write '{}': {e}", path.display()),
        )
        .path(path.display()),
    )
}
