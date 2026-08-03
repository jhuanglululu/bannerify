//! Application entry point: wire the CLI to the pipeline.
//!
//! Phase 1 ships the front half: CLI + config, wall geometry, and the streamed
//! lanczos-3 resample onto the banner wall. The solver, block matching and HTML
//! export land in phase 2, so a plain run stops after validation with a clear
//! message; `--debug` runs the resample, prints per-stage wall times and dumps
//! the resized intermediate next to the output.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use clap::Parser;
use colored::Colorize;

use crate::cli::Args;
use crate::cli::config::Config;
use crate::layout::Layout;
use crate::logger;
use crate::logger::{error_out, info};
use crate::resample::{Options, Plan, PlanarF32Sink, PlanarU8, run};

/// Parse arguments, validate, and run whatever phase 1 can run.
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

    if !config.debug {
        // Phase 1 has no solver: refuse rather than write a misleading file.
        logger::error_print(format!(
            "banner conversion is not implemented yet — phase 1 ships the resampler only.\n       \
             re-run with '{}' to write the resized intermediate and per-stage timings.",
            "--debug".yellow()
        ));
        std::process::exit(1);
    }

    run_debug(&config);
}

/// The `--debug` path: resample, dump the intermediate, print stage timings.
fn run_debug(config: &Config) {
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
    let src = PlanarU8::from_interleaved(rgb.as_raw(), src_w, src_h, 3);
    drop(rgb);
    let t_decode = t.elapsed();

    // ---- layout + weights --------------------------------------------------
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
        Options::default(),
    );
    let t_plan = t.elapsed();

    info!(
        "grid: {}x{} blocks ({} banners)",
        layout.columns,
        layout.rows + 1,
        layout.columns * layout.rows
    );
    info!(
        "wall: {}x{} px, resampling {}x{} -> {}x{}",
        layout.wall_width,
        layout.wall_height,
        src_w,
        src_h,
        layout.target_width,
        layout.target_height
    );

    // ---- resample ----------------------------------------------------------
    let sink = PlanarF32Sink::new(layout.target_width, layout.target_height, 3);
    let t = Instant::now();
    run(&plan, &src, &sink);
    let t_resample = t.elapsed();
    let planes = sink.into_planes();

    // ---- dump the resized intermediate ------------------------------------
    let t = Instant::now();
    let dump_path = resized_path(&config.output);
    let canvas = compose(&planes, &layout);
    let image =
        image::RgbImage::from_raw(layout.wall_width as u32, layout.wall_height as u32, canvas)
            .unwrap_or_else(|| error_out!("internal error: composed buffer has the wrong size"));
    image.save(&dump_path).unwrap_or_else(|e| {
        error_out!(
            "could not write '{}': {}",
            dump_path.display().to_string().yellow(),
            e.to_string().red()
        );
    });
    let t_encode = t.elapsed();

    info!(
        "wrote resized intermediate to '{}'",
        dump_path.display().to_string().yellow()
    );

    // ---- timings (printed outside every timed region) ----------------------
    let mpix = (layout.target_width * layout.target_height) as f64 / 1e6;
    debug_line("decode", t_decode, None);
    debug_line("layout+weights", t_plan, None);
    debug_line(
        "resample",
        t_resample,
        Some(format!(
            "{:.1} MPix/s ({:.1} MPix/s summed over 3 channels)",
            mpix / t_resample.as_secs_f64(),
            3.0 * mpix / t_resample.as_secs_f64()
        )),
    );
    debug_line("compose+encode", t_encode, None);
    debug_line(
        "total",
        t_decode + t_plan + t_resample + t_encode,
        Some("solver/export not implemented (phase 2)".to_string()),
    );
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

/// `<output-stem>.resized.png`, next to the output path.
fn resized_path(output: &Path) -> PathBuf {
    let stem = output
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "output".to_string());
    output.with_file_name(format!("{stem}.resized.png"))
}

/// Planar f32 planes -> interleaved RGB8 wall, padding included.
///
/// This is the encode edge (the mirror of `from_interleaved`), and the only
/// place a wall-sized buffer exists — the resampler itself never holds one.
fn compose(planes: &[Vec<f32>], layout: &Layout) -> Vec<u8> {
    let (tw, th) = (layout.target_width, layout.target_height);

    if !layout.is_padded() {
        let mut out = vec![0u8; tw * th * 3];
        for (c, plane) in planes.iter().enumerate() {
            for (i, &v) in plane.iter().enumerate() {
                out[i * 3 + c] = to_u8(v);
            }
        }
        return out;
    }

    let pad = layout.pad.unwrap_or([0, 0, 0]);
    let (ww, wh) = (layout.wall_width, layout.wall_height);
    let mut out = pad.repeat(ww * wh);
    let (ox, oy) = layout.origin;
    for (c, plane) in planes.iter().enumerate() {
        for y in 0..th {
            let dst = ((oy + y) * ww + ox) * 3;
            for x in 0..tw {
                out[dst + x * 3 + c] = to_u8(plane[y * tw + x]);
            }
        }
    }
    out
}

/// Resampled sample -> display byte (lanczos ringing is clipped here, not
/// inside the pipeline — the f32 intermediate keeps the overshoot).
#[inline]
fn to_u8(v: f32) -> u8 {
    v.clamp(0.0, 255.0).round() as u8
}
