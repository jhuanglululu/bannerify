//! bannerify CLI.
//!
//! Phase 1 exposes the streamed resampler: `bannerify resize` decodes an image,
//! resizes it with the band pipeline and either encodes a PNG or discards the
//! result (pure resample timing). Per-stage wall times are printed after all
//! timed regions have finished.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use bannerify::resample::{ChecksumSink, Options, Plan, PlanarF32Sink, PlanarU8, run};
use clap::{Args, Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "bannerify",
    version,
    about = "Minecraft banner approximation of images"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Resize an image with the streamed lanczos-3 resampler.
    Resize(ResizeArgs),
}

#[derive(Args)]
struct ResizeArgs {
    /// Input image (any format the `image` crate decodes).
    input: PathBuf,
    /// Output PNG path (not written with `--discard`).
    output: PathBuf,
    /// Target width in pixels.
    #[arg(long)]
    width: u32,
    /// Target height in pixels.
    #[arg(long)]
    height: u32,
    /// Skip output allocation and encoding — pure resample timing.
    #[arg(long)]
    discard: bool,
    /// Output rows per band (parallelism granularity).
    #[arg(long, default_value_t = 32)]
    band_rows: usize,
}

fn secs(d: Duration) -> f64 {
    d.as_secs_f64()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Command::Resize(args) => resize(args),
    }
}

fn resize(args: ResizeArgs) -> Result<(), Box<dyn std::error::Error>> {
    let (dst_w, dst_h) = (args.width as usize, args.height as usize);

    // ---- decode (including the one interleaved -> planar conversion) --------
    let t = Instant::now();
    let img = image::open(&args.input)?.to_rgb8();
    let (src_w, src_h) = (img.width() as usize, img.height() as usize);
    let src = PlanarU8::from_interleaved(img.as_raw(), src_w, src_h, 3);
    drop(img);
    let t_decode = t.elapsed();

    // ---- plan / weights ----------------------------------------------------
    let t = Instant::now();
    let plan = Plan::new(
        src_w,
        src_h,
        dst_w,
        dst_h,
        Options {
            band_rows: args.band_rows,
        },
    );
    let t_plan = t.elapsed();

    // ---- resample ----------------------------------------------------------
    let (t_resample, collected, checksum) = if args.discard {
        let sink = ChecksumSink::new();
        let t = Instant::now();
        run(&plan, &src, &sink);
        (t.elapsed(), None, Some(sink.checksum()))
    } else {
        let sink = PlanarF32Sink::new(dst_w, dst_h, 3);
        let t = Instant::now();
        run(&plan, &src, &sink);
        (t.elapsed(), Some(sink.into_planes()), None)
    };

    // ---- encode ------------------------------------------------------------
    let t = Instant::now();
    if let Some(planes) = collected {
        // Planar -> interleaved at the encode edge, the mirror of decode.
        let mut buf = vec![0u8; dst_w * dst_h * 3];
        for (c, plane) in planes.iter().enumerate() {
            for (i, &v) in plane.iter().enumerate() {
                buf[i * 3 + c] = v.clamp(0.0, 255.0).round() as u8;
            }
        }
        let out = image::RgbImage::from_raw(args.width, args.height, buf)
            .ok_or("output buffer size mismatch")?;
        out.save(&args.output)?;
    }
    let t_encode = t.elapsed();

    let mpix = (dst_w * dst_h) as f64 / 1e6;
    println!("input     {src_w}x{src_h}  ->  output {dst_w}x{dst_h} (3 channels)");
    println!("decode    {:8.3} s", secs(t_decode));
    println!("plan      {:8.3} s", secs(t_plan));
    println!(
        "resample  {:8.3} s   {:.1} MPix/s ({:.1} MPix/s summed over channels)",
        secs(t_resample),
        mpix / secs(t_resample),
        3.0 * mpix / secs(t_resample)
    );
    if args.discard {
        println!("encode    (skipped: --discard)");
        println!("checksum  {}", checksum.unwrap_or(0));
    } else {
        println!("encode    {:8.3} s", secs(t_encode));
    }
    Ok(())
}
