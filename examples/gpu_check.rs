//! Verification harness for the GPU backend: solves the same prepared cells
//! on the CPU solver and on the GPU and compares the results.
//!
//! ```text
//! cargo run --release --example gpu_check -- [--rows N] [-p TOP_N DUP ROUNDS]
//!     [-R passes] [-k window] [-x exact] [-F weight] [-U weight] [-C candidates]
//!     [-t threshold] [--seed S] [--budget MIB] [--profile] [--no-cpu]
//!     [--diff N] [--only CELL] [--image PATH]
//! ```
//!
//! The cells are built from the image (`docs/example.png` unless `--image`
//! names another) the way the CLI builds them:
//! layout, resample plan, one band per block column, `BandView::gather`, and
//! the variance-based layer grid. Cell `i` (row-major) gets `Rng::new(seed, i)`.

fn main() {
    check::main();
}

mod check {
    use std::collections::HashSet;
    use std::time::{Duration, Instant};

    use rayon::prelude::*;

    use bannerify::cli::config::RefinementConfig;
    use bannerify::geometry::{BLOCK_SIDE, NTOP_HW, TOP_HW};
    use bannerify::gpu::GpuBackend;
    use bannerify::layout::{Dimension, Layout, ResizingMethod};
    use bannerify::pattern;
    use bannerify::resample::{Plan, PlanarU8};
    use bannerify::simd::Chunk;
    use bannerify::solver::cell::BandView;
    use bannerify::solver::variance::layer_grid;
    use bannerify::solver::{
        Backend, Plane, PreparedCell, Rng, Solution, SolveCfg, Stages, Workspace,
    };

    struct Opts {
        rows: usize,
        perturbations: Option<(usize, usize, usize)>,
        refinement: RefinementConfig,
        feature_weight: f32,
        purity_weight: f32,
        seed: u64,
        budget: Option<usize>,
        profile: bool,
        cpu: bool,
        diff: usize,
        only: Option<usize>,
        image: String,
    }

    fn parse() -> Opts {
        let mut o = Opts {
            rows: 40,
            perturbations: None,
            refinement: RefinementConfig {
                refinement_pass: 2,
                window_size: 2,
                error_threshold: 0.7,
                refinement_candidate: 5,
                exact_candidates: 20,
            },
            feature_weight: 0.5,
            purity_weight: 0.0,
            seed: 0,
            budget: None,
            profile: false,
            cpu: true,
            diff: 0,
            only: None,
            image: concat!(env!("CARGO_MANIFEST_DIR"), "/docs/example.png").into(),
        };
        let args: Vec<String> = std::env::args().skip(1).collect();
        let mut i = 0;
        let num = |i: usize| -> &str {
            args.get(i)
                .unwrap_or_else(|| panic!("missing value after {}", args[i - 1]))
        };
        while i < args.len() {
            let flag = args[i].as_str();
            match flag {
                "--rows" => o.rows = num(i + 1).parse().expect("--rows N"),
                "-p" => {
                    let v: Vec<usize> = (1..=3)
                        .map(|k| num(i + k).parse().expect("-p TOP_N DUP ROUNDS"))
                        .collect();
                    o.perturbations = (!v.contains(&0)).then(|| (v[0], v[1], v[2]));
                    i += 2;
                }
                "-R" => o.refinement.refinement_pass = num(i + 1).parse().expect("-R N"),
                "-k" => o.refinement.window_size = num(i + 1).parse().expect("-k N"),
                "-x" => o.refinement.exact_candidates = num(i + 1).parse().expect("-x N"),
                "-C" => o.refinement.refinement_candidate = num(i + 1).parse().expect("-C N"),
                "-t" => o.refinement.error_threshold = num(i + 1).parse().expect("-t F"),
                "-F" => o.feature_weight = num(i + 1).parse().expect("-F F"),
                "-U" => o.purity_weight = num(i + 1).parse().expect("-U F"),
                "--seed" => o.seed = num(i + 1).parse().expect("--seed S"),
                "--budget" => {
                    o.budget = Some(num(i + 1).parse::<usize>().expect("--budget MIB") << 20)
                }
                "--only" => o.only = Some(num(i + 1).parse().expect("--only CELL")),
                "--diff" => o.diff = num(i + 1).parse().expect("--diff N"),
                "--image" => o.image = num(i + 1).to_owned(),
                "--profile" => {
                    o.profile = true;
                    i -= 1;
                }
                "--no-cpu" => {
                    o.cpu = false;
                    i -= 1;
                }
                other => panic!("unknown flag {other}"),
            }
            i += 2;
        }
        o
    }

    /// Every banner cell of the image at `path` at `rows` rows, row-major.
    fn cells(path: &str, rows: usize, seed: u64) -> Vec<PreparedCell> {
        let rgb = image::open(path).expect("readable image").to_rgb8();
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        let source = PlanarU8::from_interleaved(rgb.as_raw(), w, h, 3);
        let layout = Layout::compute(
            w as u32,
            h as u32,
            Dimension::Row(rows),
            ResizingMethod::Fit,
        );
        let plan = Plan::with_window(
            w,
            h,
            layout.window,
            layout.target_width,
            layout.target_height,
        );
        let layers = layer_grid(&source, &layout, (4, 6));
        let (ox, oy) = layout.origin;

        let columns: Vec<Vec<(usize, [Plane; 3])>> = (0..layout.columns)
            .into_par_iter()
            .map(|col| {
                let strip = col * BLOCK_SIDE..(col + 1) * BLOCK_SIDE;
                let lo = strip.start.clamp(ox, ox + layout.target_width);
                let hi = strip.end.clamp(ox, ox + layout.target_width);
                let target_cols = (lo - ox)..(hi - ox);
                let (x0, x1) = plan.src_cols(&target_cols);
                let band = (!target_cols.is_empty())
                    .then(|| plan.band((x0, x1), target_cols.len()).resample(&source));
                let view = BandView::new(band.as_ref(), lo, oy, layout.pad.unwrap_or([0; 3]));
                (0..layout.rows)
                    .map(|row| {
                        let mut target = [Chunk::zeroed(); 3];
                        view.gather(row, col, &mut target);
                        (layers.get(row, col), target)
                    })
                    .collect()
            })
            .collect();

        let mut out = Vec::with_capacity(layout.rows * layout.columns);
        for row in 0..layout.rows {
            for (col, column) in columns.iter().enumerate() {
                let (n_layers, target) = column[row];
                let index = row * layout.columns + col;
                out.push(PreparedCell {
                    row,
                    n_layers,
                    target,
                    rng: Rng::new(seed, index as u64),
                });
            }
        }
        out
    }

    fn solve_cpu(cells: &[PreparedCell], cfg: &SolveCfg<'_>) -> Vec<Solution> {
        let max_layers = cells.iter().map(|c| c.n_layers).max().unwrap_or(0);
        let cand = cfg.refinement.refinement_candidate;
        cells
            .par_iter()
            .map_init(
                || Box::new(Workspace::new(max_layers, cand)),
                |ws, cell| {
                    ws.begin(cell.row);
                    *ws.target_mut() = cell.target;
                    ws.solve(cell.n_layers, cfg, &mut cell.rng.clone())
                },
            )
            .collect()
    }

    fn pixels(cells: &[PreparedCell]) -> f64 {
        cells
            .iter()
            .map(|c| if c.row == 0 { TOP_HW } else { NTOP_HW } as f64)
            .sum()
    }

    fn mean_de(solutions: &[Solution], pixels: f64) -> f64 {
        solutions.iter().map(|s| f64::from(s.error)).sum::<f64>() / pixels
    }

    fn same(a: &Solution, b: &Solution) -> bool {
        a.base == b.base && a.layers == b.layers
    }

    fn fraction_same(a: &[Solution], b: &[Solution]) -> f64 {
        a.iter().zip(b).filter(|(a, b)| same(a, b)).count() as f64 / a.len() as f64
    }

    fn print_stages(s: &Stages) {
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        println!(
            "gpu stages: greedy {:.1} ms, refine {:.1} ms, perturb {:.1} ms, feature {:.1} ms, oklab {:.1} ms",
            ms(s.greedy),
            ms(s.refine),
            ms(s.perturb),
            ms(s.feature),
            ms(s.oklab)
        );
    }

    pub fn main() {
        let o = parse();
        let patterns = pattern::load(&HashSet::new());
        let cfg = SolveCfg {
            patterns: &patterns,
            refinement: &o.refinement,
            perturbations: o.perturbations,
            feature_weight: o.feature_weight,
            purity_weight: o.purity_weight,
        };

        let t = Instant::now();
        let mut cells = cells(&o.image, o.rows, o.seed);
        if let Some(i) = o.only {
            cells = vec![cells.swap_remove(i)];
        }
        let px = pixels(&cells);
        println!(
            "{} cells (rows {}), prepared in {:.2} s; layers 4/5/6: {}/{}/{}",
            cells.len(),
            o.rows,
            t.elapsed().as_secs_f64(),
            cells.iter().filter(|c| c.n_layers == 4).count(),
            cells.iter().filter(|c| c.n_layers == 5).count(),
            cells.iter().filter(|c| c.n_layers == 6).count(),
        );
        println!(
            "config: -R {} -k {} -x {} -C {} -t {} -F {} -U {} -p {:?}",
            o.refinement.refinement_pass,
            o.refinement.window_size,
            o.refinement.exact_candidates,
            o.refinement.refinement_candidate,
            o.refinement.error_threshold,
            o.feature_weight,
            o.purity_weight,
            o.perturbations
        );

        let cpu = o.cpu.then(|| {
            let t = Instant::now();
            let s = solve_cpu(&cells, &cfg);
            (s, t.elapsed())
        });

        let t = Instant::now();
        let mut backend = GpuBackend::new().unwrap_or_else(|e| panic!("{e}"));
        if let Some(b) = o.budget {
            backend = backend.with_memory_budget(b);
        }
        let setup = t.elapsed();
        let t = Instant::now();
        let gpu = backend.solve(&cells, &cfg);
        let gpu_time = t.elapsed();
        let stages = backend.stages();
        let batches = backend.batches();
        let t = Instant::now();
        let gpu2 = backend.solve(&cells, &cfg);
        let gpu2_time = t.elapsed();
        let deterministic = gpu
            .iter()
            .zip(&gpu2)
            .all(|(a, b)| same(a, b) && a.error.to_bits() == b.error.to_bits());

        println!(
            "device: {}, setup {:.0} ms, peak device allocation {:.0} MiB, {} batches per run",
            backend.device_name(),
            setup.as_secs_f64() * 1e3,
            backend.peak_allocated() as f64 / (1 << 20) as f64,
            batches
        );
        let gpu_de = mean_de(&gpu, px);
        println!(
            "gpu: {:.3} s (second run {:.3} s), mean dE/px {:.6}",
            gpu_time.as_secs_f64(),
            gpu2_time.as_secs_f64(),
            gpu_de
        );
        print_stages(&stages);
        println!(
            "gpu runs bit-identical: {}",
            if deterministic { "yes" } else { "NO" }
        );

        if let Some((cpu, cpu_time)) = &cpu {
            let cpu_de = mean_de(cpu, px);
            println!(
                "cpu: {:.3} s, mean dE/px {:.6}",
                cpu_time.as_secs_f64(),
                cpu_de
            );
            println!(
                "relative difference (gpu - cpu) / cpu: {:+.4}%",
                (gpu_de - cpu_de) / cpu_de * 100.0
            );
            println!("identical cells: {:.2}%", fraction_same(cpu, &gpu) * 100.0);
            // On identical cells the composites are bit-identical, so the error
            // difference is the OKLab conversion's alone.
            let rel: Vec<f64> = cpu
                .iter()
                .zip(&gpu)
                .filter(|(c, g)| same(c, g))
                .map(|(c, g)| {
                    ((f64::from(g.error) - f64::from(c.error)) / f64::from(c.error)).abs()
                })
                .collect();
            let exact = cpu
                .iter()
                .zip(&gpu)
                .filter(|(c, g)| same(c, g) && c.error.to_bits() == g.error.to_bits())
                .count();
            println!(
                "identical cells, error difference: mean {:.2e}, max {:.2e} relative; bit-equal errors {:.1}%",
                rel.iter().sum::<f64>() / rel.len().max(1) as f64,
                rel.iter().copied().fold(0.0, f64::max),
                exact as f64 / cpu.len() as f64 * 100.0
            );
            let better = cpu
                .iter()
                .zip(&gpu)
                .filter(|(c, g)| g.error < c.error)
                .count();
            let worse = cpu
                .iter()
                .zip(&gpu)
                .filter(|(c, g)| g.error > c.error)
                .count();
            println!("cells where gpu error is lower: {better}, higher: {worse}");
            let mut shown = 0;
            for (i, (c, g)) in cpu.iter().zip(&gpu).enumerate() {
                if shown >= o.diff {
                    break;
                }
                if !same(c, g) {
                    shown += 1;
                    println!(
                        "  cell {i} (row {}, {} layers): cpu base {} {:?} err {:.6}; gpu base {} {:?} err {:.6}",
                        cells[i].row,
                        cells[i].n_layers,
                        c.base,
                        c.layers,
                        c.error,
                        g.base,
                        g.layers,
                        g.error
                    );
                }
            }
        }

        if o.profile {
            backend.set_profiling(true);
            backend.solve(&cells, &cfg);
            let times = backend.kernel_times();
            let total: Duration = times.iter().map(|t| t.1).sum();
            println!(
                "per-kernel GPU time (profiled run, {:.3} s total):",
                total.as_secs_f64()
            );
            for (name, t) in times {
                println!(
                    "  {name:16} {:9.1} ms  {:5.1}%",
                    t.as_secs_f64() * 1e3,
                    t.as_secs_f64() / total.as_secs_f64() * 100.0
                );
            }
        }
    }
}
