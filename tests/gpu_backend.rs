//! The GPU backend against the CPU solver, on the same prepared cells.

use std::collections::HashSet;

use bannerify::cli::config::RefinementConfig;
use bannerify::geometry::TOP_HW;
use bannerify::gpu::GpuBackend;
use bannerify::pattern;
use bannerify::simd::Chunk;
use bannerify::solver::{Backend, PreparedCell, Rng, SolveCfg, Workspace};

/// Patches with structure: two colour regions split at a cell-dependent column,
/// a ramp, and a little deterministic noise, on alternating banner rows and
/// layer counts 4 to 6.
fn cells() -> Vec<PreparedCell> {
    let mut noise = 0x9E37_79B9_7F4A_7C15_u64;
    (0..18)
        .map(|cell| {
            let mut target = [Chunk::zeroed(); 3];
            let split = 3.0 + (cell % 7) as f32 * 2.0;
            for i in 0..TOP_HW {
                let (x, y) = ((i % 20) as f32, (i / 20) as f32);
                noise ^= noise << 13;
                noise ^= noise >> 7;
                noise ^= noise << 17;
                let jitter = (noise % 9) as f32 - 4.0;
                let px = if x < split {
                    [200.0 - 3.0 * y, 40.0 + 5.0 * x, 60.0 + cell as f32 * 9.0]
                } else {
                    [20.0 + 2.0 * x, 90.0 + 3.0 * y, 230.0 - 6.0 * x]
                };
                for (ch, plane) in target.iter_mut().enumerate() {
                    plane[i] = (px[ch] + jitter).clamp(0.0, 255.0);
                }
            }
            PreparedCell {
                row: cell % 3,
                n_layers: 4 + cell % 3,
                target,
                rng: Rng::new(7, cell as u64),
            }
        })
        .collect()
}

fn refinement(
    passes: usize,
    window: usize,
    exact: usize,
    cand: usize,
    threshold: f32,
) -> RefinementConfig {
    RefinementConfig {
        refinement_pass: passes,
        window_size: window,
        error_threshold: threshold,
        refinement_candidate: cand,
        exact_candidates: exact,
    }
}

/// Every configuration axis the backend branches on: exact rung on and off,
/// feature weight zero and not, purity weight zero and not, gradient weight
/// zero and not, window length, beam width, threshold off, refinement off, and
/// perturbation rounds.
///
/// The GPU kernels run in fast math, so a near-tie may resolve to another
/// banner than the CPU's; what must hold is the quality. A wrong kernel shows
/// up as a total error far from the CPU solver's.
#[test]
fn the_gpu_backend_matches_the_cpu_solver_in_quality() {
    let mut gpu = match GpuBackend::new() {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("skipped: {e}");
            return;
        }
    };
    let patterns = pattern::load(&HashSet::new());
    let cells = cells();

    let cases = [
        (
            "defaults",
            refinement(2, 2, 20, 5, 0.7),
            None,
            0.5,
            0.0,
            0.0,
        ),
        (
            "no exact rung",
            refinement(2, 2, 0, 5, 0.7),
            None,
            0.5,
            0.0,
            0.0,
        ),
        (
            "no feature map",
            refinement(2, 2, 20, 5, 0.7),
            None,
            0.0,
            0.0,
            0.0,
        ),
        (
            "purity term",
            refinement(2, 2, 20, 5, 0.7),
            None,
            0.5,
            0.5,
            0.0,
        ),
        (
            "purity term, no feature map",
            refinement(2, 2, 20, 5, 0.7),
            None,
            0.0,
            0.5,
            0.0,
        ),
        (
            "long window, no threshold",
            refinement(1, 3, 8, 3, 0.0),
            None,
            0.5,
            0.0,
            0.0,
        ),
        (
            "greedy only",
            refinement(0, 2, 20, 5, 0.7),
            None,
            0.5,
            0.0,
            0.0,
        ),
        (
            "perturbation",
            refinement(2, 2, 20, 5, 0.7),
            Some((2, 2, 2)),
            0.5,
            0.0,
            0.0,
        ),
        (
            "gradient term",
            refinement(2, 2, 20, 5, 0.7),
            None,
            0.5,
            0.0,
            1.0,
        ),
        (
            "gradient and purity terms, no feature map",
            refinement(2, 2, 20, 5, 0.7),
            None,
            0.0,
            0.5,
            1.0,
        ),
    ];

    for (name, refinement, perturbations, feature_weight, purity_weight, gradient_weight) in &cases
    {
        let cfg = SolveCfg {
            patterns: &patterns,
            refinement,
            perturbations: *perturbations,
            feature_weight: *feature_weight,
            purity_weight: *purity_weight,
            gradient_weight: *gradient_weight,
        };
        let expected: Vec<_> = cells
            .iter()
            .map(|cell| {
                let mut ws = Workspace::new(cell.n_layers, refinement.refinement_candidate);
                ws.begin(cell.row);
                *ws.target_mut() = cell.target;
                ws.solve(cell.n_layers, &cfg, &mut cell.rng.clone())
            })
            .collect();
        let got = gpu.solve(&cells, &cfg);

        assert_eq!(got.len(), expected.len());
        let total =
            |s: &[bannerify::solver::Solution]| s.iter().map(|s| s.error as f64).sum::<f64>();
        let (want, have) = (total(&expected), total(&got));
        let rel = (have - want) / want;
        assert!(
            rel.abs() < 0.05,
            "{name}: total error {have} vs the CPU's {want} ({:+.2}%)",
            rel * 100.0
        );
    }
}
