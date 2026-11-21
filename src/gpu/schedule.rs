//! The step schedule as a sequence of dispatches: greedy fill, windowed
//! refinement and the final error readout, each for a whole batch of slots.
//!
//! The host knows the whole schedule in advance, because every slot of a
//! batch has the same layer count and the loop bounds of the CPU solver
//! (`refinement_pass`, the start layers, the window steps) depend only on that
//! count and the configuration. What the CPU decides per cell inside those
//! loops (beam survivors, chosen layers, the base dye) is decided per slot by
//! the kernels, so nothing is read back until a refine has finished.
//!
//! Every pass runs for every slot. The CPU stops a cell once a pass changes
//! nothing; a pass is a function of the solution it starts from, so repeating
//! an unchanging pass reproduces it and the result is the same.

use crate::cli::config::RefinementConfig;

use super::kernel::Kernel;
use super::state::{PLANE, Params, Shape};
use super::wgpu_device::Encoder;

/// Plane positions, in `float4` units: the thread count per slot of the
/// element-wise kernels.
const PL: usize = PLANE / 4;

/// The parameter block every dispatch of a batch starts from.
pub(super) fn base_params(
    shape: &Shape,
    cfg: &RefinementConfig,
    lambda: f32,
    nu: f32,
    mu: f32,
    n_slots: usize,
    nl: usize,
) -> Params {
    Params {
        n_slots: n_slots as u32,
        nl: nl as u32,
        n_pat: shape.n_pat as u32,
        cand: shape.cand as u32,
        n_exact: shape.n_exact as u32,
        window: shape.window as u32,
        lambda,
        nu,
        mu,
        threshold: cfg.error_threshold,
        ..Params::default()
    }
}

/// Greedy fill (`greedy::fill`): base dye, then one layer at a time, leaving
/// the slot's layers, base and `prefixes[0..=nl]` filled in.
pub(super) fn greedy(enc: &mut Encoder<'_>, p: &Params) {
    let n = p.n_slots as usize;
    let nl = p.nl;
    enc.dispatch(Kernel::GreedyBase, p, n);
    enc.dispatch(Kernel::PrefixRange, &Params { flag: 1, ..*p }, n * PL);
    for layer in 0..nl {
        let q = Params {
            layer,
            step: layer + 1,
            ..*p
        };
        enc.dispatch(Kernel::GreedyMoments, &q, n * p.n_pat as usize);
        enc.dispatch(Kernel::GreedyPick, &q, n);
        enc.dispatch(Kernel::PrefixRange, &q, n * PL);
    }
}

/// Windowed refinement (`refine::refine`) of every slot from its current base
/// and layers, leaving `prefixes[nl]` the refined composite.
pub(super) fn refine(enc: &mut Encoder<'_>, p: &Params, cfg: &RefinementConfig) {
    let n = p.n_slots as usize;
    let rebuild = Params {
        flag: 1,
        layer: 0,
        step: p.nl,
        ..*p
    };
    enc.dispatch(Kernel::IdentitySuffix, p, n * PL);
    enc.dispatch(Kernel::PrefixRange, &rebuild, n * PL);

    for _ in 0..cfg.refinement_pass {
        for start in (0..p.nl).rev() {
            window(enc, p, start);
        }
        if cfg.exact_candidates > 0 {
            enc.dispatch(Kernel::ReoptExact, p, n * 16);
            enc.dispatch(Kernel::ReoptPick, p, n);
        } else {
            enc.dispatch(Kernel::ReoptClosed, p, n);
        }
        enc.dispatch(Kernel::PrefixRange, &rebuild, n * PL);
    }
}

/// One window (`refine::refine_window`) ending at `start`.
fn window(enc: &mut Encoder<'_>, p: &Params, start: u32) {
    let n = p.n_slots as usize;
    let window = p.window;
    let mut generation = 0;
    // The generation holding the winning chain, and its length.
    let mut last = (0, 0);

    for k in 0..window {
        let layer = start - k;
        let q = Params {
            layer,
            step: k,
            generation,
            start,
            ..*p
        };
        let entries = n * (p.cand * p.n_pat) as usize;
        enc.dispatch(Kernel::Rung1, &q, entries);
        if p.n_exact > 0 {
            enc.dispatch(Kernel::ShortlistTop, &q, entries);
            enc.dispatch(Kernel::Rung2, &q, n * p.n_exact as usize);
        } else {
            enc.dispatch(Kernel::SelectTop, &q, n);
        }
        enc.dispatch(Kernel::BeamStep, &q, n);

        // Layer 0 has nothing below it, and after the last step no later step
        // reads the candidates' suffixes: either way the window ends here.
        if layer == 0 || k == window - 1 {
            last = (generation ^ 1, k + 1);
            break;
        }
        enc.dispatch(Kernel::BeamSuffix, &q, n * p.cand as usize * PL);
        generation ^= 1;
    }

    let q = Params {
        generation: last.0,
        step: last.1,
        start,
        ..*p
    };
    enc.dispatch(Kernel::CommitWindow, &q, n);
    enc.dispatch(Kernel::ChainSuffix, &q, n * PL);

    // The prefixes the next window reads; see `refine::refine_window`.
    let from = (start + 1).saturating_sub(window);
    let to = start.saturating_sub(1);
    if from < to {
        let q = Params {
            layer: from,
            step: to,
            ..*p
        };
        enc.dispatch(Kernel::PrefixRange, &q, n * PL);
    }
}

/// The ranking and pure-target errors of every slot's `prefixes[nl]`.
pub(super) fn finish(enc: &mut Encoder<'_>, p: &Params) {
    enc.dispatch(Kernel::Finish, p, p.n_slots as usize);
}
