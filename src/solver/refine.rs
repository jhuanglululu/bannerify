use std::borrow::Cow;

use wide::f32x8;

use crate::banner::{Banner, BannerResult, PrefixPatternCache, SuffixPatternCache};
use crate::cli::config::RefinementConfig;
use crate::color::{COLORS_F32, COLORS_WSQ_SUM, NUM_COLORS, W_PERCEPTUAL};
use crate::macros::uninit;
use crate::solver::build::{build_empty_suffix, build_prefix, build_suffix};

type Candidate<'a, const HW: usize> = (Vec<(usize, usize)>, Cow<'a, SuffixPatternCache<HW>>);

pub fn refinement_pass<const HW: usize>(
    banner: &Banner<HW>,
    result: &mut BannerResult,
    prefixes: &mut [PrefixPatternCache<HW>],
    config: &RefinementConfig,
    patterns: &[[f32; HW]],
) -> Vec<SuffixPatternCache<HW>> {
    let mut suffixes: Vec<SuffixPatternCache<HW>> = uninit!(banner.n_layers + 1);
    build_empty_suffix(&mut suffixes[banner.n_layers]);

    for pass in 0..config.refinement_pass {
        let old_result = result.clone();

        for start in (0..banner.n_layers).rev() {
            refine_window(
                banner,
                result,
                prefixes,
                &mut suffixes,
                start,
                config,
                patterns,
            );
        }

        for layer in 0..banner.n_layers - 1 {
            let (left, right) = prefixes.split_at_mut(layer + 1);
            build_prefix(
                patterns,
                &left[layer],
                &mut right[0],
                result.patterns[layer].0,
                result.patterns[layer].1,
            );
        }

        if &old_result == result {
            break;
        }
    }

    suffixes
}

fn refine_window<const HW: usize>(
    banner: &Banner<HW>,
    result: &mut BannerResult,
    prefixes: &mut [PrefixPatternCache<HW>],
    suffixes: &mut [SuffixPatternCache<HW>],
    start_layer: usize,
    config: &RefinementConfig,
    patterns: &[[f32; HW]],
) {
    let cand_size = config.refine_candidate;
    let target = &banner.target;

    let mut curr_cand: Vec<Candidate<HW>> =
        vec![(Vec::new(), Cow::Borrowed(&suffixes[start_layer + 1])); cand_size];
    let mut prev_cands: Vec<Candidate<HW>> =
        vec![(Vec::new(), Cow::Borrowed(&suffixes[start_layer + 1])); cand_size];

    let mut n_cand = 1;

    'sliding_loop: for k in 0..config.window_size {
        let layer_idx = start_layer - k;

        let mut best_errs: Vec<(f32, usize, usize, usize)> =
            vec![(f32::INFINITY, 0, 0, 0); cand_size];

        let prefix = prefixes[layer_idx].2;

        for (cand_idx, cand) in prev_cands[0..n_cand].iter().enumerate() {
            let sfx_mul = &cand.1.2[3];
            let sfx_add = &cand.1.2[0..3];

            for (p_idx, pattern) in patterns.iter().enumerate() {
                let mut res2 = [0.0f32; 3];
                let mut res_2a = [0.0f32; 3];
                let mut eff_alpha2 = 0.0f32;

                for ch in 0..3 {
                    let mut res2_acc = f32x8::ZERO;
                    let mut res_a_acc = f32x8::ZERO;
                    let mut ea2_acc = f32x8::ZERO;

                    for px in (0..HW).step_by(8) {
                        let pre = f32x8::from(&prefix[ch][px..px + 8]);
                        let tar = f32x8::from(&target[ch][px..px + 8]);
                        let alp = f32x8::from(&pattern[px..px + 8]);
                        let mul = f32x8::from(&sfx_mul[px..px + 8]);
                        let add = f32x8::from(&sfx_add[ch][px..px + 8]);

                        let eff_alp = alp * mul;
                        let res = pre * (f32x8::ONE - alp) * mul + add - tar;

                        res2_acc += res * res;
                        res_a_acc += res * eff_alp;
                        if ch == 0 {
                            ea2_acc += eff_alp * eff_alp;
                        }
                    }

                    res2[ch] = res2_acc.reduce_add();
                    res_2a[ch] = 2.0 * res_a_acc.reduce_add();
                    if ch == 0 {
                        eff_alpha2 = ea2_acc.reduce_add();
                    }
                }

                for c_idx in 0..NUM_COLORS {
                    let c = COLORS_F32[c_idx];
                    let err = W_PERCEPTUAL[0] * (res2[0] + res_2a[0] * c[0])
                        + W_PERCEPTUAL[1] * (res2[1] + res_2a[1] * c[1])
                        + W_PERCEPTUAL[2] * (res2[2] + res_2a[2] * c[2])
                        + eff_alpha2 * COLORS_WSQ_SUM[c_idx];

                    if err < best_errs[cand_size - 1].0 {
                        best_errs[cand_size - 1] = (err, cand_idx, p_idx, c_idx);
                        best_errs.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                    }
                }
            }
        }

        n_cand = if config.error_threshold > 0.0 {
            let bound = best_errs[0].0 / config.error_threshold;
            best_errs.partition_point(|&c| c.0 <= bound)
        } else {
            cand_size
        };

        for cand_idx in 0..n_cand {
            let cand = best_errs[cand_idx];

            curr_cand[cand_idx].0 = prev_cands[cand.1].0.clone();
            curr_cand[cand_idx].0.push((cand.2, cand.3));

            if layer_idx == 0 {
                break 'sliding_loop;
            }

            build_suffix(
                patterns,
                curr_cand[cand_idx].1.to_mut(),
                prev_cands[cand.1].1.as_ref(),
                cand.2,
                cand.3,
            );
        }

        if k < config.window_size - 1 {
            (curr_cand, prev_cands) = (prev_cands, curr_cand)
        }
    }

    for (layer, pc) in curr_cand[0].0.iter().enumerate() {
        result.patterns[start_layer - layer] = *pc;
    }

    let (left, right) = suffixes.split_at_mut(start_layer + 1);
    build_suffix(
        patterns,
        &mut left[start_layer],
        &right[0],
        result.patterns[start_layer].0,
        result.patterns[start_layer].1,
    );

    let pfx_update_start = (start_layer + 1).saturating_sub(config.window_size);
    let pfx_update_end = start_layer.saturating_sub(1);

    for layer_idx in pfx_update_start..pfx_update_end {
        let (left, right) = prefixes.split_at_mut(layer_idx + 1);
        build_prefix(
            patterns,
            &left[layer_idx],
            &mut right[0],
            result.patterns[layer_idx].0,
            result.patterns[layer_idx].1,
        );
    }
}
