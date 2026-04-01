use wide::f32x8;

use crate::banner::{Banner, BannerResult, PrefixPatternCache};
use crate::color::{COLORS_F32, COLORS_WSQ_SUM, NUM_COLORS, W_PERCEPTUAL};
use crate::macros::uninit;
use crate::solver::build::build_prefix;

#[allow(clippy::needless_range_loop)]
pub fn intial_fill<const HW: usize>(
    banner: &Banner<HW>,
    patterns: &[[f32; HW]],
    alpha2: &[f32],
) -> (BannerResult, Vec<PrefixPatternCache<HW>>) {
    let mut base = 0;
    {
        let mut min_base_err = f32::INFINITY;
        let mut t2 = [0.0_f32; 3];
        let mut n2t = [0.0_f32; 3];

        for ch_idx in 0..3 {
            let channel = banner.target[ch_idx];
            let mut t2_acc = f32x8::ZERO;
            let mut n2t_acc = f32x8::ZERO;

            for px_idx in (0..HW).step_by(8) {
                let px = f32x8::from(&channel[px_idx..px_idx + 8]);
                t2_acc += px * px;
                n2t_acc += px;
            }

            t2[ch_idx] = t2_acc.reduce_add();
            n2t[ch_idx] = -2.0_f32 * n2t_acc.reduce_add();
        }

        for c_idx in 0..NUM_COLORS {
            let color = COLORS_F32[c_idx];
            let r = color[0];
            let g = color[1];
            let b = color[2];

            let r_err = W_PERCEPTUAL[0] * (t2[0] + n2t[0] * r);
            let g_err = W_PERCEPTUAL[1] * (t2[1] + n2t[1] * g);
            let b_err = W_PERCEPTUAL[2] * (t2[2] + n2t[2] * b);
            let c_err = r_err + g_err + b_err + HW as f32 * COLORS_WSQ_SUM[c_idx];

            if c_err < min_base_err {
                base = c_idx;
                min_base_err = c_err;
            }
        }
    }

    let mut b_patterns = Vec::with_capacity(banner.n_layers);
    let mut prefixes: Vec<(usize, usize, [[f32; HW]; 3])> = uninit!(banner.n_layers);

    prefixes[0].2[0] = [COLORS_F32[base][0]; HW];
    prefixes[0].2[1] = [COLORS_F32[base][1]; HW];
    prefixes[0].2[2] = [COLORS_F32[base][2]; HW];

    for layer in 0..banner.n_layers {
        let mut best: (usize, usize) = (0, 0);
        let mut min_err = f32::INFINITY;

        for (p_idx, pattern) in patterns.iter().enumerate() {
            let mut res2 = [0.0_f32; 3];
            let mut res_2a = [0.0_f32; 3];

            // compute residual
            for ch_idx in 0..3 {
                let mut res2_acc = f32x8::ZERO;
                let mut res_a_acc = f32x8::ZERO;

                for px in (0..HW).step_by(8) {
                    let alp = f32x8::from(&pattern[px..px + 8]);
                    let inv_alp = f32x8::ONE - alp;

                    let pre_px = f32x8::from(&prefixes[layer].2[ch_idx][px..px + 8]);
                    let tar_px = f32x8::from(&banner.target[ch_idx][px..px + 8]);
                    let res = pre_px * inv_alp - tar_px;

                    res2_acc += res * res;
                    res_a_acc += res * alp;
                }

                res2[ch_idx] = res2_acc.reduce_add();
                res_2a[ch_idx] = 2.0_f32 * res_a_acc.reduce_add();
            }

            for c_idx in 0..NUM_COLORS {
                let r = COLORS_F32[c_idx][0];
                let g = COLORS_F32[c_idx][1];
                let b = COLORS_F32[c_idx][2];

                let err_r = W_PERCEPTUAL[0] * (res2[0] + res_2a[0] * r);
                let err_g = W_PERCEPTUAL[1] * (res2[1] + res_2a[1] * g);
                let err_b = W_PERCEPTUAL[2] * (res2[2] + res_2a[2] * b);

                let err = err_r + err_g + err_b + alpha2[p_idx] * COLORS_WSQ_SUM[c_idx];

                if err < min_err {
                    best = (p_idx, c_idx);
                    min_err = err;
                }
            }
        }

        b_patterns.push(best);

        let (left, right) = prefixes.split_at_mut(layer + 1);
        if layer != banner.n_layers - 1 {
            build_prefix(patterns, &left[layer], &mut right[0], best.0, best.1);
        }
    }

    (BannerResult::new(base, b_patterns), prefixes)
}
