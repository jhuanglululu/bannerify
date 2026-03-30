use wide::f32x8;

use crate::color::{COLORS_F32, COLORS_WSQ_SUM, W_PERCEPTUAL};
use crate::solver::distance::nearest_color;

#[inline]
#[allow(clippy::uninit_assumed_init)]
pub fn optimal_color<const HW: usize, const HW_DIV_8: usize>(
    prefix: &[[f32; HW]; 3], // 3*HW
    target: &[[f32; HW]; 3], // 3*HW
    alpha: &[f32; HW],       // HW
    alp2: f32,
) -> (usize, f32) {
    let mut res2_acc = [f32x8::ZERO; 3];
    let mut res_alp_acc = [f32x8::ZERO; 3];

    for ch in 0..3 {
        for px in (0..HW).step_by(8) {
            let pre = f32x8::from(&prefix[ch][px..px + 8]);
            let tar = f32x8::from(&target[ch][px..px + 8]);
            let alp = f32x8::from(&alpha[px..px + 8]);

            let res = tar - pre + pre * alp;

            res2_acc[ch] += res * res;
            res_alp_acc[ch] += res * alp;
        }
    }

    let res_alp_r = res_alp_acc[0].reduce_add();
    let res_alp_g = res_alp_acc[1].reduce_add();
    let res_alp_b = res_alp_acc[2].reduce_add();

    let opt_r = res_alp_r / alp2;
    let opt_g = res_alp_g / alp2;
    let opt_b = res_alp_b / alp2;

    let c_idx = nearest_color(opt_r, opt_g, opt_b);

    let color = COLORS_F32[c_idx];
    let err_r = W_PERCEPTUAL[0] * (res2_acc[0].reduce_add() - 2.0_f32 * res_alp_r * color[0]);
    let err_g = W_PERCEPTUAL[1] * (res2_acc[1].reduce_add() - 2.0_f32 * res_alp_g * color[1]);
    let err_b = W_PERCEPTUAL[2] * (res2_acc[2].reduce_add() - 2.0_f32 * res_alp_b * color[2]);

    let err = err_r + err_g + err_b + COLORS_WSQ_SUM[c_idx] * alp2;

    (c_idx, err)
}

#[inline]
#[allow(clippy::uninit_assumed_init)]
pub fn optimal_color_suffix<const HW: usize, const HW_DIV_8: usize>(
    prefix: &[[f32; HW]; 3],               // 3*HW
    target: &[[f32; HW]; 3],               // 3*HW
    alpha: &[f32; HW],                     // HW
    suffix: (&[f32; HW], &[[f32; HW]; 3]), // (sfx_m: HW, sfx_a: 3*HW)
) -> (usize, f32) {
    let mut res2_acc = [f32x8::ZERO; 3];
    let mut res_alp_acc = [f32x8::ZERO; 3];
    let mut alp2_acc = f32x8::ZERO;

    let (sfx_m, sfx_a) = suffix;

    for ch in 0..3 {
        for px in (0..HW).step_by(8) {
            let pre = f32x8::from(&prefix[ch][px..px + 8]);
            let tar = f32x8::from(&target[ch][px..px + 8]);
            let alp = f32x8::from(&alpha[px..px + 8]);

            let sm = f32x8::from(&sfx_m[px..px + 8]);
            let eff_alp = alp * sm;
            let inv_a_sm = (f32x8::ONE - alp) * sm;

            let res = tar - f32x8::from(&sfx_a[ch][px..px + 8]) - pre * inv_a_sm;

            res2_acc[ch] += res * res;
            res_alp_acc[ch] += res * eff_alp;
            if ch == 0 {
                alp2_acc += eff_alp * eff_alp
            }
        }
    }

    let alp2 = alp2_acc.reduce_add();

    let res_alp_r = res_alp_acc[0].reduce_add();
    let res_alp_g = res_alp_acc[1].reduce_add();
    let res_alp_b = res_alp_acc[2].reduce_add();

    let opt_r = res_alp_r / alp2;
    let opt_g = res_alp_g / alp2;
    let opt_b = res_alp_b / alp2;

    let c_idx = nearest_color(opt_r, opt_g, opt_b);

    let color = COLORS_F32[c_idx];
    let err_r = W_PERCEPTUAL[0] * (res2_acc[0].reduce_add() - 2.0_f32 * res_alp_r * color[0]);
    let err_g = W_PERCEPTUAL[1] * (res2_acc[1].reduce_add() - 2.0_f32 * res_alp_g * color[1]);
    let err_b = W_PERCEPTUAL[2] * (res2_acc[2].reduce_add() - 2.0_f32 * res_alp_b * color[2]);

    let err = err_r + err_g + err_b + COLORS_WSQ_SUM[c_idx] * alp2;

    (c_idx, err)
}
