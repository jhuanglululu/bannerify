use std::mem::MaybeUninit;

use wide::f32x8;

use crate::color::{COLORS_F32, W_PERCEPTUAL_X8};
use crate::solver::distance::nearest_color;

const F32_1_X8: f32x8 = f32x8::splat(1.0_f32);

#[inline]
#[allow(clippy::uninit_assumed_init)]
pub fn optimal_color<const HW: usize, const HW_DIV_8: usize>(
    prefix: &[[f32; HW]; 3],                       // 3*HW
    target: &[[f32; HW]; 3],                       // 3*HW
    alpha: &[f32; HW],                             // HW
    suffix: Option<(&[f32; HW], &[[f32; HW]; 3])>, // (sfx_m: HW, sfx_a: 3*HW)
) -> (usize, f32) {
    let mut b_r = f32x8::splat(0.0);
    let mut b_g = f32x8::splat(0.0);
    let mut b_b = f32x8::splat(0.0);
    let mut a_sum = f32x8::splat(0.0);

    let mut residual_r: [f32x8; HW_DIV_8] = unsafe { MaybeUninit::uninit().assume_init() };
    let mut residual_g: [f32x8; HW_DIV_8] = unsafe { MaybeUninit::uninit().assume_init() };
    let mut residual_b: [f32x8; HW_DIV_8] = unsafe { MaybeUninit::uninit().assume_init() };
    let mut eff_a_arr: [f32x8; HW_DIV_8] = unsafe { MaybeUninit::uninit().assume_init() };

    for i in (0..HW).step_by(8) {
        let pre_r = f32x8::from(&prefix[0][i..i + 8]);
        let pre_g = f32x8::from(&prefix[1][i..i + 8]);
        let pre_b = f32x8::from(&prefix[2][i..i + 8]);

        let tar_r = f32x8::from(&target[0][i..i + 8]);
        let tar_g = f32x8::from(&target[1][i..i + 8]);
        let tar_b = f32x8::from(&target[2][i..i + 8]);

        let a = f32x8::from(&alpha[i..i + 8]);

        let (eff_a, res_r, res_g, res_b) = if let Some((sfx_m, sfx_a)) = suffix {
            let sm = f32x8::from(&sfx_m[i..i + 8]);
            let ea = a * sm;
            let inv_a_sm = (F32_1_X8 - a) * sm;
            (
                ea,
                tar_r - f32x8::from(&sfx_a[0][i..i + 8]) - pre_r * inv_a_sm,
                tar_g - f32x8::from(&sfx_a[1][i..i + 8]) - pre_g * inv_a_sm,
                tar_b - f32x8::from(&sfx_a[2][i..i + 8]) - pre_b * inv_a_sm,
            )
        } else {
            (
                a,
                tar_r - pre_r + pre_r * a,
                tar_g - pre_g + pre_g * a,
                tar_b - pre_b + pre_b * a,
            )
        };

        residual_r[i / 8] = res_r;
        residual_g[i / 8] = res_g;
        residual_b[i / 8] = res_b;
        eff_a_arr[i / 8] = eff_a;

        b_r += res_r * eff_a;
        b_g += res_g * eff_a;
        b_b += res_b * eff_a;
        a_sum += eff_a * eff_a;
    }

    let a: f32 = a_sum.reduce_add();

    let opt_r = b_r.reduce_add() / a;
    let opt_g = b_g.reduce_add() / a;
    let opt_b = b_b.reduce_add() / a;

    let color_idx = nearest_color(opt_r, opt_g, opt_b);

    let color = COLORS_F32[color_idx];
    let color_x8 = [
        f32x8::splat(color[0]),
        f32x8::splat(color[1]),
        f32x8::splat(color[2]),
    ];
    let mut error = 0.0_f32;

    for i in 0..HW / 8 {
        let res_r = residual_r[i];
        let res_g = residual_g[i];
        let res_b = residual_b[i];
        let eff_a = eff_a_arr[i];

        let dr = res_r - color_x8[0] * eff_a;
        let dg = res_g - color_x8[1] * eff_a;
        let db = res_b - color_x8[2] * eff_a;

        error += (W_PERCEPTUAL_X8[0] * dr * dr
            + W_PERCEPTUAL_X8[1] * dg * dg
            + W_PERCEPTUAL_X8[2] * db * db)
            .reduce_add();
    }

    (color_idx, error)
}
