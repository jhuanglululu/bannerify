use std::f32;

use crate::banner::{Banner, BannerResult, PrefixPatternCache};
use crate::color::COLORS_F32;
use crate::math::mean_2d;
use crate::solver::distance::nearest_color;
use crate::solver::next_layer::next_layer;
use crate::solver::optimal::optimal_color::optimal_color;

#[allow(clippy::uninit_vec)]
pub fn initial_fill_optimal<const HW: usize, const HW_DIV_8: usize>(
    banner: &Banner<HW, HW_DIV_8>,
    patterns: &[[f32; HW]],
    alp2: &[f32],
) -> (BannerResult, PrefixPatternCache<HW>) {
    let mean_color = mean_2d(&banner.target);
    let base = nearest_color(mean_color[0], mean_color[1], mean_color[2]);

    let mut b_patterns = Vec::with_capacity(banner.n_layers);
    let mut prefix_cache: Vec<(usize, usize, [[f32; HW]; 3])> =
        Vec::with_capacity(banner.n_layers + 1);

    unsafe {
        prefix_cache.set_len(banner.n_layers + 1);
    }

    prefix_cache[0].2[0] = [COLORS_F32[base][0]; HW];
    prefix_cache[0].2[1] = [COLORS_F32[base][1]; HW];
    prefix_cache[0].2[2] = [COLORS_F32[base][2]; HW];

    for layer in 0..banner.n_layers {
        let mut best = (0, 0);
        let mut min_err = f32::INFINITY;

        for (p_idx, pattern) in patterns.iter().enumerate() {
            let (color, err) = optimal_color::<_, HW_DIV_8>(
                &prefix_cache[layer].2,
                &banner.target,
                pattern,
                alp2[p_idx],
            );

            if err < min_err {
                best = (p_idx, color);
                min_err = err;
            }
        }

        b_patterns.push(best);
        next_layer(patterns, &mut prefix_cache, layer, best.0, best.1);
    }

    (BannerResult::new(base, b_patterns), prefix_cache)
}
