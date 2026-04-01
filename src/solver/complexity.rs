use rayon::prelude::*;

use crate::banner::Banner;
use crate::math::{mean_2d, square_mean_2d};

pub fn sort_banner<const HW: usize>(
    banners: &mut Vec<Banner<HW>>,
    layer_min_max: (usize, usize),
    layer_distr: &mut [usize],
) {
    let variances: Vec<_> = banners
        .par_iter()
        .map(|banner| find_variance(&banner.target))
        .collect();

    let (var_min, var_max) = {
        let (mut min, mut max) = (f32::INFINITY, f32::NEG_INFINITY);
        for &var in &variances {
            if min > var {
                min = var;
            }
            if max < var {
                max = var
            }
        }
        (min, max)
    };

    let layer_min = layer_min_max.0 as f32;
    let layer_range = layer_min_max.1 as f32 - layer_min;
    let inv_var_range = if var_max == var_min {
        0.0
    } else {
        1.0 / (var_max - var_min)
    };

    (banners.par_iter_mut(), variances.par_iter())
        .into_par_iter()
        .for_each(|(banner, var)| {
            banner.n_layers =
                (layer_min + (var - var_min) * layer_range * inv_var_range).round() as usize;
        });

    for banner in banners {
        layer_distr[banner.n_layers - layer_min_max.0] += 1;
    }
}

fn find_variance<const HW: usize>(image: &[[f32; HW]; 3]) -> f32 {
    let square_mean: f32 = square_mean_2d(image).iter().sum();
    let mean = mean_2d(image).iter().map(|m| m * m).sum::<f32>();
    square_mean - mean
}
