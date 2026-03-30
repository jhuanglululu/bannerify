use rayon::prelude::*;

use crate::banner::Banner;
use crate::cli::config::ComplexityOptions;
use crate::color::NUM_COLORS;
use crate::math::mean_2d;
use crate::solver::distance::{color_dist, nearest_color, sort_freq};

pub fn sort_banner<const HW: usize, const HW_DIV_8: usize>(
    banners: &mut Vec<Banner<HW, HW_DIV_8>>,
    option: &ComplexityOptions,
    layer_distr: &mut [usize],
    color_distr: &mut [usize],
) {
    let min_layers = option.layers.0;
    let min_colors = option.colors.0;

    banners.into_par_iter().for_each(|banner| {
        let (n_layers, colors) = eval_complexity(&banner.target, option);
        banner.n_layers = n_layers;
        banner.color_candidates = colors;
    });

    for banner in banners {
        let n_colors = banner.color_candidates.len();
        layer_distr[banner.n_layers - min_layers] += 1;
        color_distr[n_colors - min_colors] += 1;
    }
}

#[allow(clippy::needless_range_loop)]
fn eval_complexity<const HW: usize>(
    image: &[[f32; HW]; 3],
    option: &ComplexityOptions,
) -> (usize, Vec<usize>) {
    let mut freq = [0; NUM_COLORS];
    for px in 0..HW {
        let color = nearest_color(image[0][px], image[1][px], image[2][px]);
        freq[color] += 1;
    }

    let (sorted, n_candidate) = sort_freq(freq);
    let n_layers = n_candidate.clamp(option.layers.0, option.layers.1);

    let (min_colors, max_colors) = option.colors;
    let n_colors = n_candidate.clamp(min_colors, max_colors);

    let color_candidates = fill_color_candidates(image, &sorted, n_candidate, n_colors);
    (n_layers, color_candidates)
}

fn fill_color_candidates<const HW: usize>(
    image: &[[f32; HW]; 3],
    sorted: &[usize; 16],
    n_candidate: usize,
    n_colors: usize,
) -> Vec<usize> {
    if n_candidate < n_colors {
        let mean = mean_2d(image);

        let dist = color_dist(mean[0], mean[1], mean[2]);
        let mut closer_mean: [usize; NUM_COLORS] = std::array::from_fn(|i| i);

        closer_mean.sort_by(|&a, &b| dist[a].total_cmp(&dist[b]));

        let mut candidates = Vec::with_capacity(n_colors);
        let mut in_candidates = [false; NUM_COLORS];
        for &color in &sorted[..n_candidate] {
            candidates.push(color);
            in_candidates[color] = true;
        }

        for color in closer_mean {
            if !in_candidates[color] {
                candidates.push(color);
                in_candidates[color] = true;

                if candidates.len() >= n_colors {
                    break;
                }
            }
        }

        candidates
    } else {
        Vec::from(&sorted[0..n_colors])
    }
}
