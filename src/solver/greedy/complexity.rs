use rayon::iter::{IntoParallelIterator, ParallelIterator};
use wide::f32x8;

use crate::banner::{Banner, GreedyBanner};
use crate::cli::complexity::GreedyOption;
use crate::color::NUM_COLORS;
use crate::solver::distance::{color_dist, nearest_color, sort_freq};

pub fn sort_banner_greedy<const HW: usize>(
    banners: Vec<Banner<HW>>,
    option: &GreedyOption,
    layer_distr: &mut [usize],
    color_distr: &mut [usize],
) -> Vec<Vec<GreedyBanner<HW>>> {
    let (min_layers, max_layers) = option.layers;
    let (min_colors, max_colors) = option.colors;

    let banners: Vec<_> = banners
        .into_par_iter()
        .map(|mut banner| {
            let (n_layers, colors) = eval_complexity_greedy(&banner.target, option);
            banner.n_layers = n_layers;
            GreedyBanner::from_banner(banner, colors)
        })
        .collect();

    let mut sorted: Vec<Vec<GreedyBanner<HW>>> = (0..=(max_layers - min_layers + max_colors
        - min_colors))
        .map(|_| Vec::new())
        .collect();

    for banner in banners {
        let n_colors = banner.color_candidates.len();
        let idx = banner.n_layers - min_layers + n_colors - min_colors;
        layer_distr[banner.n_layers - min_layers] += 1;
        color_distr[n_colors - min_colors] += 1;
        sorted[idx].push(banner);
    }
    sorted
}

fn eval_complexity_greedy<const HW: usize>(
    image: &[[f32; HW]; 3],
    option: &GreedyOption,
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
        let mut mean = [0.0_f32; 3];

        for ch in 0..3 {
            let channel = &image[ch];
            let mut sum = f32x8::splat(0.0_f32);
            for px in (0..HW).step_by(8) {
                let pxs = f32x8::from(&channel[px..px + 8]);
                sum += pxs;
            }
            mean[ch] = sum.reduce_add() / HW as f32;
        }

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
