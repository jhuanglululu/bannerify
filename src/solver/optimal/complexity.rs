use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::banner::Banner;
use crate::cli::complexity::OptimalOption;
use crate::color::NUM_COLORS;
use crate::solver::distance::{nearest_color, sort_freq};

pub fn sort_banner_optimal<const HW: usize>(
    banners: Vec<Banner<HW>>,
    option: &OptimalOption,
    layer_distr: &mut [usize],
) -> Vec<Vec<Banner<HW>>> {
    let (min_layers, max_layers) = option.layers;

    let banners: Vec<_> = banners
        .into_par_iter()
        .map(|mut banner| {
            banner.n_layers = eval_complexity_optimal(&banner.target, option);
            banner
        })
        .collect();

    let mut sorted: Vec<Vec<_>> = (min_layers..=max_layers).map(|_| Vec::new()).collect();

    for banner in banners {
        let idx = banner.n_layers - min_layers;
        layer_distr[idx] += 1;
        sorted[idx].push(banner);
    }
    sorted
}

fn eval_complexity_optimal<const HW: usize>(
    image: &[[f32; HW]; 3],
    option: &OptimalOption,
) -> usize {
    let mut freq = [0; NUM_COLORS];

    for px in 0..HW {
        let color = nearest_color(image[0][px], image[1][px], image[2][px]);
        freq[color] += 1;
    }

    sort_freq(freq).1.clamp(option.layers.0, option.layers.1)
}
