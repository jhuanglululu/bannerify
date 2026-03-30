use rayon::prelude::*;

use crate::banner::{BannerResult, NTopBanner, PrefixPatternCache, TopBanner};
use crate::cli::config::Config;
use crate::geometry::*;
use crate::logger::info;
use crate::logger::profiler::timed;
use crate::pattern::Patterns;
use crate::solver::complexity::sort_banner;
use crate::solver::greedy::initial_fill::initial_fill_greedy;
use crate::solver::optimal::initial_fill::initial_fill_optimal;

pub fn process_banners(
    config: &Config,
    patterns: Patterns,
    mut top_banners: Vec<TopBanner>,
    mut ntop_banners: Vec<NTopBanner>,
) -> (
    Vec<BannerResult>,
    Vec<PrefixPatternCache<TOP_HW>>,
    Vec<PrefixPatternCache<NTOP_HW>>,
) {
    let optimal_option = config.optimal;
    let has_greedy = optimal_option.has_greedy();

    let complexity = config.complexity;

    let mut layer_dist = vec![0; complexity.layers.1 - complexity.layers.0 + 1];
    let mut color_dist = vec![0; complexity.colors.1 - complexity.colors.0 + 1];
    sort_banner(
        &mut top_banners,
        &complexity,
        &mut layer_dist,
        &mut color_dist,
    );
    sort_banner(
        &mut ntop_banners,
        &complexity,
        &mut layer_dist,
        &mut color_dist,
    );
    timed!("banners sorted");
    info!(
        "layer distribution: {}",
        format_distrbution(complexity.layers.0, layer_dist)
    );
    if has_greedy {
        info!(
            "color distribution: {}",
            format_distrbution(complexity.colors.0, color_dist)
        );
    }

    let (mut top_results, top_cache): (Vec<_>, Vec<_>) = top_banners
        .into_par_iter()
        .map(|tb| {
            if optimal_option.initial {
                initial_fill_optimal(&tb, &patterns.alphas, &patterns.alpha_square)
            } else {
                initial_fill_greedy(&tb, &patterns.alphas, &patterns.alpha_square)
            }
        })
        .unzip();

    let (mut ntop_results, ntop_cache): (Vec<_>, Vec<_>) = ntop_banners
        .into_par_iter()
        .map(|ntb| {
            if optimal_option.initial {
                initial_fill_optimal(&ntb, &patterns.ntop_alphas, &patterns.ntop_alpha_square)
            } else {
                initial_fill_greedy(&ntb, &patterns.ntop_alphas, &patterns.ntop_alpha_square)
            }
        })
        .unzip();

    timed!("temp(initial fill)");
    top_results.append(&mut ntop_results);
    (top_results, top_cache, ntop_cache)
}

fn format_distrbution(offset: usize, dist: Vec<usize>) -> String {
    format!(
        "{{ {} }}",
        dist.iter()
            .enumerate()
            .map(|(i, count)| format!("{}: {}", i + offset, count))
            .collect::<Vec<_>>()
            .join(", ")
    )
}
