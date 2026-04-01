use rayon::prelude::*;

use crate::banner::{BannerResult, NTopBanner, PrefixPatternCache, TopBanner};
use crate::cli::config::Config;
use crate::geometry::*;
use crate::logger::info;
use crate::logger::profiler::timed;
use crate::macros::uninit;
use crate::pattern::Patterns;
use crate::solver::build::build_prefix;
use crate::solver::complexity::sort_banner;
use crate::solver::fill::intial_fill;
use crate::solver::refine::refinement_pass;

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
    let mut layer_dist = vec![0; config.n_layers.1 - config.n_layers.0 + 1];
    sort_banner(&mut top_banners, config.n_layers, &mut layer_dist);
    sort_banner(&mut ntop_banners, config.n_layers, &mut layer_dist);
    timed!("banners sorted");
    info!(
        "layer distribution: {}",
        format_distrbution(config.n_layers.0, layer_dist)
    );

    let (mut top_results, top_cache): (Vec<_>, Vec<_>) = top_banners
        .into_par_iter()
        .map(|tb| {
            let (mut result, mut prefixes) =
                intial_fill(&tb, &patterns.alphas, &patterns.alpha_square);

            let suffixes = refinement_pass(
                &tb,
                &mut result,
                &mut prefixes,
                &config.refinement,
                &patterns.alphas,
            );

            let mut last_layer: PrefixPatternCache<TOP_HW> = uninit!();
            build_prefix(
                &patterns.alphas,
                &prefixes[tb.n_layers - 1],
                &mut last_layer,
                result.patterns.last().unwrap().0,
                result.patterns.last().unwrap().1,
            );

            (result, last_layer)
        })
        .unzip();

    let (mut ntop_results, ntop_cache): (Vec<_>, Vec<_>) = ntop_banners
        .into_par_iter()
        .map(|ntb| {
            let (mut result, mut prefixes) =
                intial_fill(&ntb, &patterns.ntop_alphas, &patterns.ntop_alpha_square);

            let suffixes = refinement_pass(
                &ntb,
                &mut result,
                &mut prefixes,
                &config.refinement,
                &patterns.ntop_alphas,
            );

            let mut last_layer: PrefixPatternCache<NTOP_HW> = uninit!();
            build_prefix(
                &patterns.ntop_alphas,
                &prefixes[ntb.n_layers - 1],
                &mut last_layer,
                result.patterns.last().unwrap().0,
                result.patterns.last().unwrap().1,
            );
            (result, last_layer)
        })
        .unzip();

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
