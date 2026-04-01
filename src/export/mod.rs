use crate::banner::PrefixPatternCache;
use crate::export::html::ExportHtml;
use crate::export::image::{banner_to_buffer, buffer_to_base64};
use crate::geometry::*;
use crate::logger::error_out;
use std::path::Path;

pub mod html;
pub mod image;

pub fn export_to_html(
    path: &Path,
    dimension: (usize, usize),
    block_assets: &[[u8; 3 * BLOCK_PIXELS]],
    blocks: &[usize],
    top_cache: Vec<PrefixPatternCache<TOP_HW>>,
    ntop_cache: Vec<PrefixPatternCache<NTOP_HW>>,
) {
    let top_caches: Vec<_> = top_cache.iter().map(|v| v.2).collect();
    let ntop_caches: Vec<_> = ntop_cache.iter().map(|v| v.2).collect();

    let html_string = ExportHtml {
        image: buffer_to_base64(
            dimension,
            banner_to_buffer(block_assets, dimension, blocks, &top_caches, &ntop_caches),
        ),
    }
    .to_string();

    std::fs::write(path, html_string).unwrap_or_else(|e| error_out!("{}", e));
}
