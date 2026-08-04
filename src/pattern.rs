//! The banner pattern masks, embedded in the binary and decoded into the
//! solver's lane-view tables.
//!
//! `assets/banners/` holds one 20×40 PNG per Minecraft banner pattern. Only the
//! **alpha channel** carries information: it is the mask the dye is laid
//! through, so a pattern contributes `α` of its dye and `1 − α` of whatever is
//! already composited underneath. The RGB channels of the assets are ignored.
//!
//! A patch is row-major, [`BANNER_W`] wide by 40 (or 24) rows tall, one plane
//! per quantity, so a pattern plane and a target plane can be zipped lane for
//! lane with no addressing arithmetic in the kernels. A banner's top rows are
//! covered by the banner in front of it, so lower rows solve the lane-aligned
//! *tail* of the same plane — only `Σ α²`, which sums over exactly the solved
//! pixels, differs between the two cases and so is stored twice.

use std::collections::HashSet;

use image::GenericImageView;
use rayon::prelude::*;
use rust_embed::Embed;

use crate::geometry::{BANNER_H, BANNER_W, NTOP_HW, TOP_HW};
use crate::logger::error_out;
use crate::simd::Chunk;

#[derive(Embed)]
#[folder = "assets/banners/"]
struct PatternAssets;

/// Every pattern the solver may use, decoded into lane-view tables.
///
/// Field `i` of every vector describes pattern `names[i]`; the solver only ever
/// carries the index.
pub struct Patterns {
    /// Pattern ids, sorted, in table order.
    pub names: Vec<String>,
    /// Full 20×40 alpha patches. Lower rows use the tail of the same plane.
    pub top: Vec<Chunk<TOP_HW>>,
    /// `Σ α²` over the whole plane, for banner row 0.
    pub top_alpha2: Vec<f32>,
    /// `Σ α²` over the bottom 24 rows, for every other banner row.
    pub lower_alpha2: Vec<f32>,
}

impl Patterns {
    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

/// Decode the embedded patterns, dropping every id in `exclude`.
///
/// An id in `exclude` that names no pattern is a user mistake, not a silent
/// no-op: it exits with the list.
pub fn load(exclude: &HashSet<String>) -> Patterns {
    // rust-embed's iteration order follows the directory walk; sort so the
    // pattern indices (and therefore every tie-break in the solver) are the
    // same on every machine and every run.
    let mut all: Vec<String> = PatternAssets::iter()
        .filter_map(|p| p.strip_suffix(".png").map(str::to_string))
        .collect();
    all.sort_unstable();

    let unknown: Vec<&str> = exclude
        .iter()
        .map(String::as_str)
        .filter(|id| !all.iter().any(|name| name == id))
        .collect();
    if !unknown.is_empty() {
        let mut unknown = unknown;
        unknown.sort_unstable();
        error_out!("unknown pattern id(s): {}", unknown.join(", "));
    }

    let names: Vec<String> = all.into_iter().filter(|id| !exclude.contains(id)).collect();
    if names.is_empty() {
        error_out!("every banner pattern was excluded — nothing left to solve with");
    }

    let decoded: Vec<Chunk<TOP_HW>> = names.par_iter().map(|id| decode(id)).collect();

    let top_alpha2 = decoded.iter().map(|p| sum_squares(p)).collect();
    let lower_alpha2 = decoded
        .iter()
        .map(|p| sum_squares(&p[TOP_HW - NTOP_HW..]))
        .collect();

    Patterns {
        names,
        top: decoded,
        top_alpha2,
        lower_alpha2,
    }
}

fn decode(id: &str) -> Chunk<TOP_HW> {
    let file = PatternAssets::get(&format!("{id}.png"))
        .unwrap_or_else(|| error_out!("internal error: pattern '{id}' vanished from the binary"));
    let img = image::load_from_memory_with_format(&file.data, image::ImageFormat::Png)
        .unwrap_or_else(|e| error_out!("could not decode pattern '{id}': {e}"));

    if img.dimensions() != (BANNER_W as u32, BANNER_H as u32) {
        error_out!("pattern '{id}' is not {BANNER_W}x{BANNER_H} pixels");
    }

    const INV_255: f32 = 1.0 / 255.0;
    let mut out = Chunk::<TOP_HW>::zeroed();
    for (dst, rgba) in out
        .iter_mut()
        .zip(img.into_rgba8().as_raw().chunks_exact(4))
    {
        *dst = f32::from(rgba[3]) * INV_255;
    }
    out
}

fn sum_squares(patch: &[f32]) -> f32 {
    patch.iter().map(|a| a * a).sum()
}
