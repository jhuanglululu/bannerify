use crate::geometry::*;

pub struct Banner<const HW: usize, const HW_DIV_8: usize> {
    pub target: Box<[[f32; HW]; 3]>,
    pub n_layers: usize,
    pub color_candidates: Vec<usize>,
}

const TOP_HW_DIV_8: usize = TOP_HW / 8;
const NTOP_HW_DIV_8: usize = NTOP_HW / 8;

pub type TopBanner = Banner<TOP_HW, TOP_HW_DIV_8>;
pub type NTopBanner = Banner<NTOP_HW, NTOP_HW_DIV_8>;
/// (pattern_idx, color_idx, cache)
pub type PrefixPatternCache<const HW: usize> = Vec<(usize, usize, [[f32; HW]; 3])>;
/// (pattern_idx, color_idx, cache)
pub type SuffixPatternCache<const HW: usize> = Vec<(usize, usize, [[f32; HW]; 3])>;

pub struct BannerResult {
    pub base: usize,
    pub patterns: Vec<(usize, usize)>, // (pattern_idx, color_idx)
}

impl<const HW: usize, const HW_DIV_8: usize> Banner<HW, HW_DIV_8> {
    pub fn new(target: Box<[[f32; HW]; 3]>) -> Self {
        Self {
            target,
            n_layers: 0,
            color_candidates: Vec::new(),
        }
    }
}

impl BannerResult {
    pub fn new(base: usize, patterns: Vec<(usize, usize)>) -> Self {
        Self { base, patterns }
    }
}
