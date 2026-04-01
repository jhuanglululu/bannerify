use crate::geometry::*;

pub struct Banner<const HW: usize> {
    pub target: Box<[[f32; HW]; 3]>,
    pub n_layers: usize,
}

pub type TopBanner = Banner<TOP_HW>;
pub type NTopBanner = Banner<NTOP_HW>;

/// (pattern_idx, color_idx, cache)
/// there's base at index 0
pub type PrefixPatternCache<const HW: usize> = (usize, usize, [[f32; HW]; 3]);
/// (pattern_idx, color_idx, cache)
/// there's no base at index 0
/// with alpha channel
pub type SuffixPatternCache<const HW: usize> = (usize, usize, [[f32; HW]; 4]);

#[derive(Clone, PartialEq)]
pub struct BannerResult {
    pub base: usize,
    pub patterns: Vec<(usize, usize)>, // (pattern_idx, color_idx)
}

impl<const HW: usize> Banner<HW> {
    pub fn new(target: Box<[[f32; HW]; 3]>) -> Self {
        Self {
            target,
            n_layers: 0,
        }
    }
}

impl BannerResult {
    pub fn new(base: usize, patterns: Vec<(usize, usize)>) -> Self {
        Self { base, patterns }
    }
}
