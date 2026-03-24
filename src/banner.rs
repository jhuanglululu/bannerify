use crate::geometry::{NTOP_HW, TOP_HW};

pub struct Banner<const HW: usize> {
    pub row: usize,
    pub column: usize,
    pub target: Box<[[f32; HW]; 3]>,
    pub n_layers: usize,
}

pub type TopBanner = Banner<TOP_HW>;
pub type NTopBanner = Banner<NTOP_HW>;

pub struct GreedyBanner<const HW: usize> {
    pub row: usize,
    pub column: usize,
    pub target: Box<[[f32; HW]; 3]>,
    pub n_layers: usize,
    pub color_candidates: Vec<usize>,
}

pub type TopGreedyBanner = GreedyBanner<TOP_HW>;
pub type NTopGreedyBanner = GreedyBanner<NTOP_HW>;

pub struct BannerResult {
    pub row: usize,
    pub column: usize,
    pub base: usize,
    pub patterns: Vec<(usize, usize)>, // (pattern_idx, color_idx)
}

impl<const HW: usize> Banner<HW> {
    pub fn new(row: usize, column: usize, target: Box<[[f32; HW]; 3]>) -> Self {
        Self {
            row,
            column,
            target,
            n_layers: 0,
        }
    }
}

impl<const HW: usize> GreedyBanner<HW> {
    #[inline]
    pub fn from_banner(value: Banner<HW>, color_candidates: Vec<usize>) -> Self {
        Self {
            row: value.row,
            column: value.column,
            target: value.target,
            n_layers: value.n_layers,
            color_candidates,
        }
    }
}
