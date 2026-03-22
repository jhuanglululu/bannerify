use ndarray::Array3;

use crate::color::DyeColor;

pub struct Banner {
    pub is_top: bool,
    pub row: usize,
    pub column: usize,
    pub target: Array3<f32>,
    pub base: DyeColor,
    pub patterns: Vec<(DyeColor, usize)>, // (color, pattern index)
}
