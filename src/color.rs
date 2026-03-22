pub const NUM_COLORS: usize = 16;

#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum DyeColor {
    White = 0,
    Orange = 1,
    Magenta = 2,
    LightBlue = 3,
    Yellow = 4,
    Lime = 5,
    Pink = 6,
    Gray = 7,
    LightGray = 8,
    Cyan = 9,
    Purple = 10,
    Blue = 11,
    Brown = 12,
    Green = 13,
    Red = 14,
    Black = 15,
}

pub const COLOR_NAMES: [&str; NUM_COLORS] = [
    "white",
    "orange",
    "magenta",
    "light_blue",
    "yellow",
    "lime",
    "pink",
    "gray",
    "light_gray",
    "cyan",
    "purple",
    "blue",
    "brown",
    "green",
    "red",
    "black",
];

pub const COLORS_RGB: [[u8; 3]; NUM_COLORS] = [
    [255, 255, 255], // white
    [249, 128, 29],  // orange
    [199, 78, 189],  // magenta
    [58, 179, 218],  // light_blue
    [254, 216, 61],  // yellow
    [128, 199, 31],  // lime
    [243, 139, 170], // pink
    [71, 79, 82],    // gray
    [157, 157, 151], // light_gray
    [22, 156, 156],  // cyan
    [137, 50, 184],  // purple
    [60, 68, 170],   // blue
    [131, 84, 50],   // brown
    [94, 124, 22],   // green
    [176, 46, 38],   // red
    [29, 29, 33],    // black
];

/// All 16 colors as f32, shape (16, 3) flattened row-major
pub const COLORS_F32: [[f32; 3]; NUM_COLORS] = {
    let mut out = [[0.0f32; 3]; NUM_COLORS];
    let mut i = 0;
    while i < NUM_COLORS {
        out[i][0] = COLORS_RGB[i][0] as f32;
        out[i][1] = COLORS_RGB[i][1] as f32;
        out[i][2] = COLORS_RGB[i][2] as f32;
        i += 1;
    }
    out
};

/// Perceptual RGB weights (luma)
pub const W_PERCEPTUAL: [f32; 3] = [0.299, 0.587, 0.114];

/// Weighted squared sum per color: sum(c_i^2 * w_i) for each of 16 colors
pub const COLORS_WSQ_SUM: [f32; NUM_COLORS] = {
    let mut out = [0.0f32; NUM_COLORS];
    let mut i = 0;
    while i < NUM_COLORS {
        let r = COLORS_F32[i][0];
        let g = COLORS_F32[i][1];
        let b = COLORS_F32[i][2];
        out[i] = r * r * W_PERCEPTUAL[0] + g * g * W_PERCEPTUAL[1] + b * b * W_PERCEPTUAL[2];
        i += 1;
    }
    out
};

impl DyeColor {
    #[inline]
    pub fn name(&self) -> &'static str {
        COLOR_NAMES[*self as usize]
    }

    #[inline]
    pub fn rgb(&self) -> [u8; 3] {
        COLORS_RGB[*self as usize]
    }

    #[inline]
    pub fn f32(&self) -> [f32; 3] {
        COLORS_F32[*self as usize]
    }

    #[inline]
    pub fn wsq_sum(&self) -> f32 {
        COLORS_WSQ_SUM[*self as usize]
    }
}
