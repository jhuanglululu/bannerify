use wide::f32x8;

pub const NUM_COLORS: usize = 16;

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
    let mut out = [[0.0_f32; 3]; NUM_COLORS];
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

/// Perceptual RGB weights (luma)
pub const W_PERCEPTUAL_X8: [f32x8; 3] = [
    f32x8::splat(W_PERCEPTUAL[0]),
    f32x8::splat(W_PERCEPTUAL[1]),
    f32x8::splat(W_PERCEPTUAL[2]),
];

/// Weighted squared sum per color: sum(c_i^2 * w_i) for each of 16 colors
pub const COLORS_WSQ_SUM: [f32; NUM_COLORS] = {
    let mut out = [0.0_f32; NUM_COLORS];
    let mut i = 0;
    while i < NUM_COLORS {
        let g = COLORS_F32[i][1];
        let r = COLORS_F32[i][0];
        let b = COLORS_F32[i][2];
        out[i] = W_PERCEPTUAL[0] * r * r + W_PERCEPTUAL[1] * g * g + W_PERCEPTUAL[2] * b * b;
        i += 1;
    }
    out
};

pub const COLORS_R: [f32x8; 2] = {
    let mut red = [[0.0; 8]; 2];
    let mut i = 0;
    while i < NUM_COLORS {
        red[i / 8][i % 8] = COLORS_F32[i][0];
        i += 1;
    }
    [f32x8::new(red[0]), f32x8::new(red[1])]
};

pub const COLORS_G: [f32x8; 2] = {
    let mut green = [[0.0; 8]; 2];
    let mut i = 0;
    while i < NUM_COLORS {
        green[i / 8][i % 8] = COLORS_F32[i][1];
        i += 1;
    }
    [f32x8::new(green[0]), f32x8::new(green[1])]
};

pub const COLORS_B: [f32x8; 2] = {
    let mut blue = [[0.0; 8]; 2];
    let mut i = 0;
    while i < NUM_COLORS {
        blue[i / 8][i % 8] = COLORS_F32[i][2];
        i += 1;
    }
    [f32x8::new(blue[0]), f32x8::new(blue[1])]
};
