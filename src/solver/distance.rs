use std::mem::MaybeUninit;

use crate::color::{COLORS_B, COLORS_G, COLORS_R, NUM_COLORS, W_PERCEPTUAL_X8};
use wide::f32x8;

#[inline]
#[allow(clippy::uninit_assumed_init, invalid_value)]
pub fn color_dist(r: f32, g: f32, b: f32) -> [f32; NUM_COLORS] {
    let pr = f32x8::splat(r);
    let pg = f32x8::splat(g);
    let pb = f32x8::splat(b);

    // Colors 0-7
    let dr = pr - COLORS_R[0];
    let dg = pg - COLORS_G[0];
    let db = pb - COLORS_B[0];
    #[rustfmt::skip]
    let dist0 = W_PERCEPTUAL_X8[0] * dr * dr +
                W_PERCEPTUAL_X8[1] * dg * dg +
                W_PERCEPTUAL_X8[2] * db * db;

    // Colors 8-15
    let dr = pr - COLORS_R[1];
    let dg = pg - COLORS_G[1];
    let db = pb - COLORS_B[1];
    #[rustfmt::skip]
    let dist1 = W_PERCEPTUAL_X8[0] * dr * dr +
                W_PERCEPTUAL_X8[1] * dg * dg +
                W_PERCEPTUAL_X8[2] * db * db;

    // Extract and find argmin

    let mut out: [f32; NUM_COLORS] = unsafe { MaybeUninit::uninit().assume_init() };
    out[..8].copy_from_slice(&<[f32; 8]>::from(dist0));
    out[8..].copy_from_slice(&<[f32; 8]>::from(dist1));

    out
}

#[inline]
pub fn nearest_color(r: f32, g: f32, b: f32) -> usize {
    let dist = color_dist(r, g, b);
    let mut best = 0;
    let mut best_dist = dist[0];
    for (i, a) in dist.iter().enumerate() {
        if *a < best_dist {
            best_dist = *a;
            best = i;
        }
    }
    best
}

#[inline]
pub fn sort_freq(freq: [usize; NUM_COLORS]) -> ([usize; NUM_COLORS], usize) {
    let count = freq.iter().filter(|&x| *x != 0).count();
    let mut indices: [usize; NUM_COLORS] = std::array::from_fn(|i| i);

    indices.sort_by_key(|&i| std::cmp::Reverse(freq[i]));

    (indices, count)
}
