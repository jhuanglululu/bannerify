use wide::f32x8;

use crate::banner::PrefixPatternCache;
use crate::color::COLORS_F32;

#[inline]
#[allow(clippy::needless_range_loop)]
pub fn next_layer<const HW: usize>(
    patterns: &[[f32; HW]],
    prefix_cache: &mut PrefixPatternCache<HW>,
    layer: usize,
    pattern_idx: usize,
    color_idx: usize,
) {
    let color = COLORS_F32[color_idx];
    let color_x8 = [
        f32x8::splat(color[0]),
        f32x8::splat(color[1]),
        f32x8::splat(color[2]),
    ];

    for ch in 0..3 {
        for px in (0..HW).step_by(8) {
            let prefix = f32x8::from(&prefix_cache[layer].2[ch][px..px + 8]);
            let alpha = f32x8::from(&patterns[pattern_idx][px..px + 8]);
            let new_px = prefix * (f32x8::ONE - alpha) + color_x8[ch] * alpha;
            prefix_cache[layer + 1].2[ch][px..px + 8].copy_from_slice(&<[f32; 8]>::from(new_px));
        }
    }

    prefix_cache[layer + 1].0 = pattern_idx;
    prefix_cache[layer + 1].1 = color_idx;
}
