use wide::f32x8;

use crate::banner::{PrefixPatternCache, SuffixPatternCache};
use crate::color::COLORS_F32;

#[inline]
#[allow(clippy::needless_range_loop)]
pub fn build_prefix<const HW: usize>(
    patterns: &[[f32; HW]],
    prefix: &PrefixPatternCache<HW>,
    layer: &mut PrefixPatternCache<HW>,
    pattern_idx: usize,
    color_idx: usize,
) {
    let color_x8 = [
        f32x8::splat(COLORS_F32[color_idx][0]),
        f32x8::splat(COLORS_F32[color_idx][1]),
        f32x8::splat(COLORS_F32[color_idx][2]),
    ];

    for ch_idx in 0..3 {
        for px_dx in (0..HW).step_by(8) {
            let pfx_px = f32x8::from(&prefix.2[ch_idx][px_dx..px_dx + 8]);
            let alpha = f32x8::from(&patterns[pattern_idx][px_dx..px_dx + 8]);
            let new_px = pfx_px * (f32x8::ONE - alpha) + color_x8[ch_idx] * alpha;
            layer.2[ch_idx][px_dx..px_dx + 8].copy_from_slice(&<[f32; 8]>::from(new_px));
        }
    }

    layer.0 = pattern_idx;
    layer.1 = color_idx;
}

#[inline]
#[allow(clippy::needless_range_loop)]
pub fn build_suffix<const HW: usize>(
    patterns: &[[f32; HW]],
    layer: &mut SuffixPatternCache<HW>,
    suffix: &SuffixPatternCache<HW>,
    pattern_idx: usize,
    color_idx: usize,
) {
    let color = [
        f32x8::splat(COLORS_F32[color_idx][0]),
        f32x8::splat(COLORS_F32[color_idx][1]),
        f32x8::splat(COLORS_F32[color_idx][2]),
    ];

    for px_idx in (0..HW).step_by(8) {
        let sfx_mult = f32x8::from(&suffix.2[3][px_idx..px_idx + 8]);
        let pat_mult = f32x8::from(&patterns[pattern_idx][px_idx..px_idx + 8]);

        let mult = (f32x8::ONE - pat_mult) * sfx_mult;
        layer.2[3][px_idx..px_idx + 8].copy_from_slice(&<[f32; 8]>::from(mult));

        for ch_idx in 0..3 {
            let sfx_add = f32x8::from(&suffix.2[ch_idx][px_idx..px_idx + 8]);
            let new_add = color[ch_idx] * pat_mult * sfx_mult + sfx_add;
            layer.2[ch_idx][px_idx..px_idx + 8].copy_from_slice(&<[f32; 8]>::from(new_add));
        }
    }

    layer.0 = pattern_idx;
    layer.1 = color_idx;
}

#[inline]
pub fn build_empty_suffix<const HW: usize>(suffix_cache: &mut SuffixPatternCache<HW>) {
    suffix_cache.2[0].fill(0.0_f32);
    suffix_cache.2[1].fill(0.0_f32);
    suffix_cache.2[2].fill(0.0_f32);
    suffix_cache.2[3].fill(1.0_f32);

    suffix_cache.0 = 0;
    suffix_cache.1 = 0;
}
