use wide::{CmpGt, f32x8};

const XYZ_D65_MAT: [[f32x8; 3]; 3] = {
    let bits = [
        [0x3ed324e3, 0x3eb7154b, 0x3e38cff5],
        [0x3e59be0a, 0x3f37154b, 0x3d93d991],
        [0x3c9e5baa, 0x3df41c65, 0x3f735613],
    ];
    let mut out = [[f32x8::ZERO; 3]; 3];

    let mut r = 0;
    while r < 3 {
        let mut c = 0;
        while c < 3 {
            out[r][c] = f32x8::splat(f32::from_bits(bits[r][c]));
            c += 1;
        }
        r += 1;
    }

    out
};

const D65_REF_WHITE: [f32x8; 3] = [
    f32x8::splat(f32::from_bits(0x3f735114)),
    f32x8::splat(f32::from_bits(0x3f800000)),
    f32x8::splat(f32::from_bits(0x3f8b663f)),
];

#[inline]
#[allow(dead_code)]
pub fn rgb_to_lab(r: &[f32], g: &[f32], b: &[f32]) -> (f32x8, f32x8, f32x8) {
    let lin_r = f32x8::from(std::array::from_fn(|i| LINEARIZE_TABLE[r[i] as usize]));
    let lin_g = f32x8::from(std::array::from_fn(|i| LINEARIZE_TABLE[g[i] as usize]));
    let lin_b = f32x8::from(std::array::from_fn(|i| LINEARIZE_TABLE[b[i] as usize]));

    let x = lin_r * XYZ_D65_MAT[0][0] + lin_g * XYZ_D65_MAT[0][1] + lin_b * XYZ_D65_MAT[0][2];
    let y = lin_r * XYZ_D65_MAT[1][0] + lin_g * XYZ_D65_MAT[1][1] + lin_b * XYZ_D65_MAT[1][2];
    let z = lin_r * XYZ_D65_MAT[2][0] + lin_g * XYZ_D65_MAT[2][1] + lin_b * XYZ_D65_MAT[2][2];

    let fx = lab_f(x / D65_REF_WHITE[0]);
    let fy = lab_f(y / D65_REF_WHITE[1]);
    let fz = lab_f(z / D65_REF_WHITE[2]);

    let l = 116.0 * fy - 16.0;
    let a = 500.0 * (fx - fy);
    let b = 200.0 * (fy - fz);

    (l, a, b)
}

// 0.46180555224418640137
const A: f32 = f32::from_bits(0x3eec71c7);
// 1.0 - 0.46180555224418640137
const B: f32 = 1.0_f32 - A;

const LINEARIZE_TABLE: [f32; 256] = {
    let mut out = [0.0_f32; 256];
    let mut i = 0;
    while i < 256 {
        let c = i as f32 / 255.0;
        out[i] = if c <= 0.04045 {
            c / 12.92
        } else {
            let x = (c + 0.055) / 1.055;
            x * x * (A * x + B)
        };
        i += 1;
    }

    out
};

const DELTA3: f32x8 = f32x8::splat(216.0 / 24389.0);
const KAPPA: f32x8 = f32x8::splat(24389.0 / 27.0);

fn lab_f(t: f32x8) -> f32x8 {
    let mask = t.simd_gt(DELTA3);
    let high = cube_root(t);
    const ONE_OVER_116: f32x8 = f32x8::splat(1.0 / 116.0);
    let low = (KAPPA * t + 16.0) * ONE_OVER_116;

    mask.blend(high, low)
}

#[inline]
fn cube_root(x: f32x8) -> f32x8 {
    const ONE_THIRD: f32x8 = f32x8::splat(1.0 / 3.0);
    (x.ln() * ONE_THIRD).exp()
}
