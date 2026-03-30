use wide::f32x8;

#[inline]
pub fn mean<const SIZE: usize>(input: &[f32; SIZE]) -> f32 {
    let mut sum = f32x8::ZERO;
    for idx in (0..SIZE).step_by(8) {
        let vals = f32x8::from(&input[idx..idx + 8]);
        sum += vals;
    }
    sum.reduce_add() / SIZE as f32
}

#[inline]
pub fn mean_2d<const SIZE: usize, const DIM: usize>(input: &[[f32; SIZE]; DIM]) -> [f32; DIM] {
    let mut means = [0.0_f32; DIM];
    for dim in 0..DIM {
        means[dim] = mean(&input[dim]);
    }
    means
}
