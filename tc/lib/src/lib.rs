use tract_core::prelude::*;
use tract_ndarray::*;

pub fn eval_t<D: Datum + tract_core::num_traits::Float>(
    p: usize,
    input: Arc<Tensor>,
) -> TractResult<TVec<Arc<Tensor>>> {
    let array = input.to_array_view::<D>()?;
    let n = array.shape()[0];
    let c = array.shape()[1];
    let mut final_shape = array.shape().to_vec();
    for dim in final_shape[2..].iter_mut() {
        *dim = 1;
    }
    let divisor = array.len() / (n * c);
    let input = array.into_shape(((n * c), divisor))?;
    let divisor = D::from(divisor).unwrap().recip();
    let result = if p == 1 {
        input.fold_axis(Axis(1), D::zero(), |&a, &b| a + b.abs()).map(|a| *a * divisor)
    } else if p == 2 {
        input.fold_axis(Axis(1), D::zero(), |&a, &b| a + b * b).map(|a| a.sqrt() * divisor)
    } else {
        input
            .fold_axis(Axis(1), D::zero(), |&a, &b| a + b.abs().powi(p as i32))
            .map(|a| a.powf(D::from(p).unwrap().recip()) * divisor)
    };
    Ok(tvec!(result.into_shape(final_shape)?.into_arc_tensor()))
}
