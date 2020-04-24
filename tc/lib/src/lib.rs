use tract_core::internal::*;
use tract_ndarray::*;

#[derive(Debug, Clone, Default)]
pub struct GlobalLpPool {
    p: usize,
}

impl GlobalLpPool {
    fn eval_t<D: Datum + tract_core::num_traits::Float>(
        &self,
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
        let result =
            input
            .fold_axis(Axis(1), D::zero(), |&a, &b| a + b.abs().powi(self.p as i32))
            .map(|a| a.powf(D::from(self.p).unwrap().recip()) * divisor);
        Ok(tvec!(result.into_shape(final_shape)?.into_arc_tensor()))
    }
}

impl Op for GlobalLpPool {
    fn name(&self) -> Cow<str> {
        "GlobalLpPool".into()
    }
    fn validation(&self) -> Validation {
        Validation::Rounding
    }
    op_as_typed_op!();
    not_a_pulsed_op!();
}

impl StatelessOp for GlobalLpPool {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        dispatch_floatlike!(Self::eval_t(input.datum_type())(self, input))
    }
}
impl TypedOp for GlobalLpPool {
    as_op!();

    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        todo!();
    }
}
