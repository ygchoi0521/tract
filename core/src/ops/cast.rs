use ops::prelude::*;

#[derive(Debug, Clone, new)]
pub struct Cast {
    to: DatumType,
}

impl Cast {
    /// Evaluates the operation given the input tensors.
    fn eval_t<T: Datum>(input: Value) -> TractResult<Value> {
        Ok(input.cast_to::<T>()?.into())
    }
}

impl Op for Cast {
    fn name(&self) -> &str {
        "Cast"
    }
}

impl StatelessOp for Cast {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<Value>) -> TractResult<TVec<Value>> {
        let input = args_1!(inputs);
        let output = dispatch_datum!(Self::eval_t(self.to)(input))?;
        Ok(tvec!(output))
    }
}

impl InferenceRulesOp for Cast {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) -> InferenceResult {
        s.equals(&inputs.len, 1)?;
        s.equals(&outputs.len, 1)?;
        s.equals(&outputs[0].datum_type, self.to)?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        Ok(())
    }
}