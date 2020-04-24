/*
use tract_onnx::prelude::*;

fn main() -> TractResult<()> {
    let onnx = tract_onnx::onnx();
    let model = onnx.model_for_path("foo")?;
    model.into_optimized()?;
    Ok(())
}
*/

/*
BUG !
use tract_core::prelude::*;

fn main() -> TractResult<()> {
    let mut model = TypedModel::default();
    let s = model.add_source("source", TypedFact::dt_shape(DatumType::F64, [1usize, 1].as_ref())?)?;
    model.wire_node("pool", tract_core::ops::nn::GlobalLpPool::default(), &[s])?;
    let plan = SimplePlan::new(model)?;
    let result = plan.run(tvec!(tensor2(&[[1.0f64, 2.0]])))?;
    println!("{:?}", result);
    Ok(())
}
*/

use tract_core::prelude::*;
use tract_core::ops::StatefullOp;

fn main() -> TractResult<()> {
    let op = tract_core::ops::nn::GlobalLpPool::default();
    let result = op.as_stateless().unwrap().eval(tvec!(rctensor2(&[[1.0f64, 2.0]])))?;
    println!("{:?}", result);
    Ok(())
}

/* NO BUG
use tract_core::prelude::*;

fn main() {
    let t = rctensor4(&[[[[12.0f64]]]]);
    let p = std::env::args().nth(1).unwrap().parse().unwrap();
    let t = tclib::eval_t::<f64>(p, t);
    println!("{:?}", t);
}
*/
