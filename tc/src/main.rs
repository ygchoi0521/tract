use tract_onnx::prelude::*;

fn main() -> TractResult<()> {
    let onnx = tract_onnx::onnx();
    let model = onnx.model_for_path("foo")?;
    model.into_optimized()?;
    Ok(())
}
