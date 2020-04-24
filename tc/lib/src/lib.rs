use ndarray::*;

#[derive(Debug, Clone, Default)]
pub struct GlobalLpPool {
    p: usize,
}

impl GlobalLpPool {
    pub fn eval_t(
        &self,
        array: ArrayViewD<f64>,
        ) -> ArrayD<f64> {
        let n = array.shape()[0];
        let c = array.shape()[1];
        let mut final_shape = array.shape().to_vec();
        for dim in final_shape[2..].iter_mut() {
            *dim = 1;
        }
        let divisor = array.len() / (n * c);
        let input = array.into_shape(((n * c), divisor)).unwrap();
        let divisor = (divisor as f64).recip();
        let result = if self.p == 1 {
            input.fold_axis(Axis(1), 0.0, |&a, &b| a + b.abs()).map(|a| *a * divisor)
        } else if self.p == 2 {
            input.fold_axis(Axis(1), 0.0, |&a, &b| a + b * b).map(|a| a.sqrt() * divisor)
        } else {
             input
                .fold_axis(Axis(1), 0.0, |&a, &b| a + b.abs().powi(self.p as i32))
                .map(|a| a.powf((self.p as f64).recip()) * divisor)
        };
        result.into_dyn()
    }
}
