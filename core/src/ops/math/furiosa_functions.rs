use num_traits::{Float, Zero};
use tract_linalg::f16::f16;

pub trait FuriosaExp {
    fn furiosa_exp(&mut self) -> Self;
}

impl FuriosaExp for f32 {
    fn furiosa_exp(&mut self) -> Self {
        self.exp()
    }
}

impl FuriosaExp for f16 {
    fn furiosa_exp(&mut self) -> Self {
        self.exp()
    }
}

impl FuriosaExp for f64 {
    fn furiosa_exp(&mut self) -> Self {
        self.exp()
    }
}
