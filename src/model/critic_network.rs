use candle_core::{Module, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

pub struct CriticNetwork {
    linear_1: Linear,
    linear_2: Linear,
}

impl CriticNetwork {
    pub fn new(vb: &VarBuilder) -> CriticNetwork {
        let linear_1 = linear(4096, 8192, vb.pp("critic_linear_1")).unwrap();
        let linear_2 = linear(8192, 1, vb.pp("critic_linear_2")).unwrap();

        CriticNetwork { linear_1, linear_2 }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut x = self.linear_1.forward(&x).unwrap();
        x = x.relu().unwrap();
        x = self.linear_2.forward(&x).unwrap();
        return x;
    }
}
