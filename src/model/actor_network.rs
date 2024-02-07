use candle_core::{Module, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

pub struct ActorNetwork {
    linear_1: Linear,
    linear_2: Linear,
}

impl ActorNetwork {
    pub fn new(vb: &VarBuilder, action_count: usize) -> ActorNetwork {
        let linear_1 = linear(4096, 8192, vb.pp("actor_linear_1")).unwrap();
        let linear_2 = linear(8192, action_count, vb.pp("actor_linear_2")).unwrap();

        ActorNetwork { linear_1, linear_2 }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut x = self.linear_1.forward(x).unwrap();
        x = x.relu().unwrap();
        x = self.linear_2.forward(&x).unwrap();
        // TODO: check dimension here
        x = candle_nn::ops::softmax(&x, candle_core::D::Minus1).unwrap();
        x
    }
}
