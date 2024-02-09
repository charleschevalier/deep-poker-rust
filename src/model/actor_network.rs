use candle_core::{Module, Tensor};
use candle_nn::{linear, seq, Activation, Sequential, VarBuilder};

pub struct ActorNetwork {
    model: Sequential,
}

impl ActorNetwork {
    pub fn new(vb: &VarBuilder, action_count: usize) -> Result<ActorNetwork, candle_core::Error> {
        Ok(ActorNetwork {
            model: seq()
                .add(linear(128, 256, vb.pp("actor_linear_1"))?)
                .add(Activation::Relu)
                .add(linear(256, action_count, vb.pp("actor_linear_2"))?),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, candle_core::Error> {
        let mut x = self.model.forward(x)?;
        x = candle_nn::ops::softmax(&x, candle_core::D::Minus1)?;
        Ok(x)
    }
}
