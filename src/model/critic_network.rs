use candle_core::{Module, Tensor};
use candle_nn::{linear, seq, Activation, Sequential, VarBuilder};

pub struct CriticNetwork {
    model: Sequential,
}

impl CriticNetwork {
    pub fn new(vb: &VarBuilder) -> Result<CriticNetwork, candle_core::Error> {
        Ok(CriticNetwork {
            model: seq()
                .add(linear(128, 256, vb.pp("critic_linear_1"))?)
                .add(Activation::Relu)
                .add(linear(256, 1, vb.pp("critic_linear_2"))?),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, candle_core::Error> {
        self.model.forward(x)
    }
}
