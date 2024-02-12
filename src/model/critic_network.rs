use candle_core::{Module, Tensor};
use candle_nn::{linear, seq, Activation, Sequential, VarBuilder};

pub struct CriticNetwork {
    model: Sequential,
}

impl CriticNetwork {
    pub fn new(vb: &VarBuilder) -> Result<CriticNetwork, candle_core::Error> {
        let weight_dims: Vec<Vec<usize>> = vec![vec![256, 256], vec![1, 256]];

        Ok(CriticNetwork {
            model: seq()
                .add(linear(
                    weight_dims[0][1],
                    weight_dims[0][0],
                    vb.pp("critic_linear_1"),
                )?)
                .add(Activation::Relu)
                .add(linear(
                    weight_dims[1][1],
                    weight_dims[1][0],
                    vb.pp("critic_linear_2"),
                )?),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, candle_core::Error> {
        self.model.forward(x)
    }
}
