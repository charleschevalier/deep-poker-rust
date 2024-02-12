use candle_core::{Module, Tensor};
use candle_nn::{linear, seq, Activation, Sequential, VarBuilder};

pub struct ActorNetwork {
    model: Sequential,
}

impl ActorNetwork {
    pub fn new(
        vb: &VarBuilder,
        action_count: usize,
    ) -> Result<ActorNetwork, Box<dyn std::error::Error>> {
        let weight_dims: Vec<Vec<usize>> = vec![vec![256, 256], vec![action_count, 256]];

        Ok(ActorNetwork {
            model: seq()
                .add(linear(
                    weight_dims[0][1],
                    weight_dims[0][0],
                    vb.pp("actor_linear_1"),
                )?)
                .add(Activation::Relu)
                .add(linear(
                    weight_dims[1][1],
                    weight_dims[1][0],
                    vb.pp("actor_linear_2"),
                )?),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
        let mut x = self.model.forward(x)?;
        x = candle_nn::ops::softmax(&x, candle_core::D::Minus1)?;
        Ok(x)
    }
}
