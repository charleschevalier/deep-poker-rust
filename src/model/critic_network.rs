use candle_core::{Module, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

pub struct CriticNetwork {
    linear_1: Linear,
    linear_2: Linear,
}

impl CriticNetwork {
    pub fn new(vb: &VarBuilder) -> Result<CriticNetwork, candle_core::Error> {
        let weight_dims: Vec<Vec<usize>> = vec![vec![1024, 1024], vec![1, 1024]];

        Ok(CriticNetwork {
            linear_1: linear(
                weight_dims[0][1],
                weight_dims[0][0],
                vb.pp("critic_linear_1"),
            )?,
            linear_2: linear(
                weight_dims[1][1],
                weight_dims[1][0],
                vb.pp("critic_linear_2"),
            )?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, candle_core::Error> {
        let mut y = self.linear_1.forward(x)?;
        y = y.relu()?;
        y = self.linear_2.forward(&y)?;
        Ok(y)
    }
}
