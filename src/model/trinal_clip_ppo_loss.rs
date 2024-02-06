use std::ops::Sub;

use candle_core::Tensor;

#[derive(Debug)]
pub struct TrinalClipLoss {
    epsilon_clip: f32,
    c_1: f32,
    c_2: f32,
}

impl TrinalClipLoss {
    pub fn new(epsilon_clip: f32, c_1: f32, c_2: f32) -> Self {
        TrinalClipLoss {
            epsilon_clip,
            c_1,
            c_2,
        }
    }

    pub fn call(
        &self,
        advantages: &Tensor,
        actions: &Tensor,
        old_probs: &Tensor,
        new_probs: &Tensor,
    ) -> Result<Tensor, String> {
        // Validate input shapes
        if advantages.shape() != actions.shape()
            || advantages.shape() != old_probs.shape()
            || advantages.shape() != new_probs.shape()
        {
            return Err("Input tensors must have matching shapes".to_string());
        }

        // Calculate probability ratios
        let ratio = new_probs.div(old_probs).unwrap();

        // Clip probability ratios using epsilon_clip
        let clipped_ratio: Result<Tensor, candle_core::Error> =
            ratio.clamp(1.0 - self.epsilon_clip, 1.0 + self.epsilon_clip);

        // Calculate surrogate objective 1
        let surrogate1 = clipped_ratio * advantages;

        // Clip advantages using c_1
        let clipped_advantages = advantages.clamp(-self.c_1, self.c_1);

        // Calculate surrogate objective 2
        let surrogate2 = ratio.clone() * clipped_advantages;

        // Clip probability ratios using c_2
        let clipped_ratio_2 = ratio.clamp(1.0 - self.c_2, 1.0 + self.c_2);

        // Calculate surrogate objective 3
        let surrogate3 = clipped_ratio_2 * advantages;

        // Combine objectives and take minimum
        let loss = Self::minimum(
            &surrogate1.unwrap(),
            &Self::minimum(&surrogate2.unwrap(), &surrogate3.unwrap()),
        );

        Ok(loss)
    }

    fn minimum(a: &Tensor, b: &Tensor) -> Tensor {
        let mask = a.lt(b).unwrap();
        let mask_inv = Tensor::ones(a.shape(), a.dtype(), a.device())
            .unwrap()
            .sub(mask.clone());
        (mask * a).unwrap().add(&(mask_inv * b).unwrap()).unwrap()
    }
}

// def trinal_clip_ppo_loss(rt_theta, advantage, epsilon, delta1, delta2, delta3):
//     # Outer clipping for both policy and value losses
//     clipped_rt_theta = torch.clamp(rt_theta, 1 - epsilon, 1 + epsilon)

//     # Policy loss: additional clipping for negative advantage
//     if advantage < 0:
//         inner_clipped_rt_theta = torch.clamp(rt_theta, 1 - epsilon, delta1)
//         policy_loss = -(inner_clipped_rt_theta * advantage).mean()
//     else:
//         policy_loss = -(clipped_rt_theta * advantage).mean()

//     # Value loss: clip return before calculating loss
//     clipped_return = torch.clamp(R_gamma_t, -delta2, delta3)
//     value_loss = torch.nn.functional.mse_loss(V_theta(st), clipped_return)

//     return policy_loss, value_loss
