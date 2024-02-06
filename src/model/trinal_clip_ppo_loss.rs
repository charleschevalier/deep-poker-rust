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
        let clipped_ratio = ratio.clip(1.0 - self.epsilon_clip, 1.0 + self.epsilon_clip);

        // Calculate surrogate objective 1
        let surrogate1 = clipped_ratio * advantages;

        // Clip advantages using c_1
        let clipped_advantages = advantages.clip(-self.c_1, self.c_1);

        // Calculate surrogate objective 2
        let surrogate2 = ratio * clipped_advantages;

        // Clip probability ratios using c_2
        let clipped_ratio_2 = ratio.clip(1.0 - self.c_2, 1.0 + self.c_2);

        // Calculate surrogate objective 3
        let surrogate3 = clipped_ratio_2 * advantages;

        // Combine objectives and take minimum
        let loss = -(surrogate1.min(surrogate2.min(surrogate3))).mean();

        Ok(loss)
    }
}
