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

// def calculate_discounted_rewards(rewards, gamma=0.999):
//     discounted_rewards = []
//     R = 0
//     for reward in reversed(rewards):
//         R = reward + gamma * R
//         discounted_rewards.insert(0, R)
//     return torch.tensor(discounted_rewards)

// def calculate_gae(next_value, rewards, masks, values, gamma=0.999, tau=0.95):
//     gae = 0
//     returns = []
//     for step in reversed(range(len(rewards))):
//         delta = rewards[step] + gamma * next_value * masks[step] - values[step]
//         gae = delta + gamma * tau * masks[step] * gae
//         next_value = values[step]
//         returns.insert(0, gae + values[step])
//     return returns

// def trinal_clip_policy_loss(advantages, old_log_probs, log_probs, epsilon=0.2, delta1=3):
//     ratio = torch.exp(log_probs - old_log_probs)
//     surr1 = ratio * advantages
//     surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
//     surr3 = torch.clamp(ratio, 1 - epsilon, delta1) * advantages
//     policy_loss = -torch.min(torch.min(surr1, surr2), surr3)
//     return policy_loss.mean()

// def clipped_value_loss(old_values, values, rewards, delta2, delta3):
//     clipped_values = torch.clamp(rewards, -delta2, delta3)
//     value_loss = (clipped_values - values).pow(2)
//     return value_loss.mean()

// # Assuming we have the necessary inputs:
// # actions, log_probs, old_log_probs, values, rewards, masks, next_value
// # You would need to modify this to fit into your environment and training loop.

// # Example usage within a training loop:
// advantages = calculate_gae(next_value, rewards, masks, values)
// discounted_rewards = calculate_discounted_rewards(rewards)
// policy_loss = trinal_clip_policy_loss(advantages, old_log_probs, log_probs)
// value_loss = clipped_value_loss(old_values, values, discounted_rewards, delta2, delta3)
