use super::Agent;
use crate::game::hand_state::HandState;
use crate::model::poker_network::PokerNetwork;
use candle_core::Tensor;

use rand::Rng;

pub struct AgentNetwork {
    network: PokerNetwork,
}

impl<'a> Agent<'a> for AgentNetwork {
    fn choose_action(
        &self,
        hand_state: &HandState,
        valid_action_mask: &[bool],
        street: u8,
        action_config: &crate::game::action::ActionConfig,
        device: &candle_core::Device,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let (card_tensor, action_tensor) = hand_state.to_input(
            street,
            action_config,
            device,
            hand_state.action_states.len(),
            valid_action_mask,
        )?;

        let (proba_tensor, _) = self
            .network
            .forward(&card_tensor.unsqueeze(0)?, &action_tensor.unsqueeze(0)?)?;

        Ok(Self::choose_action_from_net(
            &proba_tensor,
            valid_action_mask,
        )?)
    }
}

impl<'a> AgentNetwork {
    pub fn new(network: PokerNetwork) -> AgentNetwork {
        AgentNetwork { network }
    }

    pub fn choose_action_from_net(
        proba_tensor: &Tensor,
        valid_action_mask: &[bool],
    ) -> Result<usize, Box<dyn std::error::Error>> {
        // Apply valid action mask to tensor
        let mut probas = proba_tensor.squeeze(0)?.to_vec1()?;
        // for i in 0..probas.len() {
        //     if i >= valid_action_mask.len() || !valid_action_mask[i] {
        //         probas[i] = 0.0;
        //     }
        // }

        // Normalize probas
        let sum: f32 = probas.iter().sum();
        for p in &mut probas {
            *p /= sum;
        }

        // Choose action based on the probability distribution
        let mut rng = rand::thread_rng();
        let random_float_0_1: f32 = rng.gen();
        let mut sum: f32 = 0.0;
        let mut action_index: usize = 0;
        for (i, p) in probas.iter().enumerate() {
            sum += p;
            if sum > random_float_0_1 {
                action_index = i;
                break;
            }
        }

        if action_index >= valid_action_mask.len() || !valid_action_mask[action_index] {
            println!("Invalid action index: {}", action_index);
            println!("Probas: {:?}", probas);
            return Err("Invalid action index".into());
        }

        Ok(action_index)
    }
}
