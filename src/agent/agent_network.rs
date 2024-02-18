use super::Agent;
use crate::game::hand_state::HandState;
use crate::model::poker_network::PokerNetwork;
use candle_core::Tensor;

use rand::distributions::Distribution;

pub struct AgentNetwork {
    network: PokerNetwork,
}

impl Agent for AgentNetwork {
    fn clone_box(&self) -> Box<dyn Agent> {
        Box::new(AgentNetwork {
            network: self.network.clone(),
        })
    }

    fn choose_action(
        &self,
        hand_state: &HandState,
        valid_actions_mask: &[bool],
        street: u8,
        action_config: &crate::game::action::ActionConfig,
        device: &candle_core::Device,
        no_invalid: bool,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let (card_tensor, action_tensor) = hand_state.to_input(
            street,
            action_config,
            device,
            hand_state.action_states.len(),
        )?;

        let proba_tensor = self
            .network
            .forward_embedding_actor(&card_tensor.unsqueeze(0)?, &action_tensor.unsqueeze(0)?)?
            .detach()?;

        Self::choose_action_from_net(&proba_tensor, valid_actions_mask, no_invalid)
    }
}

impl AgentNetwork {
    pub fn new(network: PokerNetwork) -> AgentNetwork {
        AgentNetwork { network }
    }

    pub fn choose_action_from_net(
        proba_tensor: &Tensor,
        valid_actions_mask: &[bool],
        no_invalid: bool,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        // Apply valid action mask to tensor
        let mut probas = proba_tensor.squeeze(0)?.to_vec1()?;
        for i in 0..probas.len() {
            if no_invalid && (i >= valid_actions_mask.len() || !valid_actions_mask[i]) {
                probas[i] = 0.0;
            }
        }

        // Normalize probas
        let sum_norm: f32 = probas.iter().sum();
        if sum_norm > 1e-8 {
            for p in &mut probas {
                *p /= sum_norm;
            }
        } else {
            // Count positive values in valid_actions_mask
            let true_count = if no_invalid {
                valid_actions_mask.iter().filter(|&&x| x).count()
            } else {
                probas.len()
            };
            for (i, p) in probas.iter_mut().enumerate() {
                if i < valid_actions_mask.len() && valid_actions_mask[i] {
                    *p = 1.0 / (true_count as f32);
                }
            }
        }

        // Choose action based on the probability distribution
        let mut rng = rand::thread_rng();
        // let random_float_0_1: f32 = rng.gen();
        // let mut sum: f32 = 0.0;
        // let mut action_index: usize = 0;
        // for (i, p) in probas.iter().enumerate() {
        //     sum += p;
        //     if sum > random_float_0_1 {
        //         action_index = i;
        //         break;
        //     }
        // }
        let distribution = rand::distributions::WeightedIndex::new(probas).unwrap();
        let action_index = distribution.sample(&mut rng);

        if no_invalid
            && (action_index >= valid_actions_mask.len() || !valid_actions_mask[action_index])
        {
            // println!("Invalid action index: {}", action_index);
            // println!("Probas: {:?}", probas);
            return Err("Invalid action index".into());
        }

        Ok(action_index)
    }
}
