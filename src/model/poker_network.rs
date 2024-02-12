use super::actor_network::ActorNetwork;
use super::critic_network::CriticNetwork;
use super::siamese_network::SiameseNetwork;
use crate::game::action::ActionConfig;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};

pub struct PokerNetwork<'a> {
    siamese_network: SiameseNetwork,
    actor_network: ActorNetwork,
    critic_network: Option<CriticNetwork>,
    pub var_map: VarMap,
    pub var_builder: VarBuilder<'a>,
}

impl<'a> PokerNetwork<'a> {
    pub fn new(
        player_count: u32,
        action_config: &ActionConfig,
        device: Device,
        train: bool,
    ) -> Result<PokerNetwork, Box<dyn std::error::Error>> {
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);

        let siamese_network = SiameseNetwork::new(
            player_count,
            3 + action_config.postflop_raise_sizes.len() as u32, // Each raise size + fold, call, check
            player_count as usize * 3, // 3 actions max per player per street => TODO: prevent situations where we have more than 3 actions
            &vb,
        )?;

        let actor_network = ActorNetwork::new(&vb, 3 + action_config.postflop_raise_sizes.len())?;

        let critic_network = if train {
            Some(CriticNetwork::new(&vb)?)
        } else {
            None
        };

        Ok(PokerNetwork {
            siamese_network,
            actor_network,
            critic_network,
            var_map,
            var_builder: vb,
        })
    }

    pub fn forward(
        &self,
        card_tensor: &Tensor,
        action_tensor: &Tensor,
    ) -> Result<(Tensor, Option<Tensor>), Box<dyn std::error::Error>> {
        let x = self.siamese_network.forward(card_tensor, action_tensor)?;
        let actor_output = self.actor_network.forward(&x)?;

        if let Some(critic_network) = &self.critic_network {
            let critic_output = critic_network.forward(&x)?;
            Ok((actor_output, Some(critic_output)))
        } else {
            Ok((actor_output, None))
        }
    }

    // pub fn get_policy_vars(&self) -> Vec<&candle_core::Var> {
    //     let mut result: Vec<&candle_core::Var> = Vec::new();
    //     result.append(self.actor_network.get_weights(self.var_builder)?);
    //     self.actor_network.get_vars()
    // }
}
