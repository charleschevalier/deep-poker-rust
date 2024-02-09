use super::actor_network::ActorNetwork;
use super::critic_network::CriticNetwork;
use super::siamese_network::SiameseNetwork;
use crate::game::action::ActionConfig;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};

pub struct PokerNetwork {
    siamese_network: SiameseNetwork,
    actor_network: ActorNetwork,
    critic_network: Option<CriticNetwork>,
    pub var_map_actor: VarMap,
    pub var_map_critic_encoder: VarMap,
}

impl PokerNetwork {
    pub fn new(
        player_count: u32,
        action_config: &ActionConfig,
        device: Device,
        train: bool,
    ) -> Result<PokerNetwork, candle_core::Error> {
        let var_map_actor = VarMap::new();
        let var_map_critic_encoder = VarMap::new();
        let vb_actor = VarBuilder::from_varmap(&var_map_actor, DType::F32, &device);
        let vb_critic_encoder =
            VarBuilder::from_varmap(&var_map_critic_encoder, DType::F32, &device);

        let siamese_network = SiameseNetwork::new(
            player_count,
            3 + action_config.postflop_raise_sizes.len() as u32, // Each raise size + fold, call, check
            player_count as usize * 3, // 3 actions max per player per street => TODO: prevent situations where we have more than 3 actions
            &vb_critic_encoder,
        )?;

        let actor_network =
            ActorNetwork::new(&vb_actor, 3 + action_config.postflop_raise_sizes.len())?;

        let critic_network = if train {
            Some(CriticNetwork::new(&vb_critic_encoder)?)
        } else {
            None
        };

        Ok(PokerNetwork {
            siamese_network,
            actor_network,
            critic_network,
            var_map_actor,
            var_map_critic_encoder,
        })
    }

    pub fn forward(
        &self,
        card_tensor: &Tensor,
        action_tensor: &Tensor,
    ) -> Result<(Tensor, Option<Tensor>), candle_core::Error> {
        let x = self.siamese_network.forward(card_tensor, action_tensor)?;
        let actor_output = self.actor_network.forward(&x)?;

        if let Some(critic_network) = &self.critic_network {
            let critic_output = critic_network.forward(&x)?;
            Ok((actor_output, Some(critic_output)))
        } else {
            Ok((actor_output, None))
        }
    }
}
