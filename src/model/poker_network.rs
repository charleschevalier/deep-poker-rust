use super::actor_network::ActorNetwork;
use super::critic_network::CriticNetwork;
use super::siamese_network::SiameseNetwork;
use crate::game::{
    action::{ActionConfig, ActionType},
    action_state::{self, ActionState},
    hand_state::HandState,
};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use poker::card;

pub struct PokerNetwork<'a> {
    player_count: u32,
    action_config: &'a ActionConfig,
    siamese_network: SiameseNetwork,
    actor_network: ActorNetwork,
    critic_network: Option<CriticNetwork>,
    device: Device,
}

impl<'a> PokerNetwork<'a> {
    pub fn new(
        player_count: u32,
        action_config: &ActionConfig,
        device: Device,
        train: bool,
    ) -> PokerNetwork {
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &Device::Cpu);

        let siamese_network = SiameseNetwork::new(
            player_count,
            3 + action_config.postflop_raise_sizes.len() as u32, // Each raise size + fold, call, check
            player_count as usize * 3, // 3 actions max per player per street => TODO: prevent situations where we have more than 3 actions
            &vb,
        );

        let actor_network =
            ActorNetwork::new(&vb, 3 + action_config.postflop_raise_sizes.len() as usize);

        let critic_network = if train {
            Some(CriticNetwork::new(&vb))
        } else {
            None
        };

        PokerNetwork {
            player_count,
            action_config,
            device,
            siamese_network,
            actor_network,
            critic_network,
        }
    }

    pub fn forward(
        &self,
        card_tensor: &Tensor,
        action_tensor: &Tensor,
    ) -> (Tensor, Option<Tensor>) {
        let x = self.siamese_network.forward(card_tensor, action_tensor);
        let actor_output = self.actor_network.forward(&x);

        if let Some(critic_network) = &self.critic_network {
            let critic_output = critic_network.forward(&x);
            (actor_output, Some(critic_output))
        } else {
            (actor_output, None)
        }
    }
}
