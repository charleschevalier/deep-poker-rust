use super::actor_network::ActorNetwork;
use super::critic_network::CriticNetwork;
use super::siamese_network_conv::SiameseNetworkConv;
// use super::siamese_network_linear::SiameseNetworkLinear;
use crate::game::action::ActionConfig;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};

pub struct PokerNetwork<'a> {
    siamese_network: SiameseNetworkConv,
    // siamese_network: SiameseNetworkLinear,
    actor_network: ActorNetwork,
    critic_network: CriticNetwork,
    pub var_map: VarMap,

    player_cnt: u32,
    action_config: &'a ActionConfig,
    device: Device,
    train: bool,
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

        let siamese_network = SiameseNetworkConv::new(
            // let siamese_network = SiameseNetworkLinear::new(
            player_count,
            3 + action_config.postflop_raise_sizes.len() as u32, // Each raise size + fold, call, check
            player_count as usize * 3, // 3 actions max per player per street => TODO: prevent situations where we have more than 3 actions
            &vb,
        )?;

        let actor_network = ActorNetwork::new(&vb, 3 + action_config.postflop_raise_sizes.len())?;

        let critic_network = CriticNetwork::new(&vb)?;

        Ok(PokerNetwork {
            siamese_network,
            actor_network,
            critic_network,
            var_map,
            player_cnt: player_count,
            action_config,
            device,
            train,
        })
    }

    pub fn forward_embedding_actor(
        &self,
        card_tensor: &Tensor,
        action_tensor: &Tensor,
        train: bool,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let x = self
            .siamese_network
            .forward(card_tensor, action_tensor, train)?;
        self.actor_network.forward(&x)
    }

    pub fn forward_embedding(
        &self,
        card_tensor: &Tensor,
        action_tensor: &Tensor,
        train: bool,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        Ok(self
            .siamese_network
            .forward(card_tensor, action_tensor, train)?)
    }

    pub fn forward_actor(&self, x: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
        self.actor_network.forward(x)
    }

    pub fn forward_critic(&self, x: &Tensor) -> Result<Option<Tensor>, Box<dyn std::error::Error>> {
        if self.train {
            let critic_output = self.critic_network.forward(x)?;
            Ok(Some(critic_output))
        } else {
            Ok(None)
        }
    }

    // pub fn _print_weights(&self) {
    //     self.siamese_network._print_weights();
    // }
}

impl<'a> Clone for PokerNetwork<'a> {
    // The clone is not trainable
    fn clone(&self) -> PokerNetwork<'a> {
        let mut copy_net = Self::new(
            self.player_cnt,
            self.action_config,
            self.device.clone(),
            false,
        )
        .unwrap();

        let var_map = self.var_map.data().lock().unwrap();
        // We perform a deep copy of the varmap using Tensor::copy on Var
        var_map.iter().for_each(|(k, v)| {
            copy_net
                .var_map
                .set_one(k, v.as_tensor().copy().unwrap())
                .unwrap();
        });

        copy_net
    }
}
