use std::collections::HashMap;

use super::actor_network::ActorNetwork;
use super::critic_network::CriticNetwork;
use super::siamese_network::SiameseNetwork;
use crate::{game::action::ActionConfig, helper};
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{VarBuilder, VarMap};

pub struct PokerNetwork {
    siamese_network: SiameseNetwork,
    actor_network: ActorNetwork,
    critic_network: CriticNetwork,
    var_map: VarMap,
    player_cnt: u32,
    action_config: ActionConfig,
    clone_device: Device,
    train: bool,
}

impl PokerNetwork {
    pub fn new(
        player_count: u32,
        action_config: ActionConfig,
        device: Device,
        clone_device: Device,
        train: bool,
    ) -> Result<PokerNetwork, Box<dyn std::error::Error>> {
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);

        let siamese_network = SiameseNetwork::new(
            // let siamese_network = SiameseNetworkLinear::new(
            player_count,
            3 + action_config.postflop_raise_sizes.len() as u32, // Each raise size + fold, call, check
            player_count as usize * 3, // 3 actions max per player per street => TODO: prevent situations where we have more than 3 actions
            vb.pp("siamese"),
        )?;

        let actor_network =
            ActorNetwork::new(vb.pp("actor"), 3 + action_config.postflop_raise_sizes.len())?;

        let critic_network = CriticNetwork::new(vb.pp("critic"))?;

        Ok(PokerNetwork {
            siamese_network,
            actor_network,
            critic_network,
            var_map,
            player_cnt: player_count,
            action_config,
            clone_device,
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

    pub fn get_var_map(&self) -> &VarMap {
        &self.var_map
    }

    pub fn load_var_map<P: AsRef<std::path::Path>>(
        &mut self,
        file_path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.var_map.load(file_path)?;
        self.set_batch_norm_tensors(self.var_map.clone())?;
        Ok(())
    }

    pub fn set_batch_norm_tensors(&mut self, var_map: VarMap) -> Result<(), candle_core::Error> {
        let mut siamese_tensors = HashMap::new();
        for (k, v) in var_map.data().lock().unwrap().iter() {
            if let Some(stripped) = k.strip_prefix("siamese.") {
                siamese_tensors.insert(stripped.to_string(), v.as_tensor().copy()?);
            }
        }
        self.siamese_network.set_batch_norm_tensors(siamese_tensors);
        Ok(())
    }

    pub fn save_var_map<P: AsRef<std::path::Path>>(
        &self,
        file_path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let batch_norm_tensors = self.get_batch_norm_tensors();

        for (k, v) in self.var_map.data().lock().unwrap().iter() {
            if batch_norm_tensors.contains_key(k) {
                let tensor = batch_norm_tensors.get(k).unwrap();
                let vt = v.as_tensor().clone();
                let diff = vt
                    .sub(tensor)
                    .unwrap()
                    .abs()?
                    .sum_all()?
                    .to_scalar::<f32>()?;

                if diff > 1e-6 {
                    v.set(batch_norm_tensors.get(k).unwrap())?;
                }
            }
        }

        self.var_map.save(file_path)?;

        Ok(())
    }

    pub fn get_siamese_vars(&self) -> Vec<Var> {
        helper::filter_var_map_by_prefix(&self.var_map, &["siamese"])
    }

    pub fn get_actor_vars(&self) -> Vec<Var> {
        helper::filter_var_map_by_prefix(&self.var_map, &["actor"])
    }

    pub fn get_critic_vars(&self) -> Vec<Var> {
        helper::filter_var_map_by_prefix(&self.var_map, &["critic"])
    }

    pub fn get_batch_norm_tensors(&self) -> HashMap<String, Tensor> {
        let mut result = HashMap::new();
        let tensors = self.siamese_network.get_batch_norm_tensors();
        for (k, v) in tensors {
            result.insert(format!("siamese.{}", k), v);
        }
        result
    }
}

impl Clone for PokerNetwork {
    // The clone is not trainable and on CPU by default
    fn clone(&self) -> PokerNetwork {
        let mut copy_net = Self::new(
            self.player_cnt,
            self.action_config.clone(),
            self.clone_device.clone(),
            self.clone_device.clone(),
            false,
        )
        .unwrap();

        {
            let var_map = self.var_map.data().lock().unwrap();
            let new_var_map = copy_net.var_map.data().lock().unwrap();

            // We perform a deep copy of the varmap
            var_map.iter().for_each(|(k, v)| {
                let new_tensor = candle_core::Var::from_tensor(
                    &v.copy().unwrap().to_device(&self.clone_device).unwrap(),
                )
                .unwrap();
                if let Some(v) = new_var_map.get(k) {
                    v.set(&new_tensor).unwrap();
                }
            });
        }
        copy_net
            .set_batch_norm_tensors(copy_net.var_map.clone())
            .unwrap();

        copy_net
    }
}
