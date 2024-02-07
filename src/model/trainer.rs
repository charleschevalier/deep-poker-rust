use super::poker_network::PokerNetwork;
use super::trainer_config::TrainerConfig;
use crate::game::action::ActionConfig;
use crate::game::hand_state::{self, HandState};
use crate::game::tree::Tree;
use candle_core::{DType, Device, Tensor};

use rand::seq::SliceRandom;
use rand::thread_rng;

struct Trainer<'a> {
    player_cnt: u32,
    action_config: &'a ActionConfig,
    trainer_config: &'a TrainerConfig,
    tree: Tree<'a>,
    device: Device,
}

impl<'a> Trainer<'a> {
    pub fn new(
        player_cnt: u32,
        action_config: &'a ActionConfig,
        trainer_config: &'a TrainerConfig,
        device: Device,
    ) -> Trainer<'a> {
        Trainer {
            player_cnt,
            action_config,
            trainer_config,
            tree: Tree::new(player_cnt, action_config),
            device,
        }
    }

    pub fn train(&mut self) {
        let mut trained_network = PokerNetwork::new(
            self.player_cnt,
            self.action_config,
            self.device.clone(),
            true,
        );

        for _ in 0..self.trainer_config.max_iters {
            let mut batch_states: Vec<(HandState, usize)> = Vec::new();

            for _ in 0..self.trainer_config.hands_per_player_per_iteration {
                for player in 0..self.player_cnt {
                    // TODO: select networks to use here
                    self.tree.traverse(player, &vec![]);

                    for i in 0..self.tree.hand_state.as_ref().unwrap().action_states.len() {
                        if self.tree.hand_state.as_ref().unwrap().action_states[i].player_to_move
                            == player
                        {
                            batch_states.push((self.tree.hand_state.as_mut().unwrap().clone(), i));
                        }
                    }
                }
            }

            // Iterate through batch states
            /* vec![vec![vec![0.0; 4]; 13]; 5]; */

            for batch in batch_states.chunks(self.trainer_config.batch_size) {
                // Calculate inputs for each batch
                let mut card_tensor_vec = Vec::new();
                let mut action_tensor_vec = Vec::new();

                for (hand_state, action_state_index) in batch {
                    let (card_tensor, action_tensor) =
                        hand_state.to_input(*action_state_index, self.action_config, &self.device);
                    card_tensor_vec.push(card_tensor);
                    action_tensor_vec.push(action_tensor);
                }

                let card_tensors = Tensor::stack(&card_tensor_vec, 0).unwrap();
                let action_tensors = Tensor::stack(&action_tensor_vec, 0).unwrap();

                // Forward pass
                let (actor_output, critic_output) =
                    trained_network.forward(&card_tensors, &action_tensors);

                // Calculate advantage GAE
                let mut advantage_gae = Vec::new();
                for (hand_state, action_state_index) in batch {
                    let gae = self.calculate_advantage_gae(hand_state, 0.999, 0.95);
                    advantage_gae.push(gae);
                }
            }

            // if dataset.len() >= self.trainer_config.max_dataset_cache {
            //     while dataset.len() > self.trainer_config.batch_size {
            //         // Take a data batch
            //         let batch: Vec<_> = dataset.drain(0..self.trainer_config.batch_size).collect();

            //         // Train the model
            //         self.train_model(&batch);
            //     }
            // }
            // if dataset.len() >= self.trainer_config.max_dataset_cache {
            //     while dataset.len() > self.trainer_config.batch_size {
            //         // Take a data batch
            //         let batch: Vec<_> = dataset.drain(0..self.trainer_config.batch_size).collect();

            //         // Train the model
            //         self.train_model(&batch);
            //     }
            // }
        }
    }

    fn train_model(&self, batch: &Vec<Option<HandState>>) {
        // Get inputs and rewards
    }

    fn calculate_advantage_gae(&self, hand_state: &HandState, gamma: f32, tau: f32) -> Vec<f32> {
        let mut gae = 0.0;
        let mut returns = Vec::new();

        // for action_state in hand_state.action_states.iter().rev() {
        //     let delta = action_state.reward + gamma * action_state.next_value * action_state.mask
        //         - action_state.value;
        //     gae = delta + gamma * tau * action_state.mask * gae;
        //     returns.insert(0, gae + action_state.value);
        // }

        returns
    }

    // def calculate_gae(next_value, rewards, masks, values, gamma=0.999, tau=0.95):
    //     gae = 0
    //     returns = []
    //     for step in reversed(range(len(rewards))):
    //         delta = rewards[step] + gamma * next_value * masks[step] - values[step]
    //         gae = delta + gamma * tau * masks[step] * gae
    //         next_value = values[step]
    //         returns.insert(0, gae + values[step])
    //     return returns

    fn get_inputs(&self, hand: &HandState) -> Vec<f32> {
        // let mut card_inputs = Vec::new();
        // let mut action_inputs = Vec::new();

        // for action_state in hand.action_states {
        //     //inputs.push(action_state.get_input());
        // }

        //inputs
        Vec::new()
    }

    fn get_input(&self, hand_state: &HandState) -> Vec<f32> {
        let mut inputs = Vec::new();

        // for card in hand_state.hand {
        //     //inputs.push(card.get_input());
        // }

        // for card in hand_state.board {
        //     //inputs.push(card.get_input());
        // }

        inputs
    }

    fn get_discounted_rewards(&self, hand_state: &HandState, gamma: f32) -> Vec<f32> {
        let mut rewards = Vec::new();
        let mut reward = 0.0;

        for action_state in hand_state.action_states.iter().rev() {
            reward = action_state.reward + gamma * reward;
            rewards.push(reward);
        }

        rewards.reverse();
        rewards
    }

    fn get_card_inputs(&self, hand_state: &HandState) -> Vec<Vec<Tensor>> {
        let mut inputs = Vec::new();

        // for card in hand_state.hand {
        //     //inputs.push(card.get_input());
        // }

        // for card in hand_state.board {
        //     //inputs.push(card.get_input());
        // }

        inputs
    }

    // def calculate_discounted_rewards(rewards, gamma=0.999):
    //     discounted_rewards = []
    //     R = 0
    //     for reward in reversed(rewards):
    //         R = reward + gamma * R
    //         discounted_rewards.insert(0, R)
    //     return torch.tensor(discounted_rewards)
}
