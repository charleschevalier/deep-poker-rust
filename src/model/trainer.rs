use std::vec;

use super::poker_network::PokerNetwork;
use super::trainer_config::TrainerConfig;
use crate::game::action::ActionConfig;
use crate::game::hand_state::{self, HandState};
use crate::game::tree::Tree;
use candle_core::{DType, Device, Tensor};

use rand::seq::SliceRandom;
use rand::thread_rng;

pub struct Trainer<'a> {
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
        let gamma = 0.99;
        let lamda = 0.95;

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
                    self.tree.traverse(player, &vec![&trained_network; 3]);

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
                let mut next_card_tensor_vec = Vec::new();
                let mut next_action_tensor_vec = Vec::new();
                let mut rewards = Vec::new();
                let mut terminal_mask = Vec::new();

                for (hand_state, action_state_index) in batch {
                    let (
                        card_tensor,
                        action_tensor,
                        next_card_tensor,
                        next_action_tensor,
                        is_terminal,
                    ) = hand_state.get_tensors(
                        *action_state_index,
                        self.action_config,
                        &self.device,
                    );
                    card_tensor_vec.push(card_tensor);
                    action_tensor_vec.push(action_tensor);
                    next_card_tensor_vec.push(next_card_tensor);
                    next_action_tensor_vec.push(next_action_tensor);
                    rewards.push(hand_state.action_states[*action_state_index].reward);
                    terminal_mask.push(if !is_terminal { 1.0 } else { 0.0 });
                }

                // println!("{:?}", card_tensor_vec.len());
                // println!("{:?}", action_tensor_vec.len());
                // println!("{:?}", next_card_tensor_vec.len());
                // println!("{:?}", next_action_tensor_vec.len());
                // println!("{:?}", rewards.len());
                // println!("{:?}", terminal_mask.len());

                let card_tensors = Tensor::stack(&card_tensor_vec, 0).unwrap();
                let action_tensors = Tensor::stack(&action_tensor_vec, 0).unwrap();
                let next_card_tensors = Tensor::stack(&next_card_tensor_vec, 0).unwrap();
                let next_action_tensors = Tensor::stack(&next_action_tensor_vec, 0).unwrap();

                // println!("{:?}", card_tensors.shape());
                // println!("{:?}", action_tensors.shape());
                // println!("{:?}", next_card_tensors.shape());
                // println!("{:?}", next_action_tensors.shape());

                // Forward pass
                let (actor_output, critic_output) =
                    trained_network.forward(&card_tensors, &action_tensors);
                let (_, next_critic_output) =
                    trained_network.forward(&next_card_tensors, &next_action_tensors);

                // println!("{:?}", actor_output.shape());
                // println!("{:?}", critic_output.as_ref().unwrap().shape());
                // println!("{:?}", next_critic_output.as_ref().unwrap().shape());

                // Calculate advantage GAE
                let (advantage_gae, values_target) = self.calculate_advantage_gae(
                    &rewards,
                    &critic_output
                        .unwrap()
                        .squeeze(1)
                        .unwrap()
                        .to_vec1()
                        .unwrap(),
                    &next_critic_output
                        .unwrap()
                        .squeeze(1)
                        .unwrap()
                        .to_vec1()
                        .unwrap(),
                    gamma,
                    lamda,
                    &terminal_mask,
                );

                // println!("{:?}", advantage_gae);
                // println!("{:?}", values_target);
            }
        }
    }

    fn train_model(&self, batch: &Vec<Option<HandState>>) {
        // Get inputs and rewards
    }

    fn calculate_advantage_gae(
        &self,
        rewards: &Vec<f32>,
        values: &Vec<f32>,
        next_values: &Vec<f32>,
        gamma: f32,
        lamda: f32,
        terminal_mask: &Vec<f32>,
    ) -> (Vec<f32>, Vec<f32>) {
        let batch_size = rewards.len();
        let mut advantage = vec![0.0; batch_size + 1];

        for t in (0..batch_size).rev() {
            let delta = rewards[t] + (gamma * next_values[t] * terminal_mask[t]) - values[t];
            advantage[t] = delta + (gamma * lamda * advantage[t + 1] * terminal_mask[t]);
        }
        let mut value_target = values.clone();
        for i in 0..batch_size {
            value_target[i] += advantage[i];
        }

        advantage.truncate(batch_size);
        (advantage, value_target)
    }

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
