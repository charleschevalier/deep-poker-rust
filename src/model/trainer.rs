use std::vec;

use super::poker_network::PokerNetwork;
use super::trainer_config::TrainerConfig;
use crate::game::action::ActionConfig;
use crate::game::hand_state::{self, HandState};
use crate::game::tree::Tree;
use candle_core::{DType, Device, Tensor};

use rand::seq::{index, SliceRandom};
use rand::{thread_rng, Error};

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

    pub fn train(&mut self) -> Result<(), candle_core::Error> {
        let gae_gamma = 0.99;
        let gae_lamda = 0.95;
        let reward_gamma = 0.999;

        let mut trained_network = PokerNetwork::new(
            self.player_cnt,
            self.action_config,
            self.device.clone(),
            true,
        );

        for _ in 0..self.trainer_config.max_iters {
            let mut hand_states = Vec::new();

            for _ in 0..self.trainer_config.hands_per_player_per_iteration {
                for player in 0..self.player_cnt {
                    // TODO: select networks to use here
                    self.tree.traverse(player, &vec![&trained_network; 3]);

                    // Make sure the hand state has at least one state for the traverser
                    let hs = self.tree.hand_state.clone();
                    if let Some(hs) = hs {
                        if hs.action_states.is_empty() {
                            continue;
                        }
                        hand_states.push(hs);
                    }
                }
            }

            // Calculate cumulative rewards for each hand state
            let mut rewards = Vec::new();
            let mut rewards_flat: Vec<f32> = Vec::new();
            let mut cnt = 0;
            let mut indexes = Vec::new();

            for hand_state in hand_states.iter_mut() {
                indexes.push(cnt);
                let hand_rewards = self.get_discounted_rewards(hand_state, reward_gamma);

                cnt += hand_rewards.len();
                rewards_flat.extend(hand_rewards.iter());
                rewards.push(hand_rewards);
            }

            let rewards_flat_tensor = Tensor::new(rewards_flat, &self.device)?;

            // Get network inputs
            let mut card_input_vec = Vec::new();
            let mut action_input_vec = Vec::new();

            for hand_state in hand_states.iter() {
                let (card_tensors, action_tensors) =
                    hand_state.get_all_tensors(self.action_config, &self.device);

                card_input_vec.push(card_tensors);
                action_input_vec.push(action_tensors);
            }

            println!("Card input vec: {:?}", card_input_vec[0].shape());
            println!("Action input vec: {:?}", action_input_vec[0].shape());

            let card_input_tensor = Tensor::cat(&card_input_vec, 0)?;
            let action_input_tensor = Tensor::cat(&action_input_vec, 0)?;

            println!("Card input tensor: {:?}", card_input_tensor.shape());
            println!("Action input tensor: {:?}", action_input_tensor.shape());

            // Run all states through network
            let (base_actor_outputs, _) =
                trained_network.forward(&card_input_tensor, &action_input_tensor);

            // Get action probabilities for actions taken by the agent
            let old_probs_raw: Vec<Vec<f32>> = base_actor_outputs.to_vec2()?;
            let old_probs_tensor = Tensor::new(
                self.get_action_probs(&hand_states, &old_probs_raw, &indexes),
                &self.device,
            )?
            .log()?;

            for _ in 0..self.trainer_config.update_step {
                // Run all states through network
                let (actor_outputs, critic_outputs) =
                    trained_network.forward(&card_input_tensor, &action_input_tensor);

                let probs_raw: Vec<Vec<f32>> = actor_outputs.to_vec2()?;
                let probs = self.get_action_probs(&hand_states, &probs_raw, &indexes);

                let critic_outputs_vec: Vec<f32> =
                    critic_outputs.as_ref().unwrap().squeeze(1)?.to_vec1()?;

                println!("actor_outputs: {:?}", actor_outputs.shape());
                println!(
                    "critic_outputs: {:?}",
                    critic_outputs.as_ref().unwrap().shape()
                );

                // Calculate advantage GAE for each hand state
                let mut advantage_gae: Vec<f32> = Vec::new();

                for i in 0..rewards.len() {
                    let mut advantage = self.calculate_advantage_gae(
                        &rewards[i],
                        &critic_outputs_vec[indexes[i]..indexes[i] + rewards[i].len()],
                        gae_gamma,
                        gae_lamda,
                    );
                    advantage_gae.append(&mut advantage);
                }

                let advantage_tensor = Tensor::new(advantage_gae, &self.device)?;

                // Get trinal clip PPO
                let policy_loss = self.get_trinal_clip_policy_loss(
                    &advantage_tensor,
                    &Tensor::new(probs, &self.device)?.log()?,
                    &old_probs_tensor,
                );

                println!("Policy loss: {:?}", policy_loss?.to_scalar::<f32>());

                let value_loss = self.get_trinal_clip_value_loss(
                    critic_outputs.as_ref().unwrap(),
                    &rewards_flat_tensor,
                );

                println!("Value loss: {:?}", value_loss?.to_scalar::<f32>());
                println!("TOTO");
            }
        }

        Ok(())
    }

    fn get_action_probs(
        &self,
        hand_states: &[HandState],
        probs_raw: &[Vec<f32>],
        indexes: &[usize],
    ) -> Vec<f32> {
        let mut probs = Vec::new();

        for (i, hand_state) in hand_states.iter().enumerate() {
            for action_state in hand_state.action_states.iter() {
                probs.push(probs_raw[indexes[i]][action_state.action_taken_index]);
            }
        }

        probs
    }

    fn calculate_advantage_gae(
        &self,
        rewards: &[f32],
        values: &[f32],
        gamma: f32,
        lamda: f32,
    ) -> Vec<f32> {
        let size = rewards.len();
        let mut advantage = vec![0.0; size + 1];

        for t in (0..size).rev() {
            let delta =
                rewards[t] + gamma * (if t == size - 1 { 0.0 } else { values[t + 1] }) - values[t];
            advantage[t] = delta + gamma * lamda * advantage[t + 1];
        }
        let mut value_target = values.to_vec();
        for i in 0..size {
            value_target[i] += advantage[i];
        }

        value_target
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

    // def trinal_clip_policy_loss(advantages, old_log_probs, log_probs, epsilon=0.2, delta1=3):
    //     ratio = torch.exp(log_probs - old_log_probs)
    //     surr1 = ratio * advantages
    //     surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    //     surr3 = torch.clamp(ratio, 1 - epsilon, delta1) * advantages
    //     policy_loss = -torch.min(torch.min(surr1, surr2), surr3)
    //     return policy_loss.mean()

    fn get_trinal_clip_policy_loss(
        &self,
        advantages: &Tensor,
        log_probs: &Tensor,
        old_log_probs: &Tensor,
    ) -> Result<Tensor, candle_core::Error> {
        let ratio = (log_probs - old_log_probs)?.exp();
        let clip1 = ratio?.clamp(
            1.0 - self.trainer_config.ppo_epsilon,
            1.0 + self.trainer_config.ppo_epsilon,
        )?;
        let clip2 = clip1.clamp(
            1.0 - self.trainer_config.ppo_epsilon,
            self.trainer_config.ppo_delta_1,
        )?;

        let neg = advantages.lt(0.0)?;

        // Works because clip1 & 2 have only one dim like neg
        let final_clip = neg.where_cond(&clip2, &clip1)?;

        let policy_loss = final_clip * advantages;
        policy_loss?.mean(0)?.neg()
    }

    fn get_trinal_clip_value_loss(
        &self,
        values: &Tensor,
        rewards: &Tensor,
    ) -> Result<Tensor, candle_core::Error> {
        let clipped = rewards.clamp(
            -self.trainer_config.ppo_delta_2,
            self.trainer_config.ppo_delta_3,
        )?;
        let diff = (clipped - values.squeeze(1))?;
        (diff.as_ref() * diff.as_ref())?.mean(0)
    }
}
