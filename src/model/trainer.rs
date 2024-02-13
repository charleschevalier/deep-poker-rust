use std::vec;

use super::poker_network::PokerNetwork;
use super::trainer_config::TrainerConfig;
use crate::agent::agent_network::AgentNetwork;
use crate::agent::agent_pool::AgentPool;
use crate::game::hand_state::HandState;
use crate::game::tree::Tree;
use crate::{agent::Agent, game::action::ActionConfig};
use candle_core::{Device, Tensor};
use candle_nn::Optimizer;

use std::path::Path;

pub struct Trainer<'a> {
    player_cnt: u32,
    action_config: &'a ActionConfig,
    trainer_config: &'a TrainerConfig,
    tree: Tree<'a>,
    device: Device,
    output_path: &'a str,
}

impl<'a> Trainer<'a> {
    pub fn new(
        player_cnt: u32,
        action_config: &'a ActionConfig,
        trainer_config: &'a TrainerConfig,
        device: Device,
        output_path: &'a str,
    ) -> Trainer<'a> {
        Trainer {
            player_cnt,
            action_config,
            trainer_config,
            tree: Tree::new(player_cnt, action_config),
            device,
            output_path,
        }
    }

    pub fn train(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let gae_gamma = 0.99;
        let gae_lamda = 0.95;
        let reward_gamma = 0.999;

        let mut trained_network = PokerNetwork::new(
            self.player_cnt,
            self.action_config,
            self.device.clone(),
            true,
        )?;

        let mut agent_pool = AgentPool::new();
        let temp_net = PokerNetwork::new(
            self.player_cnt,
            self.action_config,
            self.device.clone(),
            true,
        )?;
        agent_pool.add_agent(Box::new(AgentNetwork::new(temp_net)));

        // List files in output path
        let trained_network_path = Path::new(&self.output_path);
        let trained_network_files = trained_network_path.read_dir()?;
        let mut latest_iteration = 0;

        for file in trained_network_files {
            let file = file?;
            let file_name = file.file_name();
            let file_name = file_name.to_str().unwrap();
            let file_name = file_name.split('_').collect::<Vec<&str>>();

            if file_name.len() > 2 {
                let iteration = file_name[2].split('.').collect::<Vec<&str>>()[0].parse::<u32>()?;
                if iteration > latest_iteration {
                    latest_iteration = iteration;
                }
            }
        }

        if latest_iteration > 0 {
            trained_network.var_map.load(
                trained_network_path.join(format!("poker_network_{}.pt", latest_iteration)),
            )?;
        }

        let mut policy_data = Vec::new();
        let mut critic_data = Vec::new();

        {
            let var_data = trained_network.var_map.data().lock().unwrap();
            var_data.iter().for_each(|(k, v)| {
                if k.starts_with("siamese") {
                    policy_data.push(v.clone());
                    critic_data.push(v.clone());
                }
                if k.starts_with("actor") {
                    policy_data.push(v.clone());
                }
                if k.starts_with("critic") {
                    critic_data.push(v.clone());
                }
            });
        }

        let mut optimizer_policy = candle_nn::AdamW::new_lr(policy_data, 3e-4)?;
        let mut optimizer_critic = candle_nn::AdamW::new_lr(critic_data, 3e-4)?;

        // We split the optimizers just like here:
        // https://github.com/facebookresearch/drqv2/blob/c0c650b76c6e5d22a7eb5f2edffd1440fe94f8ef/drqv2.py#L198
        // let all_vars = trained_network.var_map.all_vars();

        // let mut optimizer_actor =
        //     candle_nn::AdamW::new_lr(trained_network.var_map.all_vars(), 3e-4)?;
        // let mut optimizer_encoder_critic =
        //     candle_nn::AdamW::new_lr(trained_network.var_map_critic_encoder.all_vars(), 3e-4)?;

        for iteration in (latest_iteration as usize + 1)..self.trainer_config.max_iters {
            println!("Iteration: {}", iteration);

            let mut hand_states = Vec::new();

            for _ in 0..self.trainer_config.hands_per_player_per_iteration {
                for player in 0..self.player_cnt {
                    let mut agents: Vec<Option<&Box<dyn Agent>>> = Vec::new();
                    for p in 0..self.player_cnt {
                        let agent: Option<&Box<dyn Agent<'_>>> = if p != player {
                            Some(agent_pool.get_agent())
                        } else {
                            None
                        };
                        agents.push(agent);
                    }
                    // Select agents
                    self.tree
                        .traverse(player, &trained_network, &agents, &self.device)?;

                    // Make sure the hand state has at least one state for the traverser
                    let hs = self.tree.hand_state.clone();
                    if let Some(hs) = hs {
                        if hs.action_states.is_empty() {
                            continue;
                        }
                        hand_states.push(hs);
                    }

                    // // Print progress
                    // if hand_states.len() % 100 == 0 {
                    //     println!("Hand states: {}", hand_states.len());
                    // }
                }
            }

            // Calculate cumulative rewards for each hand state
            let mut rewards = Vec::new();
            let mut rewards_flat: Vec<f32> = Vec::new();
            let mut cnt = 0;
            let mut indexes = Vec::new();
            let mut min_rewards = Vec::new();
            let mut max_rewards = Vec::new();

            for hand_state in hand_states.iter_mut() {
                indexes.push(cnt);
                let hand_rewards = self.get_discounted_rewards(hand_state, reward_gamma);

                for action_state in hand_state.action_states.iter() {
                    min_rewards.push(action_state.min_reward);
                    max_rewards.push(action_state.max_reward);
                }

                cnt += hand_rewards.len();
                rewards_flat.extend(hand_rewards.iter());
                rewards.push(hand_rewards);
            }

            let rewards_flat_tensor = Tensor::new(rewards_flat, &self.device)?;
            let min_rewards_tensor = Tensor::new(min_rewards, &self.device)?;
            let max_rewards_tensor = Tensor::new(max_rewards, &self.device)?;

            // Get network inputs
            let mut card_input_vec = Vec::new();
            let mut action_input_vec = Vec::new();

            for hand_state in hand_states.iter() {
                let (card_tensors, action_tensors) =
                    hand_state.get_all_tensors(self.action_config, &self.device)?;

                card_input_vec.push(card_tensors);
                action_input_vec.push(action_tensors);
            }

            // println!("Card input vec: {:?}", card_input_vec[0].shape());
            // println!("Action input vec: {:?}", action_input_vec[0].shape());

            let card_input_tensor = Tensor::cat(&card_input_vec, 0)?;
            let action_input_tensor = Tensor::cat(&action_input_vec, 0)?;

            // println!("Card input tensor: {:?}", card_input_tensor.shape());
            // println!("Action input tensor: {:?}", action_input_tensor.shape());

            // Run all states through network
            let (base_actor_outputs, _) =
                trained_network.forward(&card_input_tensor, &action_input_tensor)?;

            // Get action probabilities for actions taken by the agent
            let old_probs_raw: Vec<Vec<f32>> = base_actor_outputs.to_vec2()?;
            let old_probs_tensor = Tensor::new(
                self.get_action_probs(&hand_states, &old_probs_raw, &indexes),
                &self.device,
            )?
            .log()?;

            for _update_step in 0..self.trainer_config.update_step {
                // Run all states through network
                let (actor_outputs, critic_outputs) =
                    trained_network.forward(&card_input_tensor, &action_input_tensor)?;

                let probs_raw: Vec<Vec<f32>> = actor_outputs.to_vec2()?;
                let probs = self.get_action_probs(&hand_states, &probs_raw, &indexes);

                let critic_outputs_vec: Vec<f32> =
                    critic_outputs.as_ref().unwrap().squeeze(1)?.to_vec1()?;

                // println!("actor_outputs: {:?}", actor_outputs.shape());
                // println!(
                //     "critic_outputs: {:?}",
                //     critic_outputs.as_ref().unwrap().shape()
                // );

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

                // Normalize and return advantage_gae
                let mean = advantage_gae.iter().sum::<f32>() / advantage_gae.len() as f32;
                let variance = advantage_gae
                    .iter()
                    .map(|x| (x - mean).powf(2.0))
                    .sum::<f32>()
                    / advantage_gae.len() as f32;
                let std_dev = variance.sqrt();
                for x in advantage_gae.iter_mut() {
                    *x = (*x - mean) / std_dev;
                }

                let advantage_tensor = Tensor::new(advantage_gae, &self.device)?;

                // Get trinal clip PPO
                let policy_loss = self.get_trinal_clip_policy_loss(
                    &advantage_tensor,
                    &Tensor::new(probs, &self.device)?.log()?,
                    &old_probs_tensor,
                );

                let value_loss = self.get_trinal_clip_value_loss(
                    critic_outputs.as_ref().unwrap(),
                    &rewards_flat_tensor,
                    &min_rewards_tensor,
                    &max_rewards_tensor,
                );

                // println!("Update step: {}", update_step);
                println!(
                    "Policy loss: {:?}",
                    policy_loss.as_ref().unwrap().to_scalar::<f32>()
                );
                println!(
                    "Value loss: {:?}",
                    value_loss.as_ref().unwrap().to_scalar::<f32>()
                );

                optimizer_policy.backward_step(&policy_loss?)?;
                optimizer_critic.backward_step(&value_loss?)?;
            }
            self.tree
                .print_first_actions(&trained_network, &self.device)?;

            if iteration > 0 && iteration % 100 == 0 {
                trained_network.var_map.save(
                    Path::new(&self.output_path).join(&format!("poker_network_{}.pt", iteration)),
                )?;
                self.tree
                    .print_first_actions(&trained_network, &self.device)?;
            }

            let mut copy_net = PokerNetwork::new(
                self.player_cnt,
                self.action_config,
                self.device.clone(),
                true,
            )?;
            copy_net.var_map.clone_from(&trained_network.var_map);
            agent_pool.add_agent(Box::new(AgentNetwork::new(copy_net)));
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
        policy_loss?.mean(0)
    }

    fn get_trinal_clip_value_loss(
        &self,
        values: &Tensor,
        rewards: &Tensor,
        max_bet: &Tensor,
        min_bet: &Tensor,
    ) -> Result<Tensor, candle_core::Error> {
        let clipped = rewards.minimum(min_bet)?.maximum(max_bet)?;
        let diff = (clipped - values.squeeze(1))?;
        (diff.as_ref() * diff.as_ref())?.mean(0)
    }
}
