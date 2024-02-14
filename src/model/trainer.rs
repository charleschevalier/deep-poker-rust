use std::vec;

use super::poker_network::PokerNetwork;
use super::trainer_config::TrainerConfig;
use crate::agent::agent_network::AgentNetwork;
use crate::agent::agent_pool::AgentPool;
use crate::game::action::ActionConfig;
use crate::game::hand_state::HandState;
use crate::game::tree::Tree;
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
        // let temp_net = PokerNetwork::new(
        //     self.player_cnt,
        //     self.action_config,
        //     self.device.clone(),
        //     true,
        // )?;
        // agent_pool.add_agent(Box::new(AgentNetwork::new(temp_net)));

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

        let policy_data: Vec<candle_core::Var>;
        let critic_data: Vec<candle_core::Var>;

        {
            // Here, Var::clone is not a deep copy, but a shallow copy. That's what we need for the
            // optimizer to work properly.
            let var_data = trained_network.var_map.data().lock().unwrap();

            policy_data = var_data
                .iter()
                .filter(|(k, _)| k.starts_with("siamese") || k.starts_with("actor"))
                .map(|(_, v)| v.clone())
                .collect::<Vec<_>>();

            critic_data = var_data
                .iter()
                .filter(|(k, _)| k.starts_with("siamese") || k.starts_with("critic"))
                .map(|(_, v)| v.clone())
                .collect::<Vec<_>>();
        }

        let mut optimizer_policy = candle_nn::AdamW::new_lr(policy_data, 3e-4)?;
        let mut optimizer_critic = candle_nn::AdamW::new_lr(critic_data, 3e-4)?;

        for iteration in (latest_iteration as usize + 1)..self.trainer_config.max_iters {
            println!("Iteration: {}", iteration);

            let mut hand_states = Vec::new();

            for _ in 0..self.trainer_config.hands_per_player_per_iteration {
                for traverser in 0..self.player_cnt {
                    let mut agents = Vec::new();
                    for p in 0..self.player_cnt {
                        let agent = if p != traverser {
                            let (_, ag) = agent_pool.get_agent();
                            Some(ag)
                        } else {
                            None
                        };
                        agents.push(agent);
                    }
                    // Select agents
                    self.tree.traverse(
                        traverser,
                        &trained_network,
                        &agents,
                        &self.device,
                        self.trainer_config.no_invalid_for_traverser,
                    )?;

                    // Make sure the hand state has at least one state for the traverser
                    let hs = self.tree.hand_state.clone();
                    if let Some(hs) = hs {
                        // count number of action states for traverser
                        if hs.get_traverser_action_states().is_empty() {
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
            let mut step_cnt = 0;
            let mut indexes = Vec::new();
            let mut min_rewards = Vec::new();
            let mut max_rewards = Vec::new();
            let mut gamma_rewards = Vec::new();

            for hand_state in hand_states.iter_mut() {
                indexes.push(step_cnt);
                let hand_rewards: Vec<f32> = hand_state
                    .get_traverser_action_states()
                    .iter()
                    .map(|ast| ast.reward)
                    .collect();

                for action_state in hand_state.get_traverser_action_states().iter() {
                    min_rewards.push(action_state.min_reward);
                    max_rewards.push(action_state.max_reward);
                }

                gamma_rewards.append(&mut self.get_discounted_rewards(hand_state, reward_gamma));

                step_cnt += hand_rewards.len();
                rewards_flat.extend(hand_rewards.iter());
                rewards.push(hand_rewards);
            }

            assert!(rewards_flat.len() == step_cnt);
            assert!(min_rewards.len() == step_cnt);
            assert!(max_rewards.len() == step_cnt);
            assert!(gamma_rewards.len() == step_cnt);

            let min_rewards_tensor = Tensor::new(min_rewards, &self.device)?;
            let max_rewards_tensor = Tensor::new(max_rewards, &self.device)?;
            let gamma_rewards_tensor = Tensor::new(gamma_rewards, &self.device)?;

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

            assert!(card_input_tensor.dim(0)? == step_cnt);
            assert!(action_input_tensor.dim(0)? == step_cnt);

            // println!("Card input tensor: {:?}", card_input_tensor.shape());
            // println!("Action input tensor: {:?}", action_input_tensor.shape());

            // Run all states through network
            let (base_actor_outputs, base_critic_outputs) =
                trained_network.forward(&card_input_tensor, &action_input_tensor)?;

            // Get action probabilities for actions taken by the agent
            let old_probs_raw: Vec<Vec<f32>> = base_actor_outputs.to_vec2()?;
            let old_probs_tensor = Tensor::new(
                self.get_action_probs(&hand_states, &old_probs_raw, &indexes),
                &self.device,
            )?
            .log()?;

            // Calculate advantage GAE for each hand state
            let mut advantage_gae: Vec<f32> = Vec::new();
            let base_critic_outputs_vec: Vec<f32> = base_critic_outputs
                .as_ref()
                .unwrap()
                .squeeze(1)?
                .to_vec1()?;

            for i in 0..rewards.len() {
                let (mut advantage, _) = self.calculate_advantage_gae(
                    &rewards[i],
                    &base_critic_outputs_vec[indexes[i]..indexes[i] + rewards[i].len()],
                    gae_gamma,
                    gae_lamda,
                );
                advantage_gae.append(&mut advantage);
            }

            assert!(advantage_gae.len() == step_cnt);

            // Normalize advantage_gae
            Self::normalize_mean_std(&mut advantage_gae);

            let advantage_tensor = Tensor::new(advantage_gae, &self.device)?;

            for _update_step in 0..self.trainer_config.update_step {
                // Run all states through network
                let (actor_outputs, critic_outputs) =
                    trained_network.forward(&card_input_tensor, &action_input_tensor)?;

                let probs_raw: Vec<Vec<f32>> = actor_outputs.to_vec2()?;
                let probs = self.get_action_probs(&hand_states, &probs_raw, &indexes);

                // println!("actor_outputs: {:?}", actor_outputs.shape());
                // println!(
                //     "critic_outputs: {:?}",
                //     critic_outputs.as_ref().unwrap().shape()
                // );

                // Get trinal clip PPO
                let policy_loss = self.get_trinal_clip_policy_loss(
                    &advantage_tensor,
                    &Tensor::new(probs, &self.device)?.log()?,
                    &old_probs_tensor,
                );

                let value_loss = self.get_trinal_clip_value_loss(
                    critic_outputs.as_ref().unwrap(),
                    &gamma_rewards_tensor,
                    &max_rewards_tensor,
                    &min_rewards_tensor,
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
            self.tree.print_first_actions(
                &trained_network,
                &self.device,
                self.trainer_config.no_invalid_for_traverser,
            )?;

            if iteration > 0 && iteration % 50 == 0 {
                trained_network.var_map.save(
                    Path::new(&self.output_path).join(&format!("poker_network_{}.pt", iteration)),
                )?;
                self.tree.print_first_actions(
                    &trained_network,
                    &self.device,
                    self.trainer_config.no_invalid_for_traverser,
                )?;
            }

            if iteration % 50 == 0 {
                let mut copy_net = PokerNetwork::new(
                    self.player_cnt,
                    self.action_config,
                    self.device.clone(),
                    true,
                )?;

                let var_data = trained_network.var_map.data().lock().unwrap();
                // We perform a deep copy of the varmap using Tensor::copy on Var
                var_data.iter().for_each(|(k, v)| {
                    copy_net
                        .var_map
                        .set_one(k, v.as_tensor().copy().unwrap())
                        .unwrap();
                });
                agent_pool.add_agent(Box::new(AgentNetwork::new(copy_net)));
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
            for (j, action_state) in hand_state.get_traverser_action_states().iter().enumerate() {
                probs.push(probs_raw[indexes[i] + j][action_state.action_taken_index]);
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
    ) -> (Vec<f32>, Vec<f32>) {
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

        (advantage[0..size].to_vec(), value_target)
    }

    fn get_discounted_rewards(&self, hand_state: &HandState, gamma: f32) -> Vec<f32> {
        let mut rewards = Vec::new();
        let mut reward = 0.0;

        for action_state in hand_state.get_traverser_action_states().iter().rev() {
            reward = action_state.reward + gamma * reward;
            rewards.push(reward);
        }

        rewards.reverse();
        rewards
    }

    fn get_trinal_clip_policy_loss(
        &self,
        advantages: &Tensor,
        log_probs: &Tensor,
        old_log_probs: &Tensor,
    ) -> Result<Tensor, candle_core::Error> {
        let ratio = (log_probs - old_log_probs)?.exp()?;

        let clip1 = ratio.clamp(
            1.0 - self.trainer_config.ppo_epsilon,
            1.0 + self.trainer_config.ppo_epsilon,
        )?;

        let ppo_term_1 = ratio.as_ref() * advantages;
        let ppo_term_2 = clip1.as_ref() * advantages;
        let ppo = ppo_term_1?.minimum(&ppo_term_2?)?;

        let clip2 = ratio.clamp(&clip1, self.trainer_config.ppo_delta_1)?;
        let trinal_clip_ppo = (clip2.as_ref() * advantages)?;

        // Get negative advantage values
        let neg = advantages.lt(0.0)?;

        // Apply trinal-clip PPO for negative advantages
        let policy_loss = neg.where_cond(&trinal_clip_ppo, &ppo)?;

        // NOTE: we take the negative min of the surrogate losses because we're trying to maximize
        // the performance function, but Adam minimizes the loss. So minimizing the negative
        // performance function maximizes it.
        policy_loss.mean(0)?.neg()
    }

    fn get_trinal_clip_value_loss(
        &self,
        values: &Tensor,
        rewards: &Tensor,
        max_bet: &Tensor,
        min_bet: &Tensor,
    ) -> Result<Tensor, candle_core::Error> {
        let clipped = rewards.clamp(min_bet, max_bet)?;
        let diff = (clipped - values.squeeze(1))?;
        (diff.as_ref() * diff.as_ref())?.mean(0)
    }

    fn normalize_mean_std(vec: &mut [f32]) {
        let mean = vec.iter().sum::<f32>() / vec.len() as f32;
        let variance = vec.iter().map(|x| (x - mean).powf(2.0)).sum::<f32>() / vec.len() as f32;
        let std_dev = variance.sqrt();
        for x in vec.iter_mut() {
            *x = (*x - mean) / (std_dev + 1e-10);
        }
    }
}
