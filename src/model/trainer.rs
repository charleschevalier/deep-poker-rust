use std::vec;

use super::poker_network::PokerNetwork;
use super::trainer_config::TrainerConfig;
use crate::agent::agent_network::AgentNetwork;
use crate::agent::agent_pool::AgentPool;
use crate::game::action::ActionConfig;
use crate::game::hand_state::HandState;
use crate::game::tree::Tree;
use candle_core::{Device, NdArray, Tensor};
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
                let split = file_name[3].split('.').collect::<Vec<&str>>();
                if (split.len() == 2) && (split[1] == "pt") {
                    let iteration = split[0].parse::<u32>()?;
                    if iteration > latest_iteration {
                        latest_iteration = iteration;
                    }
                }
            }
        }

        if latest_iteration > 0 {
            trained_network.var_map_actor.load(
                trained_network_path.join(format!("poker_network_actor_{}.pt", latest_iteration)),
            )?;
            trained_network.var_map_critic_siamese.load(
                trained_network_path.join(format!("poker_network_cs_{}.pt", latest_iteration)),
            )?;

            // TODO: Also load 5 previous iterations
        }

        let mut optimizer_policy =
            candle_nn::AdamW::new_lr(trained_network.var_map_actor.all_vars(), 0.01)?;
        let mut optimizer_critic =
            candle_nn::AdamW::new_lr(trained_network.var_map_critic_siamese.all_vars(), 0.01)?;

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
            let mut rewards_by_hand_state = Vec::new();
            let mut indexes = Vec::new();
            let min_rewards_tensor;
            let max_rewards_tensor;
            let gamma_rewards_tensor;
            let mut step_cnt = 0;

            {
                let mut min_rewards = Vec::new();
                let mut max_rewards = Vec::new();
                let mut gamma_rewards = Vec::new();
                let reward_ratio = self.action_config.buy_in as f32 * (self.player_cnt - 1) as f32;

                for hand_state in hand_states.iter_mut() {
                    indexes.push(step_cnt);
                    let hand_rewards: Vec<f32> = hand_state
                        .get_traverser_action_states()
                        .iter()
                        .map(|ast| ast.reward / reward_ratio)
                        .collect();

                    for action_state in hand_state.get_traverser_action_states().iter() {
                        min_rewards.push(action_state.min_reward / reward_ratio);
                        max_rewards.push(action_state.max_reward / reward_ratio);
                    }

                    gamma_rewards
                        .append(&mut self.get_discounted_rewards(&hand_rewards, reward_gamma));

                    step_cnt += hand_rewards.len();
                    rewards_by_hand_state.push(hand_rewards);
                }

                assert!(min_rewards.len() == step_cnt);
                assert!(max_rewards.len() == step_cnt);
                assert!(gamma_rewards.len() == step_cnt);

                min_rewards_tensor = Tensor::new(min_rewards, &self.device)?;
                max_rewards_tensor = Tensor::new(max_rewards, &self.device)?;
                gamma_rewards_tensor = Tensor::new(gamma_rewards, &self.device)?;
            }

            // Get network inputs
            let card_input_tensor;
            let action_input_tensor;

            {
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

                card_input_tensor = Tensor::cat(&card_input_vec, 0)?;
                action_input_tensor = Tensor::cat(&action_input_vec, 0)?;

                assert!(card_input_tensor.dim(0)? == step_cnt);
                assert!(action_input_tensor.dim(0)? == step_cnt);
            }

            // println!("Card input tensor: {:?}", card_input_tensor.shape());
            // println!("Action input tensor: {:?}", action_input_tensor.shape());

            // Run all states through network
            let (base_actor_outputs, base_critic_outputs) =
                trained_network.forward(&card_input_tensor, &action_input_tensor)?;

            // Get action probabilities for actions taken by the agent
            let old_probs_raw: Vec<Vec<f32>> = base_actor_outputs.to_vec2()?;
            let old_probs_log_tensor = Tensor::new(
                self.get_action_probs(&hand_states, &old_probs_raw),
                &self.device,
            )?
            .log()?;

            // Calculate advantage GAE for each hand state
            let mut advantage_gae: Vec<f32> = Vec::new();
            {
                let base_critic_outputs_vec: Vec<f32> = base_critic_outputs
                    .as_ref()
                    .unwrap()
                    .squeeze(1)?
                    .to_vec1()?;

                for i in 0..hand_states.len() {
                    let (mut advantage, _) = self.calculate_advantage_gae(
                        &rewards_by_hand_state[i],
                        &base_critic_outputs_vec
                            [indexes[i]..indexes[i] + rewards_by_hand_state[i].len()],
                        gae_gamma,
                        gae_lamda,
                    );
                    advantage_gae.append(&mut advantage);
                }

                assert!(advantage_gae.len() == step_cnt);

                // Normalize advantage_gae
                // Self::normalize_mean_std(&mut advantage_gae);
            }

            let advantage_tensor = Tensor::new(advantage_gae, &self.device)?;

            for _update_step in 0..self.trainer_config.update_step {
                // Run all states through network
                let (actor_outputs, critic_outputs) =
                    trained_network.forward(&card_input_tensor, &action_input_tensor)?;

                let probs_raw: Vec<Vec<f32>> = actor_outputs.to_vec2()?;
                let probs = self.get_action_probs(&hand_states, &probs_raw);

                let critic_outputs_vec: Vec<f32> =
                    critic_outputs.as_ref().unwrap().squeeze(1)?.to_vec1()?;
                println!("critic_outputs_vec: {:?}", critic_outputs_vec.len());
                let gamma_rewards_vec: Vec<f32> = gamma_rewards_tensor.to_vec1()?;
                println!("gamma_rewards_vec: {:?}", gamma_rewards_vec.len());

                // Get trinal clip PPO
                let policy_loss = self.get_trinal_clip_policy_loss(
                    &advantage_tensor,
                    &Tensor::new(probs, &self.device)?.log()?,
                    &old_probs_log_tensor,
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

                // let total_loss = policy_loss?.add(&value_loss?)?;
                // optimizer_policy.backward_step(&total_loss)?;

                optimizer_policy.backward_step(&policy_loss?)?;
                optimizer_critic.backward_step(&value_loss?)?;

                // trained_network.print_weights();
            }
            self.tree.print_first_actions(
                &trained_network,
                &self.device,
                self.trainer_config.no_invalid_for_traverser,
            )?;

            // for _ in 0..10 {
            //     self.tree._play_one_hand(
            //         &trained_network,
            //         &self.device,
            //         self.trainer_config.no_invalid_for_traverser,
            //     )?;
            // }

            if iteration > 0 && iteration % 50 == 0 {
                trained_network.var_map_actor.save(
                    Path::new(&self.output_path)
                        .join(&format!("poker_network_actor_{}.pt", iteration)),
                )?;
                trained_network.var_map_critic_siamese.save(
                    Path::new(&self.output_path)
                        .join(&format!("poker_network_cs_{}.pt", iteration)),
                )?;
                self.tree.print_first_actions(
                    &trained_network,
                    &self.device,
                    self.trainer_config.no_invalid_for_traverser,
                )?;
            }

            // Put a new agent in the pool every 100 iterations
            if iteration % 100 == 0 {
                agent_pool.add_agent(Box::new(AgentNetwork::new(trained_network.clone()?)));
            }

            // // Reset trained network every 100 iterations
            // if iteration % 100 == 0 {
            //     // Save varmaps
            //     trained_network.var_map_actor.save(
            //         Path::new(&self.output_path)
            //             .join(&format!("poker_network_actor_reset_{}.pt", iteration)),
            //     )?;
            //     trained_network.var_map_critic_siamese.save(
            //         Path::new(&self.output_path)
            //             .join(&format!("poker_network_cs_reset_{}.pt", iteration)),
            //     )?;

            //     // Reset network
            //     trained_network = PokerNetwork::new(
            //         self.player_cnt,
            //         self.action_config,
            //         self.device.clone(),
            //         true,
            //     )?;

            //     optimizer_policy =
            //         candle_nn::AdamW::new_lr(trained_network.var_map_actor.all_vars(), 3e-4)?;
            // }
        }

        Ok(())
    }

    fn get_action_probs(&self, hand_states: &[HandState], probs_raw: &[Vec<f32>]) -> Vec<f32> {
        let mut probs = Vec::new();
        let mut index: usize = 0;

        for hand_state in hand_states.iter() {
            for action_state in hand_state.get_traverser_action_states().iter() {
                // Apply valid_actions_mask to probs
                let mut action_probs = probs_raw[index].clone();
                for i in 0..action_probs.len() {
                    if i >= action_state.valid_actions_mask.len()
                        || !action_state.valid_actions_mask[i]
                    {
                        action_probs[i] = 0.0;
                    }
                }

                // Normalize probas
                let sum_norm: f32 = action_probs.iter().sum();
                if sum_norm > 0.0 {
                    for p in &mut action_probs {
                        *p /= sum_norm;
                    }
                } else {
                    // Count positive values in valid_action_mask
                    let true_count = action_state
                        .valid_actions_mask
                        .iter()
                        .filter(|&&x| x)
                        .count();

                    for (i, p) in action_probs.iter_mut().enumerate() {
                        if i < action_state.valid_actions_mask.len()
                            && action_state.valid_actions_mask[i]
                        {
                            *p = 1.0 / (true_count as f32);
                        }
                    }
                }

                probs.push(action_probs[action_state.action_taken_index]);
                index += 1;
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

    fn get_discounted_rewards(&self, hand_rewards: &Vec<f32>, gamma: f32) -> Vec<f32> {
        let mut rewards = Vec::new();
        let mut reward = 0.0;

        for r in hand_rewards.iter().rev() {
            reward = reward * gamma + r;
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

        let clip1 = ratio.copy()?.clamp(
            1.0 - self.trainer_config.ppo_epsilon,
            1.0 + self.trainer_config.ppo_epsilon,
        )?;

        let ppo_term_1 = ratio.copy() * advantages;
        let ppo_term_2 = clip1.copy() * advantages;
        let ppo = ppo_term_1?.minimum(&ppo_term_2?)?;

        let clip2 = ratio
            .copy()?
            .clamp(&clip1, self.trainer_config.ppo_delta_1)?;
        let trinal_clip_ppo = (clip2.copy() * advantages)?;

        // Get negative advantage values
        let neg = advantages.lt(0.0)?;

        let ppo_vec: Vec<f32> = ppo.to_vec1()?;
        let trinal_clip_ppo_vec: Vec<f32> = trinal_clip_ppo.to_vec1()?;

        // Apply trinal-clip PPO for negative advantages
        let policy_loss = neg.where_cond(&trinal_clip_ppo, &ppo)?;

        // let ratio_vec: Vec<f32> = ratio.to_vec1()?;
        // let policy_loss_vec: Vec<f32> = policy_loss.copy()?.to_vec1()?;
        // let neg_vec: Vec<u8> = neg.to_vec1()?;
        // let advantages_vec: Vec<f32> = advantages.to_vec1()?;

        // println!("PPO: {:?}", ppo_vec.shape());
        // println!("Trinal-clip PPO: {:?}", trinal_clip_ppo_vec.shape());
        // println!("Policy loss: {:?}", policy_loss_vec.shape());

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
        let clipped = rewards.copy()?.clamp(min_bet, max_bet)?;

        let clipped_vec: Vec<f32> = clipped.to_vec1()?;

        let diff = (clipped - values.squeeze(1))?;

        let values_vec: Vec<f32> = values.squeeze(1)?.to_vec1()?;
        let diff_vec: Vec<f32> = diff.to_vec1()?;

        // println!("Clipped: {:?}", clipped_vec);
        // println!("Values: {:?}", values_vec);
        // println!("Diff: {:?}", diff_vec);

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
