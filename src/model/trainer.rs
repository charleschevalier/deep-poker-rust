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
                let split = file_name[2].split('.').collect::<Vec<&str>>();
                if (split.len() == 2) && (split[1] == "pt") {
                    let iteration = split[0].parse::<u32>()?;
                    if iteration > latest_iteration {
                        latest_iteration = iteration;
                    }
                }
            }
        }

        if latest_iteration > 0 {
            trained_network.var_map.load(
                trained_network_path.join(format!("poker_network_{}.pt", latest_iteration)),
            )?;

            if latest_iteration >= 100 {
                let delta = latest_iteration % 100;
                let end = latest_iteration - delta;
                let start = std::cmp::max(100, end as i32 - 400);
                for i in (start as u32..end + 100).step_by(100) {
                    let mut network = PokerNetwork::new(
                        self.player_cnt,
                        self.action_config,
                        self.device.clone(),
                        false,
                    )?;
                    network
                        .var_map
                        .load(trained_network_path.join(format!("poker_network_{}.pt", i)))?;
                    agent_pool.add_agent(Box::new(AgentNetwork::new(network)));
                }
            }
        }

        let filter_var_map_by_prefix = |varmap: &candle_nn::VarMap, prefix: &[&str]| {
            varmap
                .data()
                .lock()
                .unwrap()
                .iter()
                .filter(|(name, _)| prefix.iter().any(|&item| name.starts_with(item)))
                .map(|(_, var)| var.clone())
                .collect::<Vec<candle_core::Var>>()
        };

        let mut optimizer_embedding = candle_nn::AdamW::new_lr(
            filter_var_map_by_prefix(&trained_network.var_map, &["siamese"]),
            1e-5,
        )?;
        let mut optimizer_policy = candle_nn::AdamW::new_lr(
            filter_var_map_by_prefix(&trained_network.var_map, &["actor"]),
            1e-5,
        )?;
        let mut optimizer_critic = candle_nn::AdamW::new_lr(
            filter_var_map_by_prefix(&trained_network.var_map, &["critic"]),
            1e-5,
        )?;

        for iteration in (latest_iteration as usize + 1)..self.trainer_config.max_iters {
            println!("Iteration: {}", iteration);

            let mut hand_states = Vec::new();

            {
                // Clone trained network for inference
                let inference_network = trained_network.clone()?;

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
                            &inference_network,
                            &agents,
                            &self.device,
                            self.trainer_config.no_invalid_for_traverser,
                        )?;

                        // Make sure the hand state has at least one state for the traverser, and that we
                        // do not have too much actions per street
                        let hs = self.tree.hand_state.clone();
                        if let Some(hs) = hs {
                            // Count number of action states for traverser
                            if hs.get_traverser_action_states().is_empty() {
                                continue;
                            }
                            // Count number of action per street
                            let mut action_cnt = 0;
                            let mut too_much = false;
                            let mut street = 0;
                            for action_state in hs.get_traverser_action_states() {
                                if action_state.street == street {
                                    action_cnt += 1;
                                } else {
                                    street = action_state.street;
                                    action_cnt = 0;
                                }

                                if action_cnt > (self.action_config.player_count - 1) * 3 {
                                    too_much = true;
                                    break;
                                }
                            }

                            if too_much {
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
            }

            // Get action indexes
            let action_indexes_tensor = self.get_action_indexes(&hand_states)?;

            // Run all states through network. Detach to prevent gradient updates
            let old_embedding = trained_network
                .forward_embedding(&card_input_tensor, &action_input_tensor, false)?
                .detach()?;

            let base_actor_outputs = trained_network.forward_actor(&old_embedding)?.detach()?;

            let base_critic_outputs = trained_network
                .forward_critic(&old_embedding)?
                .unwrap()
                .detach()?;

            let old_probs_log_tensor = base_actor_outputs
                .gather(&action_indexes_tensor, 1)?
                .squeeze(1)?
                .log()?;

            // Calculate advantage GAE for each hand state
            let mut advantage_gae: Vec<f32> = Vec::new();
            {
                let base_critic_outputs_vec: Vec<f32> =
                    base_critic_outputs.as_ref().squeeze(1)?.to_vec1()?;

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

                // Normalize advantage_gae
                Self::normalize_mean_std(&mut advantage_gae);
            }

            let advantage_tensor = Tensor::new(advantage_gae, &self.device)?;

            for _update_step in 0..self.trainer_config.update_step {
                // Get embedding
                let embedding = trained_network.forward_embedding(
                    &card_input_tensor,
                    &action_input_tensor,
                    true,
                )?;

                // Run actor
                let actor_outputs = trained_network.forward_actor(&embedding)?;
                let probs_log_tensor = actor_outputs
                    .gather(&action_indexes_tensor, 1)?
                    .squeeze(1)?
                    .log()?;

                // Get trinal clip policy loss
                let policy_loss = self.get_trinal_clip_policy_loss(
                    &advantage_tensor,
                    &probs_log_tensor,
                    &old_probs_log_tensor,
                );

                println!(
                    "Policy loss: {:?}",
                    policy_loss.as_ref().unwrap().to_scalar::<f32>()
                );

                let gradients_policy = policy_loss?.backward()?;

                optimizer_policy.step(&gradients_policy)?;

                // Get critic output
                let critic_outputs = trained_network.forward_critic(&embedding)?;

                // Get trinal clip value loss
                let value_loss = self.get_trinal_clip_value_loss(
                    critic_outputs.as_ref().unwrap(),
                    &gamma_rewards_tensor,
                    &max_rewards_tensor,
                    &min_rewards_tensor,
                );

                println!(
                    "Value loss: {:?}",
                    value_loss.as_ref().unwrap().to_scalar::<f32>()
                );

                let gradients_value = value_loss?.backward()?;

                optimizer_critic.step(&gradients_value)?;

                // Calculate siamese gradients, weighted sum of actor and critic gradients
                // I did not find a better way to create a new GradStore, is there one ?
                let mut gradients_embedding =
                    Tensor::zeros((), candle_core::DType::F32, &self.device)?.backward()?;
                trained_network
                    .var_map
                    .data()
                    .lock()
                    .unwrap()
                    .iter()
                    .for_each(|(k, v)| {
                        if k.starts_with("siamese")
                            && !k.ends_with("running_var")
                            && !k.ends_with("running_mean")
                            && !k.ends_with("batch_norm.bias")
                            && !k.ends_with("batch_norm.weight")
                        {
                            // println!("Key: {}", k);
                            let grad_policy = gradients_policy.get_id(v.id()).unwrap();
                            let grad_value = gradients_value.get_id(v.id()).unwrap();
                            let grad_weighted = ((grad_policy * 0.5).unwrap()
                                + (grad_value * 0.5).unwrap())
                            .unwrap();

                            // Check for NaN
                            let flat_var = v
                                .copy()
                                .unwrap()
                                .flatten(0, grad_policy.dims().len() - 1)
                                .unwrap();

                            let flat_policy = grad_policy
                                .copy()
                                .unwrap()
                                .flatten(0, grad_policy.dims().len() - 1)
                                .unwrap();

                            // if k == "siamese_card_conv_1.bias"
                            //     || k == "siamese_card_conv_2.bias"
                            //     || k == "siamese_action_conv_1.bias"
                            //     || k == "siamese_action_conv_2.bias"
                            //     || k == "siamese_card_conv_2.weight"
                            // {
                            //     println!();
                            //     println!();
                            //     println!();
                            //     println!(
                            //         "POLICY {}: {:?}",
                            //         k,
                            //         flat_policy.as_ref().to_vec1::<f32>().unwrap()
                            //     );
                            //     println!(
                            //         "VAR {}: {:?}",
                            //         k,
                            //         flat_var.as_ref().to_vec1::<f32>().unwrap()
                            //     );
                            // }

                            let flat_value = grad_value
                                .copy()
                                .unwrap()
                                .flatten(0, grad_value.dims().len() - 1)
                                .unwrap();

                            if flat_policy
                                .as_ref()
                                .to_vec1::<f32>()
                                .unwrap()
                                .iter()
                                .any(|x: &f32| *x > 100.0)
                            {
                                println!(
                                    "flat_policy: {:?}",
                                    flat_policy.as_ref().to_vec1::<f32>().unwrap()
                                );
                                panic!("HIGH in policy gradients at key: {}", k);
                            }

                            if flat_var
                                .as_ref()
                                .to_vec1::<f32>()
                                .unwrap()
                                .iter()
                                .any(|x: &f32| *x > 100.0)
                            {
                                println!(
                                    "flat_var: {:?}",
                                    flat_var.as_ref().to_vec1::<f32>().unwrap()
                                );
                                panic!("HIGH in flat_var gradients at key: {}", k);
                            }

                            if flat_policy
                                .as_ref()
                                .to_vec1::<f32>()
                                .unwrap()
                                .iter()
                                .any(|x: &f32| x.is_nan())
                            {
                                println!(
                                    "flat_policy: {:?}",
                                    flat_policy.as_ref().to_vec1::<f32>().unwrap()
                                );

                                println!(
                                    "flat_var: {:?}",
                                    flat_var.as_ref().to_vec1::<f32>().unwrap()
                                );
                                println!("NAN in policy gradients at key: {}", k);
                            }
                            if flat_value
                                .as_ref()
                                .to_vec1::<f32>()
                                .unwrap()
                                .iter()
                                .any(|x: &f32| x.is_nan())
                            {
                                println!("NAN in value gradients at key: {}", k);
                            }

                            assert!(!flat_policy
                                .as_ref()
                                .to_vec1::<f32>()
                                .unwrap()
                                .iter()
                                .any(|x: &f32| x.is_nan()));

                            // let flat = grad_weighted
                            //     .copy()
                            //     .unwrap()
                            //     .flatten(0, grad_weighted.dims().len() - 1)
                            //     .unwrap();

                            // assert!(!flat
                            //     .as_ref()
                            //     .to_vec1::<f32>()
                            //     .unwrap()
                            //     .iter()
                            //     .any(|x: &f32| x.is_nan()));

                            gradients_embedding.insert(v, grad_weighted);
                        }
                    });

                // // Do backprop
                optimizer_embedding.step(&gradients_embedding)?;
                // optimizer_embedding.step(&gradients_value)?;
            }

            self.tree.print_first_actions(
                &trained_network.clone()?,
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

            if iteration > 0 && iteration % 10 == 0 {
                trained_network.var_map.save(
                    Path::new(&self.output_path).join(&format!("poker_network_{}.pt", iteration)),
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
        }

        Ok(())
    }

    fn get_action_indexes(&self, hand_states: &[HandState]) -> Result<Tensor, candle_core::Error> {
        let mut result = Vec::new();

        for hand_state in hand_states.iter() {
            for action_state in hand_state.get_traverser_action_states().iter() {
                result.push(action_state.action_taken_index as u32);
            }
        }

        Tensor::new(result, &self.device)?.unsqueeze(1)
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

    fn get_discounted_rewards(&self, hand_rewards: &[f32], gamma: f32) -> Vec<f32> {
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
        let clipped = rewards.copy()?.clamp(min_bet, max_bet)?;
        let diff = (clipped - values.squeeze(1))?;
        (diff.as_ref() * diff.as_ref())?.mean(0)
    }

    fn normalize_mean_std(vec: &mut [f32]) {
        let mean = vec.iter().sum::<f32>() / vec.len() as f32;
        let variance = vec.iter().map(|x| (x - mean).powf(2.0)).sum::<f32>() / vec.len() as f32;
        let std_dev = variance.sqrt();
        if std_dev == 0.0 {
            return;
        }
        for x in vec.iter_mut() {
            *x = (*x - mean) / (std_dev + 1e-10);
        }
    }
}
