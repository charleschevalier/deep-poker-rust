use super::poker_network::PokerNetwork;
use super::trainer_config::TrainerConfig;
use super::adam_optimizer::AdamWCustom;
use crate::agent::agent_network::AgentNetwork;
use crate::agent::agent_pool::AgentPool;
use crate::agent::Agent;
use crate::game::action::ActionConfig;
use crate::game::hand_state::HandState;
use crate::game::tree::Tree;
use crate::helper;

use candle_core::cuda_backend::cudarc::cublas::result;
use candle_core::{Device, Tensor};
use candle_nn::Optimizer;
use itertools::cloned;
use std::vec;
use std::path::Path;
use std::sync::{Arc, Mutex};
use threadpool::ThreadPool;
use std::time::Instant;

pub struct Trainer<'a> {
    player_cnt: u32,
    action_config: &'a ActionConfig,
    trainer_config: &'a TrainerConfig,
    device: Device,
    output_path: &'a str,
    n_workers: usize,
    thread_pool: ThreadPool
}

impl<'a> Trainer<'a> {
    pub fn new(
        player_cnt: u32,
        action_config: &'a ActionConfig,
        trainer_config: &'a TrainerConfig,
        device: Device,
        output_path: &'a str,
    ) -> Trainer<'a> {
        let n_workers = num_cpus::get() / 2;
        let thread_pool = ThreadPool::new(n_workers);

        Trainer {
            player_cnt,
            action_config,
            trainer_config,
            device,
            output_path,
            n_workers,
            thread_pool
        }
    }

    pub fn train(&'a mut self) -> Result<(), Box<dyn std::error::Error>> {
        let gae_gamma = 0.99;
        let gae_lambda = 0.95;
        let reward_gamma = 0.999;
        let log_epsilon = 1e-10;

        let mut trained_network = PokerNetwork::new(
            self.player_cnt,
            self.action_config.clone(),
            self.device.clone(),
            self.trainer_config.agents_device.clone(),
            true,
        )?;

        let mut agent_pool = AgentPool::new(self.trainer_config.agent_count);

        // Load previous training
        let latest_iteration = self.load_existing(&mut trained_network, &mut agent_pool)?;

        // Select best agents
        self.refresh_agents(&mut agent_pool)?;

        // Build optimizers
        let mut optimizer_embedding = AdamWCustom::new_lr(
            trained_network.get_siamese_vars(),
            self.trainer_config.learning_rate,
        )?;
        optimizer_embedding.set_step(latest_iteration as usize * self.trainer_config.update_step);

        // optimizer_embedding.step(grads)?;
        let mut optimizer_policy = AdamWCustom::new_lr(
           trained_network.get_actor_vars(),
            self.trainer_config.learning_rate / 5.0,
        )?;
        optimizer_policy.set_step(latest_iteration as usize * self.trainer_config.update_step);

        let mut optimizer_critic = AdamWCustom::new_lr(
            trained_network.get_critic_vars(),
            self.trainer_config.learning_rate,
        )?;
        optimizer_critic.set_step(latest_iteration as usize * self.trainer_config.update_step);

        // Main training loop
        for iteration in (latest_iteration as usize + 1)..self.trainer_config.max_iters {
            println!("Iteration: {}", iteration);

            // Rollout hands and build hand states
            let mut hand_states = self.build_hand_states(
                &trained_network, 
                &agent_pool, 
                self.trainer_config.epsilon_greedy_factor * self.trainer_config.epsilon_greedy_decay.powi(iteration as i32))?;

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

                println!("Batch size: {}", step_cnt);

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
                .detach();

            let base_actor_outputs = trained_network.forward_actor(&old_embedding)?.detach();

            let base_critic_outputs = trained_network
                .forward_critic(&old_embedding)?
                .unwrap()
                .detach();

            let old_probs_tensor = (base_actor_outputs
                .gather(&action_indexes_tensor, 1)?
                .squeeze(1)?
                + log_epsilon)?; // Add an epsilon to avoid log(0)
            let old_probs_log_tensor = old_probs_tensor.log()?;

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
                        gae_lambda,
                    );
                    advantage_gae.append(&mut advantage);
                }

                // Normalize advantage_gae
                Self::normalize_mean_std(&mut advantage_gae);
            }

            let advantage_tensor = Tensor::new(advantage_gae, &self.device)?;

            for _update_step in 0..self.trainer_config.update_step {
                // Get embedding
                let embedding =
                    trained_network.forward_embedding(&card_input_tensor, &action_input_tensor, true)?;

                // Run actor
                let actor_outputs = trained_network.forward_actor(&embedding)?;
                let probs_tensor = (actor_outputs
                    .gather(&action_indexes_tensor, 1)?
                    .squeeze(1)?
                    + log_epsilon)?; // Add an epsilon to avoid log(0)
                let probs_log_tensor = probs_tensor.log()?;

                // Get trinal clip policy loss
                let mut policy_loss = self.get_trinal_clip_policy_loss(
                    &advantage_tensor,
                    &probs_log_tensor,
                    &old_probs_log_tensor,
                );

                println!(
                    "Policy loss: {:?}",
                    policy_loss.as_ref().unwrap().to_scalar::<f32>()
                );

                // Calculate entropy regularization, to encourage exploration
                if self.trainer_config.use_entropy {
                    let entropy = (actor_outputs.detach() * (actor_outputs.detach() + log_epsilon)?.log()?)?.sum(1)?.mean(0)?;

                    println!(
                        "Entropy loss: {:?}",
                        entropy.as_ref().to_scalar::<f32>()
                    );

                    // Add entropy to loss
                    policy_loss = policy_loss - (entropy * self.trainer_config.entropy_beta)?;

                    println!(
                        "Final policy loss: {:?}",
                        policy_loss.as_ref().unwrap().to_scalar::<f32>()
                    );
                }

                let gradients_policy = policy_loss?.backward()?;

                // Get critic output
                let critic_outputs = trained_network.forward_critic(&embedding)?;

                // Get trinal clip value loss
                let value_loss = self.get_trinal_clip_value_loss(
                    &critic_outputs.unwrap().copy()?,
                    &gamma_rewards_tensor,
                    &max_rewards_tensor,
                    &min_rewards_tensor,
                );

                println!(
                    "Value loss: {:?}",
                    value_loss.as_ref().unwrap().to_scalar::<f32>()
                );

                let gradients_value = value_loss?.backward()?;

                // Calculate siamese gradients, weighted sum of actor and critic gradients
                // I did not find a better way to create a new GradStore, is there one ?
                let mut gradients_embedding =
                    Tensor::zeros((), candle_core::DType::F32, &self.device)?.backward()?;
                trained_network
                .get_var_map()
                    .data()
                    .lock()
                    .unwrap()
                    .iter()
                    .for_each(|(k, v)| {
                        let grad_policy = gradients_policy.get_id(v.id());
                        let grad_value = gradients_value.get_id(v.id());

                        if grad_policy.is_some() {
                            if let Err(e) = helper::check_tensor(grad_policy.unwrap()) {
                                println!("PROBS: {:?}", helper::fast_flatten(&probs_tensor));
                                println!(
                                    "PROBS LOG: {:?}",
                                    helper::fast_flatten(&probs_log_tensor)
                                );
                                println!(
                                    "OLD PROBS LOG: {:?}",
                                    helper::fast_flatten(&old_probs_log_tensor)
                                );
                                println!("Policy gradient error at key {}: {}", k, e);
                                panic!("Policy gradient error");
                            }
                        }

                        if k.starts_with("siamese") && grad_policy.is_some() && grad_value.is_some()
                        {
                            let grad_weighted = ((grad_policy.unwrap() * 0.5).unwrap()
                                + (grad_value.unwrap() * 0.5).unwrap())
                            .unwrap();
                            gradients_embedding.insert(v, grad_weighted);
                        }
                    });

                // Do backprop
                optimizer_policy.step(&gradients_policy)?;
                optimizer_critic.step(&gradients_value)?;
                optimizer_embedding.step(&gradients_embedding)?;
            }

            self.test_clone(&trained_network, iteration as u32)?;

            let file_name = Path::new(&self.output_path).join(format!("poker_network_{}.pt", iteration));
            trained_network.save_var_map(
                file_name.clone(),
            )?;
            let mut cloned_network = PokerNetwork::new(
                self.player_cnt,
                self.action_config.clone(),
                self.device.clone(),
                self.trainer_config.agents_device.clone(),
                false,
            )?;
            cloned_network.load_var_map(file_name)?;

            Tree::print_first_actions(
                &cloned_network,
                &self.device.clone(),
                self.trainer_config.no_invalid_for_traverser,
                self.action_config
            )?;

            // for _ in 0..10 {
            //     self.tree._play_one_hand(
            //         &trained_network,
            //         &self.device,
            //         self.trainer_config.no_invalid_for_traverser,
            //     )?;
            // }

            // if iteration > 0 && (iteration % self.trainer_config.save_interval as usize == 0 || iteration == 25) {
                // trained_network.var_map.save(
                //     Path::new(&self.output_path).join(&format!("poker_network_{}.pt", iteration)),
                // )?;
            // }

            // Put a new agent in the pool every 100 iterations
            if iteration % self.trainer_config.new_agent_interval as usize == 0 || iteration == 25 {
                self.refresh_agents(&mut agent_pool)?;
            }
        }

        Ok(())
    }

    fn build_hand_states(
        &self,
        trained_network: &PokerNetwork,
        agent_pool: &AgentPool,
        epsilon_greedy: f32,
    ) -> Result<Vec<HandState>, Box<dyn std::error::Error>> {
        let start_time = Instant::now();
        let hand_states_base = Arc::new(Mutex::new(Vec::new()));

        // let n_workers = num_cpus::get() / 2;
        // let thread_pool = ThreadPool::new(n_workers);

        // Clone trained network for inference
        for _ in 0..self.n_workers {
            let hand_states = Arc::clone(&hand_states_base);
            let trained_agent: Box<dyn Agent> = Box::new(AgentNetwork::new(trained_network.clone()));
            let agent_pool_clone = agent_pool.clone();
            let player_cnt = self.player_cnt;
            let action_config = self.action_config.clone();
            let no_invalid_for_traverser = self.trainer_config.no_invalid_for_traverser;
            let iterations = self.trainer_config.hands_per_player_per_iteration / self.n_workers;
            let use_epsilon_greedy = self.trainer_config.use_epsilon_greedy;
            let agent_device = self.trainer_config.agents_device.clone();

            self.thread_pool.execute(move || {
                let mut new_hand_states = Vec::new();

                for _ in 0..iterations {
                    let mut tree = Tree::new(player_cnt, &action_config);

                    for traverser in 0..player_cnt {
                    // Select agents
                    let mut agents = Vec::new();
                    for p in 0..player_cnt {
                        let agent = if p != traverser {
                            agent_pool_clone.get_agent().1
                        } else {
                            &*trained_agent
                        };
                        agents.push(agent);
                    }

                    // Traverse tree
                    if tree
                        .traverse(
                            traverser,
                            &agents,
                            &agent_device,
                            no_invalid_for_traverser,
                            if use_epsilon_greedy {epsilon_greedy} else {0.0}
                        )
                        .is_err()
                    {
                        continue;
                    }

                    // Make sure the hand state has at least one state for the traverser, and that we
                    // do not have too much actions per street. Action count should already be checked
                    // in the tree.traverse method, so this check may be removed in the future.
                    let hs = tree.hand_state.clone();
                    if let Some(hs) = hs {
                        // Count number of action states for traverser
                        if hs.get_traverser_action_states().is_empty() {
                            continue;
                        }
                        // Count number of action per street
                        let mut action_cnt = 0;
                        let mut too_much = false;
                        let mut street = 0;
                        for action_state in hs.action_states.iter() {
                            if action_state.street == street {
                                action_cnt += 1;
                            } else {
                                street = action_state.street;
                                action_cnt = 0;
                            }

                            if action_cnt >= action_config.max_actions_per_street {
                                too_much = true;
                                println!("Trainer: WARNING Too much actions per street, we shouldn't be here.");
                                break;
                            }
                        }

                        if too_much {
                            continue;
                        }

                        new_hand_states.push(hs);
                    }
                }
            }
            hand_states.lock().unwrap().append(&mut new_hand_states); 
            }); 
        }

        self.thread_pool.join();

        let duration = start_time.elapsed();
        println!("Rollout duration: {:?}", duration);

        Ok(Arc::try_unwrap(hand_states_base).unwrap().into_inner().unwrap())
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
        lambda: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let size = rewards.len();
        let mut advantage = vec![0.0; size + 1];

        for t in (0..size).rev() {
            let delta =
                rewards[t] + gamma * (if t == size - 1 { 0.0 } else { values[t + 1] }) - values[t];
            advantage[t] = delta + gamma * lambda * advantage[t + 1];
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

    fn load_existing(
        &self,
        trained_network: &mut PokerNetwork,
        agent_pool: &mut AgentPool,
    ) -> Result<u32, Box<dyn std::error::Error>> {
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
            trained_network.load_var_map(
                trained_network_path.join(format!("poker_network_{}.pt", latest_iteration)),
            )?;

            if latest_iteration >= self.trainer_config.new_agent_interval {
                let delta = latest_iteration % self.trainer_config.new_agent_interval;
                let end = latest_iteration - delta;
                let start = std::cmp::max(
                    self.trainer_config.new_agent_interval as i32,
                    end as i32 - 9 * self.trainer_config.new_agent_interval as i32,
                );
                for i in (start as u32..end + self.trainer_config.new_agent_interval)
                    .step_by(self.trainer_config.new_agent_interval as usize)
                {
                    let mut network = PokerNetwork::new(
                        self.player_cnt,
                        self.action_config.clone(),
                        self.trainer_config.agents_device.clone(),
                        self.trainer_config.agents_device.clone(),
                        false,
                    )?;
                    network
                        .load_var_map(trained_network_path.join(format!("poker_network_{}.pt", i)))?;
                    agent_pool.add_agent(Box::new(AgentNetwork::new(network)));
                }
            }
        }

        Ok(latest_iteration)
    }

    fn refresh_agents(&self, agent_pool: &mut AgentPool) -> Result<(), Box<dyn std::error::Error>> {
        // List files in output path
        let trained_network_path = Path::new(&self.output_path);
        let trained_network_files = trained_network_path.read_dir()?;
        let mut model_files = Vec::new();

        for file in trained_network_files {
            let file = file?;
            let file_name = file.file_name();
            let file_name = file_name.to_str().unwrap();
            let file_name = file_name.split('_').collect::<Vec<&str>>();

            if file_name.len() > 2 {
                let split = file_name[2].split('.').collect::<Vec<&str>>();
                if (split.len() == 2) && (split[1] == "pt") {
                    let iteration = split[0].parse::<u32>()?;
                    if iteration == 25 || iteration % 100 == 0 {
                        model_files.push(trained_network_path.join(format!("poker_network_{}.pt", iteration)).to_str().unwrap().to_owned());
                    }
                }
            }
        }

        agent_pool.play_tournament(self.player_cnt, self.action_config, &model_files, &self.trainer_config.agents_device)
    }

    fn test_clone(&self, trained_network: &PokerNetwork, iteration: u32) -> Result<(), Box<dyn std::error::Error>> {
        let file_name = Path::new(&self.output_path).join(format!("poker_network_{}.pt", iteration));
        trained_network.save_var_map(
            file_name.clone(),
        )?;

        // {
        //     let content = trained_network.var_map.data().lock().unwrap();
        //     let mut tensors = std::collections::HashMap::new();
        //     for (tensor_name, var) in content.iter() {
        //         tensors.insert(tensor_name.to_string(), var.as_tensor().clone());
        //     }
        //     candle_core::safetensors::save(&tensors, file_name.clone())?;
        // }

        // let loaded_tensors = candle_core::safetensors::load(file_name.clone(), &Device::Cpu)?;
        // let tenors = trained_network.get
        let real_batch_norm_tensors = trained_network.get_batch_norm_tensors();

        let mut cloned_network = PokerNetwork::new(
            self.player_cnt,
            self.action_config.clone(),
            Device::Cpu,
            self.trainer_config.agents_device.clone(),
            false,
        )?;

        cloned_network.load_var_map(file_name.clone())?;
        let clone_batch_norm_tensors = trained_network.get_batch_norm_tensors();

        println!("-------------------");
        println!("Comparing VarMaps");
        cloned_network.get_var_map().data().lock().unwrap().iter().for_each(|(k, v)| {
            // trained_network.var_map.data().lock().unwrap().iter().for_each(|(kt, vt)| {
            //     if kt == k {
            //         let diff = (v.as_tensor() - vt.to_device(&Device::Cpu).unwrap()).unwrap().abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
            //         println!("Diff: {:?} for key: {:?}", diff, k);
            //     }
            // });

            let flat = v.as_tensor().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            //println!("CLONED {}: {:?}", k, flat);

            if real_batch_norm_tensors.contains_key(k) {
                let real = real_batch_norm_tensors.get(k).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
                //println!("REAL {}: {:?}", k, real);
                if real != flat {
                    println!("REAL Diff: {:?} for key: {:?}", k, k);
                }
            }

            if clone_batch_norm_tensors.contains_key(k) {
                let clone = clone_batch_norm_tensors.get(k).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
                //println!("CLONE {}: {:?}", k, clone);
                if clone != flat {
                    println!("CLONE Diff: {:?} for key: {:?}", k, k);
                }
            }

            if trained_network.get_var_map().data().lock().unwrap().contains_key(k) {
                let real = trained_network.get_var_map().data().lock().unwrap().get(k).unwrap().as_tensor().flatten_all().unwrap().to_vec1::<f32>().unwrap();
                //println!("REAL {}: {:?}", k, real);
                if real != flat {
                    println!("TRAINED Diff: {:?} for key: {:?}", k, k);
                }
            }
        });
        println!("-------------------");

        let mut cloned_gpu = PokerNetwork::new(
            self.player_cnt,
            self.action_config.clone(),
            self.device.clone(),
            self.trainer_config.agents_device.clone(),
            false,
        )?;
        cloned_gpu.load_var_map(file_name.clone())?;

        let rank1= 5;
        let suit1= 0;
        let rank2= 10;
        let suit2= 2;

        let mut card_vecs: Vec<Vec<Vec<f32>>> = vec![vec![vec![0.0; 13]; 4]; 6];

        // Set hand cards
        card_vecs[0][suit1][rank1] = 1.0;
        card_vecs[0][suit2][rank2] = 1.0;
        card_vecs[4][suit1][rank1] = 1.0;
        card_vecs[4][suit2][rank2] = 1.0;

        // Create action tensor
        // Shape is (street_cnt * max_actions_per_street) x (player_count + 2 for sum and legal) x max_number_of_actions
        let action_vecs: Vec<Vec<Vec<f32>>> =
            vec![
                vec![
                    vec![0.0; 3 + self.action_config.postflop_raise_sizes.len()];
                    self.action_config.player_count as usize + 2
                ];
                4 * self.action_config.max_actions_per_street as usize
            ];

        let card_tensor_gpu = Tensor::new(card_vecs.to_vec(), &self.device)?.unsqueeze(0)?;
        let action_tensor_gpu = Tensor::new(action_vecs.to_vec(), &self.device)?.unsqueeze(0)?;

        let card_tensor_cpu = Tensor::new(card_vecs.to_vec(), &Device::Cpu)?.unsqueeze(0)?;
        let action_tensor_cpu = Tensor::new(action_vecs.to_vec(), &Device::Cpu)?.unsqueeze(0)?;

        let embed_gpu = trained_network.forward_embedding(&card_tensor_gpu, &action_tensor_gpu, false)?;
        let embed_gpu_clone = cloned_gpu.forward_embedding(&card_tensor_gpu, &action_tensor_gpu, false)?;
        let embed_cpu = cloned_network.forward_embedding(&card_tensor_cpu, &action_tensor_cpu, false)?;
        
        // Compare varmaps
        let varmapgpu = trained_network.get_var_map().data().lock().unwrap();
        // let varmapgpu_clone = cloned_gpu.var_map.data().lock().unwrap();
        let varmapcpu = cloned_network.get_var_map().data().lock().unwrap();

        let mut has_diff = false;
        for (k, v) in varmapgpu.iter() {
            let vcpu = varmapcpu.get(k).unwrap();
            let diff = (v.as_tensor().to_device(&Device::Cpu)? - vcpu.as_tensor())?.abs()?.sum_all()?.to_scalar::<f32>()?;
            // println!("Diff: {:?} for key: {:?}", diff, k);
            if diff > 1e-5 {
                has_diff = true;
            }
        }

        if has_diff {
            panic!("Varmap diff");
        }

        let diff_gpu = (embed_gpu.clone() - embed_gpu_clone.clone())?.abs()?.sum_all()?.to_scalar::<f32>()?;
        println!("Diff GPU: {:?}", diff_gpu);
        let diff_cpu = (embed_gpu.to_device(&Device::Cpu)? - embed_cpu.clone())?.abs()?.sum_all()?.to_scalar::<f32>()?;
        println!("Diff CPU: {:?}", diff_cpu);
        let diff_new = (embed_gpu_clone.clone().to_device(&Device::Cpu)? - embed_cpu.clone())?.abs()?.sum_all()?.to_scalar::<f32>()?;
        println!("Diff NEW: {:?}", diff_new);


        // let result_gpu = trained_network.forward_actor(&embed.copy()?)?;
        // let result_cpu = cloned_network.forward_actor(&embed.copy()?.to_device(&Device::Cpu)?)?;

        // let card_tensor_cpu = Tensor::new(card_vecs, &self.device)?.unsqueeze(0)?;
        // let action_tensor_cpu = Tensor::new(action_vecs, &self.device)?.unsqueeze(0)?;

        // let result_cpu = cloned_network.forward_embedding(&card_tensor_cpu, &action_tensor_cpu, false)?;

        // Check if the results are the same
        // let diff = (result_gpu.to_device(&Device::Cpu)? - result_cpu)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        // println!("Diff: {:?}", diff);

        Ok(())
    }

}
