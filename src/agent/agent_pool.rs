use std::sync::{Arc, Mutex};

use crate::{
    game::{action::ActionConfig, tree::Tree},
    model::poker_network::PokerNetwork,
};

use super::Agent;
use super::{agent_network::AgentNetwork, agent_random::AgentRandom};

use rand::Rng;

use itertools::Itertools;
use threadpool::ThreadPool;

pub struct AgentPool {
    agent_count: u32,
    agents: Vec<Arc<Box<dyn Agent>>>,
    agent_random: Arc<Box<dyn Agent>>,
}

impl Clone for AgentPool {
    fn clone(&self) -> AgentPool {
        AgentPool {
            agent_count: self.agent_count,
            agents: self.agents.clone(),
            agent_random: Arc::clone(&self.agent_random),
        }
    }
}

impl AgentPool {
    pub fn new(agent_count: u32) -> AgentPool {
        AgentPool {
            agent_count,
            agents: Vec::new(),
            agent_random: Arc::new(Box::new(AgentRandom {})),
        }
    }

    pub fn add_agent(&mut self, agent: Box<dyn Agent>) {
        self.agents.push(Arc::new(agent));
    }

    pub fn get_agent(&self) -> (i32, Arc<Box<dyn Agent>>) {
        if self.agents.is_empty() {
            return (-1, Arc::clone(&self.agent_random));
        }

        let mut rng = rand::thread_rng();

        // Return random agent 25% of the time when we have less than 3 agents
        if self.agents.len() < 3 {
            let random_float_0_1: f32 = rng.gen();
            if random_float_0_1 <= 0.25 {
                return (-1, Arc::clone(&self.agent_random));
            }
        }

        // Get random index in the last agent_count networks in self.agents
        let rand_index = if self.agents.len() >= self.agent_count as usize {
            rng.gen_range(self.agents.len() - (self.agent_count as usize)..self.agents.len())
        } else {
            rng.gen_range(0..self.agents.len())
        };

        (rand_index as i32, Arc::clone(&self.agents[rand_index]))
    }

    pub fn play_tournament(
        &mut self,
        player_count: u32,
        action_config: &ActionConfig,
        model_files: &[String],
        iterations_per_match: u32,
        device: &candle_core::Device,
    ) -> Result<(), candle_core::Error> {
        println!(
            "Refreshing agents, playing tournament with {} agents",
            model_files.len()
        );

        let wanted_agents = self.agent_count as usize;
        let max_agents_per_pool = player_count * 2;
        let available_agent_cnt = model_files.len();
        let n_workers = num_cpus::get();
        let thread_pool = ThreadPool::new(n_workers);

        if model_files.len() <= wanted_agents {
            // Load everything and return
            println!("Less than wanted agent cnt, loading all");
            self.update_agents(player_count, action_config, model_files, device)?;
            return Ok(());
        }

        // Check if the number of agents is a multiple of player_count, so we can play a round robin
        // let needed_agents = if model_files.len() % player_count as usize == 0 {
        //     model_files.len()
        // } else {
        //     model_files.len() + player_count as usize - (model_files.len() % player_count as usize)
        // };

        let mut players: Vec<usize> = (0..available_agent_cnt).collect();

        // Create pools of maximum first_round_max_agents_per_pool agents, making sure the
        // same agent is not selected twice and all agents are in a pool
        let mut min_pool_count =
            (available_agent_cnt as f32 / max_agents_per_pool as f32).ceil() as usize;
        let mut first_round_max_agents_per_pool = available_agent_cnt / min_pool_count;

        // We need to make sure we have enough players for one game at least
        if first_round_max_agents_per_pool < player_count as usize {
            min_pool_count -= 1;
            first_round_max_agents_per_pool = available_agent_cnt / min_pool_count;
        }

        let mut agent_pool: Vec<Vec<usize>> = Vec::new();
        let mut pool: Vec<usize> = Vec::new();
        for _ in 0..available_agent_cnt {
            // Get a random index in players
            let mut rng = rand::thread_rng();
            let rand_index = rng.gen_range(0..players.len());

            // Add the index to the pool
            pool.push(players[rand_index]);

            // If the pool is full, add it to agent_pool
            if pool.len() == first_round_max_agents_per_pool {
                agent_pool.push(pool);
                pool = Vec::new();
            }

            // Remove the index from players
            players.remove(rand_index);
        }

        if !pool.is_empty() {
            agent_pool.push(pool);
        }

        let total_won = Arc::new(Mutex::new(vec![0.0f32; available_agent_cnt]));

        // Play a round robin tournament in each pool
        for pool in agent_pool.iter() {
            // First, load agents
            let mut pool_agents_base: Vec<Arc<Box<dyn Agent>>> = Vec::new();
            for agent_index in pool {
                let mut network = PokerNetwork::new(
                    player_count,
                    action_config.clone(),
                    device.clone(),
                    device.clone(),
                    false,
                )?;
                network.load_var_map(model_files[*agent_index].as_str())?;
                pool_agents_base.push(Arc::new(Box::new(AgentNetwork::new(network))));
            }

            let mut schedule = Vec::new();

            // Generate all combinations of players in groups of 3
            for combination in pool.iter().combinations(player_count as usize) {
                if combination.len() == player_count as usize {
                    schedule.push(combination.iter().map(|x| **x).collect::<Vec<usize>>());
                }
            }

            let pool_arc = Arc::new(pool_agents_base);

            // Play N hands for each match up
            for g in schedule {
                let pool_clone = Arc::clone(&pool_arc);
                let ac_clone = action_config.clone();
                let device_clone = device.clone();
                let total_won_clone = Arc::clone(&total_won);

                thread_pool.execute(move || {
                    let mut tree = Tree::new(player_count, &ac_clone);
                    let mut total_won_local = vec![0.0f32; available_agent_cnt];

                    for _ in 0..iterations_per_match {
                        // Assign each agent a random index between 0 and player_count
                        let mut game = g.clone();
                        let mut agents = Vec::new();
                        let mut indexes = Vec::new();
                        for _ in 0..player_count {
                            let mut rng = rand::thread_rng();
                            let rand_index = rng.gen_range(0..game.len());
                            agents.push(Arc::clone(&pool_clone[rand_index]));
                            indexes.push(game[rand_index]);
                            game.remove(rand_index);
                        }

                        // Play one game
                        let rewards = match tree.play_one_hand(&agents, &device_clone, true) {
                            Ok(r) => r,
                            Err(error) => panic!("ERROR in play_one_hand: {:?}", error),
                        };
                        for p in 0..player_count as usize {
                            total_won_local[indexes[p]] += rewards[p];
                        }
                    }

                    let mut total_won_locked = total_won_clone.lock().unwrap();
                    for p in 0..available_agent_cnt {
                        total_won_locked[p] += total_won_local[p];
                    }
                });
            }

            thread_pool.join();

            // Print results
            let total_won_locked = total_won.lock().unwrap();
            for agent_index in pool.iter() {
                println!(
                    "Agent {} won {} chips",
                    model_files[*agent_index], total_won_locked[*agent_index]
                );
            }

            println!("-------------------");
        }

        // We keep the 80% of wanted_agents as best agents
        let mut best_agents: Vec<usize> = Vec::new();
        {
            let total_won_clone = Arc::clone(&total_won);
            let total_won_locked = total_won_clone.lock().unwrap();

            for _ in 0..(wanted_agents * 8 / 10) {
                let mut max_index = 0;
                let mut max_value = -std::f32::MAX;
                let mut found = false;
                for (i, v) in total_won_locked.iter().enumerate() {
                    if !best_agents.contains(&i) && *v > max_value {
                        max_value = *v;
                        max_index = i;
                        found = true;
                    }
                }

                if !found {
                    break;
                }
                println!(
                    "Best agents: adding index {} with value {}, file: {}",
                    max_index, max_value, model_files[max_index]
                );
                best_agents.push(max_index);
            }
        }

        // We keep the 20% of wanted_agents as worst agents
        let mut worst_agents: Vec<usize> = Vec::new();

        for _ in 0..(wanted_agents * 2 / 10) {
            let mut min_index = 0;
            let mut min_value = std::f32::MAX;
            let total_won_clone = Arc::clone(&total_won);
            let total_won_locked = total_won_clone.lock().unwrap();
            let mut found = false;
            for (i, v) in total_won_locked.iter().enumerate() {
                if !worst_agents.contains(&i) && !best_agents.contains(&i) && *v < min_value {
                    min_value = *v;
                    min_index = i;
                    found = true;
                }
            }

            if !found {
                break;
            }
            println!(
                "Worst agents: adding index {} with value {}, file: {}",
                min_index, min_value, model_files[min_index]
            );
            worst_agents.push(min_index);
        }

        best_agents.append(&mut worst_agents);

        let final_agents_files: Vec<String> = best_agents
            .iter()
            .map(|x| model_files[*x].clone())
            .collect();

        self.update_agents(player_count, action_config, &final_agents_files, device)?;

        // Print selected agents
        println!("Selected agents:");
        for agent_file in final_agents_files.iter() {
            println!("{}", agent_file);
        }

        Ok(())
    }

    fn update_agents(
        &mut self,
        player_count: u32,
        action_config: &ActionConfig,
        agent_files: &[String],
        device: &candle_core::Device,
    ) -> Result<(), candle_core::Error> {
        self.agents.clear();
        for agent_file in agent_files.iter() {
            let mut network = PokerNetwork::new(
                player_count,
                action_config.clone(),
                device.clone(),
                device.clone(),
                false,
            )?;
            network.load_var_map(agent_file.as_str())?;
            self.add_agent(Box::new(AgentNetwork::new(network)));
        }
        Ok(())
    }
}
