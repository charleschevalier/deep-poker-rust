use crate::{
    game::{action::ActionConfig, tree::Tree},
    model::poker_network::PokerNetwork,
};

use super::Agent;
use super::{agent_network::AgentNetwork, agent_random::AgentRandom};

use rand::Rng;

use itertools::Itertools;

pub struct AgentPool {
    agent_count: u32,
    agents: Vec<Box<dyn Agent>>,
    agent_random: Box<dyn Agent>,
}

impl Clone for AgentPool {
    fn clone(&self) -> AgentPool {
        AgentPool {
            agent_count: self.agent_count,
            agents: self.agents.clone(),
            agent_random: self.agent_random.clone_box(),
        }
    }
}

impl AgentPool {
    pub fn new(agent_count: u32) -> AgentPool {
        AgentPool {
            agent_count,
            agents: Vec::new(),
            agent_random: Box::new(AgentRandom {}),
        }
    }

    pub fn add_agent(&mut self, agent: Box<dyn Agent>) {
        self.agents.push(agent);
    }

    pub fn get_agent(&self) -> (i32, &dyn Agent) {
        if self.agents.is_empty() {
            return (-1, &*self.agent_random);
        }

        let mut rng = rand::thread_rng();

        // Return random agent 25% of the time when we have less than 3 agents
        if self.agents.len() < 3 {
            let random_float_0_1: f32 = rng.gen();
            if random_float_0_1 <= 0.25 {
                return (-1, &*self.agent_random);
            }
        }

        // Get random index in the last agent_count networks in self.agents
        let rand_index = if self.agents.len() >= self.agent_count as usize {
            rng.gen_range(self.agents.len() - (self.agent_count as usize)..self.agents.len())
        } else {
            rng.gen_range(0..self.agents.len())
        };

        (rand_index as i32, &*self.agents[rand_index])
    }

    pub fn play_tournament(
        &mut self,
        player_count: u32,
        action_config: &ActionConfig,
        model_files: &[String],
        device: &candle_core::Device,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!(
            "Refreshing agents, playing tournament with {} agents",
            model_files.len()
        );

        let iterations_per_match = 100;
        let wanted_agents = self.agent_count as usize;
        let max_agents_per_pool = player_count * 2;
        let available_agent_cnt = model_files.len();

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

        let mut total_won = vec![0.0f32; available_agent_cnt];

        // Play a round robin tournament in each pool
        for pool in agent_pool.iter() {
            // First, load agents
            let mut pool_agents: Vec<Box<dyn Agent>> = Vec::new();
            for agent_index in pool {
                let mut network = PokerNetwork::new(
                    player_count,
                    action_config.clone(),
                    device.clone(),
                    device.clone(),
                    false,
                )?;
                network
                    .var_map
                    .load(model_files[*agent_index].as_str())
                    .unwrap();
                pool_agents.push(Box::new(AgentNetwork::new(network)));
            }

            let mut schedule = Vec::new();

            // Generate all combinations of players in groups of 3
            for combination in pool.iter().combinations(player_count as usize) {
                if combination.len() == player_count as usize {
                    schedule.push(combination.iter().map(|x| **x).collect::<Vec<usize>>());
                }
            }

            let mut tree = Tree::new(player_count, action_config);

            // Play N hands for each match up
            for g in schedule {
                for _ in 0..iterations_per_match {
                    // Assign each agent a random index between 0 and player_count
                    let mut game = g.clone();
                    let mut agents = Vec::new();
                    let mut indexes = Vec::new();
                    for _ in 0..player_count {
                        let mut rng = rand::thread_rng();
                        let rand_index = rng.gen_range(0..game.len());
                        agents.push(pool_agents[rand_index].clone_box());
                        indexes.push(game[rand_index]);
                        game.remove(rand_index);
                    }

                    // Play one game
                    let rewards = tree.play_one_hand(&agents, device, true)?;
                    for p in 0..player_count as usize {
                        total_won[indexes[p]] += rewards[p];
                    }
                }
            }

            // Print results
            for agent_index in pool.iter() {
                println!(
                    "Agent {} won {} chips",
                    model_files[*agent_index], total_won[*agent_index]
                );
            }

            println!("-------------------");
        }

        // We keep the 80% of wanted_agents as best agents
        let mut best_agents: Vec<usize> = Vec::new();
        for _ in 0..(wanted_agents * 8 / 10) {
            let mut max_index = 0;
            let mut max_value = 0.0;
            for (i, v) in total_won.iter().enumerate() {
                if !best_agents.contains(&i) && *v > max_value {
                    max_value = *v;
                    max_index = i;
                }
            }
            best_agents.push(max_index);
        }

        // We keep the 20% of wanted_agents as worst agents
        let mut worst_agents: Vec<usize> = Vec::new();
        for _ in 0..(wanted_agents * 2 / 10) {
            let mut min_index = 0;
            let mut min_value = std::f32::MAX;
            for (i, v) in total_won.iter().enumerate() {
                if !worst_agents.contains(&i) && *v < min_value {
                    min_value = *v;
                    min_index = i;
                }
            }
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
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.agents.clear();
        for agent_file in agent_files.iter() {
            let mut network = PokerNetwork::new(
                player_count,
                action_config.clone(),
                device.clone(),
                device.clone(),
                false,
            )?;
            network.var_map.load(agent_file.as_str())?;
            self.add_agent(Box::new(AgentNetwork::new(network)));
        }
        Ok(())
    }
}
