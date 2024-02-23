use std::{
    cmp::Ordering,
    sync::{Arc, Mutex},
};

use candle_core::Device;
use rand::Rng;
use threadpool::ThreadPool;

use crate::{
    game::{action::ActionConfig, tree::Tree},
    model::poker_network::PokerNetwork,
};

use super::{agent_network::AgentNetwork, Agent};
use std::fs::File;
use std::io::Read;
use std::io::Write;

struct AgentTournament {
    network_file: String,
    elo: f32,
    iteration: u32,
    agent_network: Arc<Box<dyn Agent>>,
}

pub struct Tournament {
    agents: Vec<Arc<Mutex<AgentTournament>>>,
    player_count: u32,
    action_config: ActionConfig,
    device: Device,
}

impl Tournament {
    pub fn new(player_count: u32, action_config: ActionConfig, device: Device) -> Tournament {
        Tournament {
            agents: Vec::new(),
            player_count,
            action_config,
            device,
        }
    }

    pub fn add_agent(
        &mut self,
        network_file: String,
        iteration: u32,
    ) -> Result<(), candle_core::Error> {
        let mut network = PokerNetwork::new(
            self.player_count,
            self.action_config.clone(),
            self.device.clone(),
            self.device.clone(),
            false,
        )?;
        network.load_var_map(network_file.as_str())?;

        self.agents.push(Arc::new(Mutex::new(AgentTournament {
            network_file,
            elo: 1000.0,
            iteration,
            agent_network: Arc::new(Box::new(AgentNetwork::new(network))),
        })));
        Ok(())
    }

    pub fn get_best_agents(&mut self, cnt: usize) -> Vec<Arc<Box<dyn Agent>>> {
        let mut result = Vec::new();
        self.agents
            .sort_by(|a, b| b.lock().unwrap().elo.total_cmp(&a.lock().unwrap().elo));
        for i in 0..cnt {
            let agent = self.agents[i].lock().unwrap();
            println!("Taking agent: {}, elo: {}", agent.iteration, agent.elo);
            result.push(Arc::clone(&agent.agent_network));
        }
        result
    }

    pub fn play(&mut self, rounds: usize) {
        // Dynamically select agents based on Elo for each game
        self.agents
            .sort_by(|a, b| b.lock().unwrap().elo.total_cmp(&a.lock().unwrap().elo));
        let agents_tournament_base = Arc::new(Mutex::new(self.agents.clone()));
        let n_workers = num_cpus::get();
        let thread_pool = ThreadPool::new(n_workers);
        let elo_k = 10.0;

        let batch_size = (self.agents.len() as f32 / n_workers as f32).ceil() as usize;

        for worker in 0..n_workers {
            // println!("Round: {} / {}", round, rounds);
            let player_count = self.player_count;
            let agents_length = self.agents.len();
            let start = worker * batch_size;
            let end = ((worker + 1) * batch_size).min(agents_length);
            let action_config = self.action_config.clone();
            let device = self.device.clone();
            let agents_tournament = Arc::clone(&agents_tournament_base);

            thread_pool.execute(move || {
                // Run the loop N times
                for _ in 0..rounds {
                    // We process each player on time per iteration
                    for agent_index in start..end {
                        // println!("Agent: {} / {}", agent_index, agents_length);
                        // Choose agents with closer Elo for fairer matches
                        let current_agent =
                            Arc::clone(&agents_tournament.lock().unwrap()[agent_index]);
                        let chosen_elo = current_agent.lock().unwrap().elo;

                        // Step 1: Sort the remaining players by their elo difference to the chosen player
                        let mut elo_differences = agents_tournament
                            .lock()
                            .unwrap()
                            .iter()
                            .enumerate()
                            .filter_map(|(index, agent)| {
                                if index != agent_index {
                                    let agent = agent.lock().unwrap();
                                    Some((index, (chosen_elo - agent.elo).abs()))
                                } else {
                                    None
                                }
                            })
                            .collect::<Vec<_>>();

                        elo_differences
                            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

                        // Step 2: Take N-1 agents with the closest elo difference, then add current agent
                        let mut game_agents = elo_differences
                            .iter()
                            .take((player_count - 1) as usize)
                            .map(|&(index, _)| {
                                Arc::clone(&agents_tournament.lock().unwrap()[index])
                            })
                            .collect::<Vec<_>>();

                        game_agents.push(current_agent);

                        // Play one game
                        let mut total_won = vec![0.0f32; player_count as usize];
                        let mut tree = Tree::new(player_count, &action_config);

                        // Assign each agent a random index between 0 and player_count
                        let mut game = (0..player_count).collect::<Vec<_>>();
                        let mut agents_tour = Vec::new();
                        let mut indexes = Vec::new();
                        for _ in 0..player_count {
                            let mut rng = rand::thread_rng();
                            let rand_index = rng.gen_range(0..game.len());
                            agents_tour.push(Arc::clone(&game_agents[rand_index]));
                            indexes.push(game[rand_index]);
                            game.remove(rand_index);
                        }
                        let agents: Vec<Arc<Box<dyn Agent>>> = agents_tour
                            .iter()
                            .map(|p| Arc::clone(&p.lock().unwrap().agent_network))
                            .collect::<Vec<_>>();

                        // Play one game
                        let rewards = match tree.play_one_hand(&agents, &device, true) {
                            Ok(r) => r,
                            Err(error) => panic!("ERROR in play_one_hand: {:?}", error),
                        };
                        for p in 0..player_count as usize {
                            total_won[indexes[p] as usize] += rewards[p];
                        }

                        // Check each pair of players to determine winner(s) and update ELO
                        let mut elo_diff = vec![0.0f32; player_count as usize];
                        for i in 0..player_count as usize {
                            for j in i + 1..player_count as usize {
                                let player_i_result = if total_won[i] > total_won[j] {
                                    1.0
                                } else if total_won[i] < total_won[j] {
                                    0.0
                                } else {
                                    0.5
                                };

                                let agent_i = game_agents[i].lock().unwrap();
                                let agent_j = game_agents[j].lock().unwrap();

                                let elo_i = agent_i.elo;
                                let elo_j = agent_j.elo;

                                let prob = Self::calculate_expected_win_prob(elo_i, elo_j, elo_k);

                                elo_diff[i] +=
                                    Self::calculate_elo_diff(prob, player_i_result, elo_k);
                                elo_diff[j] += Self::calculate_elo_diff(
                                    1.0 - prob,
                                    1.0 - player_i_result,
                                    elo_k,
                                );
                            }
                        }

                        // let base_elo = game_agents
                        //     .iter()
                        //     .map(|a| a.lock().unwrap().elo)
                        //     .collect::<Vec<_>>();

                        // Update ELO
                        for i in 0..player_count as usize {
                            let mut agent = game_agents[i].lock().unwrap();
                            agent.elo += elo_diff[i];
                            agent.elo = agent.elo.max(0.0);
                        }

                        // Print total won and new elo
                        // let _unused = agents_tournament.lock().unwrap();
                        // println!("Base elo: {:?}", base_elo);
                        // println!("ELO diff: {:?}", elo_diff);
                        // println!("Total won: {:?}", total_won);
                        // for agent in &game_agents {
                        //     let agent = agent.lock().unwrap();
                        //     println!("Agent: {} - ELO: {}", agent.iteration, agent.elo,);
                        // }
                    }
                }
            });
        }

        thread_pool.join();

        // Sort agents by ELO and print them
        self.agents
            .sort_by(|a, b| b.lock().unwrap().elo.total_cmp(&a.lock().unwrap().elo));
        for agent in &self.agents {
            let agent = agent.lock().unwrap();
            println!("Agent: {} - ELO: {}", agent.iteration, agent.elo,);
        }
    }

    fn calculate_expected_win_prob(elo_a: f32, elo_b: f32, k: f32) -> f32 {
        let elo_diff = elo_b - elo_a;
        1.0 / (1.0 + 10.0f32.powf(elo_diff / k))
    }

    fn calculate_elo_diff(expected_win_prob: f32, actual_win: f32, k: f32) -> f32 {
        k * (actual_win - expected_win_prob)
    }

    pub fn save_state<P: AsRef<std::path::Path>>(&self, path: P) {
        let mut file = File::create(path).expect("Failed to create file");

        for agent in &self.agents {
            let agent = agent.lock().unwrap();
            let line = format!("{};{};{}\n", agent.iteration, agent.elo, agent.network_file);
            file.write_all(line.as_bytes())
                .expect("Failed to write to file");
        }
    }

    pub fn load_state<P: AsRef<std::path::Path>>(&mut self, path: P) {
        let mut file = File::open(path).expect("Failed to open file");
        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .expect("Failed to read file");

        for line in contents.lines() {
            let parts: Vec<&str> = line.split(';').collect();
            let iteration = parts[0].parse::<u32>().unwrap();
            let elo = parts[1].parse::<f32>().unwrap();
            let network_file = parts[2].to_string();
            self.add_agent(network_file, iteration).unwrap();
            let agent = self.agents.last().unwrap();
            let mut agent = agent.lock().unwrap();
            agent.elo = elo;
        }
    }
}
