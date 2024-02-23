use std::sync::Arc;

use super::agent_random::AgentRandom;
use super::Agent;

use rand::Rng;

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

    pub fn set_agents(&mut self, agents: &[Arc<Box<dyn Agent>>]) {
        self.agents.clear();
        self.agents.append(&mut agents.to_vec());
    }

    // pub fn add_agent(&mut self, agent: Box<dyn Agent>) {
    //     self.agents.push(Arc::new(agent));
    // }

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
}
