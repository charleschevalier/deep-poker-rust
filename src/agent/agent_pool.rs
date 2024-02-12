use super::agent_random::AgentRandom;
use super::Agent;

use rand::Rng;

pub struct AgentPool<'a> {
    agents: Vec<Box<dyn Agent<'a> + 'a>>,
    agent_random: Box<dyn Agent<'a> + 'a>,
}

impl<'a> AgentPool<'a> {
    pub fn new() -> AgentPool<'a> {
        AgentPool {
            agents: Vec::new(),
            agent_random: Box::new(AgentRandom {}),
        }
    }

    pub fn add_agent(&mut self, agent: Box<dyn Agent<'a> + 'a>) {
        self.agents.push(agent);
    }

    pub fn get_agent(&self) -> &Box<dyn Agent<'a> + 'a> {
        let mut rng = rand::thread_rng();

        // Return random agent 10% of the time
        let random_float_0_1: f32 = rng.gen();
        if random_float_0_1 <= 0.25 {
            return &self.agent_random;
        }

        // Get random index in the last 10 networks in self.agents
        let rand_index = if self.agents.len() >= 10 {
            rng.gen_range(self.agents.len() - 10..self.agents.len())
        } else {
            rng.gen_range(0..self.agents.len())
        };

        &self.agents[rand_index]
    }
}
