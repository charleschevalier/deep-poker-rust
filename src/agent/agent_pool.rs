use super::agent_random::AgentRandom;
use super::Agent;

use rand::Rng;

pub struct AgentPool<'a> {
    agents: Vec<Box<dyn Agent<'a> + 'a>>,
    agent_random: Box<dyn Agent<'a> + 'a>,
    pub win_cnt: Vec<usize>,
    pub played_cnt: Vec<usize>,
}

impl<'a> AgentPool<'a> {
    pub fn new() -> AgentPool<'a> {
        AgentPool {
            agents: Vec::new(),
            agent_random: Box::new(AgentRandom {}),
            win_cnt: Vec::new(),
            played_cnt: Vec::new(),
        }
    }

    pub fn add_agent(&mut self, agent: Box<dyn Agent<'a> + 'a>) {
        if self.agents.len() > 5 {
            self.agents.remove(0);
        }
        self.agents.push(agent);
    }

    pub fn get_agent(&self) -> (i32, &(dyn Agent<'a> + 'a)) {
        if self.agents.is_empty() {
            return (-1, &*self.agent_random);
        }

        let mut rng = rand::thread_rng();

        // Return random agent 10% of the time
        let random_float_0_1: f32 = rng.gen();
        if random_float_0_1 <= 0.10 {
            return (-1, &*self.agent_random);
        }

        // Get random index in the last 10 networks in self.agents
        let rand_index = if self.agents.len() >= 10 {
            rng.gen_range(self.agents.len() - 10..self.agents.len())
        } else {
            rng.gen_range(0..self.agents.len())
        };

        (rand_index as i32, &*self.agents[rand_index])
    }
}
