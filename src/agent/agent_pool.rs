use super::Agent;

struct AgentPool<'a> {
    agents: Vec<&'a Box<dyn Agent>>,
}

impl<'a> AgentPool<'a> {
    pub fn new() -> AgentPool<'a> {
        AgentPool { agents: Vec::new() }
    }

    pub fn add_agent(&mut self, agent: &'a Box<dyn Agent>) {
        self.agents.push(agent);
    }

    pub fn get_agent(&self, index: usize) -> &Box<dyn Agent> {
        &self.agents[index]
    }
}
