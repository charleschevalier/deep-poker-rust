use crate::game::hand_state::HandState;

pub trait AgentClone {
    fn clone_box(&self) -> Box<dyn Agent>;
}

pub trait Agent: AgentClone + Sync + Send {
    fn choose_action(
        &self,
        hand_state: &HandState,
        valid_actions_mask: &[bool],
        street: u8,
        action_config: &crate::game::action::ActionConfig,
        device: &candle_core::Device,
        no_invalid: bool,
    ) -> Result<usize, Box<dyn std::error::Error>>;
}

impl Clone for Box<dyn Agent> {
    fn clone(&self) -> Box<dyn Agent> {
        self.clone_box()
    }
}

pub mod agent_network;
pub mod agent_pool;
pub mod agent_random;
