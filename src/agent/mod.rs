use crate::game::hand_state::HandState;

pub trait Agent<'a> {
    fn choose_action(
        &self,
        hand_state: &HandState,
        valid_actions_mask: &[bool],
        street: u8,
        action_config: &crate::game::action::ActionConfig,
        device: &candle_core::Device,
    ) -> Result<usize, Box<dyn std::error::Error>>;
}

pub mod agent_network;
pub mod agent_pool;
pub mod agent_random;
