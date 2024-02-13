use super::action::Action;

#[derive(Clone, Debug)]
pub struct ActionState {
    pub player_to_move: u32,
    pub reward: f32,
    pub valid_actions_mask: Vec<bool>,
    pub action_taken_index: usize,
    pub action_taken: Option<Action>,
    pub is_terminal: bool,
    pub street: u8,
    pub min_reward: f32,
    pub max_reward: f32,
    pub is_invalid: bool,
}
