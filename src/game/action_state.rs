use super::action::Action;

#[derive(Clone, Debug)]
pub struct ActionState {
    pub player_to_move: u32,
    pub reward: f32,
    pub valid_actions_mask: Vec<bool>,
    pub action_taken_index: usize,
    pub action_taken: Action,
    pub is_terminal: bool,
    pub street: u8,
}
