use super::action::Action;

#[derive(Clone, Debug)]
pub struct ActionState {
    pub history: Vec<Action>,
    pub player_to_move: u32,
    pub reward: f32,
    pub valid_actions_mask: Vec<f32>,
    pub action_taken: usize,
    pub is_terminal: bool,
}
