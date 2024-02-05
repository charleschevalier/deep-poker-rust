use super::action::Action;
use poker::Card;

#[derive(Clone, Debug)]
pub struct ActionState {
    pub history: Vec<Action>,
    pub player_to_move: u32,
    pub hand: Vec<Card>,
    pub board: Vec<Card>,
    pub rewards: Vec<f32>,
    pub valid_actions: Vec<Action>,
}
