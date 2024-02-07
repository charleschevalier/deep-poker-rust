use super::action_state::ActionState;
use poker::Card;
#[derive(Clone, Debug)]
pub struct HandState {
    pub traverser: u32,
    pub hand: Vec<Card>,
    pub board: Vec<Card>,
    pub action_states: Vec<ActionState>,
}
