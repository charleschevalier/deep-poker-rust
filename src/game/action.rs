#[derive(Clone)]
pub enum ActionType {
    None,
    Fold,
    Call,
    Raise,
    AllIn,
}

#[derive(Clone)]
pub struct Action {
    pub action_type: ActionType,
    pub raise_index: i8,
}

impl Action {
    pub fn new(action_type: ActionType, raise_index: i8) -> Action {
        Action {
            action_type,
            raise_index,
        }
    }
}

pub struct ActionConfig {
    pub commited_to_pot_percentage: u8,
    pub buy_in: u32,
    pub big_blind: u32,
    pub preflop_raise_sizes: Vec<f32>,
    pub postflop_raise_sizes: Vec<f32>,
}

impl ActionConfig {
    pub fn new(buy_in: u32, big_blind: u32) -> ActionConfig {
        ActionConfig {
            commited_to_pot_percentage: 15,
            buy_in: buy_in,
            big_blind: big_blind,
            preflop_raise_sizes: Vec::new(),
            postflop_raise_sizes: Vec::new(),
        }
    }
}
