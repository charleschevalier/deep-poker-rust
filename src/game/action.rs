pub enum ActionType {
    Fold,
    Call,
    Raise,
    AllIn,
}

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
    pub length: usize,
    pub preflop_raise_sizes: Vec<f32>,
    pub postflop_raise_sizes: Vec<f32>,
}

impl ActionConfig {
    pub fn new() -> ActionConfig {
        ActionConfig {
            length: 0,
            preflop_raise_sizes: Vec::new(),
            postflop_raise_sizes: Vec::new(),
        }
    }
}
