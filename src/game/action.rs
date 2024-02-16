#[derive(Clone, Debug)]
pub enum ActionType {
    None,
    Fold,
    Call,
    Raise,
    AllIn,
}

#[derive(Clone, Debug)]
pub struct Action {
    pub action_type: ActionType,
    pub raise_index: i8,
    pub player_index: i32,
    pub street: u8,
}

impl Action {
    pub fn to_print_string(&self) -> String {
        match self.action_type {
            ActionType::None => String::from("None"),
            ActionType::Fold => String::from("Fold"),
            ActionType::Call => String::from("Call"),
            ActionType::Raise => format!("{}{}", "Raise", self.raise_index),
            ActionType::AllIn => String::from("AllIn"),
        }
    }
}

pub struct ActionConfig {
    pub player_count: u32,
    pub commited_to_pot_percentage: u8,
    pub buy_in: u32,
    pub big_blind: u32,
    pub preflop_raise_sizes: Vec<f32>,
    pub postflop_raise_sizes: Vec<f32>,
    pub max_actions_per_street: u8,
}

impl ActionConfig {
    pub fn new(
        player_count: u32,
        buy_in: u32,
        big_blind: u32,
        max_actions_per_street: u8,
    ) -> ActionConfig {
        ActionConfig {
            player_count,
            commited_to_pot_percentage: 15,
            buy_in,
            big_blind,
            preflop_raise_sizes: Vec::new(),
            postflop_raise_sizes: Vec::new(),
            max_actions_per_street,
        }
    }
}
