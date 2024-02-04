use self::state_chance::StateChance;
use self::state_data::StateData;

use super::action::Action;
use super::action::ActionConfig;

mod state;
mod state_chance;
mod state_data;
mod state_play;
mod state_terminal;

pub struct Tree<'a> {
    player_cnt: u32,
    action_config: &'a ActionConfig,
    history: Vec<Action>,
    root: StateChance<'a>,
}

impl<'a> Tree<'a> {
    pub fn new(player_cnt: u32, action_config: &ActionConfig) -> Tree {
        let new_state = StateChance {
            action_config: action_config,
            state_data: StateData::new(player_cnt, action_config.buy_in),
            children: Vec::new(),
        };
        Tree {
            player_cnt: player_cnt,
            action_config: action_config,
            history: Vec::new(),
            root: new_state,
        }
    }

    fn reset(&mut self) -> () {
        // Shuffle the deck
        self.root = StateChance {
            action_config: self.action_config,
            state_data: StateData::new(self.player_cnt, self.action_config.buy_in),
            children: Vec::new(),
        };
    }

    pub fn play_hand(&mut self) -> () {
        self.reset();
    }
}
