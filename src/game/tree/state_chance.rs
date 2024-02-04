use super::super::action::ActionConfig;
use super::state::State;
use super::state_data::StateData;
use super::state_play::StatePlay;
use super::state_terminal::StateTerminal;

pub struct StateChance<'a> {
    pub action_config: &'a ActionConfig,
    pub state_data: StateData,
    pub children: Vec<Box<dyn State<'a> + 'a>>,
}

impl<'a> State<'a> for StateChance<'a> {
    // Overrides
    fn get_state_data(&self) -> &StateData {
        return &self.state_data;
    }

    fn create_children(&mut self) -> () {
        self.children = Vec::new();

        let mut new_state_data = self.state_data.clone();

        new_state_data.player_to_move = -1;
        new_state_data.last_player = -1;
        new_state_data.min_raise = self.action_config.big_blind;
        new_state_data.street += 1;

        if new_state_data.street == 1 {
            // Preflop, first to act is after the big blind, last to act is the big blind
            if self.state_data.player_count == 2 {
                new_state_data.player_to_move = 1;
                new_state_data.last_player = 0;
            } else {
                new_state_data.player_to_move = 2;
                new_state_data.last_player = 1;
            }
        } else {
            // Postflop, first to act is the first active player, last to act is the last active player
            for i in 0..self.state_data.player_count {
                if self.is_player_in(i) && self.state_data.stacks[i as usize] > 0 {
                    new_state_data.player_to_move = i as i32;
                    break;
                }
            }

            if new_state_data.player_to_move >= 0 {
                new_state_data.last_player = self.get_last_player(new_state_data.player_to_move);
            }
        }

        if self.get_number_of_players_that_need_to_act() >= 2 && new_state_data.street <= 4 {
            let new_state = Box::new(StatePlay {
                action_config: self.action_config,
                state_data: new_state_data,
                children: Vec::new(),
                action_count: 0,
                valid_actions: Vec::new(),
            });
            self.children.push(new_state);
        } else {
            if self.get_number_of_players_that_need_to_act() == 1 {
                panic!("We just dealt new cards but only 1 player has any actions left");
            } else if self.get_number_of_all_in_players() < 2 {
                panic!("No players left to act but we dont have 2 players all-in");
            }

            if new_state_data.street <= 4 {
                let new_state = Box::new(StateChance {
                    action_config: self.action_config,
                    state_data: new_state_data,
                    children: Vec::new(),
                });
                self.children.push(new_state);
            } else {
                let new_state = Box::new(StateTerminal {
                    state_data: new_state_data,
                });
                self.children.push(new_state);
            }
        }
    }

    fn get_reward(&self, _traverser: u32) -> f32 {
        panic!("Not implemented");
    }
}
