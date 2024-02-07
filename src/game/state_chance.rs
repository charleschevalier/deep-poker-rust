use super::action::ActionConfig;
use super::state::{State, StateType};
use super::state_data::StateData;
use super::state_play::StatePlay;
use super::state_terminal::StateTerminal;

pub struct StateChance<'a> {
    pub action_config: &'a ActionConfig,
    pub state_data: StateData,
    pub children: Vec<Option<Box<dyn State<'a> + 'a>>>,
}

impl<'a> State<'a> for StateChance<'a> {
    fn get_type(&self) -> StateType {
        StateType::Chance
    }

    fn get_child(&mut self, index: usize) -> &mut Option<Box<dyn State<'a> + 'a>> {
        &mut self.children[index]
    }

    fn get_child_count(&self) -> usize {
        self.children.len()
    }

    fn get_reward(&mut self, _traverser: u32) -> f32 {
        panic!("Not implemented");
    }

    fn get_valid_actions_mask(&self) -> Vec<bool> {
        panic!("Not implemented");
    }

    // Overrides
    fn get_state_data(&self) -> &StateData {
        &self.state_data
    }

    fn create_children(&mut self) {
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

            // Post blinds
            let sb = self.action_config.big_blind / 2;
            let bb = self.action_config.big_blind;
            if self.state_data.player_count == 2 {
                new_state_data.bets[1] = sb;
                new_state_data.bets[0] = bb;
                new_state_data.stacks[1] -= sb;
                new_state_data.stacks[0] -= bb;
            } else {
                new_state_data.bets[0] = sb;
                new_state_data.bets[1] = bb;
                new_state_data.stacks[0] -= sb;
                new_state_data.stacks[1] -= bb;
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
            self.children.push(Some(Box::new(StatePlay::new(
                self.action_config,
                new_state_data,
            ))));
        } else {
            if self.get_number_of_players_that_need_to_act() == 1 {
                self.print_actions();
                panic!("We just dealt new cards but only 1 player has any actions left");
            } else if self.get_number_of_all_in_players() < 2 {
                self.print_actions();
                panic!("No players left to act but we dont have 2 players all-in");
            }

            if new_state_data.street <= 4 {
                self.children.push(Some(Box::new(StateChance::new(
                    self.action_config,
                    new_state_data,
                ))));
            } else {
                self.children
                    .push(Some(Box::new(StateTerminal::new(new_state_data))));
            }
        }
    }
}

impl<'a> StateChance<'a> {
    pub fn new(action_config: &'a ActionConfig, state_data: StateData) -> StateChance {
        StateChance {
            action_config,
            state_data,
            children: Vec::new(),
        }
    }
}
