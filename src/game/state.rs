use crate::game::action::ActionType;

use super::action::Action;
use super::state_data::StateData;

pub enum StateType {
    Play,
    Terminal,
    Chance,
}

pub trait State<'a> {
    // Ugly getters, needed for polymorphism
    fn get_state_data(&self) -> &StateData;

    fn get_player_count(&self) -> u32 {
        return self.get_state_data().player_count;
    }
    fn get_player_to_move(&self) -> i32 {
        return self.get_state_data().player_to_move;
    }
    fn is_player_in(&self, player_index: u32) -> bool {
        return self.get_state_data().is_player_in[player_index as usize];
    }
    fn get_last_actions(&self) -> &Vec<Action> {
        return &self.get_state_data().last_actions;
    }
    fn get_to_move_bet(&self) -> u32 {
        return self.get_state_data().bets[self.get_player_to_move() as usize];
    }
    fn get_to_move_stack(&self) -> u32 {
        return self.get_state_data().stacks[self.get_player_to_move() as usize];
    }

    // Functions implemented by the state
    fn get_next_player(&self, last_to_move_temp: i32) -> i32 {
        let mut i = (self.get_player_to_move() + 1) % self.get_player_count() as i32;
        let last_to_move_temp = (last_to_move_temp + 1) % self.get_player_count() as i32;
        while i != last_to_move_temp {
            if self.is_player_in(i as u32) {
                return i;
            }
            i = (i + 1) % self.get_player_count() as i32;
        }

        -1
    }

    fn get_last_player(&self, player_that_raised: i32) -> i32 {
        let mut last: i32 = -1;
        let mut i = (player_that_raised + 1) % self.get_player_count() as i32;
        while i != player_that_raised {
            if self.is_player_in(i as u32) {
                last = i;
            }
            i = (i + 1) % self.get_player_count() as i32;
        }
        last
    }

    fn get_number_of_players_that_need_to_act(&self) -> u32 {
        // does not include all-in players
        let mut count = 0;
        for i in 0..self.get_player_count() {
            if self.is_player_in(i)
                && !matches!(
                    self.get_last_actions()[i as usize].action_type,
                    ActionType::AllIn
                )
            {
                count += 1;
            }
        }
        count
    }

    fn get_active_players(&self, new_is_player_in: &[bool]) -> u32 {
        let mut count = 0;
        for i in 0..self.get_player_count() {
            if new_is_player_in[i as usize] {
                count += 1;
            }
        }
        count
    }

    fn get_number_of_all_in_players(&self) -> u32 {
        let mut count = 0;
        for i in 0..self.get_player_count() {
            if matches!(
                self.get_last_actions()[i as usize].action_type,
                ActionType::AllIn
            ) {
                count += 1;
            }
        }
        count
    }

    fn is_player_in_hand(&self, player_index: u32) -> bool {
        self.is_player_in(player_index)
    }

    fn print_actions(&self) {
        println!("---------------------------------");
        for h in self.get_state_data().history.iter() {
            println!("{:?}", h);
        }
    }

    // Functions that need to be implemented by the state
    fn create_children(&mut self);
    fn get_reward(&mut self, traverser: u32) -> f32;
    fn get_type(&self) -> StateType;
    fn get_child(&mut self, index: usize) -> &mut Option<Box<dyn State<'a> + 'a>>;
    fn get_valid_actions_mask(&self) -> Vec<bool>;
}
