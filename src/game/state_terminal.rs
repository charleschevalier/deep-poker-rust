use std::vec;

use super::action::Action;
use super::state::{State, StateType};
use super::state_data::StateData;
use poker::{Eval, Evaluator};

pub struct StateTerminal {
    pub state_data: StateData,
    rewards_generated: bool,
}

impl<'a> State<'a> for StateTerminal {
    fn get_type(&self) -> StateType {
        return StateType::Terminal;
    }

    fn get_child(&mut self, _index: usize) -> &mut Box<dyn State<'a> + 'a> {
        panic!("Not implemented");
    }

    fn get_child_count(&self) -> usize {
        panic!("Not implemented");
    }

    fn create_children(&mut self) -> () {
        panic!("Not implemented");
    }

    fn get_valid_actions_count(&mut self) -> i32 {
        panic!("Not implemented");
    }

    fn get_valid_actions(&mut self) -> &Vec<Action> {
        panic!("Not implemented");
    }

    // Overrides
    fn get_state_data(&self) -> &StateData {
        return &self.state_data;
    }

    fn get_reward(&mut self, _traverser: u32) -> f32 {
        if !self.rewards_generated {
            self.create_rewards();
            self.rewards_generated = true;
        }

        return self.state_data.rewards[_traverser as usize];
    }
}

impl<'a> StateTerminal {
    pub fn new(state_data: StateData) -> StateTerminal {
        StateTerminal {
            state_data: state_data,
            rewards_generated: false,
        }
    }
    fn create_rewards(&mut self) -> &Vec<f32> {
        self.state_data
            .rewards
            .resize(self.state_data.player_count as usize, 0.0);

        // Remove bets as they are considered as lost
        for i in 0..self.state_data.player_count {
            self.state_data.rewards[i as usize] -= self.state_data.bets[i as usize] as f32;
        }

        self.state_data.players_in_hand = 0;

        for i in 0..self.state_data.player_count {
            if self.is_player_in(i) {
                self.state_data.players_in_hand += 1;
            }
        }

        let mut sum_bets: u32 = 0;
        for i in 0..self.state_data.player_count {
            sum_bets += self.state_data.bets[i as usize];
        }

        // Only one player left, he takes the pot
        if self.state_data.players_in_hand == 1 {
            for i in 0..self.state_data.player_count {
                if self.is_player_in(i) {
                    self.state_data.rewards[i as usize] += sum_bets as f32;
                }
            }
        } else {
            // Create a hand evaluator
            let eval = Evaluator::new();

            // Evaluate hands
            let mut evals: Vec<Eval> = vec![Eval::WORST; self.state_data.player_count as usize];
            for i in 0..self.state_data.player_count {
                if self.is_player_in(i) {
                    let mut hand = self.state_data.hands[i as usize].clone();
                    hand.append(&mut self.state_data.board.clone());
                    evals[i as usize] = eval.evaluate(hand).expect("Couldn't evaluate hand!");
                }
            }

            // Get best hand
            let mut best_hand = evals[0];
            for i in 1..self.state_data.player_count {
                if self.is_player_in(i) {
                    if evals[i as usize].is_better_than(best_hand) {
                        best_hand = evals[i as usize];
                    }
                }
            }

            // Get players with the best hand (there could be a draw)
            let mut indices_with_best_hand: Vec<u32> = Vec::new();
            for i in 0..self.state_data.player_count {
                if self.is_player_in(i) {
                    if evals[i as usize].is_equal_to(best_hand) {
                        indices_with_best_hand.push(i);
                    }
                }
            }

            // Calculate the rewards
            for i in &indices_with_best_hand {
                self.state_data.rewards[*i as usize] +=
                    (sum_bets as f32) / (indices_with_best_hand.len() as f32);
            }
        }

        return &self.state_data.rewards;
    }
}
