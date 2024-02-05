use rand::Rng;

use crate::model::fake_model::FakeModel;

use super::action::{self, ActionConfig};
use super::action_state::{self, ActionState};
use super::state::{State, StateType};
use super::state_chance::StateChance;
use super::state_data::StateData;

pub struct Tree<'a> {
    player_cnt: u32,
    action_config: &'a ActionConfig,
    root: Box<dyn State<'a> + 'a>,
    action_states: Vec<ActionState>,
}

impl<'a> Tree<'a> {
    pub fn new(player_cnt: u32, action_config: &ActionConfig) -> Tree {
        Tree {
            player_cnt: player_cnt,
            action_config: action_config,
            root: Box::new(StateChance::new(
                action_config,
                StateData::new(player_cnt, action_config.buy_in),
            )),
            action_states: Vec::new(),
        }
    }

    fn reset(&mut self) -> () {
        // Shuffle the deck
        self.root = Box::new(StateChance::new(
            self.action_config,
            StateData::new(self.player_cnt, self.action_config.buy_in),
        ));
    }

    pub fn traverse(&mut self, traverser: u32) -> () {
        self.reset();
        Tree::traverse_state(traverser, &mut self.root, &mut self.action_states);
        println!("Action states length: {}", self.action_states.len());
    }

    fn traverse_state(
        traverser: u32,
        state: &mut Box<dyn State<'a> + 'a>,
        action_states: &mut Vec<ActionState>,
    ) -> f32 {
        if matches!(state.get_type(), StateType::Terminal) {
            // Return reward here
            return state.get_reward(traverser);
        } else if !state.is_player_in_hand(traverser) {
            // Return the negative of his bet
            return -(state.get_state_data().bets[traverser as usize] as f32);
        } else if matches!(state.get_type(), StateType::Chance) {
            // Create children
            state.create_children();

            // Traverse first child
            return Tree::traverse_state(traverser, state.get_child(0), action_states);
        } else if state.is_player_turn(traverser as i32) {
            // Traverse all children
            state.create_children();

            let mut reward = 0.0;
            let mut action_state = ActionState {
                history: state.get_state_data().history.clone(),
                hand: state.get_state_data().hands[traverser as usize].clone(),
                board: state.get_state_data().board.clone(),
                player_to_move: traverser,
                rewards: vec![0.0; state.get_child_count() as usize],
                valid_actions: state.get_valid_actions().clone(),
            };
            let probs = FakeModel::get_probabilities(state.get_child_count() as u32);
            for i in 0..state.get_child_count() {
                reward +=
                    probs[i] * Tree::traverse_state(traverser, state.get_child(i), action_states);
                action_state.rewards[i] = reward;
            }

            action_states.push(action_state);
            return reward;
        } else {
            // Traverse next player
            // TODO: Get single state from strategy
            // Get rand, random number between 0 and state.get_child_count()
            state.create_children();
            let mut rng = rand::thread_rng();
            let index: usize = rng.gen_range(0..state.get_child_count());
            return Tree::traverse_state(traverser, state.get_child(index), action_states);
        }
    }
}
