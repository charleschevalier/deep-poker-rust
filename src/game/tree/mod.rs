use rand::Rng;

use self::state::{State, StateType};
use self::state_chance::StateChance;
use self::state_data::StateData;

use super::action::ActionConfig;

mod state;
mod state_chance;
mod state_data;
mod state_play;
mod state_terminal;

pub struct Tree<'a> {
    player_cnt: u32,
    action_config: &'a ActionConfig,
    root: Box<dyn State<'a> + 'a>,
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
        let state = &mut self.root;
        Tree::traverse_state(traverser, state);
    }

    fn traverse_state(traverser: u32, state: &mut Box<dyn State<'a> + 'a>) -> () {
        if matches!(state.get_type(), StateType::Terminal) {
            // Return reward here
            println!("Reward: {}", state.get_reward(traverser));
            return;
        } else if !state.is_player_in_hand(traverser) {
            println!("Reached fold node");
            // Return the negative of his bet
        } else if matches!(state.get_type(), StateType::Chance) {
            // Create children
            state.create_children();

            // Traverse first child
            println!("Traverse chance node");
            Tree::traverse_state(traverser, state.get_child(0));
        } else if state.is_player_turn(traverser as i32) {
            // Traverse all children
            state.create_children();

            for i in 0..state.get_child_count() {
                println!("Traverse play node");
                Tree::traverse_state(traverser, state.get_child(i));
            }
        } else {
            // Traverse next player
            // TODO: Get single state from strategy
            // Get rand, random number between 0 and state.get_child_count()
            state.create_children();
            let mut rng = rand::thread_rng();
            let index: usize = rng.gen_range(0..state.get_child_count());
            println!(
                "Traverse random node, node count: {}",
                state.get_child_count()
            );
            Tree::traverse_state(traverser, state.get_child(index));
        }
    }
}
