use rand::Rng;

use super::action::ActionConfig;
use super::action_state::ActionState;
use super::hand_state::HandState;
use super::state::{State, StateType};
use super::state_chance::StateChance;
use super::state_data::StateData;
use crate::model::poker_network::PokerNetwork;

pub struct Tree<'a> {
    player_cnt: u32,
    action_config: &'a ActionConfig,
    root: Option<Box<dyn State<'a> + 'a>>,
    pub hand_state: Option<HandState>,
}

impl<'a> Tree<'a> {
    pub fn new(player_cnt: u32, action_config: &ActionConfig) -> Tree {
        Tree {
            player_cnt,
            action_config,
            root: None,
            hand_state: None,
        }
    }

    fn reset(&mut self, traverser: u32) {
        // Shuffle the deck
        self.root = Some(Box::new(StateChance::new(
            self.action_config,
            StateData::new(self.player_cnt, self.action_config.buy_in),
        )));

        self.hand_state = Some(HandState {
            traverser,
            hand: self.root.as_ref().unwrap().get_state_data().hands[traverser as usize].clone(),
            board: self.root.as_ref().unwrap().get_state_data().board.clone(),
            action_states: Vec::new(),
        });
    }

    pub fn traverse(&mut self, traverser: u32, networks: &Vec<&PokerNetwork>) {
        self.reset(traverser);
        Tree::traverse_state(
            traverser,
            &mut self.root,
            self.hand_state.as_mut().unwrap(),
            networks,
        );
        println!(
            "Action states length: {}",
            self.hand_state.as_ref().unwrap().action_states.len()
        );
    }

    fn traverse_state(
        traverser: u32,
        state_option: &mut Option<Box<dyn State<'a> + 'a>>,
        hand_state: &mut HandState,
        networks: &Vec<&PokerNetwork>,
    ) {
        // If state is None, panic
        if state_option.is_none() {
            panic!("State is None");
        }

        let state = state_option.as_mut().unwrap();

        if matches!(state.get_type(), StateType::Terminal) {
            // Use reward from terminal state
            let last_state = hand_state.action_states.last_mut().unwrap();
            last_state.reward = state.get_reward(traverser);
            last_state.is_terminal = true;
        } else if !state.is_player_in_hand(traverser) {
            // Use the negative of his bet as reward
            let last_state = hand_state.action_states.last_mut().unwrap();
            last_state.reward = -(state.get_state_data().bets[traverser as usize] as f32);
            last_state.is_terminal = true;
        } else if matches!(state.get_type(), StateType::Chance) {
            // Create children
            state.create_children();

            // Traverse first child
            return Tree::traverse_state(traverser, state.get_child(0), hand_state, networks);
        } else if state.is_player_turn(traverser as i32) {
            // Traverse all children
            state.create_children();

            // let siamese_output = siamese_networks[traverser as usize].forward(
            //     &state.get_state_data().get_input(traverser),
            //     &state.get_valid_actions_mask(),
            // );
            // let actor_output = actor_networks[traverser as usize].forward(
            //     &state.get_state_data().get_input(traverser),
            //     &state.get_valid_actions_mask(),
            // );

            let action_index = 0;

            hand_state.action_states.push(ActionState {
                history: state.get_state_data().history.clone(),
                player_to_move: traverser,
                reward: 0.0,
                valid_actions_mask: state.get_valid_actions_mask(),
                action_taken: action_index,
                is_terminal: false,
            });

            Tree::traverse_state(
                traverser,
                state.get_child(action_index),
                hand_state,
                networks,
            );
        } else {
            // Traverse next player
            // TODO: Get single state from strategy
            // Get rand, random number between 0 and state.get_child_count()

            state.create_children();
            let mut rng = rand::thread_rng();
            let action_index: usize = rng.gen_range(0..state.get_child_count());

            Tree::traverse_state(
                traverser,
                state.get_child(action_index),
                hand_state,
                networks,
            );
        }
    }
}
