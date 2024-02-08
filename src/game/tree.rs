use candle_core::Tensor;
use poker::card;
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
            self.action_config,
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
        action_config: &ActionConfig,
    ) {
        // If state is None, panic
        if state_option.is_none() {
            panic!("State is None");
        }

        let state = state_option.as_mut().unwrap();

        if matches!(state.get_type(), StateType::Terminal) {
            // Use reward from terminal state. We may have no action states if every player folded
            // except the traverser in BB
            if !hand_state.action_states.is_empty() {
                let last_state = hand_state.action_states.last_mut().unwrap();
                last_state.reward = state.get_reward(traverser);
                last_state.is_terminal = true;
            }
        } else if !state.is_player_in_hand(traverser) {
            // Use the negative of his bet as reward
            let last_state = hand_state.action_states.last_mut().unwrap();
            last_state.reward = -(state.get_state_data().bets[traverser as usize] as f32);
            last_state.is_terminal = true;
        } else if matches!(state.get_type(), StateType::Chance) {
            // Create children
            state.create_children();

            // Traverse first child
            return Self::traverse_state(
                traverser,
                state.get_child(0),
                hand_state,
                networks,
                action_config,
            );
        } else {
            // Traverse next player
            // TODO: Get single state from strategy
            // Get rand, random number between 0 and state.get_child_count()

            state.create_children();

            let (card_tensor, action_tensor) = hand_state.to_input(
                state.get_state_data().street,
                action_config,
                &candle_core::Device::Cpu,
                hand_state.action_states.len(),
                state.get_valid_actions_mask(),
            );

            let (proba_tensor, _) = networks[state.get_state_data().player_to_move as usize]
                .forward(
                    &card_tensor.unsqueeze(0).unwrap(),
                    &action_tensor.unsqueeze(0).unwrap(),
                );

            let valid_actions_mask = state.get_valid_actions_mask();
            let action_index = Self::choose_action(proba_tensor, state.get_valid_actions_mask());

            if !valid_actions_mask[action_index] {
                panic!("Invalid action chosen");
            }

            if state.get_player_to_move() == traverser as i32 {
                hand_state
                    .action_states
                    .push(Self::build_action_state(state, action_index));
            }

            Self::traverse_state(
                traverser,
                state.get_child(action_index),
                hand_state,
                networks,
                action_config,
            );
        }
    }

    fn build_action_state(state: &mut Box<dyn State<'a> + 'a>, action_index: usize) -> ActionState {
        ActionState {
            player_to_move: state.get_player_to_move() as u32,
            reward: 0.0,
            valid_actions_mask: state.get_valid_actions_mask(),
            action_taken_index: action_index,
            action_taken: state
                .get_child(action_index)
                .as_ref()
                .unwrap()
                .get_state_data()
                .history
                .last()
                .unwrap()
                .clone(),
            is_terminal: false,
            street: state.get_state_data().street,
        }
    }

    fn choose_action(proba_tensor: Tensor, valid_action_mask: Vec<bool>) -> usize {
        // Apply valid action mask to tensor
        let mut probas = proba_tensor.squeeze(0).unwrap().to_vec1().unwrap();
        for i in 0..probas.len() {
            if i >= valid_action_mask.len() || !valid_action_mask[i] {
                probas[i] = 0.0;
            }
        }

        // Normalize probas
        let sum: f32 = probas.iter().sum();
        for p in &mut probas {
            *p /= sum;
        }

        // Choose action based on the probability distribution
        let mut rng = rand::thread_rng();
        let random_float_0_1: f32 = rng.gen();
        let mut sum: f32 = 0.0;
        let mut action_index: usize = 0;
        for (i, p) in probas.iter().enumerate() {
            sum += p;
            if sum > random_float_0_1 {
                action_index = i;
                break;
            }
        }
        action_index
    }
}
