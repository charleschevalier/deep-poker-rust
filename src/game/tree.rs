use candle_core::Tensor;

use super::action::ActionConfig;
use super::action_state::ActionState;
use super::hand_state::HandState;
use super::state::{State, StateType};
use super::state_chance::StateChance;
use super::state_data::StateData;
use crate::agent::{self, Agent};
use crate::model::poker_network::PokerNetwork;
use colored::*;

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

    pub fn traverse(
        &mut self,
        traverser: u32,
        agents: &Vec<&Box<dyn Agent>>,
        device: &candle_core::Device,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.reset(traverser);

        Tree::traverse_state(
            traverser,
            &mut self.root,
            self.hand_state.as_mut().unwrap(),
            agents,
            self.action_config,
            device,
        )?;
        // println!(
        //     "Action states length: {}",
        //     self.hand_state.as_ref().unwrap().action_states.len()
        // );

        Ok(())
    }

    fn traverse_state(
        traverser: u32,
        state_option: &mut Option<Box<dyn State<'a> + 'a>>,
        hand_state: &mut HandState,
        agents: &Vec<&Box<dyn Agent>>,
        action_config: &ActionConfig,
        device: &candle_core::Device,
    ) -> Result<(), Box<dyn std::error::Error>> {
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
                // let reward_ratio =
                //     (action_config.player_count - 1) as f32 * action_config.buy_in as f32;
                // last_state.reward = state.get_reward(traverser) / reward_ratio;
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
                agents,
                action_config,
                device,
            );
        } else {
            // Traverse next player
            // TODO: Get single state from strategy
            // Get rand, random number between 0 and state.get_child_count()

            state.create_children();

            let (card_tensor, action_tensor) = hand_state.to_input(
                state.get_state_data().street,
                action_config,
                device,
                hand_state.action_states.len(),
                &state.get_valid_actions_mask(),
            )?;

            let valid_actions_mask = state.get_valid_actions_mask();

            let action_index = agents[state.get_player_to_move() as usize].choose_action(
                &hand_state,
                &valid_actions_mask,
                state.get_state_data().street,
                action_config,
                device,
            )?;

            if !valid_actions_mask[action_index] {
                panic!("Invalid action chosen");
            }

            if state.get_player_to_move() == traverser as i32 {
                hand_state.action_states.push(Self::build_action_state(
                    state,
                    action_index,
                    action_config,
                ));
            }

            Self::traverse_state(
                traverser,
                state.get_child(action_index),
                hand_state,
                agents,
                action_config,
                device,
            )?;
        }

        Ok(())
    }

    fn build_action_state(
        state: &mut Box<dyn State<'a> + 'a>,
        action_index: usize,
        action_config: &ActionConfig,
    ) -> ActionState {
        let mut max_reward: u32 = 0;
        for i in 0..state.get_player_count() {
            if i as i32 != state.get_player_to_move() {
                max_reward += state.get_state_data().bets[i as usize];
            }
        }

        // let reward_ratio = (action_config.player_count - 1) as f32 * action_config.buy_in as f32;

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
            // min_reward: -(state.get_state_data().bets[state.get_player_to_move() as usize] as f32)
            //     / reward_ratio,
            // max_reward: max_reward as f32 / reward_ratio,
            min_reward: -(state.get_state_data().bets[state.get_player_to_move() as usize] as f32),
            max_reward: max_reward as f32,
        }
    }

    pub fn print_first_actions(
        &self,
        network: &PokerNetwork,
        device: &candle_core::Device,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let action_count = self.action_config.postflop_raise_sizes.len() + 3;
        let mut result: Vec<Vec<Vec<f32>>> = vec![vec![vec![0.0; 13]; 13]; action_count];
        let mut count: Vec<Vec<Vec<u32>>> = vec![vec![vec![0; 13]; 13]; action_count];

        // Iterate through card combinations
        for i in 0..52 {
            for j in i + 1..52 {
                let rank1: usize = i / 4;
                let suit1: usize = i % 4;
                let rank2: usize = j / 4;
                let suit2: usize = j % 4;

                let mut card_vecs: Vec<Vec<Vec<f32>>> = vec![vec![vec![0.0; 13]; 4]; 5];

                // Set hand cards
                card_vecs[0][suit1][rank1] = 1.0;
                card_vecs[0][suit2][rank2] = 1.0;
                card_vecs[4][suit1][rank1] = 1.0;
                card_vecs[4][suit2][rank2] = 1.0;

                // Create action tensor
                // Shape is (street_cnt * max_actions_per_street) x (player_count + 2 for sum and legal) x max_number_of_actions
                let action_vecs: Vec<Vec<Vec<f32>>> =
                    vec![
                        vec![
                            vec![0.0; 3 + self.action_config.postflop_raise_sizes.len()];
                            self.action_config.player_count as usize + 2
                        ];
                        4 * self.action_config.max_actions_per_street as usize
                    ];

                let card_tensor = Tensor::new(card_vecs, device)?.unsqueeze(0)?;
                let action_tensor = Tensor::new(action_vecs, device)?.unsqueeze(0)?;

                let (proba_tensor, _) = network.forward(&card_tensor, &action_tensor)?;

                let is_suited = suit1 == suit2;

                let min_rank = if rank1 < rank2 { rank1 } else { rank2 };
                let max_rank = if rank1 > rank2 { rank1 } else { rank2 };

                let mut probs: Vec<f32> = proba_tensor.squeeze(0)?.to_vec1()?;

                // Normalize probs
                let sum: f32 = probs.iter().sum();
                for p in &mut probs {
                    *p /= sum;
                }

                for action_index in 0..action_count {
                    if is_suited {
                        result[action_index][min_rank][max_rank] += probs[action_index];
                        count[action_index][min_rank][max_rank] += 1;
                    } else {
                        result[action_index][max_rank][min_rank] += probs[action_index];
                        count[action_index][max_rank][min_rank] += 1;
                    }
                }
            }
        }

        for action_index in 0..action_count {
            println!();
            println!("Action index: {}", action_index);
            println!();
            println!("     2    3    4    5    6    7    8    9    T    J    Q    K    A (suited)");
            for i in 0..13 {
                let real_rank = i + 2;
                let real_rank_str = real_rank.to_string();
                print!(
                    "{} ",
                    if real_rank == 14 {
                        "A"
                    } else if real_rank == 13 {
                        "K"
                    } else if real_rank == 12 {
                        "Q"
                    } else if real_rank == 11 {
                        "J"
                    } else if real_rank == 10 {
                        "T"
                    } else {
                        real_rank_str.as_str()
                    }
                );

                for j in 0..13 {
                    let value = if count[action_index][i][j] > 0 {
                        result[action_index][i][j] / count[action_index][i][j] as f32
                    } else {
                        0.0
                    };

                    let c = if value > 0.75 {
                        "green"
                    } else if value > 0.5 {
                        "yellow"
                    } else if value > 0.25 {
                        "magenta"
                    } else {
                        "red"
                    };

                    print!("{} ", format!("{:.2}", value).color(c),);
                }

                println!();
            }
        }

        Ok(())
    }

    // pub const fn card_from_i32(val: i32) -> Card {
    //     let rank = val / 4;
    //     let suit = val % 4;

    //     let rank_f = match rank {
    //         0 => Rank::Two,
    //         1 => Rank::Three,
    //         2 => Rank::Four,
    //         3 => Rank::Five,
    //         4 => Rank::Six,
    //         5 => Rank::Seven,
    //         6 => Rank::Eight,
    //         7 => Rank::Nine,
    //         8 => Rank::Ten,
    //         9 => Rank::Jack,
    //         10 => Rank::Queen,
    //         11 => Rank::King,
    //         _ => Rank::Ace,
    //     };

    //     let suit_f = match suit {
    //         0 => Suit::Spades,
    //         1 => Suit::Hearts,
    //         2 => Suit::Diamonds,
    //         _ => Suit::Clubs,
    //     };

    //     Card::new(rank_f, suit_f)
    // }
}
