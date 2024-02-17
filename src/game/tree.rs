use candle_core::Tensor;

use super::action::ActionConfig;
use super::action_state::ActionState;
use super::hand_state::HandState;
use super::state::{State, StateType};
use super::state_chance::StateChance;
use super::state_data::StateData;
use crate::agent::agent_network::AgentNetwork;
use crate::agent::Agent;
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
        network: &PokerNetwork,
        agents: &Vec<Option<&dyn Agent>>,
        device: &candle_core::Device,
        no_invalid_for_traverser: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.reset(traverser);

        Tree::traverse_state(
            &mut self.root,
            self.hand_state.as_mut().unwrap(),
            network,
            agents,
            self.action_config,
            device,
            no_invalid_for_traverser,
        )?;
        // println!(
        //     "Action states length: {}",
        //     self.hand_state.as_ref().unwrap().action_states.len()
        // );

        Ok(())
    }

    fn traverse_state(
        state_option: &mut Option<Box<dyn State<'a> + 'a>>,
        hand_state: &mut HandState,
        network: &PokerNetwork,
        agents: &Vec<Option<&dyn Agent>>,
        action_config: &ActionConfig,
        device: &candle_core::Device,
        no_invalid_for_traverser: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // If state is None, panic
        if state_option.is_none() {
            panic!("State is None");
        }

        let state = state_option.as_mut().unwrap();
        let traverser = hand_state.traverser;

        if matches!(state.get_type(), StateType::Terminal) {
            // Use reward from terminal state. We may have no action states if every player folded
            // except the traverser in BB
            Self::update_last_traverser_reward(hand_state, state.get_reward(traverser), false);
        } else if !state.is_player_in_hand(traverser) {
            // Use the negative of his bet as reward
            Self::update_last_traverser_reward(
                hand_state,
                -(state.get_state_data().bets[traverser as usize] as f32),
                false,
            );
        } else if matches!(state.get_type(), StateType::Chance) {
            // Create children
            state.create_children();

            // Traverse first child
            return Self::traverse_state(
                state.get_child(0),
                hand_state,
                network,
                agents,
                action_config,
                device,
                no_invalid_for_traverser,
            );
        } else {
            // Traverse next player
            // TODO: Get single state from strategy
            // Get rand, random number between 0 and state.get_child_count()

            state.create_children();

            let valid_actions_mask = state.get_valid_actions_mask();

            let action_index = if traverser == state.get_player_to_move() as u32 {
                let (card_tensor, action_tensor) = hand_state.to_input(
                    state.get_state_data().street,
                    action_config,
                    device,
                    hand_state.action_states.len(),
                )?;

                let proba_tensor = network.forward_embedding_actor(
                    &card_tensor.unsqueeze(0)?,
                    &action_tensor.unsqueeze(0)?,
                )?;

                let valid_actions_mask = state.get_valid_actions_mask();
                AgentNetwork::choose_action_from_net(
                    &proba_tensor,
                    &valid_actions_mask,
                    no_invalid_for_traverser,
                )?
            } else {
                agents[state.get_player_to_move() as usize]
                    .as_ref()
                    .unwrap()
                    .choose_action(
                        hand_state,
                        &valid_actions_mask,
                        state.get_state_data().street,
                        action_config,
                        device,
                    )?
            };

            if action_index > valid_actions_mask.len() || !valid_actions_mask[action_index] {
                if state.get_player_to_move() == traverser as i32 {
                    hand_state.action_states.push(Self::build_action_state(
                        traverser,
                        state,
                        action_index,
                        true,
                    ));

                    Self::update_last_traverser_reward(
                        hand_state,
                        -((action_config.player_count - 1) as f32 * action_config.buy_in as f32),
                        true,
                    );
                    return Ok(());
                } else {
                    panic!("Invalid action index");
                }
            }

            hand_state.action_states.push(Self::build_action_state(
                traverser,
                state,
                action_index,
                false,
            ));

            Self::traverse_state(
                state.get_child(action_index),
                hand_state,
                network,
                agents,
                action_config,
                device,
                no_invalid_for_traverser,
            )?;
        }

        Ok(())
    }

    fn build_action_state(
        traverser: u32,
        state: &mut Box<dyn State<'a> + 'a>,
        action_index: usize,
        is_invalid: bool,
    ) -> ActionState {
        let mut max_reward: u32 = 0;
        for i in 0..state.get_player_count() {
            if i != traverser {
                max_reward += state.get_state_data().bets[i as usize];
            }
        }

        ActionState {
            player_to_move: state.get_player_to_move() as u32,
            reward: 0.0,
            valid_actions_mask: state.get_valid_actions_mask(),
            action_taken_index: action_index,
            action_taken: if !state.get_child(action_index).is_none() {
                Some(
                    state
                        .get_child(action_index)
                        .as_ref()
                        .unwrap()
                        .get_state_data()
                        .history
                        .last()
                        .unwrap()
                        .clone(),
                )
            } else {
                None
            },
            is_terminal: false,
            street: state.get_state_data().street,
            min_reward: -(state.get_state_data().bets[traverser as usize] as f32),
            max_reward: max_reward as f32,
            is_invalid,
        }
    }

    pub fn print_first_actions(
        &self,
        network: &PokerNetwork,
        device: &candle_core::Device,
        no_invalid_for_traverser: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let action_count = self.action_config.postflop_raise_sizes.len() + 3;
        let mut result: Vec<Vec<Vec<f32>>> = vec![vec![vec![0.0; 13]; 13]; action_count];
        let mut count: Vec<Vec<Vec<u32>>> = vec![vec![vec![0; 13]; 13]; action_count];
        let mut valid_actions_mask: Vec<bool> = self
            .action_config
            .preflop_raise_sizes
            .iter()
            .map(|&x| x > 0.0)
            .collect();

        valid_actions_mask.insert(0, true);
        valid_actions_mask.insert(0, true);
        valid_actions_mask.push(true);

        // Iterate through card combinations
        for i in 0..52 {
            for j in i + 1..52 {
                let rank1: usize = i / 4;
                let suit1: usize = i % 4;
                let rank2: usize = j / 4;
                let suit2: usize = j % 4;

                let mut card_vecs: Vec<Vec<Vec<f32>>> = vec![vec![vec![0.0; 13]; 4]; 6];

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

                let proba_tensor = network.forward_embedding_actor(&card_tensor, &action_tensor)?;

                let is_suited = suit1 == suit2;

                let min_rank = if rank1 < rank2 { rank1 } else { rank2 };
                let max_rank = if rank1 > rank2 { rank1 } else { rank2 };

                let mut probas: Vec<f32> = proba_tensor.squeeze(0)?.to_vec1()?;

                // Normalize probs
                for i in 0..probas.len() {
                    if no_invalid_for_traverser
                        && (i >= valid_actions_mask.len() || !valid_actions_mask[i])
                    {
                        probas[i] = 0.0;
                    }
                }

                // Normalize probas
                let sum: f32 = probas.iter().sum();
                if sum > 0.0 {
                    for p in &mut probas {
                        *p /= sum;
                    }
                } else {
                    // Count positive values in valid_actions_mask
                    let true_count = if no_invalid_for_traverser {
                        valid_actions_mask.iter().filter(|&&x| x).count()
                    } else {
                        probas.len()
                    };
                    for (i, p) in probas.iter_mut().enumerate() {
                        if i < valid_actions_mask.len() && valid_actions_mask[i] {
                            *p = 1.0 / (true_count as f32);
                        }
                    }
                }

                for action_index in 0..action_count {
                    if is_suited {
                        result[action_index][max_rank][min_rank] += probas[action_index];
                        count[action_index][max_rank][min_rank] += 1;
                    } else {
                        result[action_index][min_rank][max_rank] += probas[action_index];
                        count[action_index][min_rank][max_rank] += 1;
                    }
                }
            }
        }

        for action_index in 0..action_count {
            if no_invalid_for_traverser && !valid_actions_mask[action_index] {
                continue;
            }
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

    fn update_last_traverser_reward(hand_state: &mut HandState, reward: f32, set_min: bool) {
        if let Some(b) = hand_state
            .action_states
            .iter_mut()
            .rev()
            .find(|b| b.player_to_move == hand_state.traverser)
        {
            b.reward = reward;
            b.is_terminal = true;
            if set_min {
                b.min_reward = reward;
            }
        }
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

    pub fn _play_one_hand(
        &mut self,
        network: &PokerNetwork,
        device: &candle_core::Device,
        no_invalid_for_traverser: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.reset(0);

        let mut gs = self.root.as_mut().unwrap();
        let mut first = true;

        while !matches!(gs.get_type(), StateType::Terminal) {
            if matches!(gs.get_type(), StateType::Chance) {
                gs.create_children();
                gs = gs.get_child(0).as_mut().unwrap();

                if first {
                    println!();
                    print!("Player Cards: ");
                    for i in 0..self.player_cnt {
                        let player_cards = gs.get_state_data().hands[i as usize].clone();
                        print!(
                            "{}{} ",
                            player_cards[0].rank_suit_string(),
                            player_cards[1].rank_suit_string()
                        );
                    }
                    println!();
                    first = false;
                } else if gs.get_state_data().street >= 2 && !gs.get_state_data().board.is_empty() {
                    print!("Table Cards: ");
                    let board_cnt = match gs.get_state_data().street {
                        2 => 3,
                        3 => 4,
                        4 => 5,
                        _ => 0,
                    };

                    for i in 0..board_cnt {
                        print!(
                            "{} ",
                            gs.get_state_data().board[i as usize].rank_suit_string()
                        );
                    }
                    println!();
                }
            } else {
                let p_to_move = gs.get_player_to_move();
                print!("Player {}'s turn: ", p_to_move);

                let (card_tensor, action_tensor) = self.hand_state.as_ref().unwrap().to_input(
                    gs.get_state_data().street,
                    self.action_config,
                    device,
                    self.hand_state.as_ref().unwrap().action_states.len(),
                )?;

                let proba_tensor = network.forward_embedding_actor(
                    &card_tensor.unsqueeze(0)?,
                    &action_tensor.unsqueeze(0)?,
                )?;

                gs.create_children();

                let valid_actions_mask = gs.get_valid_actions_mask();
                let action_index = AgentNetwork::choose_action_from_net(
                    &proba_tensor,
                    &valid_actions_mask,
                    no_invalid_for_traverser,
                )?;

                gs = gs.get_child(action_index).as_mut().unwrap();

                print!(
                    "{} ({} {})",
                    gs.get_state_data()
                        .history
                        .last()
                        .unwrap()
                        ._to_print_string(),
                    gs.get_state_data().bets[p_to_move as usize],
                    gs.get_state_data().stacks[p_to_move as usize],
                );
                println!();
            }
        }

        println!();
        print!("Rewards: ");
        for i in 0..self.player_cnt {
            print!("{} ", gs.get_reward(i));
        }
        println!();

        Ok(())
    }
}
