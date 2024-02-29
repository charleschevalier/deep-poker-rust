use std::sync::{Arc, Mutex};

use candle_core::Tensor;
use rand::Rng;
use threadpool::ThreadPool;

use super::action::ActionConfig;
use super::action_state::ActionState;
use super::hand_state::HandState;
use super::state::{State, StateType};
use super::state_chance::StateChance;
use super::state_data::StateData;
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
        agents: &Vec<Arc<Box<dyn Agent>>>,
        device: &candle_core::Device,
        no_invalid_for_traverser: bool,
        epsilon_greedy: f32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.reset(traverser);

        Tree::traverse_state(
            &mut self.root,
            self.hand_state.as_mut().unwrap(),
            agents,
            self.action_config,
            device,
            no_invalid_for_traverser,
            epsilon_greedy,
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
        agents: &Vec<Arc<Box<dyn Agent>>>,
        action_config: &ActionConfig,
        device: &candle_core::Device,
        no_invalid_for_traverser: bool,
        epsilon_greedy: f32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // If state is None, panic
        if state_option.is_none() {
            panic!("State is None");
        }

        // // Make sure we do not have too much actions in hand_state
        // {
        //     let mut action_cnt = 0;
        //     let mut too_much = false;
        //     let mut street = 0;
        //     for action_state in hand_state.action_states.iter() {
        //         if action_state.street == street {
        //             action_cnt += 1;
        //         } else {
        //             street = action_state.street;
        //             action_cnt = 0;
        //         }

        //         if action_cnt >= action_config.max_actions_per_street {
        //             too_much = true;
        //             break;
        //         }
        //     }

        //     if too_much {
        //         return Err("Too much actions".into());
        //     }
        // }

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
                agents,
                action_config,
                device,
                no_invalid_for_traverser,
                epsilon_greedy,
            );
        } else {
            // Traverse for player to move
            state.create_children();

            let valid_actions_mask = state.get_valid_actions_mask();

            let mut rng = rand::thread_rng();
            let random_float_0_1: f32 = rng.gen();

            let action_index = if random_float_0_1 >= epsilon_greedy || epsilon_greedy == 0.0 {
                // Regular traversal, we choose an action from the network
                agents[state.get_player_to_move() as usize].choose_action(
                    hand_state,
                    &valid_actions_mask,
                    state.get_state_data().street,
                    action_config,
                    device,
                    if state.get_player_to_move() == traverser as i32 {
                        no_invalid_for_traverser
                    } else {
                        true
                    },
                )?
            } else {
                // Epsilon greedy, we choose a random action to favor exploration
                let mut index: usize = rng.gen_range(0..valid_actions_mask.len());
                while (traverser as i32 != state.get_player_to_move() || no_invalid_for_traverser)
                    && !valid_actions_mask[index]
                {
                    index = rng.gen_range(0..valid_actions_mask.len());
                }
                index
            };

            if action_index > valid_actions_mask.len() || !valid_actions_mask[action_index] {
                if state.get_player_to_move() == traverser as i32 && !no_invalid_for_traverser {
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
                    panic!("Invalid action index in tree traversal");
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
                agents,
                action_config,
                device,
                no_invalid_for_traverser,
                epsilon_greedy,
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
        base_network: &PokerNetwork,
        base_device: &candle_core::Device,
        base_no_invalid_for_traverser: bool,
        base_action_config: &ActionConfig,
    ) -> Result<(), candle_core::Error> {
        let n_workers = num_cpus::get();
        let thread_pool = ThreadPool::new(n_workers);

        let action_count = base_action_config.postflop_raise_sizes.len() + 3;
        let base_result: Arc<Mutex<Vec<Vec<Vec<f32>>>>> =
            Arc::new(Mutex::new(vec![vec![vec![0.0; 13]; 13]; action_count]));
        let base_count: Arc<Mutex<Vec<Vec<Vec<u32>>>>> =
            Arc::new(Mutex::new(vec![vec![vec![0; 13]; 13]; action_count]));
        let mut base_valid_actions_mask: Vec<bool> = base_action_config
            .preflop_raise_sizes
            .iter()
            .map(|&x| x > 0.0)
            .collect();

        base_valid_actions_mask.insert(0, true);
        base_valid_actions_mask.insert(0, true);
        base_valid_actions_mask.push(true);

        // Iterate through card combinations
        for a in 0..13 {
            let network = base_network.clone();
            let device = base_device.clone();
            let action_config = base_action_config.clone();
            let no_invalid_for_traverser = base_no_invalid_for_traverser;
            let valid_actions_mask = base_valid_actions_mask.to_vec();
            let result = Arc::clone(&base_result);
            let count = Arc::clone(&base_count);
            //let a = a_i;

            thread_pool.execute(move || {
                for b in 0..4 {
                    let i = a * 4 + b;
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
                                    vec![0.0; 3 + action_config.postflop_raise_sizes.len()];
                                    action_config.player_count as usize + 2
                                ];
                                4 * action_config.max_actions_per_street as usize
                            ];

                        let card_tensor = Tensor::new(card_vecs, &device)
                            .unwrap()
                            .unsqueeze(0)
                            .unwrap();
                        let action_tensor = Tensor::new(action_vecs, &device)
                            .unwrap()
                            .unsqueeze(0)
                            .unwrap();

                        let proba_tensor = network
                            .forward_embedding_actor(&card_tensor, &action_tensor, false)
                            .unwrap()
                            .detach();

                        let is_suited = suit1 == suit2;

                        let min_rank = if rank1 < rank2 { rank1 } else { rank2 };
                        let max_rank = if rank1 > rank2 { rank1 } else { rank2 };

                        let mut probas: Vec<f32> =
                            proba_tensor.squeeze(0).unwrap().to_vec1().unwrap();

                        // Normalize probs
                        for k in 0..probas.len() {
                            if no_invalid_for_traverser
                                && (k >= valid_actions_mask.len() || !valid_actions_mask[k])
                            {
                                probas[k] = 0.0;
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

                        let mut res = result.lock().unwrap();
                        let mut cnt = count.lock().unwrap();
                        for action_index in 0..action_count {
                            if is_suited {
                                res[action_index][max_rank][min_rank] += probas[action_index];
                                cnt[action_index][max_rank][min_rank] += 1;
                            } else {
                                res[action_index][min_rank][max_rank] += probas[action_index];
                                cnt[action_index][min_rank][max_rank] += 1;
                            }
                        }
                    }
                }
            });
        }

        thread_pool.join();

        let res = base_result.lock().unwrap();
        let cnt = base_count.lock().unwrap();

        for action_index in 0..action_count {
            if base_no_invalid_for_traverser && !base_valid_actions_mask[action_index] {
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
                    let value = if cnt[action_index][i][j] > 0 {
                        res[action_index][i][j] / cnt[action_index][i][j] as f32
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

    pub fn play_one_hand(
        &mut self,
        agents: &[Arc<Box<dyn Agent>>],
        device: &candle_core::Device,
        silent: bool,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        self.reset(0);

        let mut gs = self.root.as_mut().unwrap();
        let mut first = true;

        while !matches!(gs.get_type(), StateType::Terminal) {
            if matches!(gs.get_type(), StateType::Chance) {
                gs.create_children();
                gs = gs.get_child(0).as_mut().unwrap();

                if !silent {
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
                    } else if gs.get_state_data().street >= 2
                        && !gs.get_state_data().board.is_empty()
                    {
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
                }
            } else {
                let p_to_move = gs.get_player_to_move();
                gs.create_children();

                if !silent {
                    print!("Player {}'s turn: ", p_to_move);
                }

                let action_index = agents[p_to_move as usize].as_ref().choose_action(
                    self.hand_state.as_ref().unwrap(),
                    &gs.get_valid_actions_mask(),
                    gs.get_state_data().street,
                    self.action_config,
                    device,
                    true,
                )?;

                gs = gs.get_child(action_index).as_mut().unwrap();

                if !silent {
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
        }
        if !silent {
            println!();
            print!("Rewards: ");
            for i in 0..self.player_cnt {
                print!("{} ", gs.get_reward(i));
            }
            println!();
        }

        let rewards = (0..self.player_cnt)
            .map(|i| gs.get_reward(i))
            .collect::<Vec<f32>>();

        Ok(rewards)
    }
}
