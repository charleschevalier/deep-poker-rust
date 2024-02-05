use super::super::action::{Action, ActionConfig, ActionType};
use super::state::{State, StateType};
use super::state_chance::StateChance;
use super::state_data::StateData;
use super::state_terminal::StateTerminal;

pub struct StatePlay<'a> {
    pub action_config: &'a ActionConfig,
    pub state_data: StateData,
    pub children: Vec<Box<dyn State<'a> + 'a>>,
    pub action_count: i32,
    pub valid_actions: Vec<Action>,
}

impl<'a> State<'a> for StatePlay<'a> {
    fn get_type(&self) -> StateType {
        return StateType::Play;
    }

    fn get_child(&mut self, index: usize) -> &mut Box<dyn State<'a> + 'a> {
        return &mut self.children[index];
    }

    fn get_child_count(&self) -> usize {
        return self.children.len();
    }

    // Overrides
    fn get_state_data(&self) -> &StateData {
        return &self.state_data;
    }

    fn get_reward(&mut self, _traverser: u32) -> f32 {
        panic!("Not implemented");
    }

    fn create_children(&mut self) -> () {
        if !self.children.is_empty() {
            return;
        }

        let mut pot: u32 = 0;
        let mut biggest_bet: u32 = 0;
        for j in 0..self.state_data.player_count {
            pot += self.state_data.bets[j as usize];
            if self.state_data.bets[j as usize] > biggest_bet {
                biggest_bet = self.state_data.bets[j as usize];
            }
        }

        self.handle_fold(biggest_bet);
        self.handle_call(biggest_bet);

        if self.state_data.is_betting_open {
            self.handle_raises(pot, biggest_bet);
            self.handle_all_in();
        }
    }
}

impl<'a> StatePlay<'a> {
    pub fn new(action_config: &'a ActionConfig, state_data: StateData) -> StatePlay {
        StatePlay {
            action_config: action_config,
            state_data: state_data,
            children: Vec::new(),
            action_count: 0,
            valid_actions: Vec::new(),
        }
    }

    fn get_valid_actions_count(&mut self) -> i32 {
        if self.action_count == 0 {
            self.action_count = self.get_valid_actions().len() as i32;
        }
        return self.action_count;
    }

    fn get_valid_actions(&mut self) -> &Vec<Action> {
        if self.valid_actions.is_empty() {
            if self.state_data.players_in_hand < 2 {
                self.print_actions();
                panic!("Not enough players in hand");
            }

            let mut pot: u32 = 0;
            let mut biggest_bet: u32 = 0;
            for j in 0..self.state_data.player_count {
                pot += self.state_data.bets[j as usize];
                if self.state_data.bets[j as usize] > biggest_bet {
                    biggest_bet = self.state_data.bets[j as usize];
                }
            }

            // Handle folds
            if biggest_bet > self.get_to_move_bet() {
                self.valid_actions.push(Action {
                    action_type: ActionType::Fold,
                    raise_index: -1,
                    player_index: self.get_player_to_move(),
                    street: self.state_data.street,
                });
            }

            // Handle checks & calls
            if biggest_bet - self.get_to_move_bet() < self.get_to_move_stack() {
                self.valid_actions.push(Action {
                    action_type: ActionType::Call,
                    raise_index: -1,
                    player_index: self.get_player_to_move(),
                    street: self.state_data.street,
                });
            }

            // Handle raises & all-ins
            if self.state_data.is_betting_open {
                let raises: Vec<f32>;
                if self.state_data.street == 1 {
                    raises = self.action_config.preflop_raise_sizes.clone();
                } else {
                    raises = self.action_config.postflop_raise_sizes.clone();
                }

                // Iterate through configured raise sizes
                for i in 0..raises.len() {
                    let to_call = biggest_bet - self.get_to_move_bet();
                    let raise;
                    let actual_bet;

                    // Calculate raise size in chips
                    if self.state_data.street == 1 {
                        actual_bet = (raises[i] * biggest_bet as f32).round() as u32
                            - self.get_to_move_bet();
                        raise = (raises[i] * biggest_bet as f32).round() as u32 - biggest_bet;
                    } else {
                        raise = (raises[i] * pot as f32).round() as u32;
                        actual_bet = to_call + raise;
                    }

                    let stack_left = self.get_to_move_stack() - actual_bet;

                    // Do not add raises below the minimum raise or raises that would put the player all in
                    // Or raises that would leave us with less than commited_to_pot_percentage% of our stack
                    if raise < self.state_data.min_raise
                        || actual_bet >= self.get_to_move_stack()
                        || (stack_left as f32)
                            < (self.action_config.commited_to_pot_percentage as f32
                                * self.action_config.buy_in as f32
                                / 100.0)
                    {
                        continue;
                    }

                    // Valid raise
                    self.valid_actions.push(Action {
                        action_type: ActionType::Raise,
                        raise_index: i as i8,
                        player_index: self.get_player_to_move(),
                        street: self.state_data.street,
                    });
                }

                // Handle all-ins
                if self.get_to_move_stack() > 0 {
                    self.valid_actions.push(Action {
                        action_type: ActionType::AllIn,
                        raise_index: -1,
                        player_index: self.get_player_to_move(),
                        street: self.state_data.street,
                    });
                }
            }
        }
        return &self.valid_actions;
    }

    fn handle_raises(&mut self, pot: u32, biggest_bet: u32) -> () {
        let raises: Vec<f32>;
        if self.state_data.street == 1 {
            raises = self.action_config.preflop_raise_sizes.clone();
        } else {
            raises = self.action_config.postflop_raise_sizes.clone();
        }

        // Iterate through configured raise sizes
        for i in 0..raises.len() {
            self.handle_raise(pot, biggest_bet, i, raises[i]);
        }
    }

    fn handle_fold(&mut self, biggest_bet: u32) -> () {
        if biggest_bet > self.get_to_move_bet() {
            let mut new_state_data = self.state_data.clone();
            new_state_data.is_player_in[self.get_player_to_move() as usize] = false;
            new_state_data.players_in_hand -= 1;
            let new_action = Action {
                action_type: ActionType::Fold,
                raise_index: -1,
                player_index: self.get_player_to_move(),
                street: new_state_data.street,
            };
            new_state_data.history.push(new_action.clone());
            new_state_data.last_actions[self.get_player_to_move() as usize] = new_action.clone();
            new_state_data.player_to_move = self.get_next_player(new_state_data.last_player);

            // check if there is any player that has to play..
            if self.get_active_players(&new_state_data.is_player_in) == 1 {
                self.children
                    .push(Box::new(StateTerminal::new(new_state_data)));
            } else if new_state_data.player_to_move != -1 {
                self.children
                    .push(Box::new(StatePlay::new(self.action_config, new_state_data)));
            } else {
                // Here the betting round is over, there is more than 1 player left
                if new_state_data.street != 4 {
                    self.children.push(Box::new(StateChance::new(
                        self.action_config,
                        new_state_data,
                    )));
                } else {
                    self.children
                        .push(Box::new(StateTerminal::new(new_state_data)));
                }
            }
        }
    }

    fn handle_call(&mut self, biggest_bet: u32) -> () {
        if biggest_bet - self.get_to_move_bet() < self.get_to_move_stack() {
            let mut new_state_data = self.state_data.clone();
            new_state_data.bets[self.get_player_to_move() as usize] +=
                biggest_bet - self.get_to_move_bet();
            new_state_data.stacks[self.get_player_to_move() as usize] -=
                biggest_bet - self.get_to_move_bet();
            let new_action = Action {
                action_type: ActionType::Call,
                raise_index: -1,
                player_index: self.get_player_to_move(),
                street: new_state_data.street,
            };
            new_state_data.history.push(new_action.clone());
            new_state_data.last_actions[self.get_player_to_move() as usize] = new_action.clone();
            new_state_data.player_to_move = self.get_next_player(new_state_data.last_player);

            // check if there is any player that has to play..
            if new_state_data.player_to_move != -1 {
                self.children
                    .push(Box::new(StatePlay::new(self.action_config, new_state_data)));
            } else {
                if new_state_data.street != 4 {
                    self.children.push(Box::new(StateChance::new(
                        self.action_config,
                        new_state_data,
                    )));
                } else {
                    self.children
                        .push(Box::new(StateTerminal::new(new_state_data)));
                }
            }
        }
    }

    fn handle_raise(
        &mut self,
        pot: u32,
        biggest_bet: u32,
        action_index: usize,
        action_value: f32,
    ) -> () {
        let to_call = biggest_bet - self.get_to_move_bet();
        let raise: u32;
        let actual_bet: u32;

        if self.state_data.street == 1 {
            actual_bet =
                (action_value * biggest_bet as f32).round() as u32 - self.get_to_move_bet();
            raise = (action_value * biggest_bet as f32).round() as u32 - biggest_bet;
        } else {
            raise = (action_value * pot as f32).round() as u32;
            actual_bet = to_call + raise;
        }

        let stack_left: i32 = self.get_to_move_stack() as i32 - actual_bet as i32;

        if raise < self.state_data.min_raise
            || actual_bet >= self.get_to_move_stack()
            || (stack_left as f32)
                < (self.action_config.commited_to_pot_percentage as f32
                    * self.action_config.buy_in as f32
                    / 100.0)
        {
            return;
        }

        // Valid raise
        let mut new_state_data = self.state_data.clone();
        new_state_data.bets[self.get_player_to_move() as usize] += actual_bet;
        new_state_data.stacks[self.get_player_to_move() as usize] -= actual_bet;
        let new_action = Action {
            action_type: ActionType::Raise,
            raise_index: action_index as i8,
            player_index: self.get_player_to_move(),
            street: new_state_data.street,
        };
        new_state_data.history.push(new_action.clone());
        new_state_data.last_actions[self.get_player_to_move() as usize] = new_action.clone();

        new_state_data.last_player = self.get_last_player(self.get_player_to_move());
        new_state_data.player_to_move = self.get_next_player(new_state_data.last_player);

        if new_state_data.player_to_move != -1 {
            self.children
                .push(Box::new(StatePlay::new(self.action_config, new_state_data)));
        } else {
            self.print_actions();
            panic!("Someone raised but there is noone left to play next");
        }
    }

    fn handle_all_in(&mut self) -> () {
        if self.get_to_move_stack() > 0 {
            let mut new_state_data = self.state_data.clone();
            new_state_data.bets[self.get_player_to_move() as usize] += self.get_to_move_stack();
            new_state_data.stacks[self.get_player_to_move() as usize] = 0;
            let new_action = Action {
                action_type: ActionType::AllIn,
                raise_index: -1,
                player_index: self.get_player_to_move(),
                street: new_state_data.street,
            };
            new_state_data.history.push(new_action.clone());
            new_state_data.last_actions[self.get_player_to_move() as usize] = new_action.clone();

            new_state_data.last_player = self.get_last_player(self.get_player_to_move());
            new_state_data.player_to_move = self.get_next_player(new_state_data.last_player);

            // check if there is any player that has to play..
            if new_state_data.player_to_move != -1 {
                new_state_data.is_betting_open = false;

                self.children
                    .push(Box::new(StatePlay::new(self.action_config, new_state_data)));
            } else {
                if self.state_data.street != 4 {
                    // New street
                    self.children.push(Box::new(StateChance::new(
                        self.action_config,
                        new_state_data,
                    )));
                } else {
                    // Showdown
                    self.children
                        .push(Box::new(StateTerminal::new(new_state_data)));
                }
            }
        }
    }
}
