use super::action::ActionConfig;
use super::action_state::ActionState;
use candle_core::Tensor;
use poker::Card;

#[derive(Clone, Debug)]
pub struct HandState {
    pub traverser: u32,
    pub hand: Vec<Card>,
    pub board: Vec<Card>,
    pub action_states: Vec<ActionState>,
}

impl HandState {
    pub fn get_all_tensors(
        &self,
        action_config: &ActionConfig,
        device: &candle_core::Device,
    ) -> Result<(Tensor, Tensor), Box<dyn std::error::Error>> {
        // Iterate on states for traverser
        let mut card_tensors: Vec<Tensor> = Vec::new();
        let mut action_tensors: Vec<Tensor> = Vec::new();

        // In case the last action is an invalid action, we only take the last action
        // (see get_traverser_action_states for more details)
        let last = self.action_states.last();
        let is_invalid = last.is_some() && last.unwrap().is_invalid;

        for i in 0..self.action_states.len() {
            if self.action_states[i].player_to_move == self.traverser
                && (!is_invalid || i == self.action_states.len() - 1)
            {
                match self.action_state_to_input(i, action_config, device) {
                    Ok((card_tensor, action_tensor)) => {
                        card_tensors.push(card_tensor);
                        action_tensors.push(action_tensor);
                    }
                    Err(err) => return Err(err),
                }
            }
        }

        Ok((
            Tensor::stack(&card_tensors, 0)?,
            Tensor::stack(&action_tensors, 0)?,
        ))
    }

    pub fn to_input(
        &self,
        street: u8,
        action_config: &ActionConfig,
        device: &candle_core::Device,
        current_state_index: usize,
    ) -> Result<(Tensor, Tensor), Box<dyn std::error::Error>> {
        // Create card tensor
        // Shape is (street_cnt + 1 for all cards) x number_of_suits x number_of_ranks
        let mut card_vecs: Vec<Vec<Vec<f32>>> = vec![vec![vec![0.0; 13]; 4]; 6];

        // Set hand cards
        for card in self.hand.iter() {
            card_vecs[0][card.suit() as usize][card.rank() as usize] = 1.0;
            card_vecs[5][card.suit() as usize][card.rank() as usize] = 1.0;
        }

        // Set flop cards
        if street > 1 {
            for i in 0..3 {
                let card = self.board[i];
                card_vecs[1][card.suit() as usize][card.rank() as usize] = 1.0;
                card_vecs[4][card.suit() as usize][card.rank() as usize] = 1.0;
                card_vecs[5][card.suit() as usize][card.rank() as usize] = 1.0;
            }
        }

        // Set turn cards
        if street > 2 {
            let card = self.board[3];
            card_vecs[2][card.suit() as usize][card.rank() as usize] = 1.0;
            card_vecs[4][card.suit() as usize][card.rank() as usize] = 1.0;
            card_vecs[5][card.suit() as usize][card.rank() as usize] = 1.0;
        }

        // Set river cards
        if street > 3 {
            let card = self.board[4];
            card_vecs[3][card.suit() as usize][card.rank() as usize] = 1.0;
            card_vecs[4][card.suit() as usize][card.rank() as usize] = 1.0;
            card_vecs[5][card.suit() as usize][card.rank() as usize] = 1.0;
        }

        // Print card_vecs as matrix
        // println!("-------------------");

        // for i in 0..5 {
        //     for j in 0..4 {
        //         for k in 0..13 {
        //             print!("{}", card_vecs[i][j][k]);
        //         }
        //         println!();
        //     }
        //     println!();
        // }

        // Create action tensor
        // Shape is (street_cnt * max_actions_per_street) x (player_count + 2 for sum and legal) x max_number_of_actions
        let mut action_vecs: Vec<Vec<Vec<f32>>> =
            vec![
                vec![
                    vec![0.0; 3 + action_config.postflop_raise_sizes.len()];
                    action_config.player_count as usize + 2
                ];
                4 * action_config.max_actions_per_street as usize
            ];

        let mut action_cnt: usize = 0;
        let mut current_street: u8 = 0;
        for action_state_it in self.action_states.iter().take(current_state_index + 1) {
            // Reset action count per player for a new street
            let action_street = action_state_it.street - 1; // Street starts at 1 in game tree
            if action_street > current_street {
                current_street = action_street;
                action_cnt = 0;
            }

            // Set player action in tensor
            action_vecs[current_street as usize * action_config.max_actions_per_street as usize
                + action_cnt][action_state_it.player_to_move as usize]
                [action_state_it.action_taken_index] = 1.0;

            // Increment sum of actions
            action_vecs[current_street as usize * action_config.max_actions_per_street as usize
                + action_cnt][action_config.player_count as usize]
                [action_state_it.action_taken_index] += 1.0;

            // Set legal actions
            for (i, valid) in action_state_it.valid_actions_mask.iter().enumerate() {
                if *valid {
                    action_vecs[current_street as usize
                        * action_config.max_actions_per_street as usize
                        + action_cnt][action_config.player_count as usize + 1][i] = 1.0;
                }
            }

            // Increment action count for current street
            action_cnt += 1;
        }

        // Print action_vecs as matrix
        // if self.action_states.len() > 2 {
        //     for i in 0..4 * action_config.max_actions_per_street as usize {
        //         for j in 0..action_config.player_count as usize + 2 {
        //             for k in 0..3 + action_config.postflop_raise_sizes.len() {
        //                 print!("{}", action_vecs[i][j][k]);
        //             }
        //             println!();
        //         }
        //         println!();
        //     }
        // }
        // for i in 0..4 * action_config.max_actions_per_street as usize {
        //     for j in 0..action_config.player_count as usize + 2 {
        //         for k in 0..3 + action_config.postflop_raise_sizes.len() {
        //             print!("{}", action_vecs[i][j][k]);
        //         }
        //         println!();
        //     }
        //     println!();
        // }

        Ok((
            Tensor::new(card_vecs, device)?,
            Tensor::new(action_vecs, device)?,
        ))
    }

    fn action_state_to_input(
        &self,
        action_state_index: usize,
        action_config: &ActionConfig,
        device: &candle_core::Device,
    ) -> Result<(Tensor, Tensor), Box<dyn std::error::Error>> {
        let action_state = &self.action_states[action_state_index];

        self.to_input(
            action_state.street,
            action_config,
            device,
            action_state_index,
        )
    }

    pub fn get_traverser_action_states(&self) -> Vec<&ActionState> {
        let result: Vec<&ActionState> = self
            .action_states
            .iter()
            .filter(|action_state| action_state.player_to_move == self.traverser)
            .collect();
        // Get last element
        let last = result.last();
        if last.is_some() && last.unwrap().is_invalid {
            // Take only last element so the network learns the invalid action but does not
            // propagate wrong rewards on potentially correct actions before that
            result.into_iter().rev().take(1).collect()
        } else {
            result
        }
    }
}
