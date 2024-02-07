use super::action::ActionConfig;
use super::action::ActionType;
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
    pub fn to_input(
        &self,
        action_state_index: usize,
        action_config: &ActionConfig,
        device: &candle_core::Device,
    ) -> (Tensor, Tensor) {
        let action_state = &self.action_states[action_state_index];

        // Create card tensor
        // Shape is (street_cnt + 1 for all cards) x number_of_ranks x number_of_suits
        let mut card_vecs: Vec<Vec<Vec<f32>>> = vec![vec![vec![0.0; 4]; 13]; 5];

        // Set hand cards
        for card in self.hand.iter() {
            card_vecs[0][card.rank() as usize][card.suit() as usize] = 1.0;
            card_vecs[5][card.rank() as usize][card.suit() as usize] = 1.0;
        }

        // Set flop cards
        if action_state.street > 1 {
            for i in 0..3 {
                let card = self.board[i];
                card_vecs[1][card.rank() as usize][card.suit() as usize] = 1.0;
                card_vecs[5][card.rank() as usize][card.suit() as usize] = 1.0;
            }
        }

        // Set turn cards
        if action_state.street > 2 {
            let card = self.board[3];
            card_vecs[2][card.rank() as usize][card.suit() as usize] = 1.0;
            card_vecs[5][card.rank() as usize][card.suit() as usize] = 1.0;
        }

        // Set river cards
        if action_state.street > 3 {
            let card = self.board[4];
            card_vecs[3][card.rank() as usize][card.suit() as usize] = 1.0;
            card_vecs[5][card.rank() as usize][card.suit() as usize] = 1.0;
        }

        // Create action tensor
        // Shape is (street_cnt * max_actions_per_street) x max_number_of_actions x (player_count + 2 for sum and legal)
        let mut action_vecs: Vec<Vec<Vec<f32>>> =
            vec![
                vec![
                    vec![0.0; action_config.player_count as usize + 2];
                    3 + action_config.postflop_raise_sizes.len()
                ];
                4 * action_config.max_actions_per_street as usize
            ];

        let mut action_cnt: usize = 0;
        let mut current_street: u8 = 0;
        for action_state_it in self.action_states.iter().take(action_state_index + 1) {
            // Reset action count per player for a new street
            let action_street = action_state_it.street - 1; // Street starts at 1 in game tree
            if action_street > current_street {
                current_street = action_street;
                action_cnt = 0;
            }

            let action_index = match action_state_it.action_taken.action_type {
                ActionType::Fold => 0,
                ActionType::Call => 1,
                ActionType::Raise => 2 + action_state_it.action_taken.raise_index as usize,
                ActionType::AllIn => 2 + action_config.postflop_raise_sizes.len() as usize,
                _ => 0,
            };

            // Set player action in tensor
            action_vecs[(current_street - 1) as usize
                * action_config.max_actions_per_street as usize
                + action_cnt as usize][action_index][action_state_it.player_to_move as usize] = 1.0;

            // Increment sum of actions
            action_vecs[(current_street - 1) as usize
                * action_config.max_actions_per_street as usize
                + action_cnt as usize][action_index][action_config.player_count as usize] += 1.0;

            // Set legal actions
            for i in 0..action_state.valid_actions_mask.len() {
                if action_state.valid_actions_mask[i] {
                    action_vecs[(current_street - 1) as usize
                        * action_config.max_actions_per_street as usize
                        + action_cnt as usize][action_index]
                        [action_config.player_count as usize + 2] = 1.0;
                }
            }

            // Increment action count for current street
            action_cnt += 1;
        }

        (
            Tensor::new(card_vecs, &device).unwrap(),
            Tensor::new(action_vecs, &device).unwrap(),
        )
    }
}
