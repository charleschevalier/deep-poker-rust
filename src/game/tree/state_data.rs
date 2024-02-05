use super::super::action::{Action, ActionType};
use poker::Card;

pub struct StateData {
    pub player_count: u32,
    pub board: Vec<Card>,
    pub hands: Vec<Vec<Card>>,
    pub stacks: Vec<u32>,
    pub bets: Vec<u32>,
    pub player_to_move: i32,
    pub last_player: i32,
    pub is_player_in: Vec<bool>,
    pub players_in_hand: u32,
    pub rewards: Vec<f32>,
    pub street: u8,
    pub min_raise: u32,
    pub is_betting_open: bool,
    pub action_count: u8,
    pub last_actions: Vec<Action>,
    pub history: Vec<Action>,
}

impl StateData {
    pub fn new(player_count: u32, stack_size: u32) -> StateData {
        // Create last actions
        let mut last_actions = Vec::new();
        for i in 0..player_count {
            last_actions.push(Action {
                action_type: ActionType::None,
                raise_index: -1,
                player_index: i as i32,
                street: 0,
            });
        }

        // Draw cards
        let mut deck = Card::generate_shuffled_deck();
        let board = deck.drain(..5).collect();
        let mut hands = Vec::new();
        for _ in 0..player_count {
            hands.push(deck.drain(..2).collect());
        }

        StateData {
            player_count: player_count,
            board: board,
            hands: hands,
            stacks: vec![stack_size; player_count as usize],
            bets: vec![0; player_count as usize],
            player_to_move: -1,
            last_player: -1,
            is_player_in: vec![true; player_count as usize],
            players_in_hand: player_count,
            rewards: Vec::new(),
            street: 0,
            min_raise: 0,
            is_betting_open: true,
            action_count: 0,
            last_actions: last_actions,
            history: Vec::new(),
        }
    }
}

impl Clone for StateData {
    fn clone(&self) -> Self {
        StateData {
            player_count: self.player_count,
            board: self.board.clone(),
            hands: self.hands.clone(),
            stacks: self.stacks.clone(),
            bets: self.bets.clone(),
            player_to_move: self.player_to_move,
            last_player: self.last_player,
            is_player_in: self.is_player_in.clone(),
            players_in_hand: self.players_in_hand,
            rewards: self.rewards.clone(),
            street: self.street,
            min_raise: self.min_raise,
            is_betting_open: self.is_betting_open,
            action_count: self.action_count,
            last_actions: self.last_actions.clone(),
            history: self.history.clone(),
        }
    }
}
