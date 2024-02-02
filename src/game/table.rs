use super::player::Player;
use poker::Card;

pub struct Table {
    pub board: Vec<Card>,
    pub players: Vec<Player>,
    pub player_to_move: i32,
    player_cnt: u32,
    stack_size: u32,
    street: u8,
}

impl Table {
    pub fn new(player_cnt: u32, stack_size: u32) -> Table {
        let mut players: Vec<Player> = Vec::with_capacity(player_cnt as usize);
        for _ in 0..player_cnt {
            players.push(Player::new());
        }
        Table {
            board: Vec::new(),
            players: players,
            player_to_move: -1,
            player_cnt: player_cnt,
            stack_size: stack_size,
            street: 0,
        }
    }

    pub fn reset(&mut self) -> () {
        let mut deck = Card::generate_shuffled_deck();
        self.board = deck.drain(..5).collect();
        self.players = Vec::new();
        for player in self.players.iter_mut() {
            player.reset(self.stack_size, deck.drain(..2).collect());
        }
    }
}
