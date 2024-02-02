use poker::Card;

pub struct Player {
    pub stack: u32,
    pub total_bet: u32,
    pub hand: Vec<Card>,
}

impl Player {
    pub fn new() -> Player {
        Player {
            stack: 0,
            total_bet: 0,
            hand: Vec::new(),
        }
    }

    pub fn reset(&mut self, stack_size: u32, hand: Vec<Card>) -> () {
        self.stack = stack_size;
        self.total_bet = 0;
        self.hand = hand;
    }
}
