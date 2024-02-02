use super::action::Action;
use super::action::ActionConfig;
use super::table::Table;

pub struct Tree {
    player_cnt: u32,
    table: Table,
    blind_value: u32,
    action_config: ActionConfig,
}

// Add different tree nodes implementations
mod chance;
mod play;
mod terminal;

impl Tree {
    pub fn new(
        player_cnt: u32,
        stack_size: u32,
        blind_value: u32,
        action_config: ActionConfig,
    ) -> Tree {
        Tree {
            player_cnt: player_cnt,
            table: Table::new(player_cnt, stack_size),
            blind_value: blind_value,
            action_config: action_config,
        }
    }

    fn reset(&mut self) -> () {
        // Shuffle the deck
        self.table.reset();
    }

    pub fn play_hand(&mut self) -> Vec<Action> {
        self.reset();

        // Post blinds and set first player to move
        if self.player_cnt == 2 {
            self.table.players[0].stack -= self.blind_value;
            self.table.players[0].total_bet += self.blind_value;
            self.table.players[1].stack -= self.blind_value / 2;
            self.table.players[1].total_bet += self.blind_value / 2;
            self.table.player_to_move = 1;
        } else {
            self.table.players[0].stack -= self.blind_value / 2;
            self.table.players[0].total_bet += self.blind_value / 2;
            self.table.players[1].stack -= self.blind_value;
            self.table.players[1].total_bet += self.blind_value;
            self.table.player_to_move = 0;
        }

        return self.process_play();
    }
}
