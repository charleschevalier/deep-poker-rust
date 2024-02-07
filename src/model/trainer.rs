use super::trainer_config::TrainerConfig;
use crate::game::action::ActionConfig;
use crate::game::hand_state::HandState;
use crate::game::tree::Tree;
use candle_core::Tensor;

use rand::seq::SliceRandom;
use rand::thread_rng;

struct Trainer<'a> {
    player_cnt: u32,
    action_config: &'a ActionConfig,
    trainer_config: &'a TrainerConfig,
    tree: Tree<'a>,
}

impl<'a> Trainer<'a> {
    pub fn new(
        player_cnt: u32,
        action_config: &'a ActionConfig,
        trainer_config: &'a TrainerConfig,
    ) -> Trainer<'a> {
        Trainer {
            player_cnt,
            action_config,
            trainer_config,
            tree: Tree::new(player_cnt, action_config),
        }
    }

    pub fn train(&mut self) {
        // let mut dataset = Vec::new();

        // for _ in 0..self.trainer_config.max_iters {
        //     for player in 0..self.player_cnt {
        //         self.tree.traverse(player);
        //         dataset.push(self.tree.hand_state.clone());
        //     }

        //     if dataset.len() >= self.trainer_config.max_dataset_cache {
        //         // Shuffle the dataset
        //         dataset.shuffle(&mut thread_rng());

        //         while dataset.len() > self.trainer_config.batch_size {
        //             // Take a data batch
        //             let batch: Vec<_> = dataset.drain(0..self.trainer_config.batch_size).collect();

        //             // Train the model
        //             self.train_model(&batch);
        //         }
        //     }
        // }
    }

    fn train_model(&self, batch: &Vec<Option<HandState>>) {
        // Get inputs and rewards
    }

    fn calculate_advantage_gae(&self, hand_state: &HandState, gamma: f32, tau: f32) -> Vec<f32> {
        let mut gae = 0.0;
        let mut returns = Vec::new();

        // for action_state in hand_state.action_states.iter().rev() {
        //     let delta = action_state.reward + gamma * action_state.next_value * action_state.mask
        //         - action_state.value;
        //     gae = delta + gamma * tau * action_state.mask * gae;
        //     returns.insert(0, gae + action_state.value);
        // }

        returns
    }

    // def calculate_gae(next_value, rewards, masks, values, gamma=0.999, tau=0.95):
    //     gae = 0
    //     returns = []
    //     for step in reversed(range(len(rewards))):
    //         delta = rewards[step] + gamma * next_value * masks[step] - values[step]
    //         gae = delta + gamma * tau * masks[step] * gae
    //         next_value = values[step]
    //         returns.insert(0, gae + values[step])
    //     return returns

    fn get_inputs(&self, hand: &HandState) -> Vec<f32> {
        // let mut card_inputs = Vec::new();
        // let mut action_inputs = Vec::new();

        // for action_state in hand.action_states {
        //     //inputs.push(action_state.get_input());
        // }

        //inputs
        Vec::new()
    }

    fn get_input(&self, hand_state: &HandState) -> Vec<f32> {
        let mut inputs = Vec::new();

        // for card in hand_state.hand {
        //     //inputs.push(card.get_input());
        // }

        // for card in hand_state.board {
        //     //inputs.push(card.get_input());
        // }

        inputs
    }

    fn get_discounted_rewards(&self, hand_state: &HandState, gamma: f32) -> Vec<f32> {
        let mut rewards = Vec::new();
        let mut reward = 0.0;

        for action_state in hand_state.action_states.iter().rev() {
            reward = action_state.reward + gamma * reward;
            rewards.push(reward);
        }

        rewards.reverse();
        rewards
    }

    fn get_card_inputs(&self, hand_state: &HandState) -> Vec<Vec<Tensor>> {
        let mut inputs = Vec::new();

        // for card in hand_state.hand {
        //     //inputs.push(card.get_input());
        // }

        // for card in hand_state.board {
        //     //inputs.push(card.get_input());
        // }

        inputs
    }

    // def calculate_discounted_rewards(rewards, gamma=0.999):
    //     discounted_rewards = []
    //     R = 0
    //     for reward in reversed(rewards):
    //         R = reward + gamma * R
    //         discounted_rewards.insert(0, R)
    //     return torch.tensor(discounted_rewards)
}
