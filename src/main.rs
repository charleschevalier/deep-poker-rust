// use candle_core::{DType, Device, Tensor};
// use candle_nn::{VarBuilder, VarMap};

use candle_core::Device;
use model::trainer_config::TrainerConfig;
use std::backtrace::Backtrace;
use std::env;

mod agent;
mod game;
mod helper;
mod model;

fn main() {
    let mut action_config = game::action::ActionConfig::new(3, 300, 20, 9);
    // 0.0 values are ignored for raises
    action_config.preflop_raise_sizes = vec![2.0, 3.0, 0.0, 0.0];
    action_config.postflop_raise_sizes = vec![0.25, 0.5, 0.66, 1.0];

    let trainer_config = TrainerConfig {
        learning_rate: 1e-5,
        max_iters: 500000,
        hands_per_player_per_iteration: 350,
        update_step: 10,
        ppo_epsilon: 0.2,
        ppo_delta_1: 3.0,
        no_invalid_for_traverser: false,
        new_agent_interval: 500,
        save_interval: 100,
        agent_count: 2,
        use_epsilon_greedy: true,
        epsilon_greedy_factor: 0.05, // 5% of random actions
        epsilon_greedy_decay: 0.9999,
        use_entropy: false,
        entropy_beta: 0.01,
        agents_device: Device::Cpu,
    };

    let device = Device::cuda_if_available(0).unwrap();

    let args: Vec<String> = env::args().collect();
    let output = &args[1];

    let mut trainer =
        model::trainer::Trainer::new(3, &action_config, &trainer_config, device, output);
    if let Err(err) = trainer.train() {
        println!("Error: {}", err);

        let backtrace = Backtrace::capture();
        println!("Backtrace:\n{:?}", backtrace);
    }
}
