use candle_core::Device;

pub struct TrainerConfig {
    pub learning_rate: f64,
    pub max_iters: usize,
    pub hands_per_player_per_iteration: usize,
    pub update_step: usize,
    pub ppo_epsilon: f32,
    pub ppo_delta_1: f32,
    pub no_invalid_for_traverser: bool,
    pub new_agent_interval: u32,
    pub save_interval: u32,
    pub agent_count: u32,
    pub use_epsilon_greedy: bool,
    pub epsilon_greedy_factor: f32,
    pub epsilon_greedy_decay: f32,
    pub use_entropy: bool,
    pub entropy_beta: f64,
    pub agents_device: Device,
    pub agents_iterations_per_match: u32,
}
