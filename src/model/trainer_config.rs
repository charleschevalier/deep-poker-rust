use candle_core::Device;

pub struct TrainerConfig {
    // Learning rate for siamese & critic, actor is 10x smaller
    pub learning_rate: f64,
    // Stop training after this many iterations
    pub max_iters: usize,
    // Number of hands dealt to each player per training iteration
    pub hands_per_player_per_iteration: usize,
    // Number of training loops per rollout
    pub update_step: usize,
    // Trinal-clip PPO epsilon
    pub ppo_epsilon: f32,
    // Trinal-clip PPO delta 1
    pub ppo_delta_1: f32,
    // If true, players will not be allowed to make invalid actions in the
    // game tree.
    // If false, invalid actions have very negative reward
    pub no_invalid_for_traverser: bool,
    // Number of iterations before a new agent is put is the tournament pool
    pub new_agent_interval: u32,
    // Save trained agent every this many iterations
    pub save_interval: u32,
    // Number of different agents used as opponents in the training loop
    pub agent_count: u32,
    // If true, agents will use epsilon-greedy exploration
    pub use_epsilon_greedy: bool,
    pub epsilon_greedy_factor: f32,
    pub epsilon_greedy_decay: f32,
    // If true, agents will use entropy exploration
    pub use_entropy: bool,
    pub entropy_beta: f64,
    // Device used for agents in rollout and tournament
    pub agents_device: Device,
}
