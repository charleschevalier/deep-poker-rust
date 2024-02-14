pub struct TrainerConfig {
    pub max_iters: usize,
    pub hands_per_player_per_iteration: usize,
    pub update_step: usize,
    pub ppo_epsilon: f32,
    pub ppo_delta_1: f32,
    pub no_invalid_for_traverser: bool,
}
