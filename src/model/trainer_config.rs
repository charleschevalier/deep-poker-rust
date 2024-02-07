pub struct TrainerConfig {
    pub max_iters: usize,
    pub max_dataset_cache: usize,
    pub batch_size: usize,
    pub hands_per_player_per_iteration: usize,
    pub update_step: usize,
}
