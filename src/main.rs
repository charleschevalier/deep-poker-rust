mod game;

fn main() {
    println!("Hello, world!");
    let mut action_config = game::action::ActionConfig::new();
    action_config.length = 3;
    action_config.preflop_raise_sizes = vec![2.0, 3.0];
    action_config.postflop_raise_sizes = vec![0.25, 0.5, 0.66, 1.0];

    let mut tree = game::tree::Tree::new(2, 100, 1, game::action::ActionConfig::new());
    tree.play_hand();
}
