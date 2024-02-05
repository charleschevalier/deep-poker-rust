pub struct FakeModel {}

impl FakeModel {
    pub fn get_probabilities(cnt: u32) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        let mut probs = Vec::new();
        for _ in 0..cnt {
            probs.push(1.0 / cnt as f32);
        }
        probs
    }
}
