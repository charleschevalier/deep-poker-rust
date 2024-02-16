// NOT USED, we use the CNN version instead

use candle_core::{Module, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

pub struct SiameseNetworkLinear {
    card_linear_layer_1: Linear,
    card_linear_layer_2: Linear,
    action_linear_layer_1: Linear,
    action_linear_layer_2: Linear,
    merge_layer: Linear,
    output_layer: Linear,
}

impl SiameseNetworkLinear {
    pub fn new(
        player_count: u32,
        action_abstraction_count: u32,
        max_action_per_street_cnt: usize,
        vb: &VarBuilder,
    ) -> Result<SiameseNetworkLinear, Box<dyn std::error::Error>> {
        let card_input_size = 13 * 4 * 6;
        let action_input_size = action_abstraction_count as usize
            * (player_count as usize + 2)
            * 4
            * max_action_per_street_cnt;

        /*let mut action_vecs: Vec<Vec<Vec<f32>>> =
        vec![
            vec![
                vec![0.0; 3 + action_config.postflop_raise_sizes.len()];
                action_config.player_count as usize + 2
            ];
            4 * action_config.max_actions_per_street as usize
        ]; */

        let card_linear_layer_1 = linear(card_input_size, 1024, vb.pp("siamese_card_linear_1"))?;

        let card_linear_layer_2 = linear(1024, 512, vb.pp("siamese_card_linear_2"))?;

        let action_linear_layer_1 =
            linear(action_input_size, 1024, vb.pp("siamese_action_linear_1"))?;

        let action_linear_layer_2 = linear(1024, 512, vb.pp("siamese_action_linear_2"))?;

        let merge_layer = linear(1024, 1024, vb.pp("siamese_merge"))?;

        let output_layer = linear(1024, 1024, vb.pp("siamese_output"))?;

        Ok(SiameseNetworkLinear {
            card_linear_layer_1,
            card_linear_layer_2,
            action_linear_layer_1,
            action_linear_layer_2,
            merge_layer,
            output_layer,
        })
    }

    pub fn forward(
        &self,
        card_tensor: &Tensor,
        action_tensor: &Tensor,
    ) -> Result<Tensor, candle_core::Error> {
        let mut card_output = self
            .card_linear_layer_1
            .forward(&card_tensor.copy()?.flatten(1, 3)?)?;
        card_output = card_output.relu()?;
        card_output = self.card_linear_layer_2.forward(&card_output)?;
        card_output = card_output.relu()?;
        let mut action_output = self
            .action_linear_layer_1
            .forward(&action_tensor.copy()?.flatten(1, 3)?)?;
        action_output = action_output.relu()?;
        action_output = self.action_linear_layer_2.forward(&action_output)?;
        action_output = action_output.relu()?;

        let merged = Tensor::cat(&[&card_output, &action_output], 1)?;
        let mut merged_output = self.merge_layer.forward(&merged)?;
        merged_output.relu()?;
        merged_output = self.output_layer.forward(&merged_output)?;
        merged_output.relu()?;
        Ok(merged_output)
    }

    fn calc_cnn_size_wh(
        input_size: (i32, i32),
        kernel_size: i32,
        padding: i32,
        stride: i32,
    ) -> (i32, i32) {
        (
            (input_size.0 - kernel_size + 2 * padding) / stride + 1,
            (input_size.1 - kernel_size + 2 * padding) / stride + 1,
        )
    }
}
