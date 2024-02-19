use candle_core::{Module, Tensor};
use candle_nn::{conv2d, linear, Conv2d, Conv2dConfig, Linear, VarBuilder};

pub struct SiameseNetwork {
    card_conv_layer_1: Conv2d,
    card_conv_layer_2: Conv2d,
    action_conv_layer_1: Conv2d,
    action_conv_layer_2: Conv2d,
    merge_layer: Linear,
    output_layer: Linear,
}

impl SiameseNetwork {
    pub fn new(
        player_count: u32,
        action_abstraction_count: u32,
        max_action_per_street_cnt: usize,
        vb: &VarBuilder,
    ) -> Result<SiameseNetwork, Box<dyn std::error::Error>> {
        let conv_factor = 2;

        let card_input_size = (13, 4);

        // Card conv 1 layer output size
        let mut card_conv_out_size = Self::calc_cnn_size_wh(card_input_size, 3, 1, 1);

        // Max pooling layer
        card_conv_out_size = Self::calc_cnn_size_wh(card_conv_out_size, 2, 0, 2);

        // Card conv 2 layer output size
        card_conv_out_size = Self::calc_cnn_size_wh(card_conv_out_size, 3, 1, 1);

        // Get number of outputs
        let final_card_conv_size =
            card_conv_out_size.0 * card_conv_out_size.1 * 6 * conv_factor * conv_factor;

        // Calculate action size
        let action_input_size = (action_abstraction_count as i32, player_count as i32 + 2);

        // Action conv 1 layer output size
        let mut action_conv_out_size = Self::calc_cnn_size_wh(action_input_size, 3, 1, 1);

        // Max pooling layer
        action_conv_out_size = Self::calc_cnn_size_wh(action_conv_out_size, 2, 0, 2);

        // Action conv 2 layer output size
        action_conv_out_size = Self::calc_cnn_size_wh(action_conv_out_size, 3, 1, 1);

        // Get number of outputs
        let final_action_conv_size = action_conv_out_size.0
            * action_conv_out_size.1
            * max_action_per_street_cnt as i32
            * 4
            * conv_factor
            * conv_factor;

        let card_conv_layer_1 = conv2d(
            6,
            6 * conv_factor as usize,
            3,
            Conv2dConfig {
                stride: 1,
                padding: 1,
                dilation: 1,
                groups: 1,
            },
            vb.pp("siamese_card_conv_1"),
        )?;

        let card_conv_layer_2 = conv2d(
            6 * conv_factor as usize,
            6 * conv_factor as usize * conv_factor as usize,
            3,
            Conv2dConfig {
                stride: 1,
                padding: 1,
                dilation: 1,
                groups: 1,
            },
            vb.pp("siamese_card_conv_2"),
        )?;

        let action_conv_layer_1 = conv2d(
            max_action_per_street_cnt * 4,
            max_action_per_street_cnt * 4 * conv_factor as usize,
            3,
            Conv2dConfig {
                stride: 1,
                padding: 1,
                dilation: 1,
                groups: 1,
            },
            vb.pp("siamese_action_conv_1"),
        )?;

        let action_conv_layer_2 = conv2d(
            max_action_per_street_cnt * 4 * conv_factor as usize,
            max_action_per_street_cnt * 4 * conv_factor as usize * conv_factor as usize,
            3,
            Conv2dConfig {
                stride: 1,
                padding: 1,
                dilation: 1,
                groups: 1,
            },
            vb.pp("siamese_action_conv_2"),
        )?;

        let merge_layer = linear(
            final_card_conv_size as usize + final_action_conv_size as usize,
            1024,
            vb.pp("siamese_merge"),
        )?;

        let output_layer = linear(1024, 1024, vb.pp("siamese_output"))?;

        Ok(SiameseNetwork {
            card_conv_layer_1,
            card_conv_layer_2,
            action_conv_layer_1,
            action_conv_layer_2,
            merge_layer,
            output_layer,
        })
    }

    pub fn forward(
        &self,
        card_tensor: &Tensor,
        action_tensor: &Tensor,
    ) -> Result<Tensor, candle_core::Error> {
        // Card Output
        let mut card_output = self.card_conv_layer_1.forward(card_tensor)?;
        // println!("Card output shape 1: {:?}", card_output.shape());
        card_output = card_output.max_pool2d_with_stride(2, 2)?;
        // println!("Card output shape 2: {:?}", card_output.shape());
        card_output = self.card_conv_layer_2.forward(&card_output)?;
        // println!("Card output shape 3: {:?}", card_output.shape());

        // Action Output
        let mut action_output = self.action_conv_layer_1.forward(action_tensor)?;
        // println!("Action output shape 1: {:?}", action_output.shape());
        action_output = action_output.max_pool2d_with_stride(2, 2)?;
        // println!("Action output shape 2: {:?}", action_output.shape());
        action_output = self.action_conv_layer_2.forward(&action_output)?;
        // println!("Action output shape 3: {:?}", action_output.shape());

        let card_output_flat = card_output.flatten(1, 3)?;
        let action_output_flat = action_output.flatten(1, 3)?;

        // println!("Card output flat shape: {:?}", card_output_flat.shape());
        // println!("Action output flat shape: {:?}", action_output_flat.shape());

        let merged = Tensor::cat(&[&card_output_flat, &action_output_flat], 1)?;

        // println!("Merged shape: {:?}", merged.shape());

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
