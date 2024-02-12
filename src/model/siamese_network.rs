use candle_core::{Module, Tensor};
use candle_nn::{conv2d, linear, Conv2d, Conv2dConfig, Linear, VarBuilder};

pub struct SiameseNetwork {
    card_conv: Conv2d,
    action_conv: Conv2d,
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
        // Calculate output shapes
        let card_conv_out_size =
            5 * 8 * Self::calc_cnn_size(13, 3, 1) * Self::calc_cnn_size(4, 3, 1);
        // println!("Card conv out size: {}", card_conv_out_size);

        let action_conv_out_size = max_action_per_street_cnt
            * 4
            * 8
            * Self::calc_cnn_size(action_abstraction_count as usize, 3, 1)
            * Self::calc_cnn_size(player_count as usize + 2, 3, 1);
        // println!("Action conv out size: {}", action_conv_out_size);

        // Define layers dimensions
        let weight_dims: Vec<Vec<usize>> = vec![
            // Card convolution layer
            vec![5 * 8, 5, 3, 3],
            // Action convolution layer
            vec![
                max_action_per_street_cnt * 4 * 8,
                max_action_per_street_cnt * 4,
                3,
                3,
            ],
            // Merge layer
            vec![512, card_conv_out_size + action_conv_out_size],
            // Output layer
            vec![256, 512],
        ];

        // Define card convolution layer
        // Card input shape: 5 channels for hole cards, 3 streets, all cards
        // 52 cards: 13 ranks, 4 suits
        let card_conv = conv2d(
            weight_dims[0][1],
            weight_dims[0][0],
            weight_dims[0][2],
            Conv2dConfig::default(),
            vb.pp("siamese_card_conv_1"),
        )?;

        // Define action convolution layer
        // Action input shape: 4 streets with max_action_per_street_cnt actions each
        // action_abstraction_count possible actions
        // player_count + 2 channels for sum and legal actions
        let action_conv = conv2d(
            weight_dims[1][1],
            weight_dims[1][0],
            weight_dims[1][2],
            Conv2dConfig::default(),
            vb.pp("siamese_action_conv_1"),
        )?;
        // println!("Action conv shape: {:?}", action_conv.weight().shape());

        let merge_layer = linear(weight_dims[2][1], weight_dims[2][0], vb.pp("siamese_merge"))?;

        let output_layer = linear(
            weight_dims[3][1],
            weight_dims[3][0],
            vb.pp("siamese_output"),
        )?;

        Ok(SiameseNetwork {
            card_conv,
            action_conv,
            merge_layer,
            output_layer,
        })
    }

    pub fn forward(
        &self,
        card_tensor: &Tensor,
        action_tensor: &Tensor,
    ) -> Result<Tensor, candle_core::Error> {
        let card_output = self.card_conv.forward(card_tensor)?;

        // println!("Card output shape: {:?}", card_output.shape());

        let action_output = self.action_conv.forward(action_tensor)?;

        // println!("Action output shape: {:?}", action_output.shape());

        let card_output_flat = card_output.flatten(1, 3)?;
        let action_output_flat = action_output.flatten(1, 3)?;

        // println!("Card output flat shape: {:?}", card_output_flat.shape());
        // println!("Action output flat shape: {:?}", action_output_flat.shape());

        let merged = Tensor::cat(&[&card_output_flat, &action_output_flat], 1)?;

        // println!("Merged shape: {:?}", merged.shape());

        let merged_output = self.merge_layer.forward(&merged)?;

        // println!("Merged output shape: {:?}", merged_output.shape());

        self.output_layer.forward(&merged_output)
    }

    fn calc_cnn_size(input_size: usize, kernel_size: usize, stride: usize) -> usize {
        (input_size - kernel_size) / stride + 1
    }
}
