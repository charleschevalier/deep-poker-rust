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
    ) -> SiameseNetwork {
        // Define card convolution layer
        // Card input shape: 5 channels for hole cards, 3 streets, all cards
        // 52 cards: 13 ranks, 4 suits
        let card_conv = conv2d(5, 5 * 8, 3, Conv2dConfig::default(), vb.pp("card_conv_1")).unwrap();

        // Define action convolution layer
        // Action input shape: 4 streets with max_action_per_street_cnt actions each
        // action_abstraction_count possible actions
        // player_count + 2 channels for sum and legal actions
        let action_conv = conv2d(
            max_action_per_street_cnt * 4,
            max_action_per_street_cnt * 4 * 8,
            3,
            Conv2dConfig::default(),
            vb.pp("action_conv_1"),
        )
        .unwrap();

        // Calculate output shapes
        let card_conv_out_size =
            5 * 8 * Self::calc_cnn_size(13, 3, 1) * Self::calc_cnn_size(4, 3, 1);
        println!("Card conv out size: {}", card_conv_out_size);
        let action_conv_out_size = max_action_per_street_cnt
            * 4
            * 8
            * Self::calc_cnn_size(action_abstraction_count as usize, 3, 1)
            * Self::calc_cnn_size(player_count as usize + 2, 3, 1);
        println!("Action conv out size: {}", action_conv_out_size);

        let merge_layer = linear(
            card_conv_out_size + action_conv_out_size,
            4096,
            vb.pp("merged"),
        )
        .unwrap();

        let output_layer = linear(4096, 4096, vb.pp("output_layer")).unwrap();

        SiameseNetwork {
            card_conv,
            action_conv,
            merge_layer,
            output_layer,
        }
    }

    pub fn forward(&self, card_tensor: &Tensor, action_tensor: &Tensor) -> Tensor {
        let card_output = self.card_conv.forward(&card_tensor).unwrap();

        // Print card_output dims
        for dim in card_output.shape().dims() {
            println!("Card output dim: {}", dim);
        }

        let action_output = self.action_conv.forward(&action_tensor).unwrap();

        // Print action_output dims
        for dim in action_output.shape().dims() {
            println!("Action output dim: {}", dim);
        }

        let card_output_flat = card_output.flatten(1, 3).unwrap();
        let action_output_flat = action_output.flatten(1, 3).unwrap();

        println!(
            "card_output_flat dims: {:?}",
            card_output_flat.shape().dims()
        );
        println!(
            "action_output_flat dims: {:?}",
            action_output_flat.shape().dims()
        );

        let merged = Tensor::cat(&[&card_output_flat, &action_output_flat], 1).unwrap();

        println!("merged dims: {:?}", merged.shape().dims());

        let merged_output = self.merge_layer.forward(&merged).unwrap();
        let res = self.output_layer.forward(&merged_output).unwrap();

        // Print res dims
        for dim in res.shape().dims() {
            println!("Res dim: {}", dim);
        }

        return res;
    }

    fn calc_cnn_size(input_size: usize, kernel_size: usize, stride: usize) -> usize {
        return (input_size - kernel_size) / stride + 1;
    }
}
