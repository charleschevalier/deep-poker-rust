use candle_core::Module;
use candle_core::Tensor;
use candle_nn::ops::dropout;
use candle_nn::{
    batch_norm, conv2d, linear, BatchNorm, BatchNormConfig, Conv2d, Conv2dConfig, Linear,
    VarBuilder,
};

// NOTE: we would prefer to use Sequence here, but it does not seem to support Send & Sync
// that we need for multi-threading

pub struct SiameseNetwork {
    card_conv_layers: Vec<Conv2d>,
    card_batch_norms: Vec<BatchNorm>,
    action_conv_layers: Vec<Conv2d>,
    action_batch_norms: Vec<BatchNorm>,
    merge_layer: Linear,
    output_layer: Linear,
}

impl Clone for SiameseNetwork {
    fn clone(&self) -> SiameseNetwork {
        // .to_vec() is a deep copy
        SiameseNetwork {
            card_conv_layers: self.card_conv_layers.to_vec(),
            card_batch_norms: self.card_batch_norms.to_vec(),
            action_conv_layers: self.action_conv_layers.to_vec(),
            action_batch_norms: self.action_batch_norms.to_vec(),
            merge_layer: self.merge_layer.clone(),
            output_layer: self.output_layer.clone(),
        }
    }
}

impl SiameseNetwork {
    pub fn new(
        player_count: u32,
        action_abstraction_count: u32,
        max_action_per_street_cnt: usize,
        vb: &VarBuilder,
    ) -> Result<SiameseNetwork, Box<dyn std::error::Error>> {
        let final_conv_features: usize = 128;

        let card_input_size = (13, 4);
        let card_output_size = card_input_size.0 * card_input_size.1 * final_conv_features;

        let action_input_size = (action_abstraction_count as usize, player_count as usize + 2);
        let action_output_size = action_input_size.0 * action_input_size.1 * final_conv_features;

        let card_conv =
            Self::build_conv_layers(6, &[64, 64, 128, final_conv_features], "card", vb)?;
        let action_conv = Self::build_conv_layers(
            max_action_per_street_cnt * 4,
            &[64, 64, 128, final_conv_features],
            "action",
            vb,
        )?;

        let merge_layer = linear(
            card_output_size + action_output_size,
            1024,
            vb.pp("siamese_merge"),
        )?;

        let output_layer = linear(1024, 1024, vb.pp("siamese_output"))?;

        Ok(SiameseNetwork {
            card_conv_layers: card_conv.0,
            card_batch_norms: card_conv.1,
            action_conv_layers: action_conv.0,
            action_batch_norms: action_conv.1,
            merge_layer,
            output_layer,
        })
    }

    pub fn build_conv_layers(
        input_channels: usize,
        features: &[usize],
        prefix: &str,
        vb: &VarBuilder,
    ) -> Result<(Vec<Conv2d>, Vec<BatchNorm>), Box<dyn std::error::Error>> {
        let norm_1 = batch_norm(
            features[0],
            BatchNormConfig::default(),
            vb.pp(format!("siamese_{}_norm_1", prefix)),
        )?;
        let norm_2 = batch_norm(
            features[1],
            BatchNormConfig::default(),
            vb.pp(format!("siamese_{}_norm_2", prefix)),
        )?;
        let norm_3 = batch_norm(
            features[2],
            BatchNormConfig::default(),
            vb.pp(format!("siamese_{}_norm_3", prefix)),
        )?;
        let norm_4 = batch_norm(
            features[3],
            BatchNormConfig::default(),
            vb.pp(format!("siamese_{}_norm_4", prefix)),
        )?;

        let convs = vec![
            conv2d(
                input_channels,
                features[0],
                3,
                Conv2dConfig {
                    stride: 1,
                    padding: 1,
                    dilation: 1,
                    groups: 1,
                },
                vb.pp(format!("siamese_{}_conv_1", prefix)),
            )?,
            conv2d(
                features[0],
                features[1],
                3,
                Conv2dConfig {
                    stride: 1,
                    padding: 1,
                    dilation: 1,
                    groups: 1,
                },
                vb.pp(format!("siamese_{}_conv_2", prefix)),
            )?,
            conv2d(
                features[1],
                features[2],
                3,
                Conv2dConfig {
                    stride: 1,
                    padding: 1,
                    dilation: 1,
                    groups: 1,
                },
                vb.pp(format!("siamese_{}_conv_3", prefix)),
            )?,
            conv2d(
                features[2],
                features[3],
                3,
                Conv2dConfig {
                    stride: 1,
                    padding: 1,
                    dilation: 1,
                    groups: 1,
                },
                vb.pp(format!("siamese_{}_conv_4", prefix)),
            )?,
        ];

        Ok((convs, vec![norm_1, norm_2, norm_3, norm_4]))
    }

    pub fn forward(
        &self,
        card_tensor: &Tensor,
        action_tensor: &Tensor,
        train: bool,
    ) -> Result<Tensor, candle_core::Error> {
        // Card Output
        let mut card_t = self.card_conv_layers[0].forward(card_tensor)?;
        card_t = card_t.apply_t(&self.card_batch_norms[0], train)?;
        card_t.relu()?;
        card_t = self.card_conv_layers[1].forward(&card_t)?;
        card_t = dropout(&card_t, 0.5)?;
        card_t = card_t.apply_t(&self.card_batch_norms[1], train)?;
        card_t.relu()?;

        // TODO: maybe add card_tensor to card_t like in resnet here ?

        card_t = self.card_conv_layers[2].forward(&card_t)?;
        card_t = card_t.apply_t(&self.card_batch_norms[2], train)?;
        card_t.relu()?;
        card_t = self.card_conv_layers[3].forward(&card_t)?;
        card_t = dropout(&card_t, 0.5)?;
        card_t = card_t.apply_t(&self.card_batch_norms[3], train)?;
        card_t.relu()?;

        // Action Output
        let mut action_t = self.action_conv_layers[0].forward(action_tensor)?;
        action_t = action_t.apply_t(&self.action_batch_norms[0], train)?;
        action_t.relu()?;
        action_t = self.action_conv_layers[1].forward(&action_t)?;
        action_t = dropout(&action_t, 0.5)?;
        action_t = action_t.apply_t(&self.action_batch_norms[1], train)?;
        action_t.relu()?;

        // TODO: maybe add action_tensor to action_t like in resnet here ?

        action_t = self.action_conv_layers[2].forward(&action_t)?;
        action_t = action_t.apply_t(&self.action_batch_norms[2], train)?;
        action_t.relu()?;
        action_t = self.action_conv_layers[3].forward(&action_t)?;
        action_t = dropout(&action_t, 0.5)?;
        action_t = action_t.apply_t(&self.action_batch_norms[3], train)?;
        action_t.relu()?;

        let card_output_flat = card_t.flatten(1, 3)?;
        let action_output_flat = action_t.flatten(1, 3)?;
        let merged = Tensor::cat(&[&card_output_flat, &action_output_flat], 1)?;
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
