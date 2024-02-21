use candle_core::Module;
use candle_core::Tensor;
use candle_nn::conv2d_no_bias;
use candle_nn::{
    batch_norm, linear, BatchNorm, BatchNormConfig, Conv2d, Conv2dConfig, Linear, VarBuilder,
};

#[derive(Clone)]
struct BasicBlock {
    conv_1: Conv2d,
    bn_1: BatchNorm,
    conv_2: Conv2d,
    bn_2: BatchNorm,
    conv_3: Conv2d,
    bn_3: BatchNorm,
}

// Block a bit like resnet with residual connection but stride 1
impl BasicBlock {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        source_channels: usize,
        prefix: &str,
        vb: &VarBuilder,
    ) -> Result<BasicBlock, candle_core::Error> {
        let conv_1 = conv2d_no_bias(
            in_channels,
            out_channels,
            3,
            Conv2dConfig {
                stride: 1,
                padding: 1,
                dilation: 1,
                groups: 1,
            },
            vb.pp(format!("siamese_basic_block_{}_conv_1", prefix)),
        )?;
        let bn_1 = batch_norm(
            out_channels,
            BatchNormConfig::default(),
            vb.pp(format!("siamese_basic_block_{}_bn_1", prefix)),
        )?;
        let conv_2 = conv2d_no_bias(
            out_channels,
            out_channels,
            3,
            Conv2dConfig {
                stride: 1,
                padding: 1,
                dilation: 1,
                groups: 1,
            },
            vb.pp(format!("siamese_basic_block_{}_conv_2", prefix)),
        )?;
        let bn_2 = batch_norm(
            out_channels,
            BatchNormConfig::default(),
            vb.pp(format!("siamese_basic_block_{}_bn_2", prefix)),
        )?;
        let conv_3 = conv2d_no_bias(
            source_channels,
            out_channels,
            1,
            Conv2dConfig {
                stride: 1,
                padding: 0,
                dilation: 1,
                groups: 1,
            },
            vb.pp(format!("siamese_basic_block_{}_conv_3", prefix)),
        )?;
        let bn_3 = batch_norm(
            out_channels,
            BatchNormConfig::default(),
            vb.pp(format!("siamese_basic_block_{}_bn_3", prefix)),
        )?;

        Ok(BasicBlock {
            conv_1,
            bn_1,
            conv_2,
            bn_2,
            conv_3,
            bn_3,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        base_input: &Tensor,
        train: bool,
    ) -> Result<Tensor, candle_core::Error> {
        let mut out = self.conv_1.forward(x)?;
        out = out.apply_t(&self.bn_1, train)?;
        out = out.relu()?;

        out = self.conv_2.forward(&out)?;
        out = out.apply_t(&self.bn_2, train)?;
        out = out.relu()?;

        let mut identity = self.conv_3.forward(base_input)?;
        identity = identity.apply_t(&self.bn_3, train)?;

        out = (out + identity)?;
        out = out.relu()?;

        Ok(out)
    }
}

#[derive(Clone)]
struct SiameseTwin {
    conv_block_1: BasicBlock,
    conv_block_2: BasicBlock,
}

impl SiameseTwin {
    pub fn new(
        size: &[usize],
        prefix: &str,
        vb: &VarBuilder,
    ) -> Result<SiameseTwin, candle_core::Error> {
        let conv_block_1 = BasicBlock::new(
            size[0],
            size[1],
            size[0],
            format!("{}_1", prefix).as_str(),
            vb,
        )?;
        let conv_block_2 = BasicBlock::new(
            size[1],
            size[2],
            size[0],
            format!("{}_2", prefix).as_str(),
            vb,
        )?;

        Ok(SiameseTwin {
            conv_block_1,
            conv_block_2,
        })
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor, candle_core::Error> {
        let mut out = self.conv_block_1.forward(x, x, train)?;
        out = self.conv_block_2.forward(&out, x, train)?;
        out = out.avg_pool2d((1, 1))?;
        Ok(out)
    }
}

#[derive(Clone)]
pub struct SiameseNetwork {
    card_twin: SiameseTwin,
    action_twin: SiameseTwin,
    merge_layer: Linear,
    output_layer: Linear,
}

impl SiameseNetwork {
    pub fn new(
        player_count: u32,
        action_abstraction_count: u32,
        max_action_per_street_cnt: usize,
        vb: &VarBuilder,
    ) -> Result<SiameseNetwork, candle_core::Error> {
        let features_size = [48, 96];

        let card_input_size = (13, 4);
        let card_output_size = card_input_size.0 * card_input_size.1 * features_size[1];

        let action_input_size = (action_abstraction_count as usize, player_count as usize + 2);
        let action_output_size = action_input_size.0 * action_input_size.1 * features_size[1];

        let card_twin = SiameseTwin::new(&[6, features_size[0], features_size[1]], "card", vb)?;
        let action_twin = SiameseTwin::new(
            &[
                max_action_per_street_cnt * 4,
                features_size[0],
                features_size[1],
            ],
            "action",
            vb,
        )?;

        let merge_layer = linear(
            card_output_size + action_output_size,
            512,
            vb.pp("siamese_merge"),
        )?;

        let output_layer = linear(512, 512, vb.pp("siamese_output"))?;

        Ok(SiameseNetwork {
            card_twin,
            action_twin,
            merge_layer,
            output_layer,
        })
    }

    pub fn forward(
        &self,
        card_tensor: &Tensor,
        action_tensor: &Tensor,
        train: bool,
    ) -> Result<Tensor, candle_core::Error> {
        // Card Output
        let mut card_t = self.card_twin.forward(card_tensor, train)?;
        card_t = card_t.flatten(1, 3)?;

        let mut action_t = self.action_twin.forward(action_tensor, train)?;
        action_t = action_t.flatten(1, 3)?;

        let merged = Tensor::cat(&[&card_t, &action_t], 1)?;
        let mut output = self.merge_layer.forward(&merged)?;
        output = output.relu()?;
        output = self.output_layer.forward(&output)?;
        output = output.relu()?;

        Ok(output)
    }
}
