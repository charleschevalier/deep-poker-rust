use std::collections::HashMap;

use candle_core::Device;
use candle_core::Module;
use candle_core::Tensor;
use candle_nn::conv2d_no_bias;
use candle_nn::BatchNormConfig;
use candle_nn::VarMap;
use candle_nn::{batch_norm, linear, BatchNorm, Conv2d, Conv2dConfig, Linear, VarBuilder};

#[derive(Clone)]
struct BasicBlock {
    conv_1: Conv2d,
    conv_2: Conv2d,
    conv_3: Conv2d,
    bn: [BatchNorm; 3],
    out_channels: usize,
}

// Block a bit like resnet with residual connection but stride 1
impl BasicBlock {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        source_channels: usize,
        vb: VarBuilder,
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
            vb.pp("conv_1"),
        )?;
        let bn_1 = batch_norm(
            out_channels,
            BatchNormConfig {
                eps: 1e-5,
                remove_mean: false,
                affine: true,
                momentum: 0.1,
            },
            vb.pp("bn_1"),
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
            vb.pp("conv_2"),
        )?;
        let bn_2 = batch_norm(
            out_channels,
            BatchNormConfig {
                eps: 1e-5,
                remove_mean: false,
                affine: true,
                momentum: 0.1,
            },
            vb.pp("bn_2"),
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
            vb.pp("conv_3"),
        )?;
        let bn_3 = batch_norm(
            out_channels,
            BatchNormConfig {
                eps: 1e-5,
                remove_mean: false,
                affine: true,
                momentum: 0.1,
            },
            vb.pp("bn_3"),
        )?;

        Ok(BasicBlock {
            conv_1,
            conv_2,
            conv_3,
            bn: [bn_1, bn_2, bn_3],
            out_channels,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        base_input: &Tensor,
        train: bool,
    ) -> Result<Tensor, candle_core::Error> {
        let mut out = self.conv_1.forward(x)?;
        out = out.apply_t(&self.bn[0], train)?;
        out = out.relu()?;

        out = self.conv_2.forward(&out)?;
        out = out.apply_t(&self.bn[1], train)?;
        out = out.relu()?;

        let mut identity = self.conv_3.forward(base_input)?;
        identity = identity.apply_t(&self.bn[2], train)?;

        out = (out + identity)?;
        out = out.relu()?;

        Ok(out)
    }

    fn get_batch_norm_tensors(&self) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        for i in 0..3 {
            map.insert(
                format!("bn_{}.running_mean", i + 1),
                self.bn[i].running_mean().copy().unwrap(),
            );
            map.insert(
                format!("bn_{}.running_var", i + 1),
                self.bn[i].running_var().copy().unwrap(),
            );
            let (weight, bias) = self.bn[i].weight_and_bias().unwrap();
            map.insert(format!("bn_{}.weight", i + 1), weight.copy().unwrap());
            map.insert(format!("bn_{}.bias", i + 1), bias.copy().unwrap());
        }
        map
    }

    fn set_batch_norm_tensors(&mut self, tensors: HashMap<String, Tensor>) {
        for i in 0..3 {
            // let running_mean = tensors[&format!("bn_{}.running_mean", i + 1)]
            //     .copy()
            //     .unwrap();
            // let running_mean_real = self.bn[i].running_mean();

            // // Check if tensors are equal
            // let diff = (running_mean - running_mean_real)
            //     .unwrap()
            //     .abs()
            //     .unwrap()
            //     .sum_all()
            //     .unwrap()
            //     .to_scalar::<f32>()
            //     .unwrap();

            // if diff > 1e-5 {
            //     println!("Running mean for bn_{} is different", i + 1);
            // }

            // let running_var = tensors[&format!("bn_{}.running_var", i + 1)]
            //     .copy()
            //     .unwrap();
            // let running_var_real = self.bn[i].running_var();

            // // Check if tensors are equal
            // let diff = (running_var - running_var_real)
            //     .unwrap()
            //     .abs()
            //     .unwrap()
            //     .sum_all()
            //     .unwrap()
            //     .to_scalar::<f32>()
            //     .unwrap();

            // if diff > 1e-5 {
            //     println!("Running var for bn_{} is different", i + 1);
            // }

            self.bn[i] = BatchNorm::new(
                self.out_channels,
                tensors[&format!("bn_{}.running_mean", i + 1)].clone(),
                tensors[&format!("bn_{}.running_var", i + 1)].clone(),
                tensors[&format!("bn_{}.weight", i + 1)].clone(),
                tensors[&format!("bn_{}.bias", i + 1)].clone(),
                1e-5,
            )
            .unwrap();
        }
    }
}

#[derive(Clone)]
struct SiameseTwin {
    conv_block_1: BasicBlock,
    conv_block_2: BasicBlock,
}

impl SiameseTwin {
    pub fn new(size: &[usize], vb: VarBuilder) -> Result<SiameseTwin, candle_core::Error> {
        let conv_block_1 = BasicBlock::new(size[0], size[1], size[0], vb.pp("twin_1"))?;
        let conv_block_2 = BasicBlock::new(size[1], size[2], size[0], vb.pp("twin_2"))?;

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

    fn get_batch_norm_tensors(&self) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();

        let tensors1 = self.conv_block_1.get_batch_norm_tensors();
        for (k, v) in tensors1 {
            map.insert(format!("twin_1.{}", k), v);
        }
        let tensors2 = self.conv_block_2.get_batch_norm_tensors();
        for (k, v) in tensors2 {
            map.insert(format!("twin_2.{}", k), v);
        }

        map
    }

    fn set_batch_norm_tensors(&mut self, tensors: HashMap<String, Tensor>) {
        let mut tensors1 = HashMap::new();
        let mut tensors2 = HashMap::new();
        for (k, v) in tensors {
            if k.starts_with("twin_1.") {
                tensors1.insert(k[7..].to_string(), v);
            } else if k.starts_with("twin_2.") {
                tensors2.insert(k[7..].to_string(), v);
            }
        }

        self.conv_block_1.set_batch_norm_tensors(tensors1);
        self.conv_block_2.set_batch_norm_tensors(tensors2);
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
        vb: VarBuilder,
    ) -> Result<SiameseNetwork, candle_core::Error> {
        let features_size = [48, 96];

        let card_input_size = (13, 4);
        let card_output_size = card_input_size.0 * card_input_size.1 * features_size[1];

        let action_input_size = (action_abstraction_count as usize, player_count as usize + 2);
        let action_output_size = action_input_size.0 * action_input_size.1 * features_size[1];

        let card_twin =
            SiameseTwin::new(&[6, features_size[0], features_size[1]], vb.pp("card_twin"))?;
        let action_twin = SiameseTwin::new(
            &[
                max_action_per_street_cnt * 4,
                features_size[0],
                features_size[1],
            ],
            vb.pp("action_twin"),
        )?;

        let merge_layer = linear(card_output_size + action_output_size, 512, vb.pp("merge"))?;

        let output_layer = linear(512, 512, vb.pp("output"))?;

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

    pub fn get_batch_norm_tensors(&self) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();

        let card_tensors = self.card_twin.get_batch_norm_tensors();
        for (k, v) in card_tensors {
            map.insert(format!("card_twin.{}", k), v);
        }
        let action_tensors = self.action_twin.get_batch_norm_tensors();
        for (k, v) in action_tensors {
            map.insert(format!("action_twin.{}", k), v);
        }

        map
    }

    pub fn set_batch_norm_tensors(&mut self, tensors: HashMap<String, Tensor>) {
        let mut card_tensors = HashMap::new();
        let mut action_tensors = HashMap::new();
        for (k, v) in tensors {
            if k.starts_with("card_twin.") {
                card_tensors.insert(k[10..].to_string(), v);
            } else if k.starts_with("action_twin.") {
                action_tensors.insert(k[12..].to_string(), v);
            }
        }

        self.card_twin.set_batch_norm_tensors(card_tensors);
        self.action_twin.set_batch_norm_tensors(action_tensors);
    }
}
