use candle_core::Tensor;

pub fn _fast_flatten(tensor: &Tensor) -> Vec<f32> {
    tensor
        .copy()
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

pub fn _check_tensor(tensor: &Tensor) -> Result<(), Box<dyn std::error::Error>> {
    let mut err: String = String::new();

    if _fast_flatten(tensor).iter().any(|x: &f32| x.is_nan()) {
        err += "NaN in tensor\n";
    }

    if err.is_empty() {
        Ok(())
    } else {
        Err(err.into())
    }
}

pub fn filter_var_map_by_prefix(
    var_map: &candle_nn::VarMap,
    prefix: &[&str],
) -> Vec<candle_core::Var> {
    var_map
        .data()
        .lock()
        .unwrap()
        .iter()
        .filter(|(name, _)| prefix.iter().any(|&item| name.starts_with(item)))
        .map(|(_, var)| var.clone())
        .collect::<Vec<candle_core::Var>>()
}
