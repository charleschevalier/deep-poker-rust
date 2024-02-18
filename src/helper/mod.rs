use candle_core::Tensor;

pub fn _check_tensor(tensor: &Tensor) -> Result<(), Box<dyn std::error::Error>> {
    // Check for NaN
    let flat_flat_vec = tensor
        .copy()
        .unwrap()
        .flatten_all()
        .unwrap()
        .as_ref()
        .to_vec1::<f32>()
        .unwrap();

    let mut err: String = String::new();

    // if flat_flat_vec.iter().any(|x: &f32| x.abs() > 50.0) {
    //     err += "HIGH in tensor\n";
    // }

    if flat_flat_vec.iter().any(|x: &f32| x.is_nan()) {
        err += "NaN in tensor\n";
    }

    if err.is_empty() {
        Ok(())
    } else {
        Err(err.into())
    }
}

pub fn _fast_flatten(tensor: &Tensor) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    Ok(tensor
        .copy()
        .unwrap()
        .flatten_all()
        .unwrap()
        .as_ref()
        .to_vec1::<f32>()
        .unwrap())
}
