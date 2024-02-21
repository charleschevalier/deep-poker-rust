use candle_core::Var;
use candle_nn::{Optimizer, ParamsAdamW};

// Duplicated AdamW from candle repo to be able to set step_t when
// we resume a training from a checkpoint.

#[derive(Debug)]
struct VarAdamW {
    var: Var,
    first_moment: Var,
    second_moment: Var,
}

#[derive(Debug)]
pub struct AdamWCustom {
    vars: Vec<VarAdamW>,
    step_t: usize,
    params: ParamsAdamW,
}

impl Optimizer for AdamWCustom {
    type Config = ParamsAdamW;

    fn new(vars: Vec<Var>, params: ParamsAdamW) -> Result<Self, candle_core::Error> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .map(|var| {
                let dtype = var.dtype();
                let shape = var.shape();
                let device = var.device();
                let first_moment = Var::zeros(shape, dtype, device)?;
                let second_moment = Var::zeros(shape, dtype, device)?;
                Ok(VarAdamW {
                    var,
                    first_moment,
                    second_moment,
                })
            })
            .collect::<Result<Vec<_>, candle_core::Error>>()?;
        Ok(Self {
            vars,
            params,
            step_t: 0,
        })
    }

    fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr
    }

    fn step(&mut self, grads: &candle_core::backprop::GradStore) -> Result<(), candle_core::Error> {
        self.step_t += 1;
        let lr = self.params.lr;
        let lambda = self.params.weight_decay;
        let lr_lambda = lr * lambda;
        let beta1 = self.params.beta1;
        let beta2 = self.params.beta2;
        let scale_m = 1f64 / (1f64 - beta1.powi(self.step_t as i32));
        let scale_v = 1f64 / (1f64 - beta2.powi(self.step_t as i32));
        for var in self.vars.iter() {
            let theta = &var.var;
            let m = &var.first_moment;
            let v = &var.second_moment;
            if let Some(g) = grads.get(theta) {
                // This involves locking 3 RWLocks per params, if the parameters are large this
                // should not be an issue but this may be problematic with models with lots of
                // small parameters.
                let next_m = ((m.as_tensor() * beta1)? + (g * (1.0 - beta1))?)?;
                let next_v = ((v.as_tensor() * beta2)? + (g.sqr()? * (1.0 - beta2))?)?;
                let m_hat = (&next_m * scale_m)?;
                let v_hat = (&next_v * scale_v)?;
                let next_theta = (theta.as_tensor() * (1f64 - lr_lambda))?;
                let adjusted_grad = (m_hat / (v_hat.sqrt()? + self.params.eps)?)?;
                let next_theta = (next_theta - (adjusted_grad * lr)?)?;
                m.set(&next_m)?;
                v.set(&next_v)?;
                theta.set(&next_theta)?;
            }
        }
        Ok(())
    }
}

impl AdamWCustom {
    pub fn new_lr(vars: Vec<Var>, learning_rate: f64) -> Result<Self, candle_core::Error> {
        let params = ParamsAdamW {
            lr: learning_rate,
            ..ParamsAdamW::default()
        };
        Self::new(vars, params)
    }

    /*pub fn params(&self) -> &ParamsAdamW {
        &self.params
    }

    pub fn set_params(&mut self, params: ParamsAdamW) {
        self.params = params;
    }*/

    pub fn set_step(&mut self, step: usize) {
        self.step_t = step;
    }
}
