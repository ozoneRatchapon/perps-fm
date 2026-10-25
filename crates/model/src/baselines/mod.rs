//! Non-neural baselines: persistence, rolling mean, linear AR. Any learned
//! model must beat these on the same splits to justify its complexity.

use candle_core::{Result, Tensor};

pub struct Persistence;

impl Persistence {
    /// Predicts the next value as the last observed value.
    /// Input (B, T), output (B,) = last column.
    pub fn predict(x: &Tensor) -> Result<Tensor> {
        let t = x.dims()[1] - 1;
        x.narrow(1, t, 1)?.squeeze(1)
    }
}
