//! Sequence models in candle, written from scratch (no pretrained weights):
//! - `TinyTransformer`: causal decoder with multi-market embedding and a
//!   quantile head (Phase 3)
//! - `S4d`: single-layer structured state-space model for comparison
//! - `baselines`: persistence / rolling-mean / linear AR
//!
//! Phase 3 fills the implementations; the module tree is set up now.

pub mod baselines;
pub mod s4d;
pub mod transformer;

use candle_core::Result;

/// Quantiles predicted by every probabilistic head.
pub const QUANTILES: [f64; 5] = [0.1, 0.25, 0.5, 0.75, 0.9];

/// Shared output contract so train/predict are model-agnostic.
pub trait QuantileNet {
    fn forward(
        &self,
        x: &candle_core::Tensor,
        market_ids: &candle_core::Tensor,
    ) -> Result<transformer::TransformerOutput>;
}

impl QuantileNet for transformer::TinyTransformer {
    fn forward(
        &self,
        x: &candle_core::Tensor,
        market_ids: &candle_core::Tensor,
    ) -> Result<transformer::TransformerOutput> {
        transformer::TinyTransformer::forward(self, x, market_ids)
    }
}
