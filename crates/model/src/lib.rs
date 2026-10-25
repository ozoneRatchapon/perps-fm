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

/// Quantiles predicted by every probabilistic head.
pub const QUANTILES: [f64; 5] = [0.1, 0.25, 0.5, 0.75, 0.9];
