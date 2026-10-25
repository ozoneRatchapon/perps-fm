//! Feature engineering on the curated 1-minute bars.
//!
//! Hard rule (enforced by tests, see `tests/lookahead.rs`): every feature at
//! row t may only use data with timestamp <= t. Targets use strictly-forward
//! windows and are stored separately from features.

use anyhow::Result;
use polars::prelude::*;
use std::path::Path;

use perpsfm_core::Market;

pub const FEATURE_COLUMNS: &[&str] = &[
    "ts",
    "ret_1m",
    "ret_5m",
    "ret_30m",
    "rv_30m",
    "ofi_30m",
    "vol_z_30m",
    "taker_imbalance_30m",
    "funding_rate",
    "liq_count_60m",
    "liq_notional_60m",
];

pub const TARGET_COLUMNS: &[&str] = &[
    "ts",
    // Forward 30-min return distribution targets (quantile regression)
    "fwd_ret_30m",
    // Forward liquidation-burst indicator (any liquidation in next 60m)
    "fwd_liq_burst_60m",
    // Forward volume burst (next 30m quote volume > 5x trailing 24h median)
    "fwd_vol_burst_30m",
];

/// Load curated candles/trades/funding/liquidations for one market and emit
/// feature + target parquet files joined on the 1-minute grid.
pub fn build_features(root: &Path, market: Market) -> Result<()> {
    let candles = crate::bars::load_candles(root, market)?;
    let _ = candles; // wired up in Phase 2
    anyhow::bail!("build_features: implemented in Phase 2")
}

pub mod bars {
    //! Loading helpers shared by features/train/backtest.

    use super::*;

    pub fn load_candles(root: &Path, market: Market) -> Result<DataFrame> {
        let path = perpsfm_core::candle_path(root, market);
        let file = std::fs::File::open(&path)?;
        let df = ParquetReader::new(file).finish()?;
        Ok(df)
    }
}
