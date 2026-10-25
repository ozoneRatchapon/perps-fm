//! Event-driven market simulator: taker fees + slippage, position sizing,
//! entry/exit policies driven by model forecasts. Phase 4 fills this in.

/// Fees in basis points (Velocity taker fee observed in fills: ~0.02%).
pub const TAKER_FEE_BPS: f64 = 2.0;

/// Slippage model placeholder — calibrated from fill data in Phase 4.
pub fn slippage_bps(size_quote: f64) -> f64 {
    // Larger trades eat more book; kept simple until calibrated.
    1.0 + (size_quote / 10_000.0).sqrt()
}

pub fn run() -> anyhow::Result<()> {
    anyhow::bail!("backtest: implemented in Phase 4")
}
