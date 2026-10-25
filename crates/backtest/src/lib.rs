//! Event-driven market simulator: entry/exit/sizing policies driven by model
//! forecasts, executed after taker fees + slippage (JD: "tested after costs
//! in a market simulator").
//!
//! Alignment contract: a prediction at minute t uses only data <= t and
//! targets the return close[t] -> close[t+30]. Entering at close[t] with a
//! 30-minute hold therefore realizes exactly the labeled horizon.

pub mod metrics;
pub mod policy;
pub mod sim;

use anyhow::Result;
use polars::prelude::*;
use std::path::Path;

use crate::policy::{DecisionCtx, Policy};
use perpsfm_core::Market;
use sim::{Action, SimState};

/// Fee in bps per taker side (observed taker_fee / quote_amount in fills).
pub const TAKER_FEE_BPS: f64 = 2.0;
/// Base slippage in bps per taker side (calibration stub; see docs).
pub const BASE_SLIPPAGE_BPS: f64 = 1.0;

pub fn predictions_path(root: &Path, market: Market) -> std::path::PathBuf {
    perpsfm_core::predictions_path(root, market)
}

pub struct Bars {
    pub ts: Vec<i64>,
    pub close: Vec<f64>,
}

/// Load oracle-close bars (continuous) for a market.
pub fn load_bars(root: &Path, market: Market) -> Result<Bars> {
    let path = perpsfm_core::candle_path(root, market);
    let file = std::fs::File::open(&path)?;
    let df = ParquetReader::new(file).finish()?;
    let ts: Vec<i64> = df.column("ts")?.i64()?.into_no_null_iter().collect();
    let close: Vec<f64> = df
        .column("oracle_close")?
        .f64()?
        .into_no_null_iter()
        .collect();
    Ok(Bars { ts, close })
}

/// Load model predictions (ts + quantiles) written by `perpsfm predict`.
pub fn load_predictions(root: &Path, market: Market) -> Result<DataFrame> {
    let path = predictions_path(root, market);
    let file = std::fs::File::open(&path).map_err(|e| {
        anyhow::anyhow!("open {}: {e} (run `perpsfm predict` first)", path.display())
    })?;
    ParquetReader::new(file)
        .finish()
        .map_err(|e| anyhow::anyhow!("{e}"))
}

/// Run one policy over one market's bars + predictions; returns the equity
/// curve and trade log. `start_ts` bounds evaluation (val OR sealed).
pub fn run_policy<P: Policy>(
    bars: &Bars,
    preds_ts: &[i64],
    preds_q: &[Vec<f64>], // aligned with preds_ts, ascending quantiles
    policy: &mut P,
    start_ts: i64,
    end_ts: i64,
) -> Result<SimState> {
    let n = bars.ts.len();
    anyhow::ensure!(n > 0, "no bars");
    let mut sim = SimState::new(TAKER_FEE_BPS + BASE_SLIPPAGE_BPS);
    let mut pi = 0usize; // prediction cursor

    for i in 0..n {
        let t = bars.ts[i];
        if t < start_ts || t >= end_ts {
            continue;
        }
        // advance prediction cursor: last prediction at or before t
        while pi < preds_ts.len() && preds_ts[pi] <= t {
            pi += 1;
        }
        let quantiles: &[f64] = if pi > 0 { &preds_q[pi - 1] } else { &[] };
        let ctx = DecisionCtx {
            ts: t,
            close: bars.close[i],
            quantiles,
            position: sim.position_units,
            equity: sim.equity,
        };
        let prev = if i > 0 {
            bars.close[i - 1]
        } else {
            bars.close[i]
        };
        let action = policy.decide(ctx);
        sim.apply(action, bars.close[i], prev)?;
    }
    Ok(sim)
}

/// CLI entry: backtest the threshold policy on predictions for all markets
/// that have a predictions file, printing after-cost metrics with block
/// bootstrap CIs.
pub fn report(root: &Path, min_edge: f64, exposure: f64) -> Result<()> {
    println!(
        "policy=threshold min_edge={min_edge} exposure={exposure} costs=taker {:.0}bps+slip {:.0}bps",
        TAKER_FEE_BPS, BASE_SLIPPAGE_BPS
    );
    println!(
        "{:<10} {:>6} {:>10} {:>8} {:>8} {:>7}  {:>22}",
        "market", "trades", "return", "sharpe", "mdd", "hit%", "daily-ret 95% CI"
    );
    for market in Market::ALL {
        let Ok(preds) = load_predictions(root, market) else {
            continue;
        };
        let bars = load_bars(root, market)?;
        let ts: Vec<i64> = preds.column("ts")?.i64()?.into_no_null_iter().collect();
        let qs: Vec<Vec<f64>> = (0..5)
            .map(|qi| {
                preds
                    .column(&format!("q{qi}"))
                    .or_else(|_| preds.column(&format!("q{qi}")))
                    .expect("quantile column")
                    .f64()
                    .map(|c| c.into_no_null_iter().collect())
                    .unwrap_or_default()
            })
            .collect();
        let n = ts.len();
        let transposed: Vec<Vec<f64>> =
            (0..n).map(|i| (0..5).map(|q| qs[q][i]).collect()).collect();
        if n == 0 {
            continue;
        }
        let start_ts = ts[0];
        let end_ts = ts[n - 1] + 60;
        let mut pol = policy::ThresholdPolicy::new(min_edge, 0.05, exposure, 30);
        let st = run_policy(&bars, &ts, &transposed, &mut pol, start_ts, end_ts)?;
        let rets = metrics::minute_returns(&st.equity_curve);
        let bs = perpsfm_eval::BlockBootstrap::default();
        let (point, lo, hi) = bs.mean_ci(&rets);
        let total = metrics::total_return(&st.equity_curve);
        let sharpe = metrics::sharpe(&rets);
        let mdd = metrics::max_drawdown(&st.equity_curve);
        println!(
            "{:<10} {:>6} {:>9.2}% {:>8.2} {:>8.2} {:>7.1}  {:>6.2e} [{:>6.2e}, {:>6.2e}]",
            market.symbol(),
            st.trades,
            total * 100.0,
            sharpe,
            mdd * 100.0,
            if st.trades > 0 { 100.0 } else { 0.0 },
            point,
            lo,
            hi
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::ThresholdPolicy;

    fn flat_bars(n: usize, price: f64) -> Bars {
        Bars {
            ts: (0..n as i64).map(|i| 1_700_000_000 + i * 60).collect(),
            close: vec![price; n],
        }
    }

    #[test]
    fn long_round_trip_pays_costs() {
        // rising market: enter long at t=0, hold to the end
        let n = 120;
        let mut bars = flat_bars(n, 100.0);
        for i in 0..n {
            bars.close[i] = 100.0 + i as f64 * 0.1; // +12% over 2h
        }
        let preds_ts = bars.ts.clone();
        let preds_q: Vec<Vec<f64>> = bars.ts.iter().map(|_| vec![0.01; 5]).collect();
        let mut pol = ThresholdPolicy::new(0.005, 0.0, 1.0, 30);
        let st = run_policy(&bars, &preds_ts, &preds_q, &mut pol, 0, i64::MAX).unwrap();
        assert!(st.position_units > 0.0, "should be long");
        assert!(st.total_cost > 0.0, "costs must be charged");
        assert!(st.equity > 100.0, "rising market after-cost should profit");
    }

    #[test]
    fn no_signal_no_trades() {
        let bars = flat_bars(60, 100.0);
        let preds_ts = bars.ts.clone();
        let preds_q: Vec<Vec<f64>> = bars.ts.iter().map(|_| vec![0.0; 5]).collect();
        let mut pol = ThresholdPolicy::new(0.005, 0.0, 1.0, 30);
        let st = run_policy(&bars, &preds_ts, &preds_q, &mut pol, 0, i64::MAX).unwrap();
        assert_eq!(st.trades, 0);
        assert_eq!(st.equity, 100.0);
    }
}
