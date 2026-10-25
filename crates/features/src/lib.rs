//! Feature engineering on the curated 1-minute bars.
//!
//! Hard rule (enforced by tests, see `tests/lookahead.rs`): every feature at
//! minute t may only use data with timestamp <= t. Targets use strictly-forward
//! windows and are stored in a separate file so training code cannot join
//! them by accident.
//!
//! Feature list (all keyed on the regularized 1-minute grid):
//! - ret_1m / ret_5m / ret_30m: log returns of oracle_close
//! - rv_30m: trailing 30-min realized vol (std of 1-min log returns)
//! - ofi_30m: order-flow imbalance (buy vol - sell vol) / total, trailing 30m
//! - vol_z_30m: current-minute quote volume z-scored vs trailing 24h
//! - taker_imbalance_30m: taker buy/sell fill-count balance, trailing 30m
//! - funding_rate: last funding value at or before t (as-of)
//! - liq_count_60m / liq_notional_60m: trailing liquidation activity
//!
//! Targets (strictly forward):
//! - fwd_ret_30m: log return oracle_close t+30 vs t
//! - fwd_liq_burst_60m: 1 if any liquidation in (t, t+60m]
//! - fwd_vol_burst_30m: 1 if forward 30m quote volume > 5x trailing 24h median

pub mod bars;
pub mod rolling;

use anyhow::Result;
use polars::prelude::*;
use std::path::Path;

use perpsfm_core::Market;
use rolling::{RollingCountSum, RollingMeanStd};

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
    "fwd_ret_30m",
    "fwd_liq_burst_60m",
    "fwd_vol_burst_30m",
];

pub fn features_path(root: &Path, market: Market) -> std::path::PathBuf {
    perpsfm_core::market_dir(root, market).join("features.parquet")
}
pub fn targets_path(root: &Path, market: Market) -> std::path::PathBuf {
    perpsfm_core::market_dir(root, market).join("targets.parquet")
}

/// Window sizes in grid minutes.
const W30: usize = 30;
const W60: usize = 60;
const W24H: usize = 1440;
const HORIZON: usize = 30;

/// Build feature + target parquet files for one market.
pub fn build_features(root: &Path, market: Market) -> Result<()> {
    let symbol = market.symbol();
    let candles = bars::load_candles_regularized(root, market)?;
    let trades = bars::load_trades(root, market)?;
    let funding = bars::load_funding(root, market)?;
    let liqs = bars::load_liquidations(root, Some(symbol))?;
    tracing::info!(
        "{symbol}: {} grid minutes, {} trades, {} funding, {} liquidations",
        candles.ts.len(),
        trades.len(),
        funding.len(),
        liqs.len()
    );

    let n = candles.ts.len();
    let ts = &candles.ts;
    let oc = &candles.oracle_close; // continuous backbone series
    let qv = &candles.quote_volume;

    // --- log returns of oracle_close (pure shifts) ---
    let ln_oc: Vec<f64> = oc
        .iter()
        .map(|c| if *c > 0.0 { c.ln() } else { f64::NAN })
        .collect();
    let ret = |lag: usize| -> Vec<f64> {
        (0..n)
            .map(|i| {
                if i >= lag && ln_oc[i].is_finite() && ln_oc[i - lag].is_finite() {
                    ln_oc[i] - ln_oc[i - lag]
                } else {
                    f64::NAN
                }
            })
            .collect()
    };
    let ret_1m = ret(1);
    let ret_5m = ret(5);
    let ret_30m = ret(30);

    // --- rv_30m: trailing std of 1-min returns (gaps treated as 0 return) ---
    let mut rv_30m = vec![f64::NAN; n];
    let mut rvs = RollingMeanStd::new(W30);
    for i in 0..n {
        if let Some((_, sd)) = rvs.push(if ret_1m[i].is_finite() {
            ret_1m[i]
        } else {
            0.0
        }) {
            rv_30m[i] = sd;
        }
    }

    // --- per-minute trade flow (sparse: most minutes are empty) ---
    let mut buy_vol = vec![0.0; n];
    let mut sell_vol = vec![0.0; n];
    let mut buy_cnt = vec![0.0; n];
    let mut sell_cnt = vec![0.0; n];
    if n > 0 {
        let grid_start = ts[0];
        for t in &trades {
            let s = ((t.ts - grid_start) / 60).clamp(0, n as i64 - 1) as usize;
            match t.taker_direction.as_deref() {
                Some("long") => {
                    buy_vol[s] += t.quote_amount;
                    buy_cnt[s] += 1.0;
                }
                Some("short") => {
                    sell_vol[s] += t.quote_amount;
                    sell_cnt[s] += 1.0;
                }
                _ => {}
            }
        }
    }

    let mut ofi_30m = vec![0.0; n];
    let mut taker_imbalance_30m = vec![0.0; n];
    let mut vol_z_30m = vec![f64::NAN; n];
    let mut vol_stats = RollingMeanStd::new(W24H);
    let mut bvol = rolling::RollingCountSum::new(W30);
    let mut svol = rolling::RollingCountSum::new(W30);
    let mut bcnt = rolling::RollingCountSum::new(W30);
    let mut scnt = rolling::RollingCountSum::new(W30);
    for i in 0..n {
        if let Some((m, sd)) = vol_stats.push(qv[i]) {
            vol_z_30m[i] = if sd > 0.0 { (qv[i] - m) / sd } else { 0.0 };
        }
        // sums of non-negative quantities; clamp away f64 drift near zero
        let bv = bvol.push(buy_vol[i]).unwrap_or((0, 0.0)).1.max(0.0);
        let sv = svol.push(sell_vol[i]).unwrap_or((0, 0.0)).1.max(0.0);
        let bc = bcnt.push(buy_cnt[i]).unwrap_or((0, 0.0)).1.max(0.0);
        let sc = scnt.push(sell_cnt[i]).unwrap_or((0, 0.0)).1.max(0.0);
        let denom = bv + sv;
        ofi_30m[i] = if denom > 0.0 { (bv - sv) / denom } else { 0.0 };
        let cden = bc + sc;
        taker_imbalance_30m[i] = if cden > 0.0 { (bc - sc) / cden } else { 0.0 };
    }

    // --- funding as-of (last record at or before t) ---
    let mut funding_rate = vec![0.0f64; n];
    {
        let mut f = 0usize;
        for i in 0..n {
            while f < funding.len() && funding[f].ts <= ts[i] {
                f += 1;
            }
            if f > 0 {
                funding_rate[i] = funding[f - 1].funding_rate;
            }
        }
    }

    // --- liquidations: scatter into per-minute notional, then rolling stats ---
    let mut liq_amt = vec![0.0f64; n];
    if n > 0 {
        let grid_start = ts[0];
        for l in &liqs {
            let s = ((l.ts - grid_start) / 60).clamp(0, n as i64 - 1) as usize;
            liq_amt[s] += l.quote_amount.unwrap_or(0.0).abs();
        }
    }
    let mut liq_count_60m = vec![0.0; n];
    let mut liq_notional_60m = vec![0.0; n];
    let mut lcs = RollingCountSum::new(W60);
    for i in 0..n {
        if let Some((c, s)) = lcs.push(liq_amt[i]) {
            liq_count_60m[i] = c as f64;
            liq_notional_60m[i] = s;
        }
    }

    // --- targets, strictly forward ---
    let mut fwd_ret_30m = vec![f64::NAN; n];
    for i in 0..n {
        if i + HORIZON < n && ln_oc[i].is_finite() && ln_oc[i + HORIZON].is_finite() {
            fwd_ret_30m[i] = ln_oc[i + HORIZON] - ln_oc[i];
        }
    }

    // fwd_liq_burst_60m: any liquidation in minutes (t, t+60]
    let mut fwd_liq_burst_60m = vec![0.0; n];
    {
        let mut prefix = vec![0.0f64; n + 1]; // prefix[s] = total liq amount minutes < s
        for i in 0..n {
            prefix[i + 1] = prefix[i] + liq_amt[i];
        }
        for i in 0..n {
            let hi = (i + W60).min(n); // exclusive upper bound on minute index
                                       // strictly after t: sum of minutes t+1 ..= min(t+60, n-1)
            let future_amt = prefix[hi.min(n)] - prefix[(i + 1).min(n)];
            fwd_liq_burst_60m[i] = if future_amt > 0.0 { 1.0 } else { 0.0 };
        }
    }

    // fwd_vol_burst_30m: forward 30m volume > 5x trailing 24h median volume
    let mut fwd_vol_burst_30m = vec![0.0; n];
    {
        let mut suffix = vec![0.0f64; n + 1];
        for i in (0..n).rev() {
            suffix[i] = suffix[i + 1] + qv[i];
        }
        let mut window: Vec<f64> = Vec::with_capacity(W24H + 1); // kept sorted
        for i in 0..n {
            let v = qv[i];
            let pos = window.partition_point(|&x| x < v);
            window.insert(pos, v);
            if window.len() > W24H {
                let old = qv[i - W24H];
                let rpos = window.partition_point(|&x| x < old);
                window.remove(rpos.min(window.len() - 1));
            }
            if i + HORIZON < n && window.len() >= W30 {
                // forward 30m MEAN per-minute volume vs 5x trailing 24h median
                let fwd_mean = (suffix[i + 1] - suffix[i + 1 + HORIZON]) / HORIZON as f64;
                let m = window[window.len() / 2]; // median of trailing 24h
                fwd_vol_burst_30m[i] = if m > 0.0 && fwd_mean > 5.0 * m {
                    1.0
                } else {
                    0.0
                };
            }
        }
    }

    write_features(
        root,
        market,
        &FeatureFrame {
            ts: ts.clone(),
            ret_1m,
            ret_5m,
            ret_30m,
            rv_30m,
            ofi_30m,
            vol_z_30m,
            taker_imbalance_30m,
            funding_rate,
            liq_count_60m,
            liq_notional_60m,
            fwd_ret_30m,
            fwd_liq_burst_60m,
            fwd_vol_burst_30m,
        },
    )?;
    tracing::info!("{symbol}: features written");
    Ok(())
}

pub struct FeatureFrame {
    pub ts: Vec<i64>,
    // features
    pub ret_1m: Vec<f64>,
    pub ret_5m: Vec<f64>,
    pub ret_30m: Vec<f64>,
    pub rv_30m: Vec<f64>,
    pub ofi_30m: Vec<f64>,
    pub vol_z_30m: Vec<f64>,
    pub taker_imbalance_30m: Vec<f64>,
    pub funding_rate: Vec<f64>,
    pub liq_count_60m: Vec<f64>,
    pub liq_notional_60m: Vec<f64>,
    // targets
    pub fwd_ret_30m: Vec<f64>,
    pub fwd_liq_burst_60m: Vec<f64>,
    pub fwd_vol_burst_30m: Vec<f64>,
}

fn write_features(root: &Path, market: Market, f: &FeatureFrame) -> Result<()> {
    let fdf = df! {
        "ts" => f.ts.clone(),
        "ret_1m" => f.ret_1m.clone(),
        "ret_5m" => f.ret_5m.clone(),
        "ret_30m" => f.ret_30m.clone(),
        "rv_30m" => f.rv_30m.clone(),
        "ofi_30m" => f.ofi_30m.clone(),
        "vol_z_30m" => f.vol_z_30m.clone(),
        "taker_imbalance_30m" => f.taker_imbalance_30m.clone(),
        "funding_rate" => f.funding_rate.clone(),
        "liq_count_60m" => f.liq_count_60m.clone(),
        "liq_notional_60m" => f.liq_notional_60m.clone(),
    }
    .map_err(|e| anyhow::anyhow!("features df: {e}"))?;
    let tdf = df! {
        "ts" => f.ts.clone(),
        "fwd_ret_30m" => f.fwd_ret_30m.clone(),
        "fwd_liq_burst_60m" => f.fwd_liq_burst_60m.clone(),
        "fwd_vol_burst_30m" => f.fwd_vol_burst_30m.clone(),
    }
    .map_err(|e| anyhow::anyhow!("targets df: {e}"))?;

    for (path, mut df) in [
        (features_path(root, market), fdf),
        (targets_path(root, market), tdf),
    ] {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("parquet.tmp");
        let file = std::fs::File::create(&tmp)?;
        ParquetWriter::new(file).finish(&mut df)?;
        std::fs::rename(&tmp, path)?;
    }
    Ok(())
}
