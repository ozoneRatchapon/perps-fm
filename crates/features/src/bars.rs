//! Loading helpers shared by features/train/backtest.

use anyhow::{Context, Result};
use polars::prelude::*;
use std::path::Path;

use perpsfm_core::Market;

/// Regularized candle grid: one row per calendar minute from first to last
/// observed bar, oracle prices forward-filled (trailing — safe), volumes 0,
/// fill prices NaN when the minute had no print.
pub struct RegularizedBars {
    pub ts: Vec<i64>,
    pub oracle_open: Vec<f64>,
    pub oracle_high: Vec<f64>,
    pub oracle_low: Vec<f64>,
    pub oracle_close: Vec<f64>,
    pub fill_close: Vec<f64>,
    pub quote_volume: Vec<f64>,
    pub base_volume: Vec<f64>,
}

pub fn load_candles(root: &Path, market: Market) -> Result<DataFrame> {
    let path = perpsfm_core::candle_path(root, market);
    let file = std::fs::File::open(&path).with_context(|| format!("open {}", path.display()))?;
    anyhow::Context::context(ParquetReader::new(file).finish(), "read candles parquet")
}

pub fn load_candles_regularized(root: &Path, market: Market) -> Result<RegularizedBars> {
    let df = load_candles(root, market)?;
    let ts: Vec<i64> = df.column("ts")?.i64()?.into_no_null_iter().collect();
    anyhow::ensure!(!ts.is_empty(), "no candles for {market:?}");
    let col_f64 = |name: &str| -> Result<Vec<f64>> {
        Ok(df.column(name)?.f64()?.into_no_null_iter().collect())
    };
    let fill_close = col_f64("fill_close")?;

    let start = ts[0];
    let end = ts[ts.len() - 1];
    anyhow::ensure!(end >= start, "candles not sorted for {market:?}");
    let n = ((end - start) / 60 + 1) as usize;
    anyhow::ensure!(
        n <= ts.len() * 10 + 10,
        "grid exploded: {} minutes for {} bars — data likely malformed",
        n,
        ts.len()
    );

    let mut out = RegularizedBars {
        ts: (0..n as i64).map(|i| start + i * 60).collect(),
        oracle_open: vec![f64::NAN; n],
        oracle_high: vec![f64::NAN; n],
        oracle_low: vec![f64::NAN; n],
        oracle_close: vec![f64::NAN; n],
        fill_close: vec![f64::NAN; n],
        quote_volume: vec![0.0; n],
        base_volume: vec![0.0; n],
    };

    let get = |name: &str| -> Result<Vec<f64>> { col_f64(name) };
    let oo = get("oracle_open")?;
    let oh = get("oracle_high")?;
    let ol = get("oracle_low")?;
    let ocl = get("oracle_close")?;
    let qv = get("quote_volume")?;
    let bv = get("base_volume")?;

    for (k, &t) in ts.iter().enumerate() {
        let i = ((t - start) / 60) as usize; // ts sorted, unique
        out.oracle_open[i] = oo[k];
        out.oracle_high[i] = oh[k];
        out.oracle_low[i] = ol[k];
        out.oracle_close[i] = ocl[k];
        out.fill_close[i] = fill_close[k];
        out.quote_volume[i] = qv[k];
        out.base_volume[i] = bv[k];
    }

    // Forward-fill oracle prices (trailing fill — no future leakage).
    let ffill = |v: &mut Vec<f64>| {
        let mut last = f64::NAN;
        for x in v.iter_mut() {
            if x.is_finite() {
                last = *x;
            } else {
                *x = last;
            }
        }
    };
    ffill(&mut out.oracle_open);
    ffill(&mut out.oracle_high);
    ffill(&mut out.oracle_low);
    ffill(&mut out.oracle_close);
    // fill_close stays NaN where nothing printed — volume features use volume.

    Ok(out)
}

macro_rules! c_i64 {
    ($df:expr,$n:literal) => {
        $df.column($n)?
            .i64()?
            .into_no_null_iter()
            .collect::<Vec<i64>>()
    };
}
macro_rules! c_f64 {
    ($df:expr,$n:literal) => {
        $df.column($n)?
            .f64()?
            .into_no_null_iter()
            .collect::<Vec<f64>>()
    };
}
macro_rules! c_str {
    ($df:expr,$n:literal) => {
        $df.column($n)?
            .str()?
            .iter()
            .map(|v| v.unwrap_or_default().to_string())
            .collect::<Vec<String>>()
    };
}

pub fn load_trades(root: &Path, market: Market) -> Result<Vec<perpsfm_core::Trade>> {
    let path = perpsfm_core::trades_path(root, market);
    if !path.exists() {
        return Ok(Vec::new());
    }
    let file = std::fs::File::open(&path)?;
    let df = ParquetReader::new(file).finish()?;
    let ts = c_i64!(df, "ts");
    let tx_sig = c_str!(df, "tx_sig");
    let tx_sig_index = c_i64!(df, "tx_sig_index");
    let slot = c_i64!(df, "slot");
    let base_amount = c_f64!(df, "base_amount");
    let quote_amount = c_f64!(df, "quote_amount");
    let taker_fee = c_f64!(df, "taker_fee");
    let maker_rebate = c_f64!(df, "maker_rebate");
    let price = c_f64!(df, "price");
    let taker_direction = c_str!(df, "taker_direction");
    Ok((0..df.height())
        .map(|i| perpsfm_core::Trade {
            ts: ts[i],
            tx_sig: tx_sig[i].clone(),
            tx_sig_index: tx_sig_index[i],
            slot: slot[i],
            base_amount: base_amount[i],
            quote_amount: quote_amount[i],
            taker_fee: taker_fee[i],
            maker_rebate: maker_rebate[i],
            price: price[i],
            taker_direction: if taker_direction[i].is_empty() {
                None
            } else {
                Some(taker_direction[i].clone())
            },
        })
        .collect())
}

pub fn load_funding(root: &Path, market: Market) -> Result<Vec<perpsfm_core::FundingRate>> {
    let path = perpsfm_core::funding_path(root, market);
    if !path.exists() {
        return Ok(Vec::new());
    }
    let file = std::fs::File::open(&path)?;
    let df = ParquetReader::new(file).finish()?;
    let ts = c_i64!(df, "ts");
    let fr = c_f64!(df, "funding_rate");
    let frl = c_f64!(df, "funding_rate_long");
    let frs = c_f64!(df, "funding_rate_short");
    let cl = c_f64!(df, "cum_funding_long");
    let cs = c_f64!(df, "cum_funding_short");
    Ok((0..df.height())
        .map(|i| perpsfm_core::FundingRate {
            ts: ts[i],
            funding_rate: fr[i],
            funding_rate_long: frl[i],
            funding_rate_short: frs[i],
            cumulative_funding_rate_long: cl[i],
            cumulative_funding_rate_short: cs[i],
        })
        .collect())
}

pub fn load_liquidations(
    root: &Path,
    symbol: Option<&str>,
) -> Result<Vec<perpsfm_core::Liquidation>> {
    let path = perpsfm_core::liquidations_path(root);
    if !path.exists() {
        return Ok(Vec::new());
    }
    let file = std::fs::File::open(&path)?;
    let df = ParquetReader::new(file).finish()?;
    let ts = c_i64!(df, "ts");
    let liq_type = c_str!(df, "liquidation_type");
    let user = c_str!(df, "user");
    let liquidator = c_str!(df, "liquidator");
    let margin = c_f64!(df, "margin_requirement");
    let collateral = c_f64!(df, "total_collateral");
    let market_symbol = c_str!(df, "market_symbol");
    let base_amount = df
        .column("base_amount")?
        .f64()?
        .iter()
        .collect::<Vec<Option<f64>>>();
    let quote_amount = df
        .column("quote_amount")?
        .f64()?
        .iter()
        .collect::<Vec<Option<f64>>>();
    let all: Vec<perpsfm_core::Liquidation> = (0..df.height())
        .map(|i| perpsfm_core::Liquidation {
            ts: ts[i],
            liquidation_type: liq_type[i].clone(),
            user: user[i].clone(),
            liquidator: liquidator[i].clone(),
            margin_requirement: margin[i],
            total_collateral: collateral[i],
            market_symbol: if market_symbol[i].is_empty() {
                None
            } else {
                Some(market_symbol[i].clone())
            },
            base_amount: base_amount[i],
            quote_amount: quote_amount[i],
        })
        .collect();
    Ok(match symbol {
        Some(s) => all
            .into_iter()
            .filter(|l| l.market_symbol.as_deref() == Some(s))
            .collect(),
        None => all,
    })
}
