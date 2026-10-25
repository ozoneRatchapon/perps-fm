//! Backfill orchestration: pull the full archive into curated Parquet files.
//!
//! Layout (relative to repo root):
//! - data/{market}/candles_1m.parquet
//! - data/{market}/trades.parquet
//! - data/{market}/funding_rates.parquet
//! - data/liquidations.parquet (exchange-wide)

use anyhow::{Context, Result};
use chrono::{Duration, TimeZone, Timelike, Utc};
use polars::prelude::*;
use std::path::Path;

use perpsfm_core::{candle_path, funding_path, liquidations_path, trades_path, Market};
use tracing::info;

use crate::velocity::Client;

#[derive(Clone)]
pub struct BackfillOpts {
    pub start: chrono::DateTime<Utc>,
    pub end: chrono::DateTime<Utc>,
    /// Skip work for days that already have data at/after this timestamp.
    pub incremental: bool,
}

/// Backfill everything for all markets into parquet under `root`.
pub fn backfill_all(root: &Path, opts: &mut BackfillOpts) -> Result<()> {
    std::fs::create_dir_all(root.join("data"))?;
    for market in Market::ALL {
        let symbol = market.symbol();
        let mut opts = opts.clone();
        info!("=== {symbol} candles ===");
        backfill_candles(root, market, &mut opts)?;
        info!("=== {symbol} trades ===");
        backfill_trades(root, market, &mut opts)?;
        info!("=== {symbol} funding ===");
        backfill_funding(root, market, &mut opts)?;
    }
    info!("=== liquidations (exchange-wide) ===");
    backfill_liquidations(root)?;
    Ok(())
}

fn day_range(opts: &BackfillOpts) -> impl Iterator<Item = chrono::DateTime<Utc>> {
    let start = opts.start.date_naive();
    let end = opts.end.date_naive();
    let n = (end - start).num_days();
    (0..=n).map(move |d| {
        Utc.from_utc_datetime(&(start + Duration::days(d)).and_hms_opt(0, 0, 0).unwrap())
    })
}

pub fn backfill_candles(root: &Path, market: Market, opts: &mut BackfillOpts) -> Result<()> {
    let client = Client::new();
    let symbol = market.symbol();
    let mut all: Vec<perpsfm_core::Candle> = Vec::new();

    // Merge with any existing file so repeated backfills accumulate.
    let path = candle_path(root, market);
    if opts.incremental {
        let existing = read_candles(&path)?;
        if let Some(max_ts) = existing.iter().map(|c| c.ts).max() {
            if max_ts + 3600 > opts.end.timestamp() {
                info!("{symbol}: candles already up to date (ts {max_ts})");
                return Ok(());
            }
            all.extend(existing);
            opts.start = Utc
                .timestamp_opt(max_ts + 60, 0)
                .single()
                .unwrap_or(opts.start);
        }
    }

    // One continuous pass (windows of 1000 min) instead of per-day calls —
    // ~40% fewer requests against a tightly rate-limited API.
    let chunk = client
        .candles_1m(symbol, opts.start.timestamp(), opts.end.timestamp())
        .with_context(|| format!("candles {symbol}"))?;
    info!("{symbol}: fetched {} candles", chunk.len());
    all.extend(chunk);
    all.sort_by_key(|c| c.ts);
    all.dedup_by_key(|c| c.ts);
    write_candles(&path, &all)?;
    info!(
        "{symbol}: wrote {} candles to {}",
        all.len(),
        path.display()
    );
    Ok(())
}

pub fn backfill_trades(root: &Path, market: Market, opts: &mut BackfillOpts) -> Result<()> {
    let client = Client::new();
    let symbol = market.symbol();
    let mut all: Vec<perpsfm_core::Trade> = Vec::new();
    let path = trades_path(root, market);
    if opts.incremental {
        let existing = read_trades(&path)?;
        if let Some(max_ts) = existing.iter().map(|t| t.ts).max() {
            all.extend(existing);
            opts.start = Utc
                .timestamp_opt(max_ts + 1, 0)
                .single()
                .unwrap_or(opts.start)
                .with_hour(0)
                .unwrap()
                .with_minute(0)
                .unwrap()
                .with_second(0)
                .unwrap();
        }
    }
    for day in day_range(opts) {
        let chunk = client
            .trades_day(symbol, day)
            .with_context(|| format!("trades {symbol} {day}"))?;
        if !chunk.is_empty() {
            info!("{symbol} {}: {} fills", day.date_naive(), chunk.len());
        }
        all.extend(chunk);
    }
    all.sort_by_key(|t| (t.ts, t.tx_sig_index));
    write_trades(&path, &all)?;
    info!("{symbol}: wrote {} trades to {}", all.len(), path.display());
    Ok(())
}

pub fn backfill_funding(root: &Path, market: Market, opts: &mut BackfillOpts) -> Result<()> {
    let client = Client::new();
    let symbol = market.symbol();
    let mut all: Vec<perpsfm_core::FundingRate> = Vec::new();
    let path = funding_path(root, market);
    if opts.incremental {
        let existing = read_funding(&path)?;
        if let Some(max_ts) = existing.iter().map(|f| f.ts).max() {
            all.extend(existing);
            opts.start = Utc
                .timestamp_opt(max_ts + 3600, 0)
                .single()
                .unwrap_or(opts.start);
        }
    }
    for day in day_range(opts) {
        let chunk = client
            .funding_day(symbol, day)
            .with_context(|| format!("funding {symbol} {day}"))?;
        all.extend(chunk);
    }
    all.sort_by_key(|f| f.ts);
    all.dedup_by_key(|f| f.ts);
    write_funding(&path, &all)?;
    info!(
        "{symbol}: wrote {} funding records to {}",
        all.len(),
        path.display()
    );
    Ok(())
}

/// Exchange-wide liquidations: the API only exposes a rolling `days` window
/// (max 100), which covers the full archive window anyway.
pub fn backfill_liquidations(root: &Path) -> Result<()> {
    let client = Client::new();
    let liqs = client.liquidations(100)?;
    let path = liquidations_path(root);
    write_liquidations(&path, &liqs)?;
    info!("wrote {} liquidations to {}", liqs.len(), path.display());
    Ok(())
}

// ---- typed record <-> polars DataFrame <-> parquet ----

/// Extract column vectors from a parquet file; missing file yields None.
fn open_df(path: &Path) -> Result<Option<DataFrame>> {
    if !path.exists() {
        return Ok(None);
    }
    let file = std::fs::File::open(path)?;
    let df = ParquetReader::new(file).finish()?;
    Ok(Some(df))
}

fn col_i64(df: &DataFrame, name: &str) -> Result<Vec<i64>> {
    Ok(df.column(name)?.i64()?.into_no_null_iter().collect())
}
fn col_f64(df: &DataFrame, name: &str) -> Result<Vec<f64>> {
    Ok(df.column(name)?.f64()?.into_no_null_iter().collect())
}
fn col_str(df: &DataFrame, name: &str) -> Result<Vec<String>> {
    Ok(df
        .column(name)?
        .str()?
        .iter()
        .map(|v| v.unwrap_or_default().to_string())
        .collect())
}

pub fn read_candles(path: &Path) -> Result<Vec<perpsfm_core::Candle>> {
    let Some(df) = open_df(path)? else {
        return Ok(Vec::new());
    };
    let ts = col_i64(&df, "ts")?;
    let fill_open = col_f64(&df, "fill_open")?;
    let fill_high = col_f64(&df, "fill_high")?;
    let fill_low = col_f64(&df, "fill_low")?;
    let fill_close = col_f64(&df, "fill_close")?;
    let oracle_open = col_f64(&df, "oracle_open")?;
    let oracle_high = col_f64(&df, "oracle_high")?;
    let oracle_low = col_f64(&df, "oracle_low")?;
    let oracle_close = col_f64(&df, "oracle_close")?;
    let quote_volume = col_f64(&df, "quote_volume")?;
    let base_volume = col_f64(&df, "base_volume")?;
    Ok((0..df.height())
        .map(|i| perpsfm_core::Candle {
            ts: ts[i],
            fill_open: fill_open[i],
            fill_high: fill_high[i],
            fill_low: fill_low[i],
            fill_close: fill_close[i],
            oracle_open: oracle_open[i],
            oracle_high: oracle_high[i],
            oracle_low: oracle_low[i],
            oracle_close: oracle_close[i],
            quote_volume: quote_volume[i],
            base_volume: base_volume[i],
        })
        .collect())
}

pub fn read_trades(path: &Path) -> Result<Vec<perpsfm_core::Trade>> {
    let Some(df) = open_df(path)? else {
        return Ok(Vec::new());
    };
    let ts = col_i64(&df, "ts")?;
    let tx_sig = col_str(&df, "tx_sig")?;
    let tx_sig_index = col_i64(&df, "tx_sig_index")?;
    let slot = col_i64(&df, "slot")?;
    let base_amount = col_f64(&df, "base_amount")?;
    let quote_amount = col_f64(&df, "quote_amount")?;
    let taker_fee = col_f64(&df, "taker_fee")?;
    let maker_rebate = col_f64(&df, "maker_rebate")?;
    let price = col_f64(&df, "price")?;
    let taker_direction = col_str(&df, "taker_direction")?;
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

pub fn read_funding(path: &Path) -> Result<Vec<perpsfm_core::FundingRate>> {
    let Some(df) = open_df(path)? else {
        return Ok(Vec::new());
    };
    let ts = col_i64(&df, "ts")?;
    let funding_rate = col_f64(&df, "funding_rate")?;
    let long = col_f64(&df, "funding_rate_long")?;
    let short = col_f64(&df, "funding_rate_short")?;
    let cum_long = col_f64(&df, "cum_funding_long")?;
    let cum_short = col_f64(&df, "cum_funding_short")?;
    Ok((0..df.height())
        .map(|i| perpsfm_core::FundingRate {
            ts: ts[i],
            funding_rate: funding_rate[i],
            funding_rate_long: long[i],
            funding_rate_short: short[i],
            cumulative_funding_rate_long: cum_long[i],
            cumulative_funding_rate_short: cum_short[i],
        })
        .collect())
}

pub fn candles_df(rows: &[perpsfm_core::Candle]) -> DataFrame {
    df! {
        "ts" => rows.iter().map(|r| r.ts).collect::<Vec<_>>(),
        "fill_open" => rows.iter().map(|r| r.fill_open).collect::<Vec<_>>(),
        "fill_high" => rows.iter().map(|r| r.fill_high).collect::<Vec<_>>(),
        "fill_low" => rows.iter().map(|r| r.fill_low).collect::<Vec<_>>(),
        "fill_close" => rows.iter().map(|r| r.fill_close).collect::<Vec<_>>(),
        "oracle_open" => rows.iter().map(|r| r.oracle_open).collect::<Vec<_>>(),
        "oracle_high" => rows.iter().map(|r| r.oracle_high).collect::<Vec<_>>(),
        "oracle_low" => rows.iter().map(|r| r.oracle_low).collect::<Vec<_>>(),
        "oracle_close" => rows.iter().map(|r| r.oracle_close).collect::<Vec<_>>(),
        "quote_volume" => rows.iter().map(|r| r.quote_volume).collect::<Vec<_>>(),
        "base_volume" => rows.iter().map(|r| r.base_volume).collect::<Vec<_>>(),
    }
    .unwrap()
}

pub fn trades_df(rows: &[perpsfm_core::Trade]) -> DataFrame {
    df! {
        "ts" => rows.iter().map(|r| r.ts).collect::<Vec<_>>(),
        "tx_sig" => rows.iter().map(|r| r.tx_sig.clone()).collect::<Vec<_>>(),
        "tx_sig_index" => rows.iter().map(|r| r.tx_sig_index).collect::<Vec<_>>(),
        "slot" => rows.iter().map(|r| r.slot).collect::<Vec<_>>(),
        "base_amount" => rows.iter().map(|r| r.base_amount).collect::<Vec<_>>(),
        "quote_amount" => rows.iter().map(|r| r.quote_amount).collect::<Vec<_>>(),
        "taker_fee" => rows.iter().map(|r| r.taker_fee).collect::<Vec<_>>(),
        "maker_rebate" => rows.iter().map(|r| r.maker_rebate).collect::<Vec<_>>(),
        "price" => rows.iter().map(|r| r.price).collect::<Vec<_>>(),
        "taker_direction" => rows.iter().map(|r| r.taker_direction.clone().unwrap_or_default()).collect::<Vec<_>>(),
    }
    .unwrap()
}

pub fn funding_df(rows: &[perpsfm_core::FundingRate]) -> DataFrame {
    df! {
        "ts" => rows.iter().map(|r| r.ts).collect::<Vec<_>>(),
        "funding_rate" => rows.iter().map(|r| r.funding_rate).collect::<Vec<_>>(),
        "funding_rate_long" => rows.iter().map(|r| r.funding_rate_long).collect::<Vec<_>>(),
        "funding_rate_short" => rows.iter().map(|r| r.funding_rate_short).collect::<Vec<_>>(),
        "cum_funding_long" => rows.iter().map(|r| r.cumulative_funding_rate_long).collect::<Vec<_>>(),
        "cum_funding_short" => rows.iter().map(|r| r.cumulative_funding_rate_short).collect::<Vec<_>>(),
    }
    .unwrap()
}

pub fn liquidations_df(rows: &[perpsfm_core::Liquidation]) -> DataFrame {
    df! {
        "ts" => rows.iter().map(|r| r.ts).collect::<Vec<_>>(),
        "liquidation_type" => rows.iter().map(|r| r.liquidation_type.clone()).collect::<Vec<_>>(),
        "user" => rows.iter().map(|r| r.user.clone()).collect::<Vec<_>>(),
        "liquidator" => rows.iter().map(|r| r.liquidator.clone()).collect::<Vec<_>>(),
        "margin_requirement" => rows.iter().map(|r| r.margin_requirement).collect::<Vec<_>>(),
        "total_collateral" => rows.iter().map(|r| r.total_collateral).collect::<Vec<_>>(),
        "market_symbol" => rows.iter().map(|r| r.market_symbol.clone()).collect::<Vec<_>>(),
        "base_amount" => rows.iter().map(|r| r.base_amount).collect::<Vec<_>>(),
        "quote_amount" => rows.iter().map(|r| r.quote_amount).collect::<Vec<_>>(),
    }
    .unwrap()
}

fn write_parquet(df: &DataFrame, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("parquet.tmp");
    let file = std::fs::File::create(&tmp)?;
    ParquetWriter::new(file).finish(&mut df.clone())?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

pub fn write_candles(path: &Path, rows: &[perpsfm_core::Candle]) -> Result<()> {
    write_parquet(&candles_df(rows), path)
}
pub fn write_trades(path: &Path, rows: &[perpsfm_core::Trade]) -> Result<()> {
    write_parquet(&trades_df(rows), path)
}
pub fn write_funding(path: &Path, rows: &[perpsfm_core::FundingRate]) -> Result<()> {
    write_parquet(&funding_df(rows), path)
}
pub fn write_liquidations(path: &Path, rows: &[perpsfm_core::Liquidation]) -> Result<()> {
    write_parquet(&liquidations_df(rows), path)
}
