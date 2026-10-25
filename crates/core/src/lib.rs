//! Shared types for the perps-fm workspace: curated record schemas, the
//! market universe, and the train/val/sealed time-split definitions.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Perp markets we ingest. Only a handful are active on Velocity; the last
/// two serve as held-out (never-trained-on) markets in the evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Market {
    SolPerp,
    HypePerp,
    EthPerp,
    BtcPerp,
}

impl Market {
    pub const ALL: [Market; 4] = [
        Market::SolPerp,
        Market::HypePerp,
        Market::EthPerp,
        Market::BtcPerp,
    ];

    pub fn symbol(&self) -> &'static str {
        match self {
            Market::SolPerp => "SOL-PERP",
            Market::HypePerp => "HYPE-PERP",
            Market::EthPerp => "ETH-PERP",
            Market::BtcPerp => "BTC-PERP",
        }
    }

    pub fn from_symbol(s: &str) -> Option<Market> {
        Market::ALL.iter().copied().find(|m| m.symbol() == s)
    }
}

/// One 1-minute candle from the Velocity candle store. Both fill and oracle
/// OHLC are kept: oracle prices are continuous even when no trade printed,
/// which makes them the backbone series for dense sequence modeling.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Candle {
    /// Unix seconds, candle bucket start.
    pub ts: i64,
    pub fill_open: f64,
    pub fill_high: f64,
    pub fill_low: f64,
    pub fill_close: f64,
    pub oracle_open: f64,
    pub oracle_high: f64,
    pub oracle_low: f64,
    pub oracle_close: f64,
    pub quote_volume: f64,
    pub base_volume: f64,
}

/// A single fill (taker-executed trade) with fee metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub ts: i64,
    pub tx_sig: String,
    pub tx_sig_index: i64,
    pub slot: i64,
    pub base_amount: f64,
    pub quote_amount: f64,
    pub taker_fee: f64,
    pub maker_rebate: f64,
    pub price: f64,
    /// Direction of the taker: "long" / "short", when reported.
    pub taker_direction: Option<String>,
}

/// Hourly funding rate record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingRate {
    pub ts: i64,
    pub funding_rate: f64,
    pub funding_rate_long: f64,
    pub funding_rate_short: f64,
    pub cumulative_funding_rate_long: f64,
    pub cumulative_funding_rate_short: f64,
}

/// A liquidation event (exchange-wide via /stats/liquidations).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Liquidation {
    pub ts: i64,
    pub liquidation_type: String,
    pub user: String,
    pub liquidator: String,
    pub margin_requirement: f64,
    pub total_collateral: f64,
    pub market_symbol: Option<String>,
    pub base_amount: Option<f64>,
    pub quote_amount: Option<f64>,
}

/// Live events captured by the DLOB websocket collector (trades channel).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveTrade {
    pub ts: i64,
    pub market: String,
    pub payload: serde_json::Value,
}

/// Time splits. `sealed` weeks must never be read while iterating on models;
/// they are opened exactly once at the end (see docs/00-preregistration.md).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Splits {
    pub train_start: DateTime<Utc>,
    pub train_end: DateTime<Utc>,
    pub val_start: DateTime<Utc>,
    pub val_end: DateTime<Utc>,
    pub sealed_start: DateTime<Utc>,
    pub sealed_end: DateTime<Utc>,
}

impl Splits {
    /// Default split for the 2026-08 archive start + live-collected tail.
    /// Computed from today so the sealed window is always the final 2 weeks
    /// of available data.
    pub fn default_from_archive(archive_start: DateTime<Utc>, data_end: DateTime<Utc>) -> Splits {
        let sealed_len = chrono::Duration::weeks(2);
        let val_len = chrono::Duration::weeks(2);
        let sealed_start = data_end - sealed_len;
        let val_end = sealed_start;
        let val_start = val_end - val_len;
        Splits {
            train_start: archive_start,
            train_end: val_start,
            val_start,
            val_end,
            sealed_start,
            sealed_end: data_end,
        }
    }
}

/// Storage layout for curated parquet files.
pub fn market_dir(root: &std::path::Path, market: Market) -> std::path::PathBuf {
    root.join("data").join(market.symbol().to_lowercase())
}

pub fn candle_path(root: &std::path::Path, market: Market) -> std::path::PathBuf {
    market_dir(root, market).join("candles_1m.parquet")
}

pub fn trades_path(root: &std::path::Path, market: Market) -> std::path::PathBuf {
    market_dir(root, market).join("trades.parquet")
}

pub fn funding_path(root: &std::path::Path, market: Market) -> std::path::PathBuf {
    market_dir(root, market).join("funding_rates.parquet")
}

/// Exchange-wide liquidations live in a shared file (not per market).
pub fn liquidations_path(root: &std::path::Path) -> std::path::PathBuf {
    root.join("data").join("liquidations.parquet")
}

/// Live-collected DLOB events (append-only JSONL).
pub fn live_dir(root: &std::path::Path) -> std::path::PathBuf {
    root.join("data").join("live")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbols_roundtrip() {
        for m in Market::ALL {
            assert_eq!(Market::from_symbol(m.symbol()), Some(m));
        }
        assert_eq!(Market::from_symbol("NOPE-PERP"), None);
    }

    #[test]
    fn default_splits_shape() {
        let start = Utc.with_ymd_and_hms(2026, 8, 1, 0, 0, 0).unwrap();
        let end = Utc.with_ymd_and_hms(2026, 10, 25, 0, 0, 0).unwrap();
        let s = Splits::default_from_archive(start, end);
        assert_eq!(s.sealed_end, end);
        assert_eq!(s.sealed_end - s.sealed_start, chrono::Duration::weeks(2));
        assert_eq!(s.val_end, s.sealed_start);
        assert!(s.train_start < s.train_end);
    }

    use chrono::TimeZone;
}
