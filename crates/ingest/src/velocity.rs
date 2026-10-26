//! Typed client for the Velocity Data API (https://data.velocity.exchange).
//!
//! Quirks we rely on (all verified 2026-10-25, see docs/00-data-source.md):
//! - candle endpoints want `startTs` > `endTs` and return bars descending;
//! - dated archive endpoints paginate via numeric `page` with `meta.nextPage`;
//! - no auth anywhere, but be polite: sleep between calls, back off on 5xx.

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Datelike, TimeZone, Utc};
use serde::Deserialize;
use std::time::Duration;

use perpsfm_core::{Candle, FundingRate, Liquidation, Trade};

pub const API_BASE: &str = "https://data.velocity.exchange";

/// API archive starts with the Velocity migration; nothing earlier is served.
pub fn archive_start() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 8, 1, 0, 0, 0).unwrap()
}

#[derive(Debug, Clone)]
pub struct Client {
    agent: ureq::Agent,
    /// Politeness delay between requests.
    pub delay: Duration,
}

impl Default for Client {
    fn default() -> Self {
        Self::new()
    }
}

impl Client {
    pub fn new() -> Self {
        let config = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(30)))
            .user_agent("perpsfm-ingest/0.1 (research; polite)")
            .build();
        Self {
            agent: config.into(),
            // ~1.6s effective interval incl. round-trip; measured sustainable
            // (bursts trigger CloudFront 403s that last minutes).
            delay: Duration::from_millis(1500),
        }
    }

    fn get_json(&self, path: &str) -> Result<serde_json::Value> {
        let url = format!("{API_BASE}{path}");
        let mut last_err = None;
        // The API sits behind CloudFront, which answers burst traffic with
        // 403 rather than 429 — treat both as throttle and back off hard.
        for attempt in 0..8u32 {
            if attempt > 0 {
                let backoff = Duration::from_secs(2u64.saturating_pow(attempt))
                    + Duration::from_millis(fastrand_u64() % 500);
                tracing::warn!("retry {attempt} for {url} after {backoff:?}");
                std::thread::sleep(backoff);
            }
            match self.agent.get(&url).call() {
                Ok(mut res) => {
                    let status = res.status().as_u16();
                    if status == 403 || status == 429 || status >= 500 {
                        last_err = Some(anyhow::anyhow!("HTTP {status} for {url}"));
                        // WAF penalty windows last minutes — wait it out and
                        // surface it loudly, or a long backfill sits silent.
                        if status == 403 || status == 429 {
                            let penalty = Duration::from_secs(75)
                                + Duration::from_millis(fastrand_u64() % 30_000);
                            tracing::warn!("throttled ({status}); penalty sleep {penalty:?}");
                            std::thread::sleep(penalty);
                        }
                        continue;
                    }
                    if status != 200 {
                        let body = res.body_mut().read_to_string().unwrap_or_default();
                        bail!("HTTP {status} for {url}: {}", truncate(&body, 300));
                    }
                    let body = res
                        .body_mut()
                        .read_to_string()
                        .with_context(|| format!("read body from {url}"))?;
                    let v: serde_json::Value = serde_json::from_str(&body)
                        .with_context(|| format!("decode JSON from {url}"))?;
                    std::thread::sleep(self.delay);
                    return Ok(v);
                }
                Err(e) => {
                    last_err = Some(anyhow::anyhow!("request failed for {url}: {e}"));
                }
            }
        }
        Err(last_err.unwrap())
    }

    /// 1-minute candles covering [start_ts, end_ts). Fetches in <=1000-minute
    /// windows (API max `limit`), returns bars sorted ascending, deduped.
    pub fn candles_1m(&self, symbol: &str, start_ts: i64, end_ts: i64) -> Result<Vec<Candle>> {
        let mut out = Vec::new();
        let mut window_end = end_ts; // exclusive
        while window_end > start_ts {
            // 1000 records x 60s bars per window (limit is a record cap)
            let window_start = (window_end - 1000 * 60).max(start_ts);
            // NOTE: API wants startTs > endTs and returns descending bars.
            let path = format!(
                "/market/{symbol}/candles/1?startTs={}&endTs={}&limit=1000",
                window_end - 1,
                window_start
            );
            tracing::debug!("candle window: {path}");
            let v = self.get_json(&path)?;
            let raw: Vec<ApiCandle> = parse_records(&v)?;
            tracing::debug!(
                "{symbol}: window [{window_start},{window_end}) -> {} bars",
                raw.len()
            );
            if raw.is_empty() {
                // No data in this window; skip forward to avoid a spin.
                window_end = window_start;
                continue;
            }
            for c in raw {
                if c.ts >= start_ts && c.ts < end_ts {
                    out.push(c.into_curated());
                }
            }
            window_end = window_start;
        }
        out.sort_by_key(|c| c.ts);
        out.dedup_by_key(|c| c.ts);
        Ok(out)
    }

    /// All fills for one UTC day (paginated).
    pub fn trades_day(&self, symbol: &str, day: DateTime<Utc>) -> Result<Vec<Trade>> {
        let mut out = Vec::new();
        let mut page = 1u64;
        loop {
            let path = format!(
                "/market/{symbol}/trades/{}/{}/{}?page={page}",
                day.year(),
                day.month(),
                day.day()
            );
            let v = self.get_json(&path)?;
            let raw: Vec<ApiTrade> = parse_records(&v)?;
            let n = raw.len();
            out.extend(raw.into_iter().map(|t| t.into_curated()));
            if n == 0 || !has_next_page(&v) {
                break;
            }
            page += 1;
            if page > 200 {
                bail!("pagination runaway for {path}");
            }
        }
        Ok(out)
    }

    /// Hourly funding rates for one UTC day.
    pub fn funding_day(&self, symbol: &str, day: DateTime<Utc>) -> Result<Vec<FundingRate>> {
        let path = format!(
            "/market/{symbol}/fundingRates/{}/{}/{}",
            day.year(),
            day.month(),
            day.day()
        );
        let v = self.get_json(&path)?;
        let raw: Vec<ApiFunding> = parse_records(&v)?;
        Ok(raw.into_iter().map(|f| f.into_curated()).collect())
    }

    /// Exchange-wide liquidation events over the last `days` days (max 100).
    /// The endpoint rejects `page` — it returns one large page.
    pub fn liquidations(&self, days: u32) -> Result<Vec<Liquidation>> {
        let path = format!("/stats/liquidations?days={days}");
        let v = self.get_json(&path)?;
        let raw: Vec<serde_json::Value> = parse_records(&v)?;
        let total = v
            .get("meta")
            .and_then(|m| m.get("totalRecords"))
            .and_then(|t| t.as_u64())
            .unwrap_or(0);
        if total > raw.len() as u64 {
            tracing::warn!(
                "liquidations truncated: got {} of {total} (no pagination support)",
                raw.len()
            );
        }
        Ok(raw
            .into_iter()
            .filter_map(|v| serde_json::from_value::<ApiLiquidation>(v).ok())
            .map(|l| l.into_curated())
            .collect())
    }
}

fn truncate(s: &str, n: usize) -> &str {
    match s.get(..n) {
        Some(x) => x,
        None => s,
    }
}

/// Tiny non-crypto jitter source to avoid synchronized retry storms.
fn fastrand_u64() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos() as u64;
    nanos.wrapping_mul(0x9E3779B97F4A7C15)
}

fn parse_records<T: serde::de::DeserializeOwned>(v: &serde_json::Value) -> Result<Vec<T>> {
    let recs = v
        .get("records")
        .and_then(|r| r.as_array())
        .context("missing records array")?;
    Ok(recs
        .iter()
        .filter_map(|r| serde_json::from_value::<T>(r.clone()).ok())
        .collect())
}

fn has_next_page(v: &serde_json::Value) -> bool {
    v.get("meta")
        .and_then(|m| m.get("nextPage"))
        .map(|p| !p.is_null())
        .unwrap_or(false)
}

// ---- API wire types (strings for numeric fields, per the API's conventions) ----

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ApiCandle {
    ts: i64,
    #[serde(default)]
    fill_open: Option<F64>,
    #[serde(default)]
    fill_high: Option<F64>,
    #[serde(default)]
    fill_low: Option<F64>,
    #[serde(default)]
    fill_close: Option<F64>,
    #[serde(default)]
    oracle_open: Option<F64>,
    #[serde(default)]
    oracle_high: Option<F64>,
    #[serde(default)]
    oracle_low: Option<F64>,
    #[serde(default)]
    oracle_close: Option<F64>,
    #[serde(default)]
    quote_volume: Option<F64>,
    #[serde(default)]
    base_volume: Option<F64>,
}

impl ApiCandle {
    fn into_curated(self) -> Candle {
        Candle {
            ts: self.ts,
            fill_open: self.fill_open.map(|v| v.inner()).unwrap_or(f64::NAN),
            fill_high: self.fill_high.map(|v| v.inner()).unwrap_or(f64::NAN),
            fill_low: self.fill_low.map(|v| v.inner()).unwrap_or(f64::NAN),
            fill_close: self.fill_close.map(|v| v.inner()).unwrap_or(f64::NAN),
            oracle_open: self.oracle_open.map(|v| v.inner()).unwrap_or(f64::NAN),
            oracle_high: self.oracle_high.map(|v| v.inner()).unwrap_or(f64::NAN),
            oracle_low: self.oracle_low.map(|v| v.inner()).unwrap_or(f64::NAN),
            oracle_close: self.oracle_close.map(|v| v.inner()).unwrap_or(f64::NAN),
            quote_volume: self.quote_volume.map(|v| v.inner()).unwrap_or(0.0),
            base_volume: self.base_volume.map(|v| v.inner()).unwrap_or(0.0),
        }
    }
}

/// API sends numbers either as JSON numbers or decimal strings; accept both.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum F64 {
    Num(f64),
    Str(String),
}

impl F64 {
    fn inner(&self) -> f64 {
        match self {
            F64::Num(x) => *x,
            F64::Str(s) => s.parse().unwrap_or(f64::NAN),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ApiTrade {
    ts: i64,
    tx_sig: String,
    #[serde(default)]
    tx_sig_index: Option<i64>,
    slot: i64,
    #[serde(default)]
    base_asset_amount_filled: Option<F64>,
    #[serde(default)]
    quote_asset_amount_filled: Option<F64>,
    #[serde(default)]
    taker_fee: Option<F64>,
    #[serde(default)]
    maker_rebate: Option<F64>,
    #[serde(default)]
    price: Option<F64>,
    #[serde(default)]
    taker_order_direction: Option<String>,
}

impl ApiTrade {
    fn into_curated(self) -> Trade {
        let base = self
            .base_asset_amount_filled
            .map(|v| v.inner())
            .unwrap_or(0.0);
        let quote = self
            .quote_asset_amount_filled
            .map(|v| v.inner())
            .unwrap_or(0.0);
        let price = self.price.map(|v| v.inner()).unwrap_or(if base > 0.0 {
            quote / base
        } else {
            f64::NAN
        });
        Trade {
            ts: self.ts,
            tx_sig: self.tx_sig,
            tx_sig_index: self.tx_sig_index.unwrap_or(0),
            slot: self.slot,
            base_amount: base,
            quote_amount: quote,
            taker_fee: self.taker_fee.map(|v| v.inner()).unwrap_or(0.0),
            maker_rebate: self.maker_rebate.map(|v| v.inner()).unwrap_or(0.0),
            price,
            taker_direction: self.taker_order_direction,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ApiFunding {
    ts: i64,
    #[serde(default)]
    funding_rate: Option<F64>,
    #[serde(default)]
    funding_rate_long: Option<F64>,
    #[serde(default)]
    funding_rate_short: Option<F64>,
    #[serde(default)]
    cumulative_funding_rate_long: Option<F64>,
    #[serde(default)]
    cumulative_funding_rate_short: Option<F64>,
}

impl ApiFunding {
    fn into_curated(self) -> FundingRate {
        FundingRate {
            ts: self.ts,
            funding_rate: self.funding_rate.map(|v| v.inner()).unwrap_or(f64::NAN),
            funding_rate_long: self
                .funding_rate_long
                .map(|v| v.inner())
                .unwrap_or(f64::NAN),
            funding_rate_short: self
                .funding_rate_short
                .map(|v| v.inner())
                .unwrap_or(f64::NAN),
            cumulative_funding_rate_long: self
                .cumulative_funding_rate_long
                .map(|v| v.inner())
                .unwrap_or(f64::NAN),
            cumulative_funding_rate_short: self
                .cumulative_funding_rate_short
                .map(|v| v.inner())
                .unwrap_or(f64::NAN),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ApiLiquidation {
    ts: i64,
    #[serde(default)]
    liquidation_type: Option<String>,
    #[serde(default)]
    user: Option<String>,
    #[serde(default)]
    liquidator: Option<String>,
    #[serde(default)]
    margin_requirement: Option<F64>,
    #[serde(default)]
    total_collateral: Option<F64>,
    #[serde(default)]
    market: Option<String>,
    #[serde(default)]
    market_symbol: Option<String>,
    #[serde(rename = "liquidatePerp_marketIndex", default)]
    liquidate_perp_market_index: Option<i64>,
    #[serde(rename = "liquidatePerp_baseAssetAmount", default)]
    base_asset_amount: Option<F64>,
    #[serde(rename = "liquidatePerp_quoteAssetAmount", default)]
    quote_asset_amount: Option<F64>,
}

impl ApiLiquidation {
    fn into_curated(self) -> Liquidation {
        // perp market index -> symbol (Velocity mainnet perp indexes)
        let by_index = |i: i64| match i {
            0 => Some("SOL-PERP".to_string()),
            1 => Some("BTC-PERP".to_string()),
            2 => Some("ETH-PERP".to_string()),
            3 => Some("HYPE-PERP".to_string()),
            _ => None,
        };
        Liquidation {
            ts: self.ts,
            liquidation_type: self.liquidation_type.unwrap_or_default(),
            user: self.user.unwrap_or_default(),
            liquidator: self.liquidator.unwrap_or_default(),
            margin_requirement: self
                .margin_requirement
                .map(|v| v.inner())
                .unwrap_or(f64::NAN),
            total_collateral: self.total_collateral.map(|v| v.inner()).unwrap_or(f64::NAN),
            market_symbol: self
                .market_symbol
                .or(self.market)
                .or(self.liquidate_perp_market_index.and_then(by_index)),
            base_amount: self.base_asset_amount.map(|v| v.inner()),
            quote_amount: self.quote_asset_amount.map(|v| v.inner()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn archive_start_is_migration_day() {
        assert_eq!(archive_start().to_string(), "2026-08-01 00:00:00 UTC");
    }

    #[test]
    fn f64_accepts_number_and_string() {
        let a: F64 = serde_json::from_value(serde_json::json!(1.5)).unwrap();
        let b: F64 = serde_json::from_value(serde_json::json!("1.5")).unwrap();
        assert_eq!(a.inner(), 1.5);
        assert_eq!(b.inner(), 1.5);
    }
}

#[cfg(test)]
mod liquidation_tests {
    use super::*;

    #[test]
    fn parses_liquidate_perp_fields() {
        let j = serde_json::json!({
            "ts": 1792854992,
            "liquidationType": "liquidateSpot",
            "user": "u",
            "liquidator": "l",
            "marginRequirement": "23.748226",
            "totalCollateral": "23.746595",
            "liquidatePerp_marketIndex": 0,
            "liquidatePerp_baseAssetAmount": "0.000000000",
            "liquidatePerp_quoteAssetAmount": "0.000000"
        });
        let a: ApiLiquidation = serde_json::from_value(j).unwrap();
        let c = a.into_curated();
        assert_eq!(c.market_symbol.as_deref(), Some("SOL-PERP"));
        assert_eq!(c.quote_amount, Some(0.0));
    }
}
