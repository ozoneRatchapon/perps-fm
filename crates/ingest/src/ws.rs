//! Live collector: streams DLOB websocket events to append-only JSONL files.
//!
//! Connects to wss://dlob.velocity.exchange/ws and subscribes to the `trades`
//! and `orderbook` channels for every active perp market. Messages arrive
//! with a `channel` field (e.g. "trades_perp_0") and a `data` field that is
//! often double-encoded JSON; heartbeats arrive on channel "heartbeat".
//!
//! Output: data/live/trades_{symbol}.jsonl and orderbook_{symbol}.jsonl.
//! One JSON object per line, so the files are crash-safe to append to and
//! trivially re-readable later.

use anyhow::{Context, Result};
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio_tungstenite::{connect_async, tungstenite::Message};

use perpsfm_core::{live_dir, Market};

const DLOB_WS: &str = "wss://dlob.velocity.exchange/ws";
const RECONNECT_DELAY: Duration = Duration::from_secs(5);
const STATS_EVERY: u64 = 100;

pub struct CollectorOpts {
    /// Extra channels to subscribe (default: trades + orderbook grouping 1).
    pub orderbook_grouping: u32,
}

pub async fn run_collector(root: PathBuf, opts: CollectorOpts) -> Result<()> {
    let dir = live_dir(&root);
    std::fs::create_dir_all(&dir)?;

    let counters: Vec<(Market, AtomicU64, AtomicU64)> = Market::ALL
        .iter()
        .map(|m| (*m, AtomicU64::new(0), AtomicU64::new(0)))
        .collect();

    loop {
        match collector_session(&root, &dir, &opts, &counters).await {
            Ok(()) => {
                tracing::info!("collector session ended cleanly; reconnecting");
            }
            Err(e) => {
                tracing::warn!(
                    "collector session failed: {e:#}; reconnecting in {RECONNECT_DELAY:?}"
                );
                tokio::time::sleep(RECONNECT_DELAY).await;
            }
        }
    }
}

async fn collector_session(
    root: &PathBuf,
    dir: &std::path::Path,
    opts: &CollectorOpts,
    counters: &[(Market, AtomicU64, AtomicU64)],
) -> Result<()> {
    let (ws, _resp) = connect_async(DLOB_WS)
        .await
        .context("connect to DLOB websocket")?;
    let (mut sink, mut stream) = ws.split();
    tracing::info!("connected to {DLOB_WS}");

    for m in Market::ALL {
        let sub = json!({
            "type": "subscribe",
            "channel": "trades",
            "marketType": "perp",
            "market": m.symbol(),
        });
        sink.send(Message::text(sub.to_string())).await?;
        let sub = json!({
            "type": "subscribe",
            "channel": "orderbook",
            "marketType": "perp",
            "market": m.symbol(),
            "grouping": opts.orderbook_grouping,
        });
        sink.send(Message::text(sub.to_string())).await?;
    }

    let mut trade_files: std::collections::HashMap<String, std::fs::File> =
        std::collections::HashMap::new();
    let mut book_files: std::collections::HashMap<String, std::fs::File> =
        std::collections::HashMap::new();
    for m in Market::ALL {
        let sym = m.symbol().to_lowercase();
        trade_files.insert(
            sym.clone(),
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(dir.join(format!("trades_{sym}.jsonl")))?,
        );
        book_files.insert(
            sym.clone(),
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(dir.join(format!("orderbook_{sym}.jsonl")))?,
        );
    }

    while let Some(msg) = stream.next().await {
        let msg = match msg? {
            Message::Text(t) => t,
            Message::Ping(_) | Message::Pong(_) => continue,
            Message::Close(_) => anyhow::bail!("server closed connection"),
            _ => continue,
        };
        let v: Value = match serde_json::from_str(&msg) {
            Ok(v) => v,
            Err(_) => continue, // ignore non-JSON frames
        };
        let channel = v.get("channel").and_then(|c| c.as_str()).unwrap_or("");
        if channel == "heartbeat" || channel.is_empty() {
            continue;
        }
        // data is often double-encoded JSON
        let data = match v.get("data") {
            Some(Value::String(s)) => serde_json::from_str::<Value>(s).unwrap_or(Value::Null),
            Some(other) => other.clone(),
            None => continue,
        };

        // channel looks like "trades_perp_0" / "orderbook_perp_0_grouped_1"
        let kind = channel.split('_').next().unwrap_or("");
        let market_index: usize = channel
            .split('_')
            .filter_map(|p| p.parse::<usize>().ok())
            .next()
            .unwrap_or(usize::MAX);
        let market = Market::ALL
            .iter()
            .find(|m| market_index_for(m) == market_index)
            .copied();
        let Some(market) = market else { continue };
        let sym = market.symbol().to_lowercase();

        let now_ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        let line = json!({"recv_ts": now_ts, "channel": channel, "data": data});

        match kind {
            "trades" => {
                append_line(trade_files.get_mut(&sym), &line)?;
                bump(counters, market, true);
            }
            "orderbook" => {
                append_line(book_files.get_mut(&sym), &line)?;
                bump(counters, market, false);
            }
            _ => {}
        }
    }
    let _ = root;
    Ok(())
}

fn market_index_for(m: &Market) -> usize {
    match m {
        Market::SolPerp => 0,
        Market::BtcPerp => 1,
        Market::EthPerp => 2,
        Market::HypePerp => 3,
    }
}

fn bump(counters: &[(Market, AtomicU64, AtomicU64)], market: Market, is_trade: bool) {
    if let Some((_, trades, books)) = counters.iter().find(|(m, _, _)| *m == market) {
        let c = if is_trade { trades } else { books };
        let n = c.fetch_add(1, Ordering::Relaxed) + 1;
        if n % STATS_EVERY == 0 {
            tracing::info!(
                "{} {} events: {}",
                market.symbol(),
                if is_trade { "trade" } else { "book" },
                n
            );
        }
    }
}

fn append_line(file: Option<&mut std::fs::File>, line: &Value) -> Result<()> {
    use std::io::Write;
    let Some(f) = file else { return Ok(()) };
    writeln!(f, "{line}")?;
    Ok(())
}
