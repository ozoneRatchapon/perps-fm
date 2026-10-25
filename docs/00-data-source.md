# Data source: Velocity (ex-Drift) Data API

Everything in this repo is Solana-native: the venue is **Velocity** — the protocol
formerly known as Drift, the largest perps DEX on Solana. No centralized-exchange
data anywhere.

## How we got here (2026-10-25)

Drift rebranded to Velocity in 2026 and dismantled its old data stack:

| Source | Status |
| --- | --- |
| `drift-historical-data` S3 (eu-west-1) | locked down (403 on list + get) |
| `drift-historical-data-v2` S3 | **bucket emptied** |
| `data.api.drift.trade` | CloudFront distribution deleted |
| `dlob.drift.trade` | NXDOMAIN |
| `data.velocity.exchange` | **live, no auth** ← we use this |
| `dlob.velocity.exchange` | **live** (orderbook/trades WS) ← we use this |

The old API's per-month URL scheme (`program/{pid}/market/{mkt}/{field}/{y}/{m}`)
still circulates in blog posts and papers; it no longer works. Current docs live in
the `velocity-exchange/velocity-docs` repo (`content/developers/data-api/index.mdx`).

## What we ingest

Base: `https://data.velocity.exchange` (OpenAPI at `/openapi.json`).

| Endpoint | Use | Notes |
| --- | --- | --- |
| `/market/{sym}/candles/{res}` | 1-min OHLCV backbone | fill **and** oracle OHLC; oracle is continuous even with zero fills. Resolution strings `"1".."240","D","W","M"`, `limit <= 1000`. **`startTs` must be greater than `endTs`** (inverted validation); bars come back descending. `limit` caps RECORDS — a window of 1000 one-minute bars spans 60,000 seconds. The leading edge (~last 20 min) is incomplete: cap backfills at now − 1h. |
| `/market/{sym}/trades/{y}/{m}/{d}` | fills (order flow) | taker/maker fee + rebate metadata; paginate with `page` until `meta.nextPage` is null; `format=csv` available for bulk. |
| `/market/{sym}/fundingRates/{y}/{m}/{d}` | hourly funding | 24 records/day. |
| `/stats/liquidations?days=N` | liquidation events | exchange-wide, rolling window (max 100 days — covers the whole archive anyway). |
| `/stats/markets` | universe | 4 active perps: SOL-PERP (idx 0), BTC-PERP (1), ETH-PERP (2), HYPE-PERP (3). |
| `/amm/oraclePrice`, `/amm/spreads`, `/amm/openInterest` | 20s-resolution series | rolling ~2.5-day windows only (cap 11,000 samples, 20s floor) — useful for recent-period sanity checks, not for archive. |

Live: `wss://dlob.velocity.exchange/ws`. Subscribe
`{"type":"subscribe","channel":"trades","marketType":"perp","market":"SOL-PERP"}` (+ `orderbook`
with `grouping`). Messages carry `channel` (e.g. `trades_perp_0`) and a `data` field that is
**often double-encoded JSON** (parse twice). Heartbeats on channel `heartbeat`.

## Client gotchas (all verified the hard way)

- **gzip truncation**: with `accept-encoding: gzip`, the API returns
  *truncated* bodies (~1KB instead of ~40KB) with HTTP 200. Send identity
  encoding — the `ingest` crate builds ureq with `default-features = false`
  so no gzip is negotiated.
- **Rate limiting**: bursts of fast requests trigger CloudFront 403s that
  last minutes (not seconds). ~1.6s/request spacing is sustainable; the
  client backs off 75s+ on 403/429.
- **Soft-empty leading edge**: requests whose window touches "now" return
  few or zero bars without any error — always end backfills in the past.

## Archive reality

- History starts at the Velocity migration (**2026-08-01**); the OpenAPI `minimum`
  of 2022-11-15 is a validation bound only — Drift-era data was **not** migrated.
- Post-migration the venue is small: SOL-PERP does ~7–50 fills/day, ~80
  liquidations/30d across the exchange. The interesting modeling regime is
  therefore **1-minute oracle/fill bars + sparse event streams**, not
  Binance-scale tick data. That is the honest Solana-native data problem, and
  this project treats it as a feature, not a bug (see the README).
- Live WS collection during the project supplies the sealed test weeks.
