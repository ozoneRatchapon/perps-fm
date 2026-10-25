# perps-fm — a miniature Solana-native market foundation model

A portfolio project mirroring the **"Junior Research Scientist: Market
Foundation Model"** role: one tiny sequence model trained across Solana perps
markets (Velocity — the protocol formerly known as Drift), evaluated the way a
research shop evaluates — pre-registered hypotheses, held-out markets, sealed
test weeks, confidence intervals, and after-cost backtests.

Built **entirely in Rust**: ingestion (Velocity Data API + DLOB websocket),
feature engineering (Polars/Parquet), models (candle, from scratch), training
(Metal backend), market simulation, and statistics.

> **Status**: pipeline end-to-end (data → features → train → predict →
> evaluate → backtest); results in `docs/01-results.md`.

## Why Rust (and where PyTorch would be)

The role lists Python/PyTorch as the working stack and Rust as a nice-to-have;
this project inverts that deliberately:

- Polars, Parquet, and DuckDB are all in the JD and all have first-class Rust
  APIs — the data plane doesn't need Python.
- The models (~68k-parameter transformer, S4D SSM) are small enough to write
  from scratch in candle (~1,000 lines). Implementing causal attention,
  complex-diagonal state spaces, and the pinball loss by hand demonstrates the
  mechanics a foundation-model researcher needs anyway; the same code maps 1:1
  to PyTorch.
- The venue is Solana-native: the Rust ecosystem (solana SDKs, Anchor, gRPC
  Geyser) is where real Solana market data work happens.

## The honest data regime

Velocity (ex-Drift) reset its data stack in 2026: the old S3 dumps and data
APIs are gone ([details](docs/00-data-source.md)). What remains is ~12 weeks of
1-minute bars per market plus genuinely sparse event streams: ~30 fills/day,
~70 liquidations/quarter on the flagship market. So this project models the
regime that actually exists on Solana — **dense minute bars + rare events** —
rather than pretending to have Binance-scale tick data. Rare-event prediction
under sparse supervision is arguably the more interesting research problem, and
it is exactly the regime a Solana-native trading desk faces.

| Market | 1-min bars | fills | funding | liquidations |
| --- | --- | --- | --- | --- |
| SOL-PERP | 123,023 | 3,080 | 2,050 | 71 |
| HYPE-PERP | 123,023 | 832 | 2,050 | 10 |
| ETH-PERP | 123,023 | 407 | 2,050 | 11 |
| BTC-PERP (held out) | 123,023 | 488 | 2,050 | 6 |

## Pipeline

```
just backfill      # Velocity archive -> data/<market>/*.parquet (idempotent)
just collect       # DLOB websocket -> data/live/*.jsonl (sealed weeks grow here)
just build-features# trailing-window features + strictly-forward targets
just train         # pooled transformer, early-stop on val CRPS
perpsfm predict    # checkpoint -> predictions.parquet (val split; never sealed)
perpsfm evaluate   # CRPS vs baselines + event AUCs
perpsfm backtest   # threshold policy after 3bps costs, block-bootstrap CIs
```

| Crate | Purpose |
| --- | --- |
| `core` | record schemas, market universe, train/val/sealed splits |
| `ingest` | Velocity REST client (retry/backoff, 403-aware) + DLOB WS collector |
| `features` | trailing-window features; forward-only targets; look-ahead unit tests |
| `model` | tiny transformer + S4D SSM + baselines (candle, from scratch) |
| `train` | pooled multi-market training loop, Metal/CPU device selection |
| `backtest` | event-driven simulator: fees, slippage, policies |
| `eval` | pinball/CRPS, AUC, 1-day block bootstrap CIs |
| `cli` | the `perpsfm` binary |

## How the JD maps to this repo

| JD item | Here |
| --- | --- |
| "Forecasting: small transformers and state-space models" | `crates/model` — 68k-param causal transformer + hand-rolled S4D, quantile heads |
| "Data: bring in new venues, measure what each adds" | multi-market pooled training vs held-out transfer (H2), data scorecard in results |
| "Decision models: entry/exit/sizing after costs" | `crates/backtest` — threshold policy, 3bps costs, bootstrap CIs |
| "Distillation and synthetic data" | Phase 6 (stretch) |
| "Pre-registered, negative results merged" | `docs/00-preregistration.md` committed before any sealed-data access |
| "Held-out markets, sealed test weeks, CIs" | BTC-PERP never in training; 2 sealed weeks; block-bootstrap CIs everywhere |
| "Eye for look-ahead bias" | features trailing-only (brute-force-tested); labels strictly-forward; train-only normalization |
| "Polars, DuckDB, Parquet" | Polars + Parquet throughout; DuckDB optional analytics |
| "Solana data or Rust" | both: Solana-native venue data, Rust end to end |
