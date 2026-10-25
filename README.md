# perps-fm — a miniature Solana-native market foundation model

A 2–4 week portfolio project mirroring the "Junior Research Scientist: Market
Foundation Model" role: one tiny sequence model trained across Solana perps
markets (Velocity, formerly Drift), evaluated like a research shop would —
pre-registered hypotheses, held-out markets, sealed test weeks, confidence
intervals, and after-cost backtests.

Built entirely in **Rust**: ingestion (Velocity Data API + DLOB websocket),
feature engineering (Polars), models (candle, from scratch), training (Metal
backend), simulation, and evaluation.

> Work in progress — phases land as separate commits.
> See `docs/00-data-source.md`, `docs/00-preregistration.md`, `docs/01-results.md`.

## Quick start

```sh
just backfill   # pull the Velocity archive into data/ (parquet)
just collect    # stream live trades + orderbook to data/live/ (until Ctrl-C)
just check      # fmt + clippy + tests
```

## Layout

| Crate | Purpose |
| --- | --- |
| `core` | shared record schemas, market universe, splits |
| `ingest` | Velocity REST backfill → Parquet; DLOB WS live collector |
| `features` | look-ahead-safe trailing features + forward targets |
| `model` | tiny transformer, S4D SSM, non-neural baselines (candle) |
| `train` | training loop, Metal/CPU device selection |
| `backtest` | event-driven simulator with fees + slippage |
| `eval` | pinball/CRPS, block-bootstrap CIs, sealed-week protocol |
| `cli` | the `perpsfm` binary wiring it all together |
