# Portfolio Project: Solana Perps "Foundation Model" — Full Rust

A 2–4 week, public-repo project that mirrors every work area in the JD (forecasting, data, decision models, distillation) using **Solana perps data only**, built end-to-end in **Rust**. MacBook-only training (tiny models, Metal/CPU). The repo + results write-up doubles as your writing sample for the application.

**Caveat we'll manage:** the JD's hard requirement is Python/PyTorch. Mitigation: we use Polars/DuckDB/Parquet (all named in the JD, all Rust-native), implement the models *from scratch* in candle, and frame this in the README as deeper ML understanding + the JD's "Rust" and "Solana data" nice-to-haves.

## Stack
- **Rust workspace** (edition 2021), `cargo` + `just`/`make` task runner
- **candle** (Hugging Face Rust ML) for tensors/training — Metal backend for Apple Silicon; smoke-tested in Phase 0, CPU fallback (models are tiny)
- **Polars (Rust)** for feature engineering, **Parquet** for storage, **DuckDB (Rust crate)** for analytics queries
- Data: **Drift Protocol** (Solana's largest perps DEX) — fills, funding, liquidations. No Python anywhere.

## Repo layout
```
perps-fm/
├── Cargo.toml               # workspace
├── crates/
│   ├── ingest/              # Drift dump download + parse → normalized Parquet
│   ├── features/            # trailing-window features, labels (look-ahead-safe)
│   ├── model/               # candle: small transformer + compact SSM, quantile heads
│   ├── train/               # training loop, Metal/CPU device, checkpoints
│   ├── backtest/            # event-driven simulator: fees, slippage, sizing
│   ├── eval/                # CRPS/pinball, block-bootstrap CIs, sealed-week protocol
│   └── cli/                 # download / build-features / train / eval / backtest
├── docs/
│   ├── 00-preregistration.md   # hypotheses + success criteria, written BEFORE sealed weeks
│   └── 01-results.md           # results incl. negative results (JD: "negative results get merged")
├── .github/workflows/ci.yml # fmt + clippy + tests
└── README.md                # results tables, plots, framing for the application
```

## Phases

**Phase 0 — Risk spikes (days 1–2).** Three go/no-go checks before committing: (a) verify Drift historical data dumps exist and download one month of SOL-PERP fills/funding/liquidations — fallbacks if not: Dune/Flipside Solana tables, or live collection via Drift program logs on a Helius free-tier RPC; (b) candle-on-Metal smoke test: train a toy transformer on synthetic data, confirm speed; (c) scaffold workspace + CI.

**Phase 1 — Data pipeline (JD: "Data: bring in new venues, measure what each adds").** Ingest SOL-PERP, BTC-PERP, ETH-PERP + 2–3 alt perps (several months). Normalize fills into a single event schema → partitioned Parquet. Build 1s/1m bars from fills (OHLCV, buy/sell volume imbalance), liquidation event stream, funding rate series. DuckDB-backed "data scorecard" notebook-style doc: coverage, gaps, what each market adds.

**Phase 2 — Features & labels with look-ahead hygiene.** Trailing-window features only: returns, realized vol, order-flow imbalance, liquidation intensity, funding. Targets: forward return quantiles, liquidation-burst indicator. Unit tests prove features at time *t* use only data ≤ *t* (JD: "eye for look-ahead bias" — this is a key differentiator to show explicitly).

**Phase 3 — Models (JD: "small transformers and state-space models").** Baselines first: persistence, rolling-mean, simple linear AR. Then (a) small transformer (~1–4M params) with multi-market embedding and quantile/probabilistic heads; (b) compact SSM (S4D-style layer you implement yourself). Loss: pinball/CRPS. All trained on MacBook in minutes-per-run ("small, cheap experiments").

**Phase 4 — Decision layer (JD: "entry, exit and sizing policies, tested after costs").** Event-driven simulator: taker/maker fees + slippage model. Policies driven by forecasts: threshold entry/exit, volatility-targeted sizing. Report after-cost PnL, Sharpe, max drawdown.

**Phase 5 — Evaluation protocol (JD "How we work", mirrored exactly).** Held-out markets (2 perps never seen in training), sealed test weeks (final weeks untouched until the end), block-bootstrap confidence intervals on every headline number, pre-registration doc written before opening sealed data, negative results reported alongside wins.

**Phase 6 — Stretch (JD: "Distillation and synthetic data").** Distill teacher → tiny student (fewer layers), report speed-vs-quality Pareto; bootstrap-style synthetic stress scenarios replaying rare liquidation cascades.

## Deliverables
- Working Rust repo with CI, tests, and reproducible `cli` commands
- `docs/00-preregistration.md` and `docs/01-results.md` (with plots)
- README with after-cost results tables — this is the artifact you link in the application, plus a short "why Rust for market ML" paragraph addressing the Python/PyTorch point head-on

## Known risks
1. **Drift dump availability** — resolved by Phase 0 spike before anything else; fallbacks identified.
2. **candle Metal maturity** — smoke-tested in Phase 0; CPU fallback is acceptable for 1–4M param models.
3. **SSM from scratch** — scoped to a single-layer S4D variant; transformer path is the primary deliverable, SSM is the comparison.
