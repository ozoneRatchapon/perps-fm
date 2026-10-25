# 00 — Pre-registration

> **Status: DRAFT.** Finalized and committed **before** the sealed test weeks are
> opened for the first time. After that this file is immutable — changes require a
> new section appended below, never an edit (mirrors "every study is pre-registered").

## Claims under test

- **H1 (probabilistic edge).** A small causal transformer trained on multi-market
  1-minute bars predicts the forward 30-minute return distribution with lower CRPS
  than (a) persistence, (b) a rolling-mean baseline, and (c) an S4D-style SSM of
  similar parameter count — on **validation weeks**.
- **H2 (cross-market transfer).** Training on 3 markets improves held-out-market
  (BTC-PERP) CRPS relative to single-market training — testing whether pooled
  training transfers (the "foundation model" premise at miniature scale).
- **H3 (event targets).** Liquidation-burst and volume-burst events in the next
  hour are predictable above base rate (AUC > 0.5 with bootstrap CI excluding 0.5).
- **H4 (after-cost viability).** Threshold policies on the H1 forecasts, executed
  with the taker fee + slippage model, produce positive after-cost PnL with a
  block-bootstrap 95% CI excluding zero on the **sealed weeks**.

Negative results are reported with the same prominence as positive ones.

## Protocol

- **Splits** (fixed at commit time, from `perpsfm_core::Splits::default_from_archive`):
  - train: 2026-08-01 → (end − 4 weeks)
  - val: (end − 4 weeks) → (end − 2 weeks) — used for early stopping/model selection
  - **sealed: final 2 weeks of collected data** — opened exactly once, after all
    model iteration is frozen.
- **Held-out market:** BTC-PERP never appears in any training or val loss.
- **Look-ahead hygiene:** features use trailing windows only (unit-tested); labels
  use strictly-forward windows; normalization statistics computed on train only.
- **Statistics:** every headline number reports a block-bootstrap 95% CI
  (block length = 1 day, 1000 resamples, fixed seed 42).
- **Costs:** taker fee 2 bps per side (from fill data) + slippage model calibrated
  in Phase 4; reported PnL is always after costs.
- **Runs:** small and cheap; every run gets a config hash; configs + metrics land
  in `runs/` (gitignored, summary in 01-results.md).

## Success criteria

- H1: CRPS improvement over the best baseline > 0 with bootstrap CI excluding 0.
- H2: strictly lower held-out-market CRPS for pooled vs single-market.
- H3: AUC CI excludes 0.5 on val (H3 is *not* evaluated on sealed weeks — event
  counts there are too small; sealed weeks carry H1/H4).
- H4: after-cost PnL CI excludes 0 on sealed weeks. Otherwise H4 is **rejected**
  and reported as such.

## Deviations

(none yet)
