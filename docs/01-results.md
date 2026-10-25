# 01 — Results

> All numbers below are **validation-split** (2026-09-27 → 2026-10-11). The
> sealed weeks (final 2 weeks) have NOT been opened yet — that happens once,
> after model iteration is frozen. Negative results are reported with the same
> prominence as positive ones, per the pre-registration.

## Setup

- Checkpoint: `transformer-20261025-134653.safetensors` (68k params, 11,750 steps
  (early-stopped from a 12k cap), seed 42, seq 256, batch 64, pooled
  SOL+HYPE+ETH; BTC-PERP held out).
- Val CRPS trajectory: 0.033 @ 1000 -> 0.0100 @ 4500 -> 0.0065 @ 7500 ->
  **0.005349 @ 10250 (best)** -> early stop. Monotone improvement throughout.
- Metric: CRPS from the 5-quantile grid {0.1, 0.25, 0.5, 0.75, 0.9} of the
  forward 30-minute log return.
- Baseline: climatological quantiles = empirical quantiles of train-split
  forward returns (constant per market).
- Statistics: 1-day block bootstrap, 1,000 resamples, seed 42.

## H1 — probabilistic edge: **REJECTED on val**

| market | CRPS(model) | CRPS(clim) | model/clim |
| --- | --- | --- | --- |
| SOL-PERP | 0.006630 | 0.001982 | 3.3x worse |
| HYPE-PERP | 0.004622 | 0.002371 | 2.0x worse |
| ETH-PERP | 0.004976 | 0.001486 | 3.3x worse |

The 12k-step model is 2x better than the 4k-step one (0.0053 vs 0.0100 pooled
val CRPS) but still loses to the constant climatological baseline on every val
market. Interpretation: at 1-minute granularity, 30-minute perps returns on
this venue are close to pure noise, and the input-dependent quantile spread the
model learns adds variance relative to the constant predictor. Longer training
kept helping (val CRPS fell monotonically for 10k steps) — the next lever is
capacity/inputs, not steps — but the burden of proof is now on a materially
different model, not more of the same.

## H3 — event predictability: **INCONCLUSIVE / mixed**

| market | AUC(liq burst, 60m ahead) | AUC(vol burst, 30m ahead) |
| --- | --- | --- |
| SOL-PERP | 0.423 | n/a (no positives in val) |
| HYPE-PERP | 0.174 | n/a |
| ETH-PERP | 0.867 | n/a |

With ~30 liquidations per market per quarter, positive counts in any 2-week
window are tiny; the CIs are too wide to accept or reject H3 honestly. The
honest statement is: **current event data cannot support the H3 test** —
collecting more live data (the `collect` daemon) is the fix, not more modeling.

## H4 — after-cost viability: **REJECTED at min_edge=0.002**

Threshold policy on val predictions, taker 2 bps + 1 bps slippage per side.
min_edge=0.002 (effectively always-in):

| market | trades | return | Sharpe | max DD | daily-ret 95% CI |
| --- | --- | --- | --- | --- | --- |
| SOL-PERP | 11,760 | −29.4% | −18.1 | 29.9% | [−2.4e-5, −9.2e-6] |
| HYPE-PERP | 14,582 | −20.7% | −10.0 | 23.9% | [−1.9e-5, −2.6e-6] |
| ETH-PERP | 4,376 | −29.2% | −22.9 | 29.2% | [−2.5e-5, −1.0e-5] |

min_edge=0.004 (half the trade count):

| market | trades | return | Sharpe | max DD | daily-ret 95% CI |
| --- | --- | --- | --- | --- | --- |
| SOL-PERP | 7,265 | −18.7% | −12.6 | 19.6% | [−1.7e-5, −3.6e-6] |
| HYPE-PERP | 12,667 | −15.3% | −7.7 | 19.1% | [−1.6e-5, +1.5e-6] |
| ETH-PERP | 2,588 | −19.7% | −16.4 | 21.7% | [−1.8e-5, −4.4e-6] |

At these thresholds the policy is still in the market most of the time; ~3 bps
× 2 sides × thousands of trades swamps the signal. CIs exclude zero on the
downside for 5 of 6 runs. **H4 rejected at every threshold tested.**

## Bugs found and fixed on the way (worth remembering)

1. `VarMap::load` only fills **already-registered** vars — loading into an
   empty VarMap is a silent no-op → predict ran a random-init model and
   produced ±4-scale "quantiles". Build first, load second.
2. Rolling-sum f64 drift let `ofi_30m` exceed its mathematical bound
   ([−1,1] → 33.3). Sums of non-negative quantities are clamped ≥ 0.
3. The API's gzip responses are truncated server-side (200 OK, 1KB instead of
   40KB) — client negotiates identity encoding.
4. Candle windows: `limit` caps records; a 1,000-bar window spans 60,000
   seconds (the first implementation walked 1,000-second windows: 122k
   requests instead of 123).

## What's NOT done yet

- Sealed-week eval (H1/H4 final verdicts) — model iteration should freeze
  here; the sealed weeks open once, with this checkpoint.
- S4D comparison run (H1's second arm) — training path is wired
  (`perpsfm train --model s4d`), run pending.
- H2 (held-out-market transfer) — evaluate the pooled model on BTC-PERP,
  together with the sealed eval.
- Phase 6 stretch (distillation, synthetic stress scenarios).
