# 01 — Results

> All numbers below are **validation-split** (2026-09-27 → 2026-10-11). The
> sealed weeks (final 2 weeks) have NOT been opened yet — that happens once,
> after model iteration is frozen. Negative results are reported with the same
> prominence as positive ones, per the pre-registration.

## Setup

- Checkpoint: `transformer-20261025-115255.safetensors` (68k params, 4,000 steps,
  seed 42, seq 256, batch 64, pooled SOL+HYPE+ETH; BTC-PERP held out).
- Metric: CRPS from the 5-quantile grid {0.1, 0.25, 0.5, 0.75, 0.9} of the
  forward 30-minute log return.
- Baseline: climatological quantiles = empirical quantiles of train-split
  forward returns (constant per market).
- Statistics: 1-day block bootstrap, 1,000 resamples, seed 42.

## H1 — probabilistic edge: **REJECTED on val (so far)**

| market | CRPS(model) | CRPS(clim) | model/clim |
| --- | --- | --- | --- |
| SOL-PERP | 0.012845 | 0.001982 | 6.5x worse |
| HYPE-PERP | 0.006192 | 0.002371 | 2.6x worse |
| ETH-PERP | 0.011424 | 0.001486 | 7.7x worse |

The trained model loses to the constant climatological baseline on every val
market. Interpretation: at 1-minute granularity, 30-minute perps returns on
this venue are close to pure noise, and the input-dependent quantile spread the
model learns adds variance relative to the constant predictor.

**Caveat (pre-registered as a legitimate continuation)**: val CRPS was still
improving monotonically when the run hit its step cap
(0.0262 @ 750 → 0.0139 @ 2000 → 0.0100 @ 3750). A 12,000-step run is in
flight; this section will be updated with its numbers before the sealed eval.

## H3 — event predictability: **INCONCLUSIVE / mixed**

| market | AUC(liq burst, 60m ahead) | AUC(vol burst, 30m ahead) |
| --- | --- | --- |
| SOL-PERP | 0.396 | n/a (no positives in val) |
| HYPE-PERP | 0.220 | n/a |
| ETH-PERP | 0.739 | n/a |

With ~30 liquidations per market per quarter, positive counts in any 2-week
window are tiny; the CIs are too wide to accept or reject H3 honestly. The
honest statement is: **current event data cannot support the H3 test** —
collecting more live data (the `collect` daemon) is the fix, not more modeling.

## H4 — after-cost viability: **REJECTED at min_edge=0.002**

Threshold policy on val predictions, taker 2 bps + 1 bps slippage per side:

| market | trades | return | Sharpe | max DD | daily-ret 95% CI |
| --- | --- | --- | --- | --- | --- |
| SOL-PERP | 6,524 | −28.9% | −16.9 | 29.9% | [−2.7e-5, −6.9e-6] |
| HYPE-PERP | 5,877 | −37.3% | −20.1 | 38.2% | [−3.6e-5, −1.1e-5] |
| ETH-PERP | 3,642 | −38.6% | −30.9 | 38.6% | [−3.1e-5, −1.8e-5] |

At a threshold this low the policy is essentially always-in; ~3 bps × 2 sides
× thousands of trades swamps any signal. CIs exclude zero on the downside for
all three markets. Any retest at a higher min_edge must trade far less.

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

- Sealed-week eval (H1/H4 final verdicts) — after the longer run + freezing.
- S4D comparison run (H1's second arm) — training path is wired
  (`perpsfm train --model s4d`), run pending.
- H2 (held-out-market transfer) — needs the same trained models evaluated on
  BTC-PERP, which happens together with the sealed eval.
- Phase 6 stretch (distillation, synthetic stress scenarios).
