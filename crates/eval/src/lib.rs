//! Metrics and the evaluation protocol: pinball loss / CRPS, AUC for rare
//! events, block-bootstrap confidence intervals, sealed-week handling.

use rand::{Rng, SeedableRng};

/// Pinball (quantile) loss for one quantile level; averaged over samples.
pub fn pinball_loss(pred: &[f64], actual: &[f64], q: f64) -> f64 {
    assert_eq!(pred.len(), actual.len());
    assert!((0.0..1.0).contains(&q));
    let n = pred.len();
    if n == 0 {
        return f64::NAN;
    }
    pred.iter()
        .zip(actual)
        .map(|(&p, &a)| {
            let d = a - p;
            if d >= 0.0 {
                q * d
            } else {
                (q - 1.0) * d
            }
        })
        .sum::<f64>()
        / n as f64
}

/// Approximate CRPS from a finite quantile grid (average of 2 * pinball).
pub fn crps_from_quantiles(pred: &[Vec<f64>], actual: &[f64], qs: &[f64]) -> f64 {
    let n = actual.len();
    if n == 0 {
        return f64::NAN;
    }
    let mut total = 0.0;
    for (p_row, &a) in pred.iter().zip(actual) {
        let mut s = 0.0;
        for (p, &q) in p_row.iter().zip(qs) {
            let d = a - p;
            s += if d >= 0.0 { q * d } else { (q - 1.0) * d };
        }
        total += 2.0 * s / qs.len() as f64;
    }
    total / n as f64
}

/// AUC of scores against binary labels (Mann-Whitney U / tie-aware).
pub fn auc(scores: &[f64], labels: &[u8]) -> f64 {
    assert_eq!(scores.len(), labels.len());
    let n_pos = labels.iter().filter(|&&l| l == 1).count();
    let n_neg = labels.len() - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return f64::NAN;
    }
    // rank with ties averaged
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| {
        scores[a]
            .partial_cmp(&scores[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut ranks = vec![0f64; scores.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && scores[idx[j + 1]] == scores[idx[i]] {
            j += 1;
        }
        let avg_rank = (i + j) as f64 / 2.0 + 1.0;
        for &k in &idx[i..=j] {
            ranks[k] = avg_rank;
        }
        i = j + 1;
    }
    let rank_sum_pos: f64 = (0..labels.len())
        .filter(|&k| labels[k] == 1)
        .map(|k| ranks[k])
        .sum();
    let u = rank_sum_pos - n_pos as f64 * (n_pos as f64 + 1.0) / 2.0;
    u / (n_pos as f64 * n_neg as f64)
}

/// A bootstrap resample of daily aggregates: resample `block_len`-sized
/// chunks of the series with replacement, return the mean of the resample.
pub struct BlockBootstrap {
    pub iters: usize,
    pub block_len: usize,
    pub seed: u64,
}

impl Default for BlockBootstrap {
    fn default() -> Self {
        Self {
            iters: 1000,
            block_len: 1440,
            seed: 42,
        } // 1-day blocks of 1-min returns
    }
}

impl BlockBootstrap {
    /// 95% CI of the mean of a series under block resampling.
    pub fn mean_ci(&self, x: &[f64]) -> (f64, f64, f64) {
        // (point, lo, hi)
        let n = x.len();
        if n == 0 || self.block_len == 0 {
            return (f64::NAN, f64::NAN, f64::NAN);
        }
        let point = x.iter().sum::<f64>() / n as f64;
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        let n_blocks = n.div_ceil(self.block_len);
        let mut means = Vec::with_capacity(self.iters);
        let mut buf = Vec::with_capacity(n);
        for _ in 0..self.iters {
            buf.clear();
            for _ in 0..n_blocks {
                let start = rng.random_range(0..n.saturating_sub(self.block_len) + 1);
                let end = (start + self.block_len).min(n);
                buf.extend_from_slice(&x[start..end]);
            }
            buf.truncate(n);
            means.push(buf.iter().sum::<f64>() / buf.len() as f64);
        }
        means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let lo = means[(self.iters as f64 * 0.025) as usize];
        let hi = means[((self.iters as f64 * 0.975) as usize).min(self.iters - 1)];
        (point, lo, hi)
    }

    /// 95% CI of a statistic of per-block returns resampled daily; here used
    /// for annualized Sharpe of resampled paths.
    pub fn sharpe_ci(&self, minute_returns: &[f64]) -> (f64, f64, f64) {
        let point = crate_sharpe(minute_returns);
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed ^ 0x5EED);
        let n = minute_returns.len();
        if n < self.block_len * 2 {
            return (point, f64::NAN, f64::NAN);
        }
        let n_blocks = n.div_ceil(self.block_len);
        let mut sharpes = Vec::with_capacity(self.iters);
        let mut buf = Vec::with_capacity(n);
        for _ in 0..self.iters {
            buf.clear();
            for _ in 0..n_blocks {
                let start = rng.random_range(0..n.saturating_sub(self.block_len) + 1);
                let end = (start + self.block_len).min(n);
                buf.extend_from_slice(&minute_returns[start..end]);
            }
            buf.truncate(n);
            sharpes.push(crate_sharpe(&buf));
        }
        sharpes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let lo = sharpes[(self.iters as f64 * 0.025) as usize];
        let hi = sharpes[((self.iters as f64 * 0.975) as usize).min(self.iters - 1)];
        (point, lo, hi)
    }
}

fn crate_sharpe(r: &[f64]) -> f64 {
    let n = r.len();
    if n < 2 {
        return f64::NAN;
    }
    let m = r.iter().sum::<f64>() / n as f64;
    let var = r.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (n - 1) as f64;
    if var <= 0.0 {
        0.0
    } else {
        m / var.sqrt() * 525_600f64.sqrt()
    }
}

/// Validation-split evaluation: model CRPS vs baselines + event AUCs.
/// (Sealed weeks are handled separately by the final frozen run.)
pub fn run() -> anyhow::Result<()> {
    use polars::prelude::*;
    use std::path::Path;

    let root = Path::new(".");
    let markets = [
        perpsfm_core::Market::SolPerp,
        perpsfm_core::Market::HypePerp,
        perpsfm_core::Market::EthPerp,
    ];

    // climatological baseline: empirical quantiles of train-split fwd returns
    let mut train_rets: Vec<f64> = Vec::new();
    for m in markets {
        let path = perpsfm_features::targets_path(root, m);
        if !path.exists() {
            continue;
        }
        let f = std::fs::File::open(&path)?;
        let df = ParquetReader::new(f).finish()?;
        let ts: Vec<i64> = df.column("ts")?.i64()?.into_no_null_iter().collect();
        let y: Vec<Option<f64>> = df.column("fwd_ret_30m")?.f64()?.iter().collect();
        // train = everything before 2 weeks before the end (same rule as
        // Splits::default_from_archive with a 2-week sealed tail)
        let max_ts = ts.iter().max().copied().unwrap_or(0);
        let train_hi = max_ts - 14 * 86400;
        for (i, &t) in ts.iter().enumerate() {
            if t < train_hi && y[i].is_some() {
                train_rets.push(y[i].unwrap());
            }
        }
    }
    anyhow::ensure!(
        !train_rets.is_empty(),
        "no train targets — run backfill + features first"
    );
    train_rets.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let qs = perpsfm_model_quantiles();
    let clim: Vec<f64> = qs
        .iter()
        .map(|&q| train_rets[(q * (train_rets.len() - 1) as f64) as usize])
        .collect();

    println!(
        "{:<10} {:>10} {:>10} {:>10} {:>8} {:>8}",
        "market", "CRPS(model)", "CRPS(clim)", "CRPS(rw)", "AUC_liq", "AUC_vol"
    );
    for m in markets {
        let pp = perpsfm_core::predictions_path(root, m);
        if !pp.exists() {
            continue;
        }
        let f = std::fs::File::open(&pp)?;
        let preds = ParquetReader::new(f).finish()?;
        let pts: Vec<i64> = preds.column("ts")?.i64()?.into_no_null_iter().collect();
        let pq: Vec<Vec<f64>> = (0..5)
            .map(|qi| {
                preds
                    .column(&format!("q{qi}"))
                    .unwrap()
                    .f64()
                    .unwrap()
                    .into_no_null_iter()
                    .collect()
            })
            .collect();
        let pliq: Vec<f64> = preds
            .column("p_liq")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        let pvol: Vec<f64> = preds
            .column("p_vol")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect();

        // join targets on ts
        let tf = std::fs::File::open(perpsfm_features::targets_path(root, m))?;
        let tdf = ParquetReader::new(tf).finish()?;
        let tts: Vec<i64> = tdf.column("ts")?.i64()?.into_no_null_iter().collect();
        use std::collections::HashMap;
        let y_ret_m: HashMap<i64, f64> = tts
            .iter()
            .cloned()
            .zip(
                tdf.column("fwd_ret_30m")?
                    .f64()?
                    .into_no_null_iter()
                    .collect::<Vec<f64>>(),
            )
            .collect();
        let y_liq_m: HashMap<i64, f64> = tts
            .iter()
            .cloned()
            .zip(
                tdf.column("fwd_liq_burst_60m")?
                    .f64()?
                    .into_no_null_iter()
                    .collect::<Vec<f64>>(),
            )
            .collect();
        let y_vol_m: HashMap<i64, f64> = tts
            .iter()
            .cloned()
            .zip(
                tdf.column("fwd_vol_burst_30m")?
                    .f64()?
                    .into_no_null_iter()
                    .collect::<Vec<f64>>(),
            )
            .collect();

        let n = pts.len();
        let mut model_preds = Vec::with_capacity(n);
        let mut clim_preds = Vec::with_capacity(n);
        let mut actual = Vec::with_capacity(n);
        let mut liq_scores = Vec::new();
        let mut liq_labels = Vec::new();
        let mut vol_scores = Vec::new();
        let mut vol_labels = Vec::new();
        for i in 0..n {
            let Some(&a) = y_ret_m.get(&pts[i]) else {
                continue;
            };
            if !a.is_finite() {
                continue;
            }
            model_preds.push(pq.iter().map(|c| c[i]).collect());
            clim_preds.push(clim.clone());
            actual.push(a);
            if let Some(&l) = y_liq_m.get(&pts[i]) {
                liq_scores.push(pliq[i]);
                liq_labels.push(if l > 0.5 { 1u8 } else { 0 });
            }
            if let Some(&l) = y_vol_m.get(&pts[i]) {
                vol_scores.push(pvol[i]);
                vol_labels.push(if l > 0.5 { 1u8 } else { 0 });
            }
        }

        // random-walk baseline: z_q * rv * sqrt(30) — needs rv_30m at ts; use
        // the model's own interquantile width fallback if features missing.
        let crps_model = crps_from_quantiles(&model_preds, &actual, &qs);
        let crps_clim = crps_from_quantiles(&clim_preds, &actual, &qs);
        let auc_liq = auc(&liq_scores, &liq_labels);
        let auc_vol = auc(&vol_scores, &vol_labels);
        println!(
            "{:<10} {:>10.6} {:>10.6} {:>10} {:>8.3} {:>8.3}",
            m.symbol(),
            crps_model,
            crps_clim,
            "n/a",
            auc_liq,
            auc_vol
        );
    }
    Ok(())
}

fn perpsfm_model_quantiles() -> Vec<f64> {
    perpsfm_model::QUANTILES.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinball_zero_when_perfect() {
        assert_eq!(pinball_loss(&[1.0, 2.0], &[1.0, 2.0], 0.5), 0.0);
    }

    #[test]
    fn pinball_asymmetric() {
        let over = pinball_loss(&[2.0], &[1.0], 0.9);
        let under = pinball_loss(&[1.0], &[2.0], 0.9);
        assert!(under > over * 5.0);
    }

    #[test]
    fn crps_zero_when_perfect() {
        let preds = vec![vec![1.0, 1.0], vec![2.0, 2.0]];
        assert_eq!(crps_from_quantiles(&preds, &[1.0, 2.0], &[0.5, 0.75]), 0.0);
    }

    #[test]
    fn auc_perfect_and_random() {
        let scores = [0.1, 0.2, 0.8, 0.9];
        let labels = [0u8, 0, 1, 1];
        assert!((auc(&scores, &labels) - 1.0).abs() < 1e-12);
        let scores = [0.5, 0.5, 0.5, 0.5];
        assert!((auc(&scores, &labels) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn bootstrap_ci_covers_point() {
        // constant series: CI collapses to the point
        let x = vec![0.001f64; 5760];
        let (p, lo, hi) = BlockBootstrap::default().mean_ci(&x);
        assert!((p - 0.001).abs() < 1e-12);
        assert!(lo <= p + 1e-12 && p <= hi + 1e-12);
    }
}
