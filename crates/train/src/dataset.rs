//! Dataset assembly: windows over the regularized 1-minute grid, pooled
//! across markets, with normalization statistics computed on the TRAIN split
//! only (pre-registered look-ahead rule).

use anyhow::{Context, Result};
use polars::prelude::*;
use std::path::Path;

use perpsfm_core::{Market, Splits};
use perpsfm_features::FEATURE_COLUMNS;

/// Feature columns fed to the model (order fixed = model input order).
/// FEATURE_COLUMNS[1..] plus the two calendar features appended here.
pub const CALENDAR_FEATURES: usize = 2;

pub struct MarketSeries {
    pub market: Market,
    pub ts: Vec<i64>,
    /// (n_rows, n_features), NaN where undefined
    pub x: Vec<Vec<f64>>, // row-major: x[row][col]
    pub y_ret: Vec<f64>,
    pub y_liq: Vec<f64>,
    pub y_vol: Vec<f64>,
}

impl MarketSeries {
    pub fn n_features(&self) -> usize {
        self.x.first().map(|r| r.len()).unwrap_or(0)
    }
}

fn read_f64_col(path: &Path, col: &str) -> Result<(Vec<i64>, Vec<f64>)> {
    let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let df = ParquetReader::new(file)
        .finish()
        .map_err(|e| anyhow::anyhow!("read {}: {e}", path.display()))?;
    let ts: Vec<i64> = df.column("ts")?.i64()?.into_no_null_iter().collect();
    let v: Vec<f64> = df
        .column(col)
        .map_err(|e| anyhow::anyhow!("col {col} in {}: {e}", path.display()))?
        .f64()?
        .into_no_null_iter()
        .collect();
    anyhow::ensure!(v.len() == ts.len(), "col {col} length mismatch in {path:?}");
    Ok((ts, v))
}

fn read_f64_cols(path: &Path, cols: &[&str]) -> Result<(Vec<i64>, Vec<Vec<f64>>)> {
    let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let df = ParquetReader::new(file)
        .finish()
        .map_err(|e| anyhow::anyhow!("read {}: {e}", path.display()))?;
    let ts: Vec<i64> = df.column("ts")?.i64()?.into_no_null_iter().collect();
    let mut data = Vec::with_capacity(cols.len());
    for c in cols {
        let v: Vec<f64> = df
            .column(c)
            .map_err(|e| anyhow::anyhow!("col {c} in {}: {e}", path.display()))?
            .f64()?
            .into_no_null_iter()
            .collect();
        anyhow::ensure!(v.len() == ts.len(), "col {c} length mismatch in {path:?}");
        data.push(v);
    }
    let rows = (0..ts.len())
        .map(|i| data.iter().map(|col| col[i]).collect::<Vec<_>>())
        .collect();
    Ok((ts, rows))
}

/// Load one market's features+targets; appends minute-of-day sin/cos.
pub fn load_series(root: &Path, market: Market) -> Result<MarketSeries> {
    let (ts, mut x) = read_f64_cols(
        &perpsfm_features::features_path(root, market),
        &FEATURE_COLUMNS[1..], // skip "ts"
    )?;
    let tpath = perpsfm_features::targets_path(root, market);
    let (t_ts, y_ret) = read_f64_col(&tpath, "fwd_ret_30m")?;
    let (_, y_liq) = read_f64_col(&tpath, "fwd_liq_burst_60m")?;
    let (_, y_vol) = read_f64_col(&tpath, "fwd_vol_burst_30m")?;
    anyhow::ensure!(t_ts == ts, "targets ts mismatch for {market:?}");

    for (i, row) in x.iter_mut().enumerate() {
        let ang = ((ts[i].rem_euclid(86400)) as f64 / 86400.0) * std::f64::consts::TAU;
        row.push(ang.sin());
        row.push(ang.cos());
    }

    Ok(MarketSeries {
        market,
        ts,
        x,
        y_ret,
        y_liq,
        y_vol,
    })
}

/// Per-feature (mean, std) computed over the given ts range only.
pub fn norm_stats(series: &[&MarketSeries], lo: i64, hi: i64) -> Vec<(f64, f64)> {
    let n_features = series[0].n_features();
    let mut stats = vec![(0.0f64, 1.0f64); n_features];
    for f in 0..n_features {
        let mut sum = 0.0;
        let mut sumsq = 0.0;
        let mut n = 0u64;
        for s in series {
            for (row, &t) in s.x.iter().zip(&s.ts) {
                if t < lo || t >= hi {
                    continue;
                }
                let v = row[f];
                if v.is_finite() {
                    sum += v;
                    sumsq += v * v;
                    n += 1;
                }
            }
        }
        if n > 0 {
            let m = sum / n as f64;
            let var = (sumsq / n as f64 - m * m).max(0.0);
            stats[f] = (m, var.sqrt().max(1e-8));
        }
    }
    stats
}

/// Normalized copy of a series (NaN stays NaN; converted to 0 at batch time).
pub fn normalize(s: &MarketSeries, stats: &[(f64, f64)]) -> MarketSeries {
    let x =
        s.x.iter()
            .map(|row| {
                row.iter()
                    .zip(stats)
                    .map(|(&v, &(m, sd))| {
                        if v.is_finite() {
                            (v - m) / sd
                        } else {
                            f64::NAN
                        }
                    })
                    .collect()
            })
            .collect();
    MarketSeries {
        market: s.market,
        ts: s.ts.clone(),
        x,
        y_ret: s.y_ret.clone(),
        y_liq: s.y_liq.clone(),
        y_vol: s.y_vol.clone(),
    }
}

/// A supervised sample: window end index into a specific series.
#[derive(Debug, Clone, Copy)]
pub struct Sample {
    pub series: usize,
    pub t: usize, // window ends at t (inclusive); label = target row t
}

/// Collect sampleable indices: t needs seq_len history and finite labels.
pub fn collect_samples(series: &[MarketSeries], seq_len: usize, lo: i64, hi: i64) -> Vec<Sample> {
    let mut out = Vec::new();
    for (si, s) in series.iter().enumerate() {
        for (t, &ts) in s.ts.iter().enumerate() {
            if ts < lo || ts >= hi || t < seq_len - 1 {
                continue;
            }
            if s.y_ret[t].is_finite() {
                out.push(Sample { series: si, t });
            }
        }
    }
    out
}

/// Gather a batch as (B, T, F) f32 tensor data plus targets.
pub struct Batch {
    pub x: Vec<f32>, // B*T*F row-major
    pub market_ids: Vec<i64>,
    pub y_ret: Vec<f64>,
    pub y_liq: Vec<f64>,
    pub y_vol: Vec<f64>,
}

pub fn gather_batch(
    series: &[MarketSeries],
    samples: &[Sample],
    idx: &[usize],
    seq_len: usize,
    n_features: usize,
) -> Batch {
    let b = idx.len();
    let mut x = vec![0f32; b * seq_len * n_features];
    let mut market_ids = Vec::with_capacity(b);
    let mut y_ret = Vec::with_capacity(b);
    let mut y_liq = Vec::with_capacity(b);
    let mut y_vol = Vec::with_capacity(b);
    for (bi, &si) in idx.iter().enumerate() {
        let smp = samples[si];
        let s = &series[smp.series];
        market_ids.push(s.market as i64);
        y_ret.push(s.y_ret[smp.t]);
        y_liq.push(s.y_liq[smp.t]);
        y_vol.push(s.y_vol[smp.t]);
        for (w, row) in s.x[smp.t + 1 - seq_len..=smp.t].iter().enumerate() {
            let dst = &mut x[bi * seq_len * n_features + w * n_features..][..n_features];
            for (f, &v) in row.iter().enumerate() {
                dst[f] = if v.is_finite() { v as f32 } else { 0.0 };
            }
        }
    }
    Batch {
        x,
        market_ids,
        y_ret,
        y_liq,
        y_vol,
    }
}

/// Samples over a single (already normalized) series.
pub fn collect_slices_samples(s: &MarketSeries, seq_len: usize, lo: i64, hi: i64) -> Vec<Sample> {
    (0..s.ts.len())
        .filter(|&t| {
            let ts = s.ts[t];
            ts >= lo && ts < hi && t + 1 >= seq_len && s.y_ret[t].is_finite()
        })
        .map(|t| Sample { series: 0, t })
        .collect()
}

/// Gather a batch from a single series; `samples[i].t` indexes into `s`.
pub fn gather_window_batch(
    s: &MarketSeries,
    samples: &[Sample],
    seq_len: usize,
    n_features: usize,
) -> Batch {
    let b = samples.len();
    let mut x = vec![0f32; b * seq_len * n_features];
    let mut market_ids = Vec::with_capacity(b);
    let mut y_ret = Vec::with_capacity(b);
    let mut y_liq = Vec::with_capacity(b);
    let mut y_vol = Vec::with_capacity(b);
    for (bi, smp) in samples.iter().enumerate() {
        market_ids.push(s.market as i64);
        y_ret.push(s.y_ret[smp.t]);
        y_liq.push(s.y_liq[smp.t]);
        y_vol.push(s.y_vol[smp.t]);
        for (w, row) in s.x[smp.t + 1 - seq_len..=smp.t].iter().enumerate() {
            let dst = &mut x[bi * seq_len * n_features + w * n_features..][..n_features];
            for (f, &v) in row.iter().enumerate() {
                dst[f] = if v.is_finite() { v as f32 } else { 0.0 };
            }
        }
    }
    Batch {
        x,
        market_ids,
        y_ret,
        y_liq,
        y_vol,
    }
}

/// Default splits derived from the archive window.
pub fn default_splits(series: &[MarketSeries]) -> Splits {
    let lo = series
        .iter()
        .map(|s| s.ts.first().copied().unwrap_or(i64::MAX))
        .min()
        .unwrap_or(0);
    let hi = series
        .iter()
        .map(|s| s.ts.last().copied().unwrap_or(0))
        .max()
        .unwrap_or(1);
    let start = chrono::DateTime::from_timestamp(lo, 0).unwrap_or_default();
    let end = chrono::DateTime::from_timestamp(hi, 0).unwrap_or_default();
    Splits::default_from_archive(start, end)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_series() -> MarketSeries {
        let n = 600;
        MarketSeries {
            market: Market::SolPerp,
            ts: (0..n as i64).map(|i| 1_700_000_000 + i * 60).collect(),
            x: (0..n)
                .map(|i| vec![i as f64, (i as f64 * 0.1).sin()])
                .collect(),
            y_ret: (0..n).map(|i| (i as f64 * 0.01).sin()).collect(),
            y_liq: vec![0.0; n],
            y_vol: vec![0.0; n],
        }
    }

    #[test]
    fn samples_respect_history_and_range() {
        let s = dummy_series();
        let series = vec![s];
        let seq = 100;
        let samples = collect_samples(&series, seq, series[0].ts[0], series[0].ts[600 - 1]);
        assert_eq!(samples.len(), 600 - seq + 1 - 1); // last row has NaN-free labels here; -1 for hi bound
        let smp = samples[0];
        assert_eq!(smp.t, seq - 1);
        let batch = gather_batch(&series, &samples, &[0], seq, 2);
        assert_eq!(batch.x.len(), seq * 2);
        assert_eq!(batch.y_ret.len(), 1);
    }

    #[test]
    fn normalization_uses_only_train_range() {
        let s = dummy_series();
        let series = [s];
        let train_hi = series[0].ts[300];
        let stats = norm_stats(&series.iter().collect::<Vec<_>>(), 0, train_hi);
        // feature 0 ranges over [0,300) * 1.0 -> mean ~149.5
        assert!((stats[0].0 - 149.5).abs() < 0.5);
    }
}
