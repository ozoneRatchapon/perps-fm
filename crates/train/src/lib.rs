//! Training loop and device selection. Models are tiny on purpose: every run
//! should finish in minutes on this MacBook (Phase 0b smoke test: CPU wins on
//! small configs, Metal ~3x faster at seq 256 / batch 256).

pub mod dataset;

use anyhow::{Context as _, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, Optimizer, VarBuilder, VarMap};
use polars::prelude::*;
use rand::SeedableRng;
use std::path::{Path, PathBuf};

use perpsfm_core::Market;
use perpsfm_model::transformer::{build as build_transformer, quantile_loss, TransformerConfig};
use perpsfm_model::QUANTILES;

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum DeviceChoice {
    Auto,
    Cpu,
    Metal,
}

pub fn pick_device(choice: DeviceChoice) -> Result<Device> {
    match choice {
        DeviceChoice::Cpu => Ok(Device::Cpu),
        #[cfg(feature = "metal")]
        DeviceChoice::Metal => Device::new_metal(0).context("Metal device unavailable"),
        #[cfg(not(feature = "metal"))]
        DeviceChoice::Metal => anyhow::bail!("built without --features metal"),
        #[cfg(feature = "metal")]
        DeviceChoice::Auto => Ok(Device::new_metal(0).unwrap_or(Device::Cpu)),
        #[cfg(not(feature = "metal"))]
        DeviceChoice::Auto => Ok(Device::Cpu),
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrainConfig {
    pub seq_len: usize,
    pub batch_size: usize,
    pub steps: usize,
    pub lr: f64,
    pub eval_every: usize,
    pub patience: usize,
    pub seed: u64,
    pub model: ModelKind,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            seq_len: 256,
            batch_size: 64,
            steps: 4000,
            lr: 3e-4,
            eval_every: 250,
            patience: 6,
            seed: 42,
            model: ModelKind::Transformer,
        }
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, clap::ValueEnum,
)]
pub enum ModelKind {
    Transformer,
    S4d,
}

pub struct RunOutput {
    pub checkpoint: PathBuf,
    pub best_val_crps: f64,
    pub steps_trained: usize,
}

/// Softplus (numerically stable).
fn softplus(x: &Tensor) -> Result<Tensor> {
    let zero = x.zeros_like()?;
    let a = x.maximum(&zero)?;
    let b = (x.ones_like()? + x.neg()?.exp()?)?;
    Ok(a.add(&b.log()?)?)
}

/// BCE with logits: max(x,0) - x*y + softplus(-|x|).
#[allow(dead_code)]
fn bce_with_logits(logits: &Tensor, target: &Tensor) -> Result<Tensor> {
    let abs = logits.abs()?;
    let relu = logits.maximum(&logits.zeros_like()?)?;
    let sp = softplus(&abs.neg()?)?;
    let xy = logits.broadcast_mul(&target.to_dtype(logits.dtype())?)?;
    Ok((relu - xy + sp)?.mean_all()?)
}

struct BatchTensors {
    x: Tensor,
    market_ids: Tensor,
    y_ret: Tensor,
}

fn to_device(
    batch: &dataset::Batch,
    n_features: usize,
    seq_len: usize,
    dev: &Device,
) -> Result<BatchTensors> {
    let b = batch.market_ids.len();
    Ok(BatchTensors {
        x: Tensor::from_slice(&batch.x, (b, seq_len, n_features), dev)?,
        market_ids: Tensor::from_slice(&batch.market_ids, (b,), dev)?,
        y_ret: Tensor::from_vec(batch.y_ret.clone(), (b,), dev)?,
    })
}

/// Train the transformer: quantile loss on fwd returns, early-stop on val
/// CRPS, checkpoint to runs/. Market pooling: SOL/HYPE/ETH train, BTC held
/// out (never in train OR val loss; used only in the sealed evaluation).
pub fn train_transformer(root: &Path, cfg: &TrainConfig) -> Result<RunOutput> {
    let dev = pick_device(DeviceChoice::Auto)?;
    tracing::info!("device: {dev:?}");

    let all: Vec<dataset::MarketSeries> = Market::ALL
        .iter()
        .map(|m| dataset::load_series(root, *m))
        .collect::<Result<Vec<_>>>()?;
    let n_features = all[0].n_features();

    let splits = dataset::default_splits(&all);
    tracing::info!(
        "splits: train [{}, {})  val [{}, {})  SEALED [{}, {}) (untouched)",
        splits.train_start,
        splits.train_end,
        splits.val_start,
        splits.val_end,
        splits.sealed_start,
        splits.sealed_end
    );

    let train_markets = [Market::SolPerp, Market::HypePerp, Market::EthPerp];
    let train_stats = dataset::norm_stats(
        &all.iter()
            .filter(|s| train_markets.contains(&s.market))
            .collect::<Vec<_>>(),
        splits.train_start.timestamp(),
        splits.train_end.timestamp(),
    );
    let train_series: Vec<dataset::MarketSeries> = all
        .iter()
        .filter(|s| train_markets.contains(&s.market))
        .map(|s| dataset::normalize(s, &train_stats))
        .collect();
    let val_series: Vec<dataset::MarketSeries> = all
        .iter()
        .filter(|s| train_markets.contains(&s.market))
        .map(|s| dataset::normalize(s, &train_stats))
        .collect();

    for s in &train_series {
        tracing::info!(
            "series {:?}: ts={} x={} y_ret={}",
            s.market,
            s.ts.len(),
            s.x.len(),
            s.y_ret.len()
        );
    }
    let train_samples = dataset::collect_samples(
        &train_series,
        cfg.seq_len,
        splits.train_start.timestamp(),
        splits.train_end.timestamp(),
    );
    let val_samples = dataset::collect_samples(
        &val_series,
        cfg.seq_len,
        splits.val_start.timestamp(),
        splits.val_end.timestamp(),
    );
    tracing::info!(
        "train samples: {}  val samples: {}",
        train_samples.len(),
        val_samples.len()
    );
    anyhow::ensure!(
        !train_samples.is_empty(),
        "no training samples — run `perpsfm backfill && perpsfm features` first"
    );

    let tcfg = TransformerConfig {
        n_features,
        seq_len: cfg.seq_len,
        ..Default::default()
    };
    let (model, varmap, mut opt) = build_transformer(&tcfg, &dev)?;
    tracing::info!("transformer params ~{}", tcfg.n_params_estimate());

    let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed);
    let mut best_crps = f64::INFINITY;
    let mut best_steps = 0usize;
    let mut stale = 0usize;

    for step in 1..=cfg.steps {
        use rand::Rng;
        let idx: Vec<usize> = (0..cfg.batch_size)
            .map(|_| rng.random_range(0..train_samples.len()))
            .collect();
        let batch =
            dataset::gather_batch(&train_series, &train_samples, &idx, cfg.seq_len, n_features);
        let b = to_device(&batch, n_features, cfg.seq_len, &dev)?;
        let out = model.forward(&b.x, &b.market_ids)?;
        let loss = quantile_loss(&out.ret_quantiles, &b.y_ret, &QUANTILES)?;
        opt.backward_step(&loss)?;

        if step % 100 == 0 {
            let l: f64 = loss.to_scalar::<f32>()? as f64;
            tracing::info!("step {step} pinball {l:.6}");
        }

        if step % cfg.eval_every == 0 || step == cfg.steps {
            let crps = eval_crps(&dev, &tcfg, &varmap, &val_series, &val_samples, cfg)?;
            if crps < best_crps {
                best_crps = crps;
                best_steps = step;
                stale = 0;
                save_checkpoint(root, &varmap, cfg, &tcfg, best_crps)?;
                tracing::info!("step {step} val CRPS {crps:.6} ** best (saved)");
            } else {
                stale += 1;
                tracing::info!("step {step} val CRPS {crps:.6} (best {best_crps:.6} @ {best_steps}, stale {stale}/{})", cfg.patience);
                if stale >= cfg.patience {
                    tracing::info!("early stop at step {step}");
                    break;
                }
            }
        }
    }

    Ok(RunOutput {
        checkpoint: checkpoint_path(root),
        best_val_crps: best_crps,
        steps_trained: best_steps,
    })
}

pub fn checkpoint_path(root: &Path) -> PathBuf {
    latest_checkpoint(root).unwrap_or_else(|| root.join("runs").join("transformer.safetensors"))
}

/// Newest checkpoint by mtime (predict/evaluate default).
pub fn latest_checkpoint(root: &Path) -> Option<PathBuf> {
    let dir = root.join("runs");
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    for e in std::fs::read_dir(&dir).ok()? {
        let p = e.ok()?.path();
        if p.extension()?.to_str()? == "safetensors" {
            let m = p.metadata().ok()?.modified().ok()?;
            if best.as_ref().map(|(t, _)| m > *t).unwrap_or(true) {
                best = Some((m, p));
            }
        }
    }
    best.map(|(_, p)| p)
}

fn save_checkpoint(
    root: &Path,
    varmap: &VarMap,
    cfg: &TrainConfig,
    tcfg: &TransformerConfig,
    val_crps: f64,
) -> Result<()> {
    let dir = root.join("runs");
    std::fs::create_dir_all(&dir)?;
    let stamp = chrono::Utc::now().format("%Y%m%d-%H%M%S");
    let path = dir.join(format!("transformer-{stamp}.safetensors"));
    varmap.save(&path)?;
    let meta = serde_json::json!({
        "model": "transformer",
        "checkpoint": path.file_name().unwrap().to_str().unwrap(),
        "train_config": cfg,
        "model_config": tcfg,
        "quantiles": QUANTILES,
        "best_val_crps": val_crps,
    });
    std::fs::write(
        dir.join("transformer.meta.json"),
        serde_json::to_string_pretty(&meta)?,
    )?;
    Ok(())
}

/// CRPS of the model over a sample set (batched inference, shuffled).
pub fn eval_crps(
    dev: &Device,
    tcfg: &TransformerConfig,
    varmap: &VarMap,
    series: &[dataset::MarketSeries],
    samples: &[dataset::Sample],
    cfg: &TrainConfig,
) -> Result<f64> {
    let vs = VarBuilder::from_varmap(varmap, DType::F32, dev);
    let model = perpsfm_model::transformer::TinyTransformer::new(tcfg, vs)?;
    let n_features = series[0].n_features();
    let mut preds: Vec<Vec<f64>> = Vec::with_capacity(samples.len());
    let mut actual: Vec<f64> = Vec::with_capacity(samples.len());
    let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed ^ 0xA11CE);
    let mut order: Vec<usize> = (0..samples.len()).collect();
    // Fisher-Yates (rand 0.9 moved SliceRandom::shuffle; manual keeps it simple)
    for i in (1..order.len()).rev() {
        use rand::Rng;
        order.swap(i, rng.random_range(0..=i));
    }
    for chunk in order.chunks(cfg.batch_size) {
        let b = dataset::gather_batch(series, samples, chunk, tcfg.seq_len, n_features);
        let bt = to_device(&b, n_features, tcfg.seq_len, dev)?;
        let out = model.forward(&bt.x, &bt.market_ids)?;
        let q = out.ret_quantiles.to_vec2::<f32>()?;
        for (i, row) in q.iter().enumerate() {
            preds.push(row.iter().map(|&v| v as f64).collect());
            actual.push(b.y_ret[i]);
        }
    }
    Ok(perpsfm_eval::crps_from_quantiles(
        &preds, &actual, &QUANTILES,
    ))
}

/// Placeholder kept for CLI wiring until the `Train` subcommand lands.
pub fn train() -> Result<()> {
    anyhow::bail!("use train_transformer(root, cfg) via the CLI train subcommand")
}

/// Run the trained checkpoint over `[lo, hi)` for the given markets and write
/// per-market prediction parquet files (ts + quantiles, ascending).
/// Sealed weeks are never passed here except by the final evaluation.
pub fn predict(root: &Path, markets: &[Market], lo: i64, hi: i64) -> Result<Vec<(Market, usize)>> {
    let dev = pick_device(DeviceChoice::Auto)?;
    let all: Vec<dataset::MarketSeries> = Market::ALL
        .iter()
        .map(|m| dataset::load_series(root, *m))
        .collect::<Result<Vec<_>>>()?;
    let tmeta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        root.join("runs/transformer.meta.json"),
    )?)?;
    let tcfg: TransformerConfig = serde_json::from_value(tmeta["model_config"].clone())?;
    let mut varmap = VarMap::new();
    varmap.load(latest_checkpoint(root).context("no checkpoint in runs/")?)?;

    // Same normalization rule as training: train-split stats over pooled
    // train markets, applied unchanged (no re-estimation at predict time).
    let train_markets = [Market::SolPerp, Market::HypePerp, Market::EthPerp];
    let splits = dataset::default_splits(&all);
    let stats = dataset::norm_stats(
        &all.iter()
            .filter(|s| train_markets.contains(&s.market))
            .collect::<Vec<_>>(),
        splits.train_start.timestamp(),
        splits.train_end.timestamp(),
    );

    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let model = perpsfm_model::transformer::TinyTransformer::new(&tcfg, vs)?;
    let n_features = all[0].n_features();

    let mut written = Vec::new();
    for m in markets {
        let Some(s) = all.iter().find(|s| &s.market == m) else {
            continue;
        };
        let norm = dataset::normalize(s, &stats);
        let samples = dataset::collect_slices_samples(&norm, tcfg.seq_len, lo, hi);
        anyhow::ensure!(
            !samples.is_empty(),
            "no predict samples for {m:?} in [{lo},{hi})"
        );
        let mut ts_out: Vec<i64> = Vec::with_capacity(samples.len());
        let mut q_out: Vec<Vec<f64>> = Vec::with_capacity(samples.len());
        let mut p_liq: Vec<f64> = Vec::with_capacity(samples.len());
        let mut p_vol: Vec<f64> = Vec::with_capacity(samples.len());
        for chunk_start in (0..samples.len()).step_by(256) {
            let end = (chunk_start + 256).min(samples.len());
            let batch = dataset::gather_window_batch(
                &norm,
                &samples[chunk_start..end],
                tcfg.seq_len,
                n_features,
            );
            let bt = to_device(&batch, n_features, tcfg.seq_len, &dev)?;
            let out = model.forward(&bt.x, &bt.market_ids)?;
            let q = out.ret_quantiles.to_vec2::<f32>()?;
            let logits = out.binary_logits.to_vec2::<f32>()?;
            for (i, row) in q.iter().enumerate() {
                ts_out.push(s.ts[samples[chunk_start + i].t]);
                q_out.push(row.iter().map(|&v| v as f64).collect());
                // sigmoid of logits: [liq_burst, vol_burst]
                let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
                p_liq.push(sig(logits[i][0]) as f64);
                p_vol.push(sig(logits[i][1]) as f64);
            }
        }

        let mut df = df! {
            "ts" => ts_out.clone(),
        }
        .map_err(|e| anyhow::anyhow!("predict ts df: {e}"))?;
        for qi in 0..QUANTILES.len() {
            let col: Vec<f64> = q_out.iter().map(|r| r[qi]).collect();
            df.with_column(Series::new(PlSmallStr::from(format!("q{qi}")), col).into_column())
                .map_err(|e| anyhow::anyhow!("predict q df: {e}"))?;
        }
        df.with_column(Series::new(PlSmallStr::from("p_liq"), p_liq.clone()).into_column())
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        df.with_column(Series::new(PlSmallStr::from("p_vol"), p_vol.clone()).into_column())
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let path = perpsfm_core::predictions_path(root, *m);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let f = std::fs::File::create(&path)?;
        ParquetWriter::new(f).finish(&mut df)?;
        tracing::info!("{m:?}: wrote {} predictions", ts_out.len());
        written.push((*m, ts_out.len()));
    }
    Ok(written)
}
