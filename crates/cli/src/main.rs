//! perpsfm — one binary for the whole pipeline.
//!
//! Root dir convention: commands take `--root` (default: repo root inferred
//! as the current directory). Curated data lands under `<root>/data/`.

use anyhow::Result;
use chrono::{DateTime, Utc};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "perpsfm",
    version,
    about = "Solana perps mini foundation model"
)]
struct Cli {
    /// Repo root (data/ lives here).
    #[arg(long, default_value = ".")]
    root: PathBuf,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Backfill the Velocity archive into curated parquet files.
    Backfill {
        /// Overlap-free incremental mode: fetch only newer than existing data.
        #[arg(long, default_value_t = true)]
        incremental: bool,
        /// Force full re-download from the archive start.
        #[arg(long)]
        from_scratch: bool,
        /// End date (UTC, YYYY-MM-DD). Default: now.
        #[arg(long)]
        end: Option<String>,
    },
    /// Stream live DLOB events (trades + orderbook) to data/live/*.jsonl.
    /// Runs until interrupted; reconnects automatically.
    Collect {
        #[arg(long, default_value_t = 1)]
        orderbook_grouping: u32,
    },
    /// Build look-ahead-safe features from curated data.
    Features,
    /// Train the pooled transformer on the train split (early-stops on val CRPS).
    Train {
        #[arg(long, default_value_t = 4000)]
        steps: usize,
        #[arg(long, default_value_t = 256)]
        seq_len: usize,
        #[arg(long, default_value_t = 64)]
        batch: usize,
        #[arg(long, value_enum, default_value_t = perpsfm_train::ModelKind::Transformer)]
        model: perpsfm_train::ModelKind,
    },
    /// Run the trained checkpoint over a time range, writing predictions.
    Predict {
        /// Which trained model to use.
        #[arg(long, value_enum, default_value_t = perpsfm_train::ModelKind::Transformer)]
        model: perpsfm_train::ModelKind,
        /// Range start (UTC, YYYY-MM-DD). Default: val-split start.
        #[arg(long)]
        from: Option<String>,
        /// Range end (UTC, YYYY-MM-DD). Default: now minus 1 day.
        #[arg(long)]
        to: Option<String>,
    },
    /// Evaluate on val/sealed splits (Phase 5).
    Evaluate,
    /// Backtest policies after costs on predictions.
    Backtest {
        #[arg(long, default_value_t = 0.0)]
        min_edge: f64,
        #[arg(long, default_value_t = 1.0)]
        exposure: f64,
    },
}

fn parse_date(s: &str) -> Result<DateTime<Utc>> {
    use chrono::NaiveDate;
    let d = NaiveDate::parse_from_str(s, "%Y-%m-%d")?;
    Ok(d.and_hms_opt(0, 0, 0).unwrap().and_utc())
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    let cli = Cli::parse();

    match cli.cmd {
        Cmd::Backfill {
            incremental,
            from_scratch,
            end,
        } => {
            let end = match end {
                Some(s) => parse_date(&s)?,
                None => Utc::now(),
            };
            let start = perpsfm_ingest::velocity::archive_start();
            let mut opts = perpsfm_ingest::backfill::BackfillOpts {
                start,
                end,
                incremental: incremental && !from_scratch,
            };
            perpsfm_ingest::backfill_all(&cli.root, &mut opts)
        }
        Cmd::Collect { orderbook_grouping } => {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(perpsfm_ingest::run_collector(
                cli.root,
                perpsfm_ingest::ws::CollectorOpts { orderbook_grouping },
            ))
        }
        Cmd::Predict { model, from, to } => {
            let all: Vec<perpsfm_train::dataset::MarketSeries> = perpsfm_core::Market::ALL
                .iter()
                .map(|m| perpsfm_train::dataset::load_series(&cli.root, *m))
                .collect::<Result<Vec<_>>>()?;
            let splits = perpsfm_train::dataset::default_splits(&all);
            let lo = match from {
                Some(s) => parse_date(&s)?.timestamp(),
                None => splits.val_start.timestamp(),
            };
            let hi = match to {
                Some(s) => parse_date(&s)?.timestamp(),
                None => Utc::now().timestamp() - 86400,
            };
            let markets = vec![
                perpsfm_core::Market::SolPerp,
                perpsfm_core::Market::HypePerp,
                perpsfm_core::Market::EthPerp,
            ];
            let written = perpsfm_train::predict(&cli.root, model, &markets, lo, hi)?;
            tracing::info!("predictions written: {written:?}");
            Ok(())
        }
        Cmd::Features => {
            for m in perpsfm_core::Market::ALL {
                perpsfm_features::build_features(&cli.root, m)?;
            }
            Ok(())
        }
        Cmd::Train {
            steps,
            seq_len,
            batch,
            model,
        } => {
            let cfg = perpsfm_train::TrainConfig {
                steps,
                seq_len,
                batch_size: batch,
                model,
                ..Default::default()
            };
            let out = perpsfm_train::train_transformer(&cli.root, &cfg)?;
            tracing::info!(
                "done: best val CRPS {:.6} @ step {}, checkpoint {}",
                out.best_val_crps,
                out.steps_trained,
                out.checkpoint.display()
            );
            Ok(())
        }
        Cmd::Evaluate => perpsfm_eval::run(),
        Cmd::Backtest { min_edge, exposure } => {
            perpsfm_backtest::report(&cli.root, min_edge, exposure)
        }
    }
}
