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
    /// Train a model (Phase 3).
    Train {
        #[arg(long, value_enum, default_value_t = perpsfm_train::DeviceChoice::Auto)]
        device: perpsfm_train::DeviceChoice,
    },
    /// Evaluate on val/sealed splits (Phase 5).
    Evaluate,
    /// Backtest policies after costs (Phase 4).
    Backtest,
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
        Cmd::Features => perpsfm_features::build_features(&cli.root, perpsfm_core::Market::SolPerp),
        Cmd::Train { device: _ } => perpsfm_train::train(),
        Cmd::Evaluate => perpsfm_eval::run(),
        Cmd::Backtest => perpsfm_backtest::run(),
    }
}
