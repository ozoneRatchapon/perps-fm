//! Ingest: pull Velocity/Drift perps history into curated Parquet files and
//! stream live events from the DLOB websocket into append-only JSONL.
//!
//! Data source docs: `docs/00-data-source.md`.

pub mod backfill;
pub mod velocity;
pub mod ws;

pub use backfill::backfill_all;
pub use ws::run_collector;
