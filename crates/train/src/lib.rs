//! Training loop and device selection. Models are tiny on purpose: every run
//! should finish in minutes on this MacBook (see Phase 0b smoke test notes:
//! CPU wins below ~100k params/small batches, Metal ~3x faster at seq 256).

use anyhow::Result;
use candle_core::Device;

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

/// Phase 3 fills this in: train a model on the train split, early-stop on
/// the val split (CRPS on quantile forecasts), checkpoint to disk.
pub fn train() -> Result<()> {
    anyhow::bail!("train: implemented in Phase 3")
}
