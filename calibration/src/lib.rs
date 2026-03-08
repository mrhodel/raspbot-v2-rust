//! Calibration subsystem.
//!
//! Manages camera intrinsics, IMU bias, camera-to-base extrinsics, and
//! validated camera tilt. Calibration data is persisted to JSON so it
//! survives restarts.
//!
//! # Modes (planned)
//! - `camera`    — capture checkerboard frames, solve intrinsics
//! - `imu`       — static IMU bias estimation
//! - `extrinsics`— camera-to-base pose from a known reference
//! - `tilt`      — validated camera tilt from a level surface

use anyhow::Result;
use core_types::CalibData;
use std::path::Path;
use tracing::info;

/// Calibration manager: loads, validates, and saves `CalibData`.
pub struct CalibrationManager {
    pub data: CalibData,
    path: String,
}

impl CalibrationManager {
    /// Load calibration from `path`, falling back to defaults if missing.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_string_lossy().into_owned();
        let data = if std::path::Path::new(&path).exists() {
            let text = std::fs::read_to_string(&path)?;
            let d: CalibData = serde_json::from_str(&text)?;
            info!("Calibration loaded from {path}");
            d
        } else {
            info!("No calibration file at {path} — using defaults");
            CalibData::default()
        };
        Ok(Self { data, path })
    }

    /// Persist current calibration data to the configured path.
    pub fn save(&self) -> Result<()> {
        let text = serde_json::to_string_pretty(&self.data)?;
        std::fs::write(&self.path, text)?;
        info!("Calibration saved to {}", self.path);
        Ok(())
    }
}
