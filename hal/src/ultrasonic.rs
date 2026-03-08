//! Ultrasonic sensor HAL trait and stub.
//!
//! The HC-SR04 is read via the Yahboom expansion board over I2C.
//! TODO (Phase 2): implement real I2C read via rppal.

use anyhow::Result;
use async_trait::async_trait;
use core_types::UltrasonicReading;

#[async_trait]
pub trait Ultrasonic: Send + Sync {
    /// Read the current forward distance in cm.
    async fn read_distance(&mut self) -> Result<UltrasonicReading>;
}

// ── Stub ──────────────────────────────────────────────────────────────────────

pub struct StubUltrasonic {
    t0: std::time::Instant,
}

impl StubUltrasonic {
    pub fn new() -> Self {
        Self { t0: std::time::Instant::now() }
    }
}

impl Default for StubUltrasonic { fn default() -> Self { Self::new() } }

#[async_trait]
impl Ultrasonic for StubUltrasonic {
    async fn read_distance(&mut self) -> Result<UltrasonicReading> {
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        Ok(UltrasonicReading {
            t_ms: self.t0.elapsed().as_millis() as u64,
            range_cm: 120.0, // synthetic "all clear"
        })
    }
}
