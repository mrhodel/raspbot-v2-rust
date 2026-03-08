//! IMU HAL trait and stub.
//!
//! TODO (Phase 2): implement real MPU-6050 driver via rppal I2C.

use anyhow::Result;
use async_trait::async_trait;
use core_types::ImuSample;

#[async_trait]
pub trait Imu: Send + Sync {
    /// Read one IMU sample. Blocks until data is available.
    async fn read_sample(&mut self) -> Result<ImuSample>;
}

// ── Stub ──────────────────────────────────────────────────────────────────────

/// Returns zero-noise samples with a slow synthetic yaw rotation.
pub struct StubImu {
    t0: std::time::Instant,
    interval_ms: u64,
}

impl StubImu {
    pub fn new() -> Self {
        Self { t0: std::time::Instant::now(), interval_ms: 10 }
    }
}

impl Default for StubImu {
    fn default() -> Self { Self::new() }
}

#[async_trait]
impl Imu for StubImu {
    async fn read_sample(&mut self) -> Result<ImuSample> {
        tokio::time::sleep(tokio::time::Duration::from_millis(self.interval_ms)).await;
        let t_ms = self.t0.elapsed().as_millis() as u64;
        Ok(ImuSample {
            t_ms,
            gyro_x: 0.0,
            gyro_y: 0.0,
            gyro_z: 0.02, // slow synthetic yaw, ~1.1 °/s
            accel_x: 0.0,
            accel_y: 0.0,
            accel_z: 9.81, // gravity pointing down
        })
    }
}
