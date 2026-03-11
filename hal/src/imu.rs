//! IMU HAL — MPU-6050 via rppal I2C (bit-banged I2C-6) + stub.
//!
//! # Hardware
//!
//!   MPU-6050 on I2C bus 6 (`/dev/i2c-6`, bit-banged 400 kHz, GPIO 22/23).
//!   I2C address: 0x68.
//!
//! # Configuration (baked in — no config file knob needed)
//!
//!   ACCEL_CONFIG = 0x10  → ±8 g    (LSB = 4096 counts/g)
//!   GYRO_CONFIG  = 0x08  → ±500 °/s (LSB = 65.5 counts/°/s)
//!
//! # Sign convention (confirmed on physical robot)
//!
//!   Rotate CW  (viewed from above) → gyro_z < 0
//!   Rotate CCW (viewed from above) → gyro_z > 0
//!   Gravity pulls down              → accel_z ≈ +9.81 m/s²
//!
//! # Feature flag
//!
//!   `Mpu6050Imu` is compiled only when the `mpu6050` feature is enabled
//!   (default on Pi).  Build on dev machines without rppal:
//!     cargo check --no-default-features

use anyhow::Result;
use async_trait::async_trait;
use core_types::ImuSample;

// ── Trait ──────────────────────────────────────────────────────────────────────

#[async_trait]
pub trait Imu: Send + Sync {
    /// Read one IMU sample (blocks briefly for register read).
    async fn read_sample(&mut self) -> Result<ImuSample>;
}

// ── Real MPU-6050 implementation ───────────────────────────────────────────────

#[cfg(feature = "mpu6050")]
mod mpu {
    use std::time::Instant;

    use anyhow::{Context, Result};
    use async_trait::async_trait;
    use core_types::ImuSample;
    use rppal::i2c::I2c;
    use tracing::info;

    use super::Imu;

    // ── MPU-6050 register addresses ────────────────────────────────────────────
    const PWR_MGMT_1:   u8 = 0x6B;
    const ACCEL_CONFIG: u8 = 0x1C;
    const GYRO_CONFIG:  u8 = 0x1B;
    const ACCEL_XOUT_H: u8 = 0x3B; // first of 14 contiguous registers

    // ── Scale factors ──────────────────────────────────────────────────────────
    /// ±8 g  → 4096 LSB/g  → divide by 4096 → g, × 9.80665 → m/s²
    const ACCEL_SCALE: f32 = 9.80665 / 4096.0;
    /// ±500 °/s → 65.5 LSB/°/s → divide by 65.5 → °/s → × π/180 → rad/s
    const GYRO_SCALE:  f32 = std::f32::consts::PI / (65.5 * 180.0);

    /// MPU-6050 IMU on bit-banged I2C bus 6.
    ///
    /// `rppal::i2c::I2c` is `!Sync`, so it is wrapped in a `Mutex` to satisfy
    /// the `Imu: Sync` bound.
    pub struct Mpu6050Imu {
        i2c: std::sync::Mutex<I2c>,
        t0:  Instant,
    }

    impl Mpu6050Imu {
        /// Open the MPU-6050 on the given I2C bus number (typically 6).
        pub fn new(bus: u8) -> Result<Self> {
            let addr: u16 = 0x68;

            let mut i2c = I2c::with_bus(bus)
                .with_context(|| format!("open /dev/i2c-{bus}"))?;
            i2c.set_slave_address(addr)
                .context("set MPU-6050 slave address")?;

            // Wake the device (clear SLEEP bit).
            i2c.block_write(PWR_MGMT_1, &[0x00])
                .context("MPU-6050 wake")?;

            // ±8 g
            i2c.block_write(ACCEL_CONFIG, &[0x10])
                .context("MPU-6050 accel config")?;

            // ±500 °/s
            i2c.block_write(GYRO_CONFIG, &[0x08])
                .context("MPU-6050 gyro config")?;

            info!("Mpu6050Imu: /dev/i2c-{bus} addr=0x{addr:02X} ±8g ±500°/s");

            Ok(Self { i2c: std::sync::Mutex::new(i2c), t0: Instant::now() })
        }
    }

    #[async_trait]
    impl Imu for Mpu6050Imu {
        async fn read_sample(&mut self) -> Result<ImuSample> {
            // Read 14 bytes starting at ACCEL_XOUT_H:
            //   [0..1]  ACCEL_X   [2..3]  ACCEL_Y   [4..5]  ACCEL_Z
            //   [6..7]  TEMP      [8..9]  GYRO_X    [10..11] GYRO_Y
            //   [12..13] GYRO_Z
            let mut buf = [0u8; 14];
            self.i2c
                .lock()
                .unwrap()
                .block_read(ACCEL_XOUT_H, &mut buf)
                .context("MPU-6050 read burst")?;

            let raw_ax = i16::from_be_bytes([buf[0],  buf[1]]);
            let raw_ay = i16::from_be_bytes([buf[2],  buf[3]]);
            let raw_az = i16::from_be_bytes([buf[4],  buf[5]]);
            // buf[6..7] = temperature, skip
            let raw_gx = i16::from_be_bytes([buf[8],  buf[9]]);
            let raw_gy = i16::from_be_bytes([buf[10], buf[11]]);
            let raw_gz = i16::from_be_bytes([buf[12], buf[13]]);

            Ok(ImuSample {
                t_ms:    self.t0.elapsed().as_millis() as u64,
                accel_x: raw_ax as f32 * ACCEL_SCALE,
                accel_y: raw_ay as f32 * ACCEL_SCALE,
                accel_z: raw_az as f32 * ACCEL_SCALE,
                gyro_x:  raw_gx as f32 * GYRO_SCALE,
                gyro_y:  raw_gy as f32 * GYRO_SCALE,
                gyro_z:  raw_gz as f32 * GYRO_SCALE,
            })
        }
    }
}

#[cfg(feature = "mpu6050")]
pub use mpu::Mpu6050Imu;

// ── Stub ───────────────────────────────────────────────────────────────────────

/// Returns zero-noise samples with a slow synthetic yaw rotation.
pub struct StubImu {
    t0:          std::time::Instant,
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
            gyro_x:  0.0,
            gyro_y:  0.0,
            gyro_z:  0.02, // slow synthetic yaw ~1.1°/s
            accel_x: 0.0,
            accel_y: 0.0,
            accel_z: 9.81,
        })
    }
}
