//! Ultrasonic sensor HAL trait and implementations.
//!
//! The HC-SR04 is connected to the Yahboom expansion board (I2C bus 1, addr 0x2B).
//! The board handles the HC-SR04 trigger/echo timing internally.
//!
//! # Wire protocol (from Yahboom Raspbot_Lib.py, confirmed via source inspection)
//!
//!   Enable sensor:     block_write(0x07, [1])          (write_i2c_block_data)
//!   Wait for settle:   ~80 ms
//!   Read high byte:    block_read(0x1b, 1 byte) → hi
//!   Read low byte:     block_read(0x1a, 1 byte) → lo
//!   Distance (mm):     (hi << 8) | lo
//!   Distance (cm):     mm / 10.0
//!   Disable sensor:    block_write(0x07, [0])           (optional between readings)
//!
//! A median filter over `samples_per_reading` readings reduces spike noise.

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use async_trait::async_trait;
use core_types::UltrasonicReading;
use tracing::{debug, info, warn};

// ── Yahboom board register map ────────────────────────────────────────────────

/// Write [1] to enable, [0] to disable the HC-SR04.
const REG_US_ENABLE: u8 = 0x07;
/// High byte of distance result (mm).
const REG_US_DIST_H: u8 = 0x1b;
/// Low byte of distance result (mm).
const REG_US_DIST_L: u8 = 0x1a;

/// Time from enabling sensor to first stable reading (ms).
/// Yahboom example uses 1000ms; the Python wrapper uses 80ms — 100ms is safe.
const SETTLE_MS: u64 = 100;
/// Gap between consecutive median-filter samples (ms).
const INTER_SAMPLE_MS: u64 = 20;
/// The HC-SR04 board returns dist_mm == 0 in two cases that cannot be
/// distinguished from registers alone:
///   (a) Target in blind spot < ~2 cm (too close for echo)
///   (b) No echo received — target > ~400 cm or no reflective surface
///
/// For the safety-interlock role, returning max_range for 0 is correct:
///   • Case (a): robot's chassis is already touching the obstacle — wheels
///     would have already stalled; no need to re-trigger emergency stop.
///   • Case (b): open space — safe to move.

// ── Trait ─────────────────────────────────────────────────────────────────────

#[async_trait]
pub trait Ultrasonic: Send + Sync {
    /// Read the current forward distance in cm (median-filtered).
    async fn read_distance(&mut self) -> Result<UltrasonicReading>;
}

// ── Real Yahboom implementation ───────────────────────────────────────────────

/// Reads the HC-SR04 via the Yahboom expansion board over I2C.
///
/// `rppal::i2c::I2c` is not `Sync`, so it is wrapped in a `Mutex` to satisfy
/// the `Ultrasonic: Sync` bound required by `Box<dyn Ultrasonic>`.
pub struct YahboomUltrasonic {
    i2c:       std::sync::Mutex<rppal::i2c::I2c>,
    t0:        Instant,
    max_range: f32,
    min_range: f32,
    /// Number of samples to collect per reading (median filter).
    n_samples: u8,
}

impl YahboomUltrasonic {
    /// Open the I2C bus and configure the device address.
    pub fn new(
        i2c_bus:   u8,
        i2c_addr:  u16,
        max_range: f32,
        min_range: f32,
        n_samples: u8,
    ) -> Result<Self> {
        let mut i2c = rppal::i2c::I2c::with_bus(i2c_bus)
            .with_context(|| format!("open /dev/i2c-{i2c_bus}"))?;
        i2c.set_slave_address(i2c_addr)
            .with_context(|| format!("set slave address 0x{i2c_addr:02X}"))?;
        info!(
            "YahboomUltrasonic: /dev/i2c-{i2c_bus} @ 0x{i2c_addr:02X}, \
             range {min_range}-{max_range} cm, {n_samples} sample(s)"
        );
        Ok(Self {
            i2c: std::sync::Mutex::new(i2c),
            t0: Instant::now(),
            max_range,
            min_range,
            n_samples: n_samples.max(1),
        })
    }

    /// Enable sensor, wait for settle, take N readings, disable, return median.
    fn read_median_cm(&self) -> Result<f32> {
        let i2c = self.i2c.lock().unwrap();

        // Enable the ultrasonic sensor.
        i2c.block_write(REG_US_ENABLE, &[1])
            .context("ultrasonic enable")?;
        std::thread::sleep(Duration::from_millis(SETTLE_MS));

        let mut samples: Vec<f32> = Vec::with_capacity(self.n_samples as usize);
        for idx in 0..self.n_samples {
            // Read high and low bytes separately (board doesn't support
            // reading both in one block_read call per the Raspbot_Lib source).
            let mut hi_buf = [0u8; 1];
            let mut lo_buf = [0u8; 1];
            if let Err(e) = i2c.block_read(REG_US_DIST_H, &mut hi_buf)
                .and_then(|_| i2c.block_read(REG_US_DIST_L, &mut lo_buf))
            {
                warn!("ultrasonic sample {idx} read error: {e}");
            } else {
                let dist_mm = ((hi_buf[0] as u32) << 8) | lo_buf[0] as u32;
                debug!("ultrasonic raw: hi={:#04x} lo={:#04x} dist_mm={dist_mm}", hi_buf[0], lo_buf[0]);
                let dist_cm = if dist_mm == 0 {
                    // No echo (open space) or blind spot — both safe to treat as max range.
                    self.max_range
                } else {
                    (dist_mm as f32 / 10.0).clamp(self.min_range, self.max_range)
                };
                samples.push(dist_cm);
            }
            if idx + 1 < self.n_samples {
                std::thread::sleep(Duration::from_millis(INTER_SAMPLE_MS));
            }
        }

        // Disable sensor to reduce power and electromagnetic noise.
        let _ = i2c.block_write(REG_US_ENABLE, &[0]);

        if samples.is_empty() {
            anyhow::bail!("all ultrasonic samples failed");
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Ok(samples[samples.len() / 2])
    }
}

#[async_trait]
impl Ultrasonic for YahboomUltrasonic {
    async fn read_distance(&mut self) -> Result<UltrasonicReading> {
        let cm = self.read_median_cm()?;
        Ok(UltrasonicReading {
            t_ms:     self.t0.elapsed().as_millis() as u64,
            range_cm: cm,
        })
    }
}

// ── Stub ──────────────────────────────────────────────────────────────────────

pub struct StubUltrasonic {
    t0: Instant,
}

impl StubUltrasonic {
    pub fn new() -> Self {
        Self { t0: Instant::now() }
    }
}

impl Default for StubUltrasonic {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Ultrasonic for StubUltrasonic {
    async fn read_distance(&mut self) -> Result<UltrasonicReading> {
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        Ok(UltrasonicReading {
            t_ms:     self.t0.elapsed().as_millis() as u64,
            range_cm: 120.0, // synthetic "all clear"
        })
    }
}
