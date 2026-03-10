//! Ultrasonic sensor HAL trait and implementations.
//!
//! The HC-SR04 is connected to the Yahboom expansion board (I2C bus 1, addr 0x2B).
//! The board handles the HC-SR04 trigger/echo timing internally; we just trigger
//! via I2C and read back the result.
//!
//! # Wire protocol (Yahboom expansion board, smbus2 raw path)
//!
//!   Trigger:  write single byte 0x02 to the board
//!   Wait:     ~60 ms for the HC-SR04 round-trip + MCU processing
//!   Read:     write 0x02 then read 3 bytes → 24-bit big-endian distance in mm
//!   Distance: dist_cm = dist_mm / 10.0
//!
//! A median filter over `samples_per_reading` readings reduces spike noise.

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use async_trait::async_trait;
use core_types::UltrasonicReading;
use tracing::{debug, info, warn};

// ── Yahboom board constants ───────────────────────────────────────────────────

const REG_ULTRASONIC: u8 = 0x02;
/// Wait after trigger before reading (ms). HC-SR04 max round-trip at 400 cm ≈ 24 ms;
/// the MCU adds a few ms of processing. 60 ms is conservative and reliable.
const TRIGGER_WAIT_MS: u64 = 60;
/// Clamp distance to this when the sensor returns 0 (blind-spot / too close).
const BLIND_SPOT_CM: f32 = 2.0;

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
    i2c:         std::sync::Mutex<rppal::i2c::I2c>,
    t0:          Instant,
    max_range:   f32,
    min_range:   f32,
    /// Number of samples to collect per reading (median filter).
    n_samples:   u8,
}

impl YahboomUltrasonic {
    /// Open the I2C bus and configure the device address.
    ///
    /// * `i2c_bus`    — Linux I2C bus number (1 for Pi I2C-1)
    /// * `i2c_addr`   — Yahboom board address (0x2B)
    /// * `max_range`  — clamp ceiling in cm (default 400)
    /// * `min_range`  — clamp floor in cm (default 3)
    /// * `n_samples`  — samples per reading for the median filter (1 = fastest)
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

    /// Trigger the sensor and read one raw distance value in cm.
    fn read_once_cm(&self) -> Result<f32> {
        let mut i2c = self.i2c.lock().unwrap();

        // Trigger: send the register byte as a raw I2C write.
        i2c.write(&[REG_ULTRASONIC])
            .context("ultrasonic trigger write")?;

        // Blocking sleep is deliberate here — it keeps the sampling timing
        // deterministic and is short enough not to starve the async runtime
        // in practice (called infrequently, e.g. at 5-10 Hz).
        std::thread::sleep(Duration::from_millis(TRIGGER_WAIT_MS));

        // Read: write register address, then read 3 bytes (24-bit big-endian mm).
        let mut buf = [0u8; 3];
        i2c.write_read(&[REG_ULTRASONIC], &mut buf)
            .context("ultrasonic read")?;

        let dist_mm = ((buf[0] as u32) << 16) | ((buf[1] as u32) << 8) | buf[2] as u32;
        debug!("ultrasonic raw: buf={buf:?} dist_mm={dist_mm}");

        if dist_mm == 0 {
            // Board reports 0 when the target is in the HC-SR04 blind spot (<~2 cm).
            return Ok(BLIND_SPOT_CM);
        }

        Ok((dist_mm as f32 / 10.0).clamp(self.min_range, self.max_range))
    }
}

#[async_trait]
impl Ultrasonic for YahboomUltrasonic {
    async fn read_distance(&mut self) -> Result<UltrasonicReading> {
        let mut samples: Vec<f32> = Vec::with_capacity(self.n_samples as usize);

        for i in 0..self.n_samples {
            match self.read_once_cm() {  // &self — Mutex used internally
                Ok(cm) => samples.push(cm),
                Err(e) => warn!("ultrasonic sample {i}: {e}"),
            }
            // Short gap between back-to-back samples so the HC-SR04 has time
            // to quiesce before the next trigger (board enforces this internally,
            // but an extra gap avoids any echo cross-talk).
            if i + 1 < self.n_samples {
                std::thread::sleep(Duration::from_millis(20));
            }
        }

        if samples.is_empty() {
            anyhow::bail!("all ultrasonic samples failed");
        }

        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = samples[samples.len() / 2];

        Ok(UltrasonicReading {
            t_ms:     self.t0.elapsed().as_millis() as u64,
            range_cm: median,
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
