//! Gimbal HAL — pan/tilt servo control via the Yahboom expansion board.
//!
//! # Wire protocol (Raspbot_Lib.py, Ctrl_Servo)
//!
//!   block_write(0x02, [servo_id, angle_u8])
//!
//!   Servo 1 = pan  (horizontal), Servo 2 = tilt (vertical)
//!   Angle: 0–180° raw.  Tilt (id=2) is firmware-capped at 110°.
//!
//! # Angle convention
//!
//!   Pan:  raw = PAN_CENTER (90°) + pan_deg
//!         pan_deg = 0 → forward, −90 → full left, +90 → full right
//!
//!   Tilt: raw = tilt_neutral_raw − tilt_deg
//!         tilt_deg = 0   → level   (raw = tilt_neutral, default 30°)
//!         tilt_deg = +30 → up      (raw = 0°)
//!         tilt_deg = −45 → down    (raw = tilt_neutral + 45°)
//!
//!   The subtraction sign for tilt reflects servo mounting: higher raw
//!   angle tilts the camera downward.

use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use tracing::{debug, info};

// ── Yahboom board constants ───────────────────────────────────────────────────

const REG_SERVO:     u8  = 0x02;
const SERVO_PAN:     u8  = 1;
const SERVO_TILT:    u8  = 2;
/// Raw servo angle (degrees) when pan_deg = 0 (camera pointing forward).
const PAN_CENTER:    f32 = 90.0;
/// Firmware cap on tilt servo (Yahboom MCU enforces max 110°).
const TILT_RAW_MAX:  f32 = 110.0;
/// Minimum gap between successive servo writes (ms).
const WRITE_GAP_MS:  u64 = 20;

// ── Trait ─────────────────────────────────────────────────────────────────────

#[async_trait]
pub trait Gimbal: Send + Sync {
    /// Set pan angle in degrees (clamped to configured pan_range).
    /// pan_deg = 0 is forward; negative = left, positive = right.
    async fn set_pan(&mut self, deg: f32) -> Result<()>;
    /// Set tilt angle in degrees (clamped to configured tilt_range).
    /// tilt_deg = 0 is level; positive = up, negative = down.
    async fn set_tilt(&mut self, deg: f32) -> Result<()>;
    /// Command both axes simultaneously.
    async fn set_angles(&mut self, pan_deg: f32, tilt_deg: f32) -> Result<()>;
    /// Return current (pan_deg, tilt_deg).
    fn angles(&self) -> (f32, f32);
}

// ── Real Yahboom implementation ───────────────────────────────────────────────

/// Controls the pan/tilt servos via the Yahboom expansion board over I2C.
///
/// `rppal::i2c::I2c` is not `Sync`, so it is wrapped in a `Mutex`.
pub struct YahboomGimbal {
    i2c:              std::sync::Mutex<rppal::i2c::I2c>,
    pan_deg:          f32,
    tilt_deg:         f32,
    pan_range:        [f32; 2],
    tilt_range:       [f32; 2],
    tilt_neutral_raw: f32,
}

impl YahboomGimbal {
    pub fn new(
        i2c_bus:          u8,
        i2c_addr:         u16,
        pan_range:        [f32; 2],
        tilt_range:       [f32; 2],
        tilt_neutral_raw: f32,
    ) -> Result<Self> {
        let mut i2c = rppal::i2c::I2c::with_bus(i2c_bus)
            .with_context(|| format!("open /dev/i2c-{i2c_bus}"))?;
        i2c.set_slave_address(i2c_addr)
            .with_context(|| format!("set slave address 0x{i2c_addr:02X}"))?;
        info!(
            "YahboomGimbal: /dev/i2c-{i2c_bus} @ 0x{i2c_addr:02X}, \
             pan [{},{}]° tilt [{},{}]° tilt_neutral_raw={tilt_neutral_raw}°",
            pan_range[0], pan_range[1], tilt_range[0], tilt_range[1]
        );
        let gimbal = Self {
            i2c: std::sync::Mutex::new(i2c),
            pan_deg: 0.0,
            tilt_deg: 0.0,
            pan_range,
            tilt_range,
            tilt_neutral_raw,
        };
        Ok(gimbal)
    }

    fn deg_to_raw_pan(&self, pan_deg: f32) -> u8 {
        (PAN_CENTER + pan_deg).clamp(0.0, 180.0) as u8
    }

    fn deg_to_raw_tilt(&self, tilt_deg: f32) -> u8 {
        (self.tilt_neutral_raw - tilt_deg)
            .clamp(0.0, TILT_RAW_MAX) as u8
    }

    fn write_servo(&self, servo_id: u8, raw_angle: u8) -> Result<()> {
        self.i2c
            .lock()
            .unwrap()
            .block_write(REG_SERVO, &[servo_id, raw_angle])
            .with_context(|| format!("servo write id={servo_id} angle={raw_angle}"))?;
        debug!("YahboomGimbal: servo {servo_id} raw={raw_angle}°");
        Ok(())
    }
}

#[async_trait]
impl Gimbal for YahboomGimbal {
    async fn set_pan(&mut self, deg: f32) -> Result<()> {
        self.pan_deg = deg.clamp(self.pan_range[0], self.pan_range[1]);
        let raw = self.deg_to_raw_pan(self.pan_deg);
        self.write_servo(SERVO_PAN, raw)?;
        tokio::time::sleep(Duration::from_millis(WRITE_GAP_MS)).await;
        Ok(())
    }

    async fn set_tilt(&mut self, deg: f32) -> Result<()> {
        self.tilt_deg = deg.clamp(self.tilt_range[0], self.tilt_range[1]);
        let raw = self.deg_to_raw_tilt(self.tilt_deg);
        self.write_servo(SERVO_TILT, raw)?;
        tokio::time::sleep(Duration::from_millis(WRITE_GAP_MS)).await;
        Ok(())
    }

    async fn set_angles(&mut self, pan_deg: f32, tilt_deg: f32) -> Result<()> {
        self.pan_deg = pan_deg.clamp(self.pan_range[0], self.pan_range[1]);
        self.tilt_deg = tilt_deg.clamp(self.tilt_range[0], self.tilt_range[1]);
        let raw_pan  = self.deg_to_raw_pan(self.pan_deg);
        let raw_tilt = self.deg_to_raw_tilt(self.tilt_deg);
        self.write_servo(SERVO_PAN, raw_pan)?;
        tokio::time::sleep(Duration::from_millis(WRITE_GAP_MS)).await;
        self.write_servo(SERVO_TILT, raw_tilt)?;
        tokio::time::sleep(Duration::from_millis(WRITE_GAP_MS)).await;
        Ok(())
    }

    fn angles(&self) -> (f32, f32) {
        (self.pan_deg, self.tilt_deg)
    }
}

// ── Stub ──────────────────────────────────────────────────────────────────────

pub struct StubGimbal {
    pan:  f32,
    tilt: f32,
}

impl StubGimbal {
    pub fn new(initial_tilt: f32) -> Self {
        Self { pan: 0.0, tilt: initial_tilt }
    }
}

#[async_trait]
impl Gimbal for StubGimbal {
    async fn set_pan(&mut self, deg: f32) -> Result<()> {
        self.pan = deg.clamp(-90.0, 90.0);
        debug!(pan = self.pan, "StubGimbal: pan");
        Ok(())
    }
    async fn set_tilt(&mut self, deg: f32) -> Result<()> {
        self.tilt = deg.clamp(-45.0, 30.0);
        debug!(tilt = self.tilt, "StubGimbal: tilt");
        Ok(())
    }
    async fn set_angles(&mut self, pan_deg: f32, tilt_deg: f32) -> Result<()> {
        self.set_pan(pan_deg).await?;
        self.set_tilt(tilt_deg).await
    }
    fn angles(&self) -> (f32, f32) {
        (self.pan, self.tilt)
    }
}
