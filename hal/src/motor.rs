//! Motor controller HAL — trait, stub, and Yahboom I2C implementation.
//!
//! # Yahboom I2C protocol (address 0x2B)
//!
//! Each motor is commanded individually:
//!   write_i2c_block_data(0x2B, 0x01, [motor_id, direction, abs_speed])
//!
//! Motor IDs (as used by Yahboom firmware):
//!   0 = Front-Left (FL)   1 = Rear-Left  (RL)
//!   2 = Front-Right (FR)  3 = Rear-Right (RR)
//!
//! Direction: 0 = forward, 1 = backward
//! Speed:     0–255  (we scale our duty-cycle range −100..+100 to 0–255)
//!
//! The board needs ~10 ms between writes to process each command.
//!
//! # Mecanum wheel layout (top view, front = ↑)
//!
//!   FL(/) ── FR(\)
//!   |              |
//!   RL(\) ── RR(/)
//!
//! Kinematics (all values in duty-cycle units, positive = respective direction):
//!   Forward  : FL+ FR+ RL+ RR+
//!   Backward : FL− FR− RL− RR−
//!   Rotate CW: FL+ FR− RL+ RR−   (left fwd, right rev)
//!   RotateCCW: FL− FR+ RL− RR+
//!   Strafe R : FL+ FR− RL− RR+
//!   Strafe L : FL− FR+ RL+ RR−

use std::time::Duration;
use anyhow::{Context, Result};
use async_trait::async_trait;
use core_types::MotorCommand;
use tracing::{debug, info, warn};

// ── Yahboom board constants ───────────────────────────────────────────────────

const REG_MOTOR:       u8  = 0x01;
const DIR_FORWARD:     u8  = 0;
const DIR_BACKWARD:    u8  = 1;

/// Yahboom motor IDs
const M_FL: u8 = 0; // Front-Left
const M_RL: u8 = 1; // Rear-Left
const M_FR: u8 = 2; // Front-Right
const M_RR: u8 = 3; // Rear-Right

/// Delay between successive motor writes (board needs time to process each).
const INTER_WRITE_MS: u64 = 10;

// ── Trait ────────────────────────────────────────────────────────────────────

#[async_trait]
pub trait MotorController: Send + Sync {
    async fn send_command(&mut self, cmd: MotorCommand) -> Result<()>;
    async fn emergency_stop(&mut self) -> Result<()>;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Convert a duty-cycle value (−100..+100) to (direction byte, speed byte 0–255).
fn encode_duty(duty: i8) -> (u8, u8) {
    let (dir, abs_val) = if duty >= 0 {
        (DIR_FORWARD,  duty as u16)
    } else {
        (DIR_BACKWARD, (-duty as i16) as u16)
    };
    let speed = (abs_val * 255 / 100) as u8;
    (dir, speed)
}

// ── Real Yahboom implementation ───────────────────────────────────────────────

/// Motor controller that talks to the Yahboom expansion board over I2C.
///
/// Call `YahboomMotorController::new(bus, addr, max_duty)` on the Pi;
/// it returns `Err` if the I2C device cannot be opened (e.g. on desktop),
/// allowing the caller to fall back to `StubMotorController`.
///
/// `rppal::i2c::I2c` is not `Sync`, so we wrap it in a `Mutex` to satisfy
/// the `MotorController: Sync` bound.
pub struct YahboomMotorController {
    i2c:      std::sync::Mutex<rppal::i2c::I2c>,
    max_duty: i8,
}

impl YahboomMotorController {
    /// Open the I2C bus and configure the slave address.
    ///
    /// * `i2c_bus`  — Linux I2C bus number (1 for Pi I2C-1)
    /// * `i2c_addr` — Yahboom board address (0x2B)
    /// * `max_duty` — duty-cycle ceiling (0–100); commands are clamped to ±this
    pub fn new(i2c_bus: u8, i2c_addr: u16, max_duty: i8) -> Result<Self> {
        let mut i2c = rppal::i2c::I2c::with_bus(i2c_bus)
            .with_context(|| format!("open /dev/i2c-{i2c_bus}"))?;
        i2c.set_slave_address(i2c_addr)
            .with_context(|| format!("set slave address 0x{i2c_addr:02X}"))?;
        info!("YahboomMotor: opened /dev/i2c-{i2c_bus} @ 0x{i2c_addr:02X}, max_duty={max_duty}");
        Ok(Self { i2c: std::sync::Mutex::new(i2c), max_duty })
    }

    /// Write a single motor command to the board (synchronous I2C write).
    fn write_motor(&self, motor_id: u8, duty: i8) -> Result<()> {
        let (dir, speed) = encode_duty(duty);
        self.i2c
            .lock()
            .unwrap()
            .block_write(REG_MOTOR, &[motor_id, dir, speed])
            .with_context(|| format!("I2C motor write id={motor_id}"))?;
        Ok(())
    }
}

#[async_trait]
impl MotorController for YahboomMotorController {
    async fn send_command(&mut self, cmd: MotorCommand) -> Result<()> {
        let fl = cmd.fl.clamp(-self.max_duty, self.max_duty);
        let fr = cmd.fr.clamp(-self.max_duty, self.max_duty);
        let rl = cmd.rl.clamp(-self.max_duty, self.max_duty);
        let rr = cmd.rr.clamp(-self.max_duty, self.max_duty);
        debug!(fl, fr, rl, rr, "YahboomMotor: send_command");

        // Board order: FL(0), RL(1), FR(2), RR(3)
        for (motor_id, duty) in [(M_FL, fl), (M_RL, rl), (M_FR, fr), (M_RR, rr)] {
            self.write_motor(motor_id, duty)?;
            tokio::time::sleep(Duration::from_millis(INTER_WRITE_MS)).await;
        }
        Ok(())
    }

    async fn emergency_stop(&mut self) -> Result<()> {
        warn!("YahboomMotor: EMERGENCY STOP");
        for motor_id in [M_FL, M_RL, M_FR, M_RR] {
            self.write_motor(motor_id, 0)?;
            tokio::time::sleep(Duration::from_millis(INTER_WRITE_MS)).await;
        }
        Ok(())
    }
}

// ── Stub ──────────────────────────────────────────────────────────────────────

pub struct StubMotorController;

#[async_trait]
impl MotorController for StubMotorController {
    async fn send_command(&mut self, cmd: MotorCommand) -> Result<()> {
        debug!(?cmd, "StubMotor: command");
        Ok(())
    }
    async fn emergency_stop(&mut self) -> Result<()> {
        debug!("StubMotor: emergency stop");
        Ok(())
    }
}
