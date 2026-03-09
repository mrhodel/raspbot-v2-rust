//! Hardware safety layer.
//!
//! `SafetyMonitor` is a pure, stateless evaluator. The runtime task wraps it
//! with a `tokio::time::timeout` watchdog and publishes results to the bus.
//!
//! # Emergency stop conditions
//! 1. **Proximity**: ultrasonic reading < `STOP_THRESHOLD_CM`.
//! 2. **Watchdog**: no ultrasonic reading for > `WATCHDOG_TIMEOUT_MS`.
//!
//! When triggered the runtime task:
//!   - Publishes `SafetyState::EmergencyStop` on `bus.safety_state`.
//!   - Sends `MotorCommand::stop()` on `bus.motor_command`.

use core_types::{MotorCommand, SafetyState, UltrasonicReading};
use tracing::warn;

/// Robot halts if anything is closer than this.
pub const STOP_THRESHOLD_CM: f32 = 15.0;
/// Duration (ms) without a reading before the watchdog fires.
pub const WATCHDOG_TIMEOUT_MS: u64 = 500;

pub struct SafetyMonitor {
    pub stop_threshold_cm: f32,
    pub watchdog_timeout_ms: u64,
}

impl SafetyMonitor {
    pub fn new() -> Self {
        Self {
            stop_threshold_cm: STOP_THRESHOLD_CM,
            watchdog_timeout_ms: WATCHDOG_TIMEOUT_MS,
        }
    }

    /// Evaluate a fresh ultrasonic reading.
    ///
    /// Returns `(SafetyState, Option<MotorCommand>)`. If a stop command is
    /// returned the runtime task should send it immediately on the motor bus.
    pub fn evaluate(&self, reading: &UltrasonicReading) -> (SafetyState, Option<MotorCommand>) {
        if reading.range_cm < self.stop_threshold_cm {
            warn!(
                range_cm = reading.range_cm,
                threshold_cm = self.stop_threshold_cm,
                "Safety: obstacle too close"
            );
            let state = SafetyState::EmergencyStop {
                reason: format!(
                    "Ultrasonic {:.1}cm < {:.1}cm",
                    reading.range_cm, self.stop_threshold_cm
                ),
            };
            let cmd = MotorCommand::stop(reading.t_ms);
            (state, Some(cmd))
        } else {
            (SafetyState::Ok, None)
        }
    }

    /// Evaluate a watchdog timeout (no reading within `watchdog_timeout_ms`).
    pub fn evaluate_timeout(&self, t_ms: u64) -> (SafetyState, MotorCommand) {
        warn!(
            timeout_ms = self.watchdog_timeout_ms,
            "Safety: ultrasonic watchdog timeout"
        );
        let state = SafetyState::EmergencyStop {
            reason: format!("No ultrasonic reading for {}ms", self.watchdog_timeout_ms),
        };
        (state, MotorCommand::stop(t_ms))
    }
}

impl Default for SafetyMonitor {
    fn default() -> Self {
        Self::new()
    }
}
