//! Configuration loader.
//!
//! Deserialises `robot_config.yaml` into typed structs. Only the fields
//! needed for the current milestone are defined; add sections as subsystems
//! are implemented. Unknown YAML keys are silently ignored via
//! `#[serde(default)]`.

use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::path::Path;

// ── Top-level ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobotConfig {
    pub robot: RobotParams,
    pub hal: HalConfig,
    #[serde(default)]
    pub agent: AgentConfig,
}

impl RobotConfig {
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {:?}", path.as_ref()))?;
        serde_yaml::from_str(&text)
            .with_context(|| format!("parsing {:?}", path.as_ref()))
    }
}

// ── Robot params ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobotParams {
    pub name: String,
    pub wheel_type: String,
    pub max_speed: u8,
    pub min_speed: u8,
}

// ── HAL ───────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HalConfig {
    pub motor: MotorConfig,
    pub ultrasonic: UltrasonicConfig,
    pub gimbal: GimbalConfig,
    pub camera: CameraConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotorConfig {
    pub driver: String,   // "yahboom" | "simulation"
    pub i2c_bus: u8,
    pub i2c_address: u8,
    pub max_speed: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UltrasonicConfig {
    pub i2c_bus: u8,
    pub i2c_address: u8,
    pub max_range_cm: f32,
    pub min_range_cm: f32,
    pub samples_per_reading: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GimbalConfig {
    pub driver: String,
    pub i2c_bus: u8,
    pub i2c_address: u8,
    pub pan_range: [f32; 2],
    pub tilt_range: [f32; 2],
    pub tilt_neutral: f32,
    pub move_speed_deg_s: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraConfig {
    pub driver: String,        // "usb" | "simulation"
    pub device_index: u32,
    #[serde(default)]
    pub device_path: String,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    #[serde(default = "default_freeze_reopen")]
    pub freeze_reopen_s: f32,
    #[serde(default = "default_true")]
    pub auto_exposure: bool,
    #[serde(default = "default_stream_port")]
    pub stream_port: u16,
    #[serde(default = "default_true")]
    pub stream_enabled: bool,
}

fn default_freeze_reopen() -> f32 { 3.0 }
fn default_true() -> bool { true }
fn default_stream_port() -> u16 { 8080 }

// ── Agent ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AgentConfig {
    #[serde(default = "default_loop_hz")]
    pub loop_hz: u32,
    #[serde(default)]
    pub safety: SafetyConfig,
}

fn default_loop_hz() -> u32 { 10 }

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SafetyConfig {
    #[serde(default = "default_emergency_stop_cm")]
    pub emergency_stop_cm: f32,
}

fn default_emergency_stop_cm() -> f32 { 15.0 }
