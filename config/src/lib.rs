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
    pub imu: ImuConfig,
    #[serde(default)]
    pub agent: AgentConfig,
    #[serde(default)]
    pub perception: PerceptionConfig,
    #[serde(default)]
    pub sim: SimConfig,
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
    /// Tilt angle (degrees) the gimbal homes to at startup.
    /// Positive = camera up.  Tune until the horizon sits in the upper ~40% of frame.
    #[serde(default)]
    pub tilt_home_deg: f32,
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
    #[serde(default = "default_true")]
    pub auto_exposure: bool,
    #[serde(default = "default_stream_port")]
    pub stream_port: u16,
    #[serde(default = "default_true")]
    pub stream_enabled: bool,
    /// Rows from the bottom of every frame that are permanently robot body.
    /// Set to 0 until calibrated via camera_test --save-frame.
    /// Perception layer computes mask_start_row = height - body_mask_bottom_rows.
    #[serde(default)]
    pub body_mask_bottom_rows: u32,
}

fn default_true() -> bool { true }
fn default_stream_port() -> u16 { 8080 }

// ── IMU ───────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImuConfig {
    #[serde(default)]
    pub ax_bias: f32,
    #[serde(default)]
    pub ay_bias: f32,
    #[serde(default = "default_az_bias")]
    pub az_bias: f32,  // defaults to gravity so zero-bias still yields sane accel norm
    #[serde(default)]
    pub gx_bias: f32,
    #[serde(default)]
    pub gy_bias: f32,
    #[serde(default)]
    pub gz_bias: f32,
}

impl Default for ImuConfig {
    fn default() -> Self {
        Self {
            ax_bias: 0.0,
            ay_bias: 0.0,
            az_bias: default_az_bias(),
            gx_bias: 0.0,
            gy_bias: 0.0,
            gz_bias: 0.0,
        }
    }
}

fn default_az_bias() -> f32 { 9.81 }

// ── Agent ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AgentConfig {
    #[serde(default = "default_loop_hz")]
    pub loop_hz: u32,
    #[serde(default)]
    pub safety: SafetyConfig,
    #[serde(default)]
    pub crash: CrashConfig,
    #[serde(default)]
    pub motor: MotorAgentConfig,
    #[serde(default)]
    pub gimbal: GimbalAgentConfig,
    #[serde(default)]
    pub navigation: NavigationAgentConfig,
}

fn default_loop_hz() -> u32 { 10 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyConfig {
    #[serde(default = "default_emergency_stop_cm")]
    pub emergency_stop_cm: f32,
    /// Speed is linearly scaled from 1→0 as US reading drops from us_slow_m to us_stop_m.
    #[serde(default = "default_us_slow_m")]
    pub us_slow_m: f32,
    /// US reading at or below this → full stop.
    #[serde(default = "default_us_stop_m")]
    pub us_stop_m: f32,
    /// Emergency stop latch duration (seconds) — clears only after this time, not just when US clears.
    #[serde(default = "default_emstop_latch_s")]
    pub emstop_latch_s: f64,
    /// Delay before escape reverse begins (ms).
    #[serde(default = "default_escape_delay_ms")]
    pub escape_delay_ms: u64,
    /// Duration of escape reverse motion (ms).
    #[serde(default = "default_escape_duration_ms")]
    pub escape_duration_ms: u64,
    /// Duration of escape rotation after reverse (ms).
    #[serde(default = "default_escape_rotation_ms")]
    pub escape_rotation_ms: u64,
    /// Hysteresis: obstacle_stopped clears only after range > 2×stop_m for this many seconds.
    #[serde(default = "default_clear_hold_s")]
    pub clear_hold_s: f32,
}

impl Default for SafetyConfig {
    fn default() -> Self {
        Self {
            emergency_stop_cm:  default_emergency_stop_cm(),
            us_slow_m:          default_us_slow_m(),
            us_stop_m:          default_us_stop_m(),
            emstop_latch_s:     default_emstop_latch_s(),
            escape_delay_ms:    default_escape_delay_ms(),
            escape_duration_ms: default_escape_duration_ms(),
            escape_rotation_ms: default_escape_rotation_ms(),
            clear_hold_s:       default_clear_hold_s(),
        }
    }
}

fn default_emergency_stop_cm() -> f32 { 25.0 }
fn default_us_slow_m()         -> f32 { 0.70 }
fn default_us_stop_m()         -> f32 { 0.30 }
fn default_emstop_latch_s()    -> f64 { 5.0 }
fn default_escape_delay_ms()   -> u64 { 200 }
fn default_escape_duration_ms()-> u64 { 400 }
fn default_escape_rotation_ms()-> u64 { 600 }
fn default_clear_hold_s()      -> f32 { 0.8 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrashConfig {
    /// Horizontal acceleration threshold (m/s²) that counts as a crash impact.
    #[serde(default = "default_crash_accel_threshold")]
    pub accel_threshold_m_s2: f32,
    /// Minimum ms between two consecutive crash events (debounce).
    #[serde(default = "default_crash_debounce_ms")]
    pub debounce_ms: u64,
}

impl Default for CrashConfig {
    fn default() -> Self {
        Self {
            accel_threshold_m_s2: default_crash_accel_threshold(),
            debounce_ms:          default_crash_debounce_ms(),
        }
    }
}

fn default_crash_accel_threshold() -> f32  { 15.0 }
fn default_crash_debounce_ms()     -> u64  { 2_000 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotorAgentConfig {
    /// Motors are zeroed if no command arrives for this many ms.
    #[serde(default = "default_motor_watchdog_ms")]
    pub watchdog_ms: u64,
}

impl Default for MotorAgentConfig {
    fn default() -> Self {
        Self { watchdog_ms: default_motor_watchdog_ms() }
    }
}

fn default_motor_watchdog_ms() -> u64 { 500 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GimbalAgentConfig {
    /// Sinusoidal sweep amplitude (degrees).
    #[serde(default = "default_sweep_amp_deg")]
    pub sweep_amp_deg: f32,
    /// Sinusoidal sweep period (seconds).
    #[serde(default = "default_sweep_period_s")]
    pub sweep_period_s: f32,
    /// Gain applied to depth-derived open-side bias.
    #[serde(default = "default_reactive_gain")]
    pub reactive_gain: f32,
    /// Maximum reactive correction per frame (degrees).
    #[serde(default = "default_reactive_cap_deg")]
    pub reactive_cap_deg: f32,
    /// Hard pan limit applied after sweep + reactive (degrees from centre).
    #[serde(default = "default_pan_limit_deg")]
    pub pan_limit_deg: f32,
}

impl Default for GimbalAgentConfig {
    fn default() -> Self {
        Self {
            sweep_amp_deg:   default_sweep_amp_deg(),
            sweep_period_s:  default_sweep_period_s(),
            reactive_gain:   default_reactive_gain(),
            reactive_cap_deg:default_reactive_cap_deg(),
            pan_limit_deg:   default_pan_limit_deg(),
        }
    }
}

fn default_sweep_amp_deg()    -> f32 { 20.0 }
fn default_sweep_period_s()   -> f32 { 4.0 }
fn default_reactive_gain()    -> f32 { 10.0 }
fn default_reactive_cap_deg() -> f32 { 10.0 }
fn default_pan_limit_deg()    -> f32 { 30.0 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NavigationAgentConfig {
    /// Planning failures before the goal is blacklisted.
    #[serde(default = "default_max_consecutive_failures")]
    pub max_consecutive_failures: u32,
    /// Seconds a blacklisted goal stays blocked.
    #[serde(default = "default_goal_blacklist_s")]
    pub goal_blacklist_s: u64,
}

impl Default for NavigationAgentConfig {
    fn default() -> Self {
        Self {
            max_consecutive_failures: default_max_consecutive_failures(),
            goal_blacklist_s:         default_goal_blacklist_s(),
        }
    }
}

fn default_max_consecutive_failures() -> u32 { 15 }
fn default_goal_blacklist_s()         -> u64 { 10 }

// ── Sim ───────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimConfig {
    /// Default random seed for maze generation.
    #[serde(default = "default_sim_seed")]
    pub seed: u64,
    /// Number of random rectangular obstacles (0 = empty room).
    #[serde(default = "default_sim_obstacles")]
    pub obstacles: usize,
    /// MiDaS nearest-obstacle distance that begins speed scaling (sim only; real uses 0.0).
    #[serde(default = "default_sim_obstacle_slow_m")]
    pub obstacle_slow_m: f32,
    /// MiDaS nearest-obstacle distance that triggers full stop (sim only).
    #[serde(default = "default_sim_obstacle_stop_m")]
    pub obstacle_stop_m: f32,
    /// Gaussian noise sigma on simulated gyro axes (rad/s).
    #[serde(default = "default_sim_imu_noise_gyro")]
    pub imu_noise_gyro_rad_s: f32,
    /// Gaussian noise sigma on simulated lateral accelerometer axes (m/s²).
    #[serde(default = "default_sim_imu_noise_accel")]
    pub imu_noise_accel_m_s2: f32,
    /// Fraction of pseudo-lidar rays to drop per scan (0–1).
    #[serde(default = "default_sim_range_dropout")]
    pub range_dropout: f32,
    /// Gaussian noise sigma on pseudo-lidar range (metres).
    #[serde(default = "default_sim_range_noise_m")]
    pub range_noise_m: f32,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            seed:                  default_sim_seed(),
            obstacles:             default_sim_obstacles(),
            obstacle_slow_m:       default_sim_obstacle_slow_m(),
            obstacle_stop_m:       default_sim_obstacle_stop_m(),
            imu_noise_gyro_rad_s:  default_sim_imu_noise_gyro(),
            imu_noise_accel_m_s2:  default_sim_imu_noise_accel(),
            range_dropout:         default_sim_range_dropout(),
            range_noise_m:         default_sim_range_noise_m(),
        }
    }
}

fn default_sim_seed()               -> u64   { 42 }
fn default_sim_obstacles()          -> usize  { 18 }
fn default_sim_obstacle_slow_m()    -> f32    { 0.80 }
fn default_sim_obstacle_stop_m()    -> f32    { 0.25 }
fn default_sim_imu_noise_gyro()     -> f32    { 0.01 }
fn default_sim_imu_noise_accel()    -> f32    { 0.05 }
fn default_sim_range_dropout()      -> f32    { 0.08 }
fn default_sim_range_noise_m()      -> f32    { 0.07 }

// ── Perception ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceptionConfig {
    /// Path to the MiDaS ONNX model file.
    #[serde(default = "default_midas_model_path")]
    pub midas_model_path: String,
    /// Output depth map width (fed to RL state).
    #[serde(default = "default_depth_out_width")]
    pub depth_out_width: u32,
    /// Output depth map height (fed to RL state).
    #[serde(default = "default_depth_out_height")]
    pub depth_out_height: u32,
    /// Bottom rows of the depth map to zero out (robot body mask).
    /// Expressed in depth-map pixels (not camera pixels).
    #[serde(default)]
    pub depth_mask_rows: u32,
    /// Number of ONNX intra-op CPU threads.
    #[serde(default = "default_num_threads")]
    pub num_threads: usize,
}

impl Default for PerceptionConfig {
    fn default() -> Self {
        Self {
            midas_model_path: default_midas_model_path(),
            depth_out_width:  default_depth_out_width(),
            depth_out_height: default_depth_out_height(),
            depth_mask_rows:  0,
            num_threads:      default_num_threads(),
        }
    }
}

fn default_midas_model_path() -> String { "models/midas_small.onnx".to_string() }
fn default_depth_out_width()  -> u32    { 32 }
fn default_depth_out_height() -> u32    { 32 }
fn default_num_threads()      -> usize  { 4 }
