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
    #[serde(default)]
    pub kinematics: KinematicsConfig,
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
    #[serde(default)]
    pub tof: TofConfig,
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

// ── ToF ───────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TofConfig {
    #[serde(default = "default_tof_i2c_bus")]
    pub i2c_bus: u8,
    #[serde(default = "default_tof_i2c_address")]
    pub i2c_address: u8,    // 7-bit: 0x29
    #[serde(default = "default_tof_ranging_mode")]
    pub ranging_mode: u8,   // 1=short (≤135 cm, fastest), 2=long (≤400 cm)
    #[serde(default = "default_tof_integration_time_ms")]
    pub integration_time_ms: u32,
    /// First row to include in per-column minimum (0 = topmost zone, 7 = bottommost).
    /// Row 0 points forward/upward; Row 7 points toward the floor.
    /// Raise row_min to skip top rows if sensor is mounted upside-down.
    #[serde(default = "default_tof_row_min")]
    pub row_min: u8,
    /// Last row to include (inclusive). Set row_max < 7 to exclude floor-facing
    /// bottom rows that produce false close-obstacle readings on open floor.
    /// Default 5 skips rows 6–7 (~14° and ~20° below horizontal at typical mounting).
    #[serde(default = "default_tof_row_max")]
    pub row_max: u8,
}

impl Default for TofConfig {
    fn default() -> Self {
        Self {
            i2c_bus:             default_tof_i2c_bus(),
            i2c_address:         default_tof_i2c_address(),
            ranging_mode:        default_tof_ranging_mode(),
            integration_time_ms: default_tof_integration_time_ms(),
            row_min:             default_tof_row_min(),
            row_max:             default_tof_row_max(),
        }
    }
}

fn default_tof_i2c_bus()             -> u8  { 2 }
fn default_tof_i2c_address()         -> u8  { 0x29 }
fn default_tof_ranging_mode()        -> u8  { 1 }
fn default_tof_integration_time_ms() -> u32 { 5 }
fn default_tof_row_min()             -> u8  { 0 }
fn default_tof_row_max()             -> u8  { 5 } // skip rows 6–7 (floor-facing)

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
    /// Motor duty cycle for escape reverse/rotate maneuver (0-100).
    #[serde(default = "default_escape_reverse_spd")]
    pub escape_reverse_spd: u8,
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
            escape_reverse_spd: default_escape_reverse_spd(),
        }
    }
}

fn default_emergency_stop_cm() -> f32 { 25.0 }
fn default_us_slow_m()         -> f32 { 0.70 }
fn default_us_stop_m()         -> f32 { 0.30 }
fn default_emstop_latch_s()    -> f64 { 5.0 }
fn default_escape_delay_ms()   -> u64 { 200 }
fn default_escape_duration_ms()-> u64 { 300 }  // reduced from 600ms — avoids reversing into rear walls (crash 8)
fn default_escape_rotation_ms()-> u64 { 870 }  // 110° at 35% duty: 35/100 × 6.33 rad/s × 0.87s ≈ 1.92 rad
fn default_clear_hold_s()      -> f32 { 0.8 }
fn default_escape_reverse_spd()-> u8  { 35 }

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
    /// Distance (m) within which the robot considers a path waypoint reached.
    /// Must be ≥ lookahead_dist_m to avoid oscillation at the goal.
    #[serde(default = "default_goal_tolerance_m")]
    pub goal_tolerance_m: f32,
    /// Pure-pursuit lookahead distance (m).
    /// Scale proportionally with kinematics.max_vx when swapping motors.
    #[serde(default = "default_lookahead_dist_m")]
    pub lookahead_dist_m: f32,
}

impl Default for NavigationAgentConfig {
    fn default() -> Self {
        Self {
            max_consecutive_failures: default_max_consecutive_failures(),
            goal_blacklist_s:         default_goal_blacklist_s(),
            goal_tolerance_m:         default_goal_tolerance_m(),
            lookahead_dist_m:         default_lookahead_dist_m(),
        }
    }
}

fn default_max_consecutive_failures() -> u32  { 15 }
fn default_goal_blacklist_s()         -> u64  { 10 }
fn default_goal_tolerance_m()         -> f32  { 0.25 }
fn default_lookahead_dist_m()         -> f32  { 0.20 }

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
    /// Gaussian sigma on simulated ultrasonic range (cm).
    /// Real HC-SR04 takes a median of `hal.ultrasonic.samples_per_reading` pings;
    /// sim replicates this by drawing that many noise samples and returning the median.
    #[serde(default = "default_sim_us_noise_cm")]
    pub ultrasonic_noise_cm: f32,
    /// Magnitude of the IMU accel spike injected on a simulated collision (m/s²).
    /// Must exceed `agent.crash.accel_threshold_m_s2` so crash detection fires.
    #[serde(default = "default_sim_crash_spike_accel")]
    pub crash_spike_accel_m_s2: f32,
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
            ultrasonic_noise_cm:   default_sim_us_noise_cm(),
            crash_spike_accel_m_s2: default_sim_crash_spike_accel(),
        }
    }
}

fn default_sim_seed()               -> u64   { 42 }
fn default_sim_obstacles()          -> usize  { 18 }
// Sim uses ground-truth geometric lidar (not MiDaS), so thresholds are metric.
// Match real-robot intent: slow at 70 cm, stop at 25 cm.
fn default_sim_obstacle_slow_m()    -> f32    { 0.70 }
fn default_sim_obstacle_stop_m()    -> f32    { 0.25 }
fn default_sim_imu_noise_gyro()     -> f32    { 0.01 }
fn default_sim_imu_noise_accel()    -> f32    { 0.05 }
fn default_sim_range_dropout()      -> f32    { 0.08 }
fn default_sim_range_noise_m()      -> f32    { 0.02 }
fn default_sim_us_noise_cm()        -> f32    { 1.5  }
fn default_sim_crash_spike_accel()  -> f32    { 20.0 }

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

// ── Kinematics ────────────────────────────────────────────────────────────────

/// Measured physical speed constants.  Used by SLAM to integrate motor
/// feedforward velocity into the x,y position estimate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KinematicsConfig {
    /// Maximum forward speed at 100% motor duty (m/s). Measured on real hardware.
    #[serde(default = "default_forward_speed_m_s")]
    pub forward_speed_m_s: f32,
    /// Maximum rotation rate at 100% motor duty (rad/s). Measured on real hardware.
    #[serde(default = "default_rotation_rate_rad_s")]
    pub rotation_rate_rad_s: f32,
    /// Maximum CmdVel.vx output from the pure-pursuit controller (normalised units).
    /// `cmdvel_to_motor` maps this to `max_motor_duty`.  Scale proportionally when
    /// swapping motors: half-speed motors → halve this value.
    #[serde(default = "default_max_vx")]
    pub max_vx: f32,
    /// Maximum CmdVel.omega (rad/s) the pure-pursuit is allowed to command.
    /// Set to `rotation_rate_rad_s × max_motor_duty / 100` for full-range turns,
    /// or lower to reduce spin aggressiveness.  Scale with motors like `max_vx`.
    #[serde(default = "default_max_omega_rad_s")]
    pub max_omega_rad_s: f32,
}

impl Default for KinematicsConfig {
    fn default() -> Self {
        Self {
            forward_speed_m_s:   default_forward_speed_m_s(),
            rotation_rate_rad_s: default_rotation_rate_rad_s(),
            max_vx:              default_max_vx(),
            max_omega_rad_s:     default_max_omega_rad_s(),
        }
    }
}

fn default_forward_speed_m_s()   -> f32 { 1.23 }
fn default_rotation_rate_rad_s() -> f32 { 6.33 }
fn default_max_vx()              -> f32 { 0.3  }
fn default_max_omega_rad_s()     -> f32 { 2.2  }
