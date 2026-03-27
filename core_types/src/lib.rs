//! Core data types shared across all crates.
//!
//! All types derive Debug, Clone, serde::Serialize/Deserialize so they can
//! flow through the bus and be written to telemetry logs.

use serde::{Deserialize, Serialize};

// ── Time ──────────────────────────────────────────────────────────────────────

/// Monotonic millisecond timestamp from system startup.
pub type Ms = u64;

// ── Camera ────────────────────────────────────────────────────────────────────

/// Raw RGB camera frame (packed R,G,B bytes, row-major).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraFrame {
    pub t_ms: Ms,
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>, // len = width * height * 3
}

/// Single-channel grayscale frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrayFrame {
    pub t_ms: Ms,
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>, // len = width * height
}

// ── IMU ───────────────────────────────────────────────────────────────────────

/// Raw IMU sample from gyro + accelerometer.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ImuSample {
    pub t_ms: Ms,
    pub gyro_x: f32,  // rad/s
    pub gyro_y: f32,  // rad/s
    pub gyro_z: f32,  // rad/s  (yaw rate, positive = CCW from above)
    pub accel_x: f32, // m/s²
    pub accel_y: f32, // m/s²
    pub accel_z: f32, // m/s²
}

/// Fused IMU orientation (from bias-corrected integration).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Orientation {
    pub t_ms: Ms,
    pub yaw_rad: f32,
    pub pitch_rad: f32,
    pub roll_rad: f32,
}

impl Default for Orientation {
    fn default() -> Self {
        Self { t_ms: 0, yaw_rad: 0.0, pitch_rad: 0.0, roll_rad: 0.0 }
    }
}

// ── Perception ────────────────────────────────────────────────────────────────

/// MiDaS inverse-depth map. Values in [0,1]; 1.0 = closest.
/// Rows with index >= mask_start_row are the robot-body mask and should be
/// ignored downstream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthMap {
    pub t_ms: Ms,
    pub width: u32,
    pub height: u32,
    pub data: Vec<f32>,        // len = width * height
    pub mask_start_row: u32,   // rows [mask_start_row..height) are masked
}

/// Single pseudo-lidar ray derived from the depth map.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LidarRay {
    pub angle_rad: f32, // in robot frame, 0 = forward, positive = CCW
    pub range_m: f32,
    pub confidence: f32, // [0,1]
}

/// A full pseudo-lidar scan converted from one depth frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PseudoLidarScan {
    pub t_ms: Ms,
    pub rays: Vec<LidarRay>,
}

// ── Vision features / tracking (micro-SLAM inputs) ───────────────────────────

/// A single detected feature point in image coordinates.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FeaturePoint {
    pub id: u32,
    pub x: f32, // pixels
    pub y: f32, // pixels
    pub score: f32,
}

/// Set of features detected in one frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureFrame {
    pub t_ms: Ms,
    pub points: Vec<FeaturePoint>,
}

/// Feature tracks between consecutive frames.
/// Each match is (feature_id, prev_[x,y], curr_[x,y]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackSet {
    pub t_ms: Ms,
    pub matches: Vec<(u32, [f32; 2], [f32; 2])>,
    pub mean_flow: [f32; 2], // mean optical flow vector in pixels
    pub inlier_count: usize,
}

// ── Micro-SLAM outputs ────────────────────────────────────────────────────────

/// Frame-to-frame visual motion estimate.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct VisualDelta {
    pub t_ms: Ms,
    pub dx_m: f32,
    pub dy_m: f32,
    pub dtheta_rad: f32,
    pub confidence: f32,
}

/// Fused 2-D robot pose published by micro-SLAM.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Pose2D {
    pub t_ms: Ms,
    pub x_m: f32,
    pub y_m: f32,
    pub theta_rad: f32,
    pub confidence: f32,
}

impl Default for Pose2D {
    fn default() -> Self {
        Self { t_ms: 0, x_m: 0.0, y_m: 0.0, theta_rad: 0.0, confidence: 0.0 }
    }
}

/// Emitted when a new keyframe is inserted into the local map.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyframeEvent {
    pub t_ms: Ms,
    pub keyframe_id: u32,
    pub pose: Pose2D,
}

// ── Mapping ───────────────────────────────────────────────────────────────────

/// Incremental update to the occupancy grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridDelta {
    pub t_ms: Ms,
    /// (grid_x, grid_y, log_odds_delta)
    pub cells: Vec<(i32, i32, f32)>,
}

/// A detected frontier (boundary between known-free and unknown space).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Frontier {
    pub centroid_x_m: f32,
    pub centroid_y_m: f32,
    pub size_cells: u32,
    /// Planner-assigned display status (see `frontier_status` constants).
    /// Default 0 = normal; set by the planning task after each cycle.
    #[serde(default)]
    pub status: u8,
}

/// Display status codes for [`Frontier::status`].
pub mod frontier_status {
    pub const NORMAL:           u8 = 0; // reachable candidate
    pub const TOO_SMALL:        u8 = 1; // below MIN_FRONTIER_SIZE
    pub const TOO_CLOSE:        u8 = 2; // within MIN_GOAL_DIST of robot
    pub const UNREACHABLE:      u8 = 3; // BFS-disconnected from robot
    pub const SOFT_BLACKLISTED: u8 = 4; // in goal_blacklist (short timeout)
    pub const HARD_BLACKLISTED: u8 = 5; // in hard_blacklist (crash area)
    pub const CURRENT_GOAL:     u8 = 6; // actively being pursued
}

/// Exploration statistics published after each map update.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ExploredStats {
    pub t_ms: Ms,
    pub explored_cells: u32,
    pub frontier_count: u32,
}

// ── Planning / control ────────────────────────────────────────────────────────

/// A planned path as a sequence of waypoints in world frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Path {
    pub t_ms: Ms,
    pub waypoints: Vec<[f32; 2]>, // [x_m, y_m]
}

/// Velocity command output by the pure-pursuit controller.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct CmdVel {
    pub t_ms: Ms,
    pub vx: f32,    // forward m/s equivalent (scaled to duty cycle later)
    pub vy: f32,    // strafe m/s (mecanum)
    pub omega: f32, // yaw rate rad/s
}

/// Direct motor duty-cycle command (−100 to +100 per motor).
/// Order: front-left, front-right, rear-left, rear-right.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MotorCommand {
    pub t_ms: Ms,
    pub fl: i8,
    pub fr: i8,
    pub rl: i8,
    pub rr: i8,
}

impl MotorCommand {
    pub fn stop(t_ms: Ms) -> Self {
        Self { t_ms, fl: 0, fr: 0, rl: 0, rr: 0 }
    }
}

// ── Safety ────────────────────────────────────────────────────────────────────

/// Current safety system state.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SafetyState {
    Ok,
    EmergencyStop { reason: String },
}

impl Default for SafetyState {
    fn default() -> Self {
        Self::Ok
    }
}

/// Ultrasonic distance reading.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct UltrasonicReading {
    pub t_ms: Ms,
    pub range_cm: f32,
}

// ── Exploration / RL ─────────────────────────────────────────────────────────

/// RL agent's frontier selection decision.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum FrontierChoice {
    Nearest,
    Largest,
    Leftmost,
    Rightmost,
    RandomValid,
}

// ── Executive ─────────────────────────────────────────────────────────────────

/// High-level robot executive state.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ExecutiveState {
    Idle,
    Calibrating,
    Exploring,
    Recovering,
    SafetyStopped,
    /// Browser is sending manual WASD velocity commands.
    ManualDrive,
    Fault { reason: String },
}

impl Default for ExecutiveState {
    fn default() -> Self {
        Self::Idle
    }
}

// ── Health / observability ────────────────────────────────────────────────────

/// Runtime health snapshot published periodically.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HealthMetrics {
    pub t_ms: Ms,
    pub cpu_pct: f32,
    pub mem_used_mb: u32,
    /// Loop timing jitter — difference from nominal period (ms).
    pub task_jitter_ms: f32,
}

// ── UI bridge ─────────────────────────────────────────────────────────────────

/// Status snapshot published by the WebSocket bridge.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BridgeStatus {
    pub t_ms: Ms,
    pub connected_clients: u32,
    pub bytes_sent_total: u64,
    pub last_send_ok: bool,
}

// ── Calibration ───────────────────────────────────────────────────────────────

/// Saved calibration parameters (camera intrinsics + IMU bias + extrinsics).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibData {
    // Camera intrinsics (pinhole model)
    pub camera_fx: f32,
    pub camera_fy: f32,
    pub camera_cx: f32,
    pub camera_cy: f32,
    // IMU bias estimates
    pub imu_gyro_bias: [f32; 3],  // [x, y, z] rad/s
    pub imu_accel_bias: [f32; 3], // [x, y, z] m/s²
    // Camera-to-base extrinsics [tx_m, ty_m, tz_m, rx_rad, ry_rad, rz_rad]
    pub camera_to_base: [f32; 6],
    /// Validated camera tilt angle (rad, positive = tilted down).
    pub camera_tilt_rad: f32,
}

impl Default for CalibData {
    fn default() -> Self {
        Self {
            // 110° H-FOV camera, 640×480 — approximate pinhole intrinsics
            camera_fx: 410.0,
            camera_fy: 410.0,
            camera_cx: 320.0,
            camera_cy: 240.0,
            imu_gyro_bias:  [0.0; 3],
            imu_accel_bias: [0.0; 3],
            // Camera ~10cm above base, facing forward with no tilt
            camera_to_base: [0.0, 0.0, 0.10, 0.0, 0.0, 0.0],
            camera_tilt_rad: 0.0,
        }
    }
}

// ── Telemetry ─────────────────────────────────────────────────────────────────

/// A structured event marker logged to telemetry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventMarker {
    Collision,
    EmergencyStop,
    FrontierSelected { choice: FrontierChoice },
    FrontierReached,
    ParallaxScanStarted,
    ParallaxScanCompleted,
    TrackingConfidenceDrop { confidence: f32 },
    KeyframeInserted { id: u32 },
    PlannerFailed,
    RecoveryTriggered,
    CalibrationCompleted { mode: String },
    ExecutiveTransition { from: String, to: String },
    Custom(String),
}
