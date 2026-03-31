//! Typed message bus.
//!
//! The `Bus` struct holds one channel per logical topic. Subsystems publish by
//! calling `send()` on a sender; they subscribe by calling `subscribe()` on a
//! broadcast sender or `clone()` on a watch receiver.
//!
//! Channel type rationale (from requirements_v3.md §11.4):
//!   broadcast  — high-rate sensor streams (many subscribers, bounded)
//!   watch      — latest-state topics (cheap multi-read, no history)
//!   mpsc       — command paths (point-to-point; receivers held by consumers)
//!
//! For the skeleton all sensor/state topics use broadcast. Command senders are
//! stored here; the matching receiver is returned by `Bus::take_*_rx()` once.

use std::sync::Arc;
use tokio::sync::{broadcast, mpsc, watch};

use core_types::{
    BridgeStatus, CameraFrame, CmdVel, DepthMap, EventMarker, ExecutiveState,
    ExploredStats, FeatureFrame, Frontier, FrontierChoice, GrayFrame, GridDelta,
    HealthMetrics, ImuSample, KeyframeEvent, MotorCommand, Orientation, Pose2D,
    PseudoLidarScan, SafetyState, TrackSet, UltrasonicReading, VisualDelta,
};

const DEFAULT_CMDVEL: CmdVel = CmdVel { t_ms: 0, vx: 0.0, vy: 0.0, omega: 0.0 };

/// Commands sent from the UI bridge to the executive task.
///
/// Delivered via `Bus::bridge_cmd` (watch channel).  The executive polls it
/// and drives state transitions; backend silently ignores invalid transitions.
#[derive(Clone, Debug, PartialEq, Default)]
pub enum BridgeCommand {
    #[default]
    None,
    Arm,
    Stop,
    /// Enter manual WASD drive mode (bypasses autonomous planning/control).
    Manual,
    /// Return from manual drive to Idle (autonomous mode can then be re-armed).
    Auto,
}

// Convenience alias – all Arc-wrapped to keep broadcast clone cheap.
pub type ArcFrame    = Arc<CameraFrame>;
pub type ArcGray     = Arc<GrayFrame>;
pub type ArcDepth    = Arc<DepthMap>;
pub type ArcLidar    = Arc<PseudoLidarScan>;
pub type ArcFeatures = Arc<FeatureFrame>;
pub type ArcTracks   = Arc<TrackSet>;

/// Default broadcast channel capacity (messages buffered per receiver).
pub const CAP: usize = 32;

/// The shared message bus. Wrap in `Arc<Bus>` and pass to every subsystem.
pub struct Bus {
    // ── Sensor streams (broadcast) ────────────────────────────────────────
    pub camera_frame_raw:     broadcast::Sender<ArcFrame>,
    pub camera_frame_gray:    broadcast::Sender<ArcGray>,
    pub imu_raw:              broadcast::Sender<ImuSample>,
    pub ultrasonic:           broadcast::Sender<UltrasonicReading>,

    // ── Perception (broadcast) ────────────────────────────────────────────
    pub vision_depth:         broadcast::Sender<ArcDepth>,
    pub vision_pseudo_lidar:  broadcast::Sender<ArcLidar>,
    pub vision_features:      broadcast::Sender<ArcFeatures>,
    pub vision_tracks:        broadcast::Sender<ArcTracks>,
    pub vision_visual_delta:  broadcast::Sender<VisualDelta>,

    // ── Latest-state (watch) ──────────────────────────────────────────────
    pub imu_orientation:      watch::Sender<Orientation>,
    pub slam_pose2d:          watch::Sender<Pose2D>,
    pub safety_state:         watch::Sender<SafetyState>,
    pub gimbal_pan_deg:           watch::Sender<f32>,
    pub gimbal_tilt_deg:          watch::Sender<f32>,
    pub nearest_obstacle_m:       watch::Sender<f32>,
    /// Angle (rad, robot frame: 0=forward, +CCW/left) of the nearest obstacle ray.
    pub nearest_obstacle_angle_rad: watch::Sender<f32>,

    // ── SLAM events (broadcast) ───────────────────────────────────────────
    pub slam_keyframe_event:  broadcast::Sender<KeyframeEvent>,

    // ── Mapping (broadcast) ───────────────────────────────────────────────
    pub map_grid_delta:       broadcast::Sender<GridDelta>,
    pub map_explored_stats:   broadcast::Sender<ExploredStats>,
    pub map_frontiers:        broadcast::Sender<Vec<Frontier>>,
    /// Planner-annotated frontier list (status field set). Published by the
    /// planning task each cycle. The UI bridge prefers this over `map_frontiers`.
    pub map_frontier_annotations: broadcast::Sender<Vec<Frontier>>,

    // ── Executive (watch) ─────────────────────────────────────────────────
    pub executive_state:      watch::Sender<ExecutiveState>,

    // ── Safety / episode event counters (watch) ───────────────────────────
    /// Cumulative collision count. In sim: physics wall-contact events.
    /// On real: IMU-detected impacts (accel spike > threshold while armed).
    pub collision_count:      watch::Sender<u32>,
    /// Cumulative US EmergencyStop count — near-misses that the safety
    /// system caught before a physical contact occurred.
    pub estop_count:          watch::Sender<u32>,
    /// Total rooms explored so far (incremented each time a new room starts).
    pub episode_count:        watch::Sender<u32>,
    /// Rooms that ended by timeout rather than full exploration.
    pub episode_timeout_count: watch::Sender<u32>,

    // ── Sim ground truth (broadcast, one message per episode reset) ──────
    /// Flat wall grid: `walls[y * 200 + x] == 1` means wall. 200×200 cells, 0.05 m/cell.
    pub sim_ground_truth:     broadcast::Sender<Arc<Vec<u8>>>,

    // ── Health / bridge (broadcast) ───────────────────────────────────────
    pub health_runtime:       broadcast::Sender<HealthMetrics>,
    pub ui_bridge_status:     broadcast::Sender<BridgeStatus>,

    // ── UI bridge commands (watch) ────────────────────────────────────────
    /// ARM / STOP / MANUAL / AUTO commands sent from the HTML control panel via WebSocket.
    pub bridge_cmd:           watch::Sender<BridgeCommand>,
    /// Manual WASD velocity from the browser (only applied in ManualDrive state).
    pub manual_cmd_vel:       watch::Sender<CmdVel>,
    /// Velocity actually applied by the motor task (published for SLAM feedforward).
    pub effective_cmd_vel:    watch::Sender<CmdVel>,
    /// Manual gimbal pan/tilt from the browser (only applied in ManualDrive state).
    /// Tuple is (pan_deg, tilt_deg) in degrees.
    pub manual_gimbal_cmd:    watch::Sender<(f32, f32)>,

    // ── Decision / planning / control (mpsc senders stored here) ─────────
    pub decision_frontier:    mpsc::Sender<FrontierChoice>,
    pub planner_path:         mpsc::Sender<core_types::Path>,
    pub controller_cmd_vel:   mpsc::Sender<CmdVel>,
    pub motor_command:        mpsc::Sender<MotorCommand>,

    // ── Telemetry events (mpsc) ───────────────────────────────────────────
    pub telemetry_event:      mpsc::Sender<EventMarker>,
}

/// Receivers for the mpsc command channels. Returned once by `Bus::new()`.
pub struct BusReceivers {
    pub decision_frontier:  mpsc::Receiver<FrontierChoice>,
    pub planner_path:       mpsc::Receiver<core_types::Path>,
    pub controller_cmd_vel: mpsc::Receiver<CmdVel>,
    pub motor_command:      mpsc::Receiver<MotorCommand>,
    pub telemetry_event:    mpsc::Receiver<EventMarker>,
}

/// Watch receivers for latest-state topics. Clone as needed.
pub struct BusWatchRx {
    pub imu_orientation:  watch::Receiver<Orientation>,
    pub slam_pose2d:      watch::Receiver<Pose2D>,
    pub safety_state:     watch::Receiver<SafetyState>,
    pub executive_state:  watch::Receiver<ExecutiveState>,
}

impl Bus {
    /// Create the bus and return it together with the mpsc receivers and
    /// initial watch receivers.
    pub fn new(cap: usize) -> (Arc<Self>, BusReceivers, BusWatchRx) {
        let (tx_frame_raw,  _) = broadcast::channel(cap);
        let (tx_frame_gray, _) = broadcast::channel(cap);
        let (tx_imu_raw,    _) = broadcast::channel(cap * 4);
        let (tx_ultrasonic, _) = broadcast::channel(cap);
        let (tx_depth,      _) = broadcast::channel(cap);
        let (tx_lidar,      _) = broadcast::channel(cap);
        let (tx_features,   _) = broadcast::channel(cap);
        let (tx_tracks,     _) = broadcast::channel(cap);
        let (tx_v_delta,    _) = broadcast::channel(cap);
        let (tx_keyframe,   _) = broadcast::channel(cap);
        let (tx_grid,       _) = broadcast::channel(cap);
        let (tx_explored,   _) = broadcast::channel(cap);
        let (tx_frontiers,  _) = broadcast::channel(cap);
        let (tx_frontier_ann, _) = broadcast::channel(cap);
        let (tx_health,     _) = broadcast::channel(cap);
        let (tx_bridge_st,  _) = broadcast::channel(cap);
        let (tx_sim_truth,  _) = broadcast::channel(4);

        let (tx_bridge_cmd, _)    = watch::channel(BridgeCommand::None);
        let (tx_manual_vel, _)    = watch::channel(DEFAULT_CMDVEL);
        let (tx_eff_vel,    _)    = watch::channel(DEFAULT_CMDVEL);
        let (tx_manual_gimbal, _) = watch::channel((0.0_f32, 0.0_f32));

        let (tx_orientation,   rx_orientation)  = watch::channel(Orientation::default());
        let (tx_pose2d,        rx_pose2d)       = watch::channel(Pose2D::default());
        let (tx_safety,        rx_safety)       = watch::channel(SafetyState::default());
        let (tx_exec_state,    rx_exec_state)   = watch::channel(ExecutiveState::default());
        let (tx_gimbal_pan,    _)               = watch::channel(0.0_f32);
        let (tx_gimbal_tilt,   _)               = watch::channel(0.0_f32);
        let (tx_nearest,       _)               = watch::channel(f32::MAX);
        let (tx_nearest_angle, _)               = watch::channel(0.0_f32);
        let (tx_collisions,    _)               = watch::channel(0_u32);
        let (tx_estops,        _)               = watch::channel(0_u32);
        let (tx_episodes,      _)               = watch::channel(0_u32);
        let (tx_ep_timeouts,   _)               = watch::channel(0_u32);

        let (tx_decision,  rx_decision)  = mpsc::channel(cap);
        let (tx_path,      rx_path)      = mpsc::channel(cap);
        let (tx_cmdvel,    rx_cmdvel)    = mpsc::channel(cap);
        let (tx_motor,     rx_motor)     = mpsc::channel(cap);
        let (tx_telem,     rx_telem)     = mpsc::channel(cap * 4);

        let bus = Arc::new(Bus {
            camera_frame_raw:    tx_frame_raw,
            camera_frame_gray:   tx_frame_gray,
            imu_raw:             tx_imu_raw,
            ultrasonic:          tx_ultrasonic,
            vision_depth:        tx_depth,
            vision_pseudo_lidar: tx_lidar,
            vision_features:     tx_features,
            vision_tracks:       tx_tracks,
            vision_visual_delta: tx_v_delta,
            imu_orientation:     tx_orientation,
            slam_pose2d:         tx_pose2d,
            safety_state:        tx_safety,
            gimbal_pan_deg:              tx_gimbal_pan,
            gimbal_tilt_deg:             tx_gimbal_tilt,
            nearest_obstacle_m:          tx_nearest,
            nearest_obstacle_angle_rad:  tx_nearest_angle,
            slam_keyframe_event: tx_keyframe,
            map_grid_delta:      tx_grid,
            map_explored_stats:  tx_explored,
            map_frontiers:       tx_frontiers,
            map_frontier_annotations: tx_frontier_ann,
            executive_state:     tx_exec_state,
            collision_count:     tx_collisions,
            estop_count:         tx_estops,
            episode_count:       tx_episodes,
            episode_timeout_count: tx_ep_timeouts,
            sim_ground_truth:    tx_sim_truth,
            health_runtime:      tx_health,
            ui_bridge_status:    tx_bridge_st,
            bridge_cmd:          tx_bridge_cmd,
            manual_cmd_vel:      tx_manual_vel,
            effective_cmd_vel:   tx_eff_vel,
            manual_gimbal_cmd:   tx_manual_gimbal,
            decision_frontier:   tx_decision,
            planner_path:        tx_path,
            controller_cmd_vel:  tx_cmdvel,
            motor_command:       tx_motor,
            telemetry_event:     tx_telem,
        });

        let rx = BusReceivers {
            decision_frontier:  rx_decision,
            planner_path:       rx_path,
            controller_cmd_vel: rx_cmdvel,
            motor_command:      rx_motor,
            telemetry_event:    rx_telem,
        };

        let watch_rx = BusWatchRx {
            imu_orientation:  rx_orientation,
            slam_pose2d:      rx_pose2d,
            safety_state:     rx_safety,
            executive_state:  rx_exec_state,
        };

        (bus, rx, watch_rx)
    }
}
