//! Robot runtime — Milestone 2 (mapping, planning, control, safety).
//!
//! Tasks started here:
//!   camera       → perception → pseudo-lidar
//!   imu          → dead-reckoning pose
//!   ultrasonic   → safety interlock
//!   mapping      → occupancy grid + frontier detection
//!   safety       → emergency-stop monitor + motor override
//!   planning     → A* path (triggered by FrontierChoice from exploration_rl)
//!   control      → pure-pursuit cmd_vel (triggered by new Path)
//!   telemetry    → NDJSON log writer
//!   executive    → state machine
//!   ui_bridge    → WebSocket visualization
//!
//! Run modes (first CLI argument):
//!   robot calibrate    — IMU bias collection; prints offsets for robot_config.yaml
//!   robot slam-debug   — camera + IMU + telemetry only; no planning or motors
//!   robot robot-run    — full autonomy stack (default if arg is omitted or unknown)
//!   robot robot-debug  — full autonomy stack (alias; heavy telemetry planned)

use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use bus::Bus;
use config::RobotConfig;
use control::PurePursuitController;
use core_types::{CmdVel, ExecutiveState, FrontierChoice, MotorCommand, SafetyState};
use executive::Executive;
use hal::{
    Camera, Gimbal, Imu, MotorController, Ultrasonic,
    StubCamera, StubGimbal, StubImu, StubMotorController, StubUltrasonic,
    YahboomGimbal, YahboomMotorController, YahboomUltrasonic,
};
#[cfg(feature = "usb-camera")]
use hal::V4L2Camera;
#[cfg(feature = "mpu6050")]
use hal::Mpu6050Imu;
use mapping::Mapper;
use micro_slam::ImuDeadReckon;
use perception::{DepthInference, EventGate, PseudoLidarExtractor};
use planning::AStarPlanner;
use safety::SafetyMonitor;
use telemetry::TelemetryWriter;
use ui_bridge::{UiBridgeConfig, start as start_ui_bridge};
use exploration_rl::spawn_selector_task;
use sim_fast::hal_impls::SimState;
use sim_vision::SimCamera as SimCameraHal;
use bus::BridgeCommand;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // ── Logging ───────────────────────────────────────────────────────────────
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("robot=info".parse()?)
                .add_directive("info".parse()?),
        )
        .init();

    // ── Config ────────────────────────────────────────────────────────────────
    let cfg = RobotConfig::load("robot_config.yaml")?;
    info!(name = %cfg.robot.name, "Config loaded");

    // ── Mode dispatch ─────────────────────────────────────────────────────────
    // First CLI argument selects the operating mode.  Unrecognised values fall
    // through to robot-run (full autonomy stack) so existing launch scripts
    // continue to work without modification.
    let mode = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "robot-run".to_string());
    info!(mode = %mode, "Operating mode");

    // ── Bus ───────────────────────────────────────────────────────────────────
    let (bus, rx, _watch_rx) = Bus::new(bus::CAP);
    // Extract mpsc receivers for planning/control tasks; keep rest alive.
    let plan_frontier_rx = rx.decision_frontier;
    let control_path_rx  = rx.planner_path;
    let motor_cmd_rx     = rx.motor_command;     // safety-triggered MotorCommands
    let cmdvel_rx        = rx.controller_cmd_vel; // pure-pursuit CmdVel
    let _telem_event_rx  = rx.telemetry_event;
    info!("Message bus created");

    // ── Sim mode — short-circuit before any HAL init ──────────────────────────
    if mode == "sim" {
        return run_sim_mode(bus, cfg, plan_frontier_rx, control_path_rx, motor_cmd_rx, cmdvel_rx).await;
    }

    // ── Telemetry ─────────────────────────────────────────────────────────────
    let mut telem = TelemetryWriter::open("logs/robot.ndjson").await?;
    telem.event("runtime", "system_boot").await?;
    info!("Telemetry writer opened → logs/robot.ndjson");

    // ── HAL initialisation ────────────────────────────────────────────────────
    // Real drivers are tried first; fall back to stubs so the binary still
    // runs on a dev machine without I2C/V4L2 devices.

    let camera: Box<dyn Camera> = {
        #[cfg(feature = "usb-camera")]
        {
            let dev = if cfg.hal.camera.device_path.is_empty() {
                format!("/dev/video{}", cfg.hal.camera.device_index)
            } else {
                cfg.hal.camera.device_path.clone()
            };
            let stream_port = cfg.hal.camera.stream_enabled
                .then_some(cfg.hal.camera.stream_port);
            match V4L2Camera::new(&dev, cfg.hal.camera.width, cfg.hal.camera.height,
                                  cfg.hal.camera.fps, stream_port)
            {
                Ok(c)  => { info!("Camera: V4L2 on {dev}"); Box::new(c) as Box<dyn Camera> }
                Err(e) => { warn!("Camera: V4L2 unavailable ({e:#}), using stub");
                            Box::new(StubCamera::new(&cfg.hal.camera)) }
            }
        }
        #[cfg(not(feature = "usb-camera"))]
        {
            info!("Camera: stub (usb-camera feature disabled)");
            Box::new(StubCamera::new(&cfg.hal.camera))
        }
    };

    let imu: Box<dyn Imu> = {
        #[cfg(feature = "mpu6050")]
        {
            match Mpu6050Imu::new(6) {
                Ok(i)  => { info!("IMU: MPU-6050 on /dev/i2c-6"); Box::new(i) as Box<dyn Imu> }
                Err(e) => { warn!("IMU: MPU-6050 unavailable ({e:#}), using stub");
                            Box::new(StubImu::new()) }
            }
        }
        #[cfg(not(feature = "mpu6050"))]
        {
            info!("IMU: stub (mpu6050 feature disabled)");
            Box::new(StubImu::new())
        }
    };

    let ultrasonic: Box<dyn Ultrasonic> = match YahboomUltrasonic::new(
        cfg.hal.ultrasonic.i2c_bus,
        cfg.hal.ultrasonic.i2c_address as u16,
        cfg.hal.ultrasonic.max_range_cm,
        cfg.hal.ultrasonic.min_range_cm,
        cfg.hal.ultrasonic.samples_per_reading,
    ) {
        Ok(u)  => { info!("Ultrasonic: Yahboom on /dev/i2c-{}",
                          cfg.hal.ultrasonic.i2c_bus); Box::new(u) }
        Err(e) => { warn!("Ultrasonic: Yahboom unavailable ({e:#}), using stub");
                    Box::new(StubUltrasonic::new()) }
    };

    let motor: Box<dyn MotorController> = if cfg.hal.motor.driver == "yahboom" {
        match YahboomMotorController::new(cfg.hal.motor.i2c_bus,
                                          cfg.hal.motor.i2c_address as u16,
                                          cfg.hal.motor.max_speed as i8) {
            Ok(m)  => { info!("Motor: Yahboom on /dev/i2c-{}",
                              cfg.hal.motor.i2c_bus); Box::new(m) }
            Err(e) => { warn!("Motor: Yahboom unavailable ({e:#}), using stub");
                        Box::new(StubMotorController) }
        }
    } else {
        info!("Motor: stub (driver={})", cfg.hal.motor.driver);
        Box::new(StubMotorController)
    };

    let gimbal: Box<dyn Gimbal> = if cfg.hal.gimbal.driver == "yahboom" {
        match YahboomGimbal::new(cfg.hal.gimbal.i2c_bus,
                                 cfg.hal.gimbal.i2c_address as u16,
                                 cfg.hal.gimbal.pan_range,
                                 cfg.hal.gimbal.tilt_range,
                                 cfg.hal.gimbal.tilt_neutral) {
            Ok(g)  => { info!("Gimbal: Yahboom on /dev/i2c-{}",
                              cfg.hal.gimbal.i2c_bus); Box::new(g) }
            Err(e) => { warn!("Gimbal: Yahboom unavailable ({e:#}), using stub");
                        Box::new(StubGimbal::new(cfg.hal.gimbal.tilt_neutral)) }
        }
    } else {
        info!("Gimbal: stub (driver={})", cfg.hal.gimbal.driver);
        Box::new(StubGimbal::new(cfg.hal.gimbal.tilt_neutral))
    };

    let max_motor_duty = cfg.hal.motor.max_speed as i8;
    info!("HAL initialised");

    // Early exit for modes that don't need the full autonomy stack.
    // Each branch takes ownership of only what it needs; Rust's flow analysis
    // guarantees the remaining variables are still valid for robot-run below.
    if mode == "calibrate" {
        return run_calibrate(imu).await;
    }
    if mode == "slam-debug" {
        return run_slam_debug(camera, imu, gimbal, bus, &cfg).await;
    }
    if mode != "robot-run" && mode != "robot-debug" {
        warn!("Unknown mode '{mode}' — running full autonomy stack (robot-run)");
    }

    // ── Perception ────────────────────────────────────────────────────────────
    let mut depth_infer = DepthInference::new(
        &cfg.perception.midas_model_path,
        cfg.perception.depth_out_width,
        cfg.perception.depth_out_height,
        cfg.perception.depth_mask_rows,
        cfg.perception.num_threads,
    );
    let mut event_gate = EventGate::with_default_threshold();
    let lidar_extractor = PseudoLidarExtractor::new(48, 3.0);
    info!("Perception pipeline initialised");

    // ── Micro-SLAM ────────────────────────────────────────────────────────────
    let mut slam = ImuDeadReckon::with_gz_bias(cfg.imu.gz_bias);
    info!(gz_bias = cfg.imu.gz_bias, "Micro-SLAM initialised (IMU dead-reckoning)");

    // ── Mapper (shared: mapping task writes, planning task reads) ─────────────
    let mapper = Arc::new(RwLock::new(Mapper::new()));
    info!("Occupancy grid mapper initialised (5 cm/cell)");

    // ── Executive task ────────────────────────────────────────────────────────
    // Owns the state machine.  Reacts to two event sources:
    //   1. `safety_state` watch — fires SafetyStopped when the US trips.
    //   2. arm_tx channel      — SIGUSR1 or any future arm command.
    //
    // Motor CmdVel is blocked unless ExecutiveState ∈ {Exploring, Recovering}.
    // To arm: `kill -USR1 <pid>` (or send on arm_tx from a future UI).
    let (arm_tx, mut arm_rx) = tokio::sync::mpsc::channel::<()>(4);
    let mut rx_safety_exec   = bus.safety_state.subscribe();
    let bus_exec             = Arc::clone(&bus);
    let exec_handle = tokio::spawn(async move {
        let mut exec = Executive::new(Arc::clone(&bus_exec));
        info!("Executive task started (Idle) — send SIGUSR1 to arm");
        loop {
            tokio::select! {
                biased;
                _ = arm_rx.recv() => {
                    match exec.arm() {
                        Ok(()) => info!("Executive: armed → Exploring"),
                        Err(e) => warn!("Executive arm rejected: {e}"),
                    }
                }
                Ok(()) = rx_safety_exec.changed() => {
                    let safety = rx_safety_exec.borrow().clone();
                    if matches!(safety, SafetyState::EmergencyStop { .. }) {
                        match exec.state() {
                            ExecutiveState::Exploring | ExecutiveState::Recovering => {
                                let _ = exec.transition(ExecutiveState::SafetyStopped);
                            }
                            _ => {}
                        }
                    } else {
                        // Safety cleared — auto-recover from SafetyStopped so the robot
                        // resumes exploration without requiring a manual re-arm.
                        if matches!(exec.state(), ExecutiveState::SafetyStopped) {
                            info!("Executive: safety cleared — auto-recovering to Exploring");
                            let _ = exec.arm();
                        }
                    }
                }
            }
        }
    });
    // SIGUSR1 → arm
    let arm_tx_sig = arm_tx.clone();
    tokio::spawn(async move {
        use tokio::signal::unix::{signal, SignalKind};
        let mut sig = match signal(SignalKind::user_defined1()) {
            Ok(s) => s,
            Err(e) => { warn!("SIGUSR1 handler failed: {e}"); return; }
        };
        loop {
            sig.recv().await;
            info!("SIGUSR1 received — arming robot");
            let _ = arm_tx_sig.send(()).await;
        }
    });

    // ── UI Bridge ─────────────────────────────────────────────────────────────
    start_ui_bridge(Arc::clone(&bus), UiBridgeConfig::default()).await?;

    // ── Frontier selector (exploration_rl) ────────────────────────────────────
    // Classical heuristic by default; ONNX policy loaded if model file exists.
    spawn_selector_task(Arc::clone(&bus)).await;

    // ── Camera task ───────────────────────────────────────────────────────────
    let bus_cam  = Arc::clone(&bus);
    let mut camera = camera;
    let cam_handle = tokio::spawn(async move {
        info!("Camera task started");
        loop {
            match camera.read_frame().await {
                Ok(frame) => {
                    let arc = Arc::new(frame);
                    let _ = bus_cam.camera_frame_raw.send(Arc::clone(&arc));
                    let gray = rgb_to_gray(&arc);
                    let _ = bus_cam.camera_frame_gray.send(Arc::new(gray));
                }
                Err(e) => error!("Camera error: {e}"),
            }
        }
    });

    // ── IMU task ──────────────────────────────────────────────────────────────
    let bus_imu = Arc::clone(&bus);
    let mut imu = imu;
    let imu_handle = tokio::spawn(async move {
        info!("IMU task started");
        loop {
            match imu.read_sample().await {
                Ok(s) => { let _ = bus_imu.imu_raw.send(s); }
                Err(e) => error!("IMU error: {e}"),
            }
        }
    });

    // ── Crash detection task (IMU-based) ─────────────────────────────────────
    // Monitors horizontal acceleration for sudden spikes that indicate a wall
    // impact.  Only counts while the robot is armed (Exploring/Recovering) to
    // avoid false positives from manual handling.
    //
    // Threshold: 15 m/s² horizontal (~1.5 g).  Normal cornering is < 3 m/s²;
    // a wall impact at typical nav speeds produces 10–30 m/s².
    // Debounce: 2 s between events so a single impact counts once.
    {
        const CRASH_ACCEL_THRESHOLD: f32 = 15.0; // m/s²
        const CRASH_DEBOUNCE_MS: u64 = 2_000;
        let bus_crash  = Arc::clone(&bus);
        let mut rx_imu_c  = bus.imu_raw.subscribe();
        let rx_exec_crash = bus.executive_state.subscribe();
        tokio::spawn(async move {
            let mut last_crash_ms: u64 = 0;
            loop {
                match rx_imu_c.recv().await {
                    Ok(s) => {
                        let armed = matches!(
                            *rx_exec_crash.borrow(),
                            ExecutiveState::Exploring | ExecutiveState::Recovering
                        );
                        if !armed { continue; }
                        let horiz = (s.accel_x * s.accel_x + s.accel_y * s.accel_y).sqrt();
                        if horiz > CRASH_ACCEL_THRESHOLD && s.t_ms > last_crash_ms + CRASH_DEBOUNCE_MS {
                            last_crash_ms = s.t_ms;
                            let n = *bus_crash.collision_count.borrow() + 1;
                            let _ = bus_crash.collision_count.send(n);
                            warn!(horiz_accel_m_s2 = horiz, collision_n = n, "Crash detected — IMU impact spike");
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {}
                    Err(tokio::sync::broadcast::error::RecvError::Closed)    => break,
                }
            }
        });
    }

    // ── Ultrasonic task ───────────────────────────────────────────────────────
    let bus_us = Arc::clone(&bus);
    let mut ultrasonic = ultrasonic;
    let us_handle = tokio::spawn(async move {
        info!("Ultrasonic task started");
        loop {
            match ultrasonic.read_distance().await {
                Ok(r) => { let _ = bus_us.ultrasonic.send(r); }
                Err(e) => error!("Ultrasonic error: {e}"),
            }
        }
    });

    // ── SLAM task (IMU-only, runs independently of depth inference) ──────────
    // Kept separate so that the 147ms depth inference does not starve IMU
    // integration: heading would freeze during every inference frame if both
    // were in the same select! loop.
    let bus_slam   = Arc::clone(&bus);
    let mut rx_imu = bus.imu_raw.subscribe();
    let slam_handle = tokio::spawn(async move {
        info!("SLAM task started (IMU dead-reckoning)");
        loop {
            match rx_imu.recv().await {
                Ok(sample) => {
                    let pose = slam.update(&sample);
                    let _ = bus_slam.slam_pose2d.send(pose);
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    warn!("SLAM: IMU lagged {n} samples");
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    });

    // ── Perception task (camera → depth → pseudo-lidar) ───────────────────────
    let bus_perc   = Arc::clone(&bus);
    let mut rx_gray = bus.camera_frame_gray.subscribe();
    let perc_handle = tokio::spawn(async move {
        info!("Perception task started");
        let mut last_infer = std::time::Instant::now();
        const MAX_INFER_INTERVAL: std::time::Duration = std::time::Duration::from_secs(2);
        // Rolling minimum over the last N MiDaS frames.  MiDaS on real hardware
        // produces noisy frame-to-frame readings (e.g. 0.3 m → 1.5 m → 0.3 m for
        // the same static scene) which cause false clears and drive the robot into
        // obstacles.  Publishing the rolling min means a genuine obstacle stays
        // visible for ~N frames even if one noisy frame reads much farther away.
        // N=5 at ~3 Hz ≈ 1.7 s of memory — enough to cover 4–5 frame MiDaS lag.
        const NEAR_HIST_LEN: usize = 5;
        let mut near_hist: std::collections::VecDeque<f32> = std::collections::VecDeque::with_capacity(NEAR_HIST_LEN);
        loop {
            match rx_gray.recv().await {
                Ok(gray) => {
                    let force_timer = last_infer.elapsed() >= MAX_INFER_INTERVAL;
                    if force_timer { event_gate.force(); }
                    if event_gate.should_infer(&gray) {
                        last_infer = std::time::Instant::now();
                        let rgb_stub = core_types::CameraFrame {
                            t_ms: gray.t_ms, width: gray.width, height: gray.height,
                            data: gray.data.iter().flat_map(|&v| [v, v, v]).collect(),
                        };
                        match tokio::task::block_in_place(|| depth_infer.infer(&rgb_stub)) {
                            Ok(depth) => {
                                let mut scan = lidar_extractor.extract(&depth);
                                // Nearest obstacle in camera frame (before pan offset).
                                let (near_m, near_cam_rad) = nearest_in_fov(&scan);
                                // Convert pan to robot-frame offset and apply to all rays
                                // so the mapper correctly places walls in world frame.
                                // Convention: pan_deg > 0 = looking right (negative in
                                // robot angle frame where CCW/left is positive).
                                let pan_deg = *bus_perc.gimbal_pan_deg.borrow();
                                let pan_rad = -pan_deg * std::f32::consts::PI / 180.0;
                                for ray in &mut scan.rays { ray.angle_rad += pan_rad; }
                                let near_angle_rad = near_cam_rad + pan_rad; // robot frame
                                // Rolling minimum: keep the closest reading seen in the last
                                // NEAR_HIST_LEN frames so a single noisy far reading can't
                                // cause a false clear and allow forward motion into an obstacle.
                                near_hist.push_back(near_m);
                                if near_hist.len() > NEAR_HIST_LEN { near_hist.pop_front(); }
                                let near_m_min = near_hist.iter().cloned().fold(f32::MAX, f32::min);
                                // Log approach timeline (raw near_m for diagnostics).
                                if near_m < 1.5 || near_m_min < 1.5 {
                                    let near_deg = near_angle_rad.to_degrees();
                                    let side = if near_deg > 5.0 { "left" } else if near_deg < -5.0 { "right" } else { "center" };
                                    info!(nearest_m = near_m_min, nearest_raw_m = near_m, angle_deg = near_deg, side, "Perception: nearest obstacle");
                                }
                                let _ = bus_perc.vision_depth.send(Arc::new(depth));
                                let _ = bus_perc.vision_pseudo_lidar.send(Arc::new(scan));
                                let _ = bus_perc.nearest_obstacle_m.send(near_m_min);
                                let _ = bus_perc.nearest_obstacle_angle_rad.send(near_angle_rad);
                            }
                            Err(e) => error!("Depth inference error: {e}"),
                        }
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    warn!("Perception: camera lagged {n} frames");
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    });

    // ── Mapping task ──────────────────────────────────────────────────────────
    let map_handle = spawn_mapping_task(Arc::clone(&bus), Arc::clone(&mapper));

    // ── Safety task ───────────────────────────────────────────────────────────
    let emergency_stop_cm  = cfg.agent.safety.emergency_stop_cm;
    let safety_handle = spawn_safety_task(Arc::clone(&bus), emergency_stop_cm, true);

    // ── Planning task ─────────────────────────────────────────────────────────
    // Triggered by FrontierChoice from exploration_rl (Phase 10).
    // Reads frontier list cached in the mapper, runs A*, sends Path.
    let bus_plan        = Arc::clone(&bus);
    let mapper_plan     = Arc::clone(&mapper);
    let mut plan_rx     = plan_frontier_rx;
    let mut rx_pose_p   = bus.slam_pose2d.subscribe();
    let plan_handle = tokio::spawn(async move {
        use tokio::time::Instant;
        const GOAL_BLACKLIST_SECS: u64 = 10;
        info!("Planning task started (A*, 4-cell clearance)");
        let planner = AStarPlanner::new();
        // (goal_m, expiry): goals to skip — filled when A* fails OR same goal repeats.
        let mut goal_blacklist: Vec<([f32; 2], Instant)> = Vec::new();
        let mut last_goal: Option<[f32; 2]> = None;
        let mut last_path_sent_at: Option<Instant> = None;
        // How many consecutive times the same goal must be selected before it is
        // blacklisted.  2 is too eager — a replan after an obstacle stop often
        // re-selects the nearest frontier before the robot has moved at all.
        const SAME_GOAL_BLACKLIST_COUNT: u32 = 3;
        let mut same_goal_streak: u32 = 0;
        // Safety valve: if every frontier fails A* repeatedly the blacklist
        // saturates and the robot is permanently stuck.  After this many
        // consecutive failures, clear the blacklist entirely for a fresh pass.
        const MAX_CONSECUTIVE_FAILURES: u32 = 15;
        let mut consecutive_failures: u32 = 0;
        loop {
            let Some(choice) = plan_rx.recv().await else { break };
            let now = Instant::now();
            goal_blacklist.retain(|(_, exp)| *exp > now);
            let bl: Vec<[f32; 2]> = goal_blacklist.iter().map(|(g, _)| *g).collect();

            let pose = *rx_pose_p.borrow_and_update();
            let maybe_goal = {
                let m = mapper_plan.read().await;
                select_frontier_goal(&m, &choice, &pose, &bl)
            };
            if let Some(goal) = maybe_goal {
                // Blacklist a goal selected SAME_GOAL_BLACKLIST_COUNT times in a
                // row without making progress.  A single repeat is normal after an
                // obstacle stop before the robot moves; only blacklist on persistence.
                let same = last_goal.map_or(false, |lg| {
                    let dx = lg[0] - goal[0]; let dy = lg[1] - goal[1];
                    (dx*dx + dy*dy).sqrt() < 0.10
                });
                if same {
                    same_goal_streak += 1;
                } else {
                    same_goal_streak = 0;
                }
                if same_goal_streak >= SAME_GOAL_BLACKLIST_COUNT {
                    warn!(x = goal[0], y = goal[1], streak = same_goal_streak, "Planning: same goal repeated — blacklisting");
                    let exp = now + std::time::Duration::from_secs(GOAL_BLACKLIST_SECS);
                    goal_blacklist.push((goal, exp));
                    last_goal = None;
                    same_goal_streak = 0;
                    consecutive_failures += 1;
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
                        warn!(consecutive_failures, "Planning: blacklist saturated — clearing for fresh pass");
                        goal_blacklist.clear();
                        consecutive_failures = 0;
                    }
                    continue;
                }
                // Don't interrupt an active path with a different goal.
                // 3 s is enough to protect a path in flight; goals are reached in
                // 2-7 s so the window expires shortly after arrival.
                const PATH_COMMIT_S: u64 = 3;
                let goal_matches_last = last_goal.map_or(false, |lg| {
                    let dx = lg[0] - goal[0]; let dy = lg[1] - goal[1];
                    (dx*dx + dy*dy).sqrt() < 0.25
                });
                let path_committed = !goal_matches_last
                    && last_path_sent_at.map_or(false, |t| {
                        now.duration_since(t) < std::time::Duration::from_secs(PATH_COMMIT_S)
                    });
                if path_committed {
                    continue;
                }
                last_goal = Some(goal);

                let planned = {
                    let m = mapper_plan.read().await;
                    planner.plan_with_clearance(&m, &pose, goal, 7)
                        .or_else(|| planner.plan_with_clearance(&m, &pose, goal, 5))
                };
                match planned {
                    Some(path) => {
                        info!(waypoints = path.waypoints.len(), "Planning: path found");
                        consecutive_failures = 0;
                        last_path_sent_at = Some(now);
                        let _ = bus_plan.planner_path.send(path).await;
                    }
                    None => {
                        warn!(?choice, "Planning: no path to frontier");
                        let exp = now + std::time::Duration::from_secs(GOAL_BLACKLIST_SECS);
                        goal_blacklist.push((goal, exp));
                        last_goal = None;
                        consecutive_failures += 1;
                        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
                            warn!(consecutive_failures, "Planning: blacklist saturated — clearing for fresh pass");
                            goal_blacklist.clear();
                            consecutive_failures = 0;
                        }
                    }
                }
            } else {
                warn!(?choice, "Planning: no frontiers available");
            }
        }
    });

    // ── Control task ──────────────────────────────────────────────────────────
    // Real: slow=2.00 m, stop=1.00 m — raised from 0.60 to account for 1.5s camera
    // lag at ~0.17 m/s forward speed (0.25 m travel during lag).  When camera reads
    // 1.0 m the robot is actually ~0.55 m from the obstacle — enough to stop safely.
    // Real mode has no episode reset; pass a dummy watch whose sender is dropped
    // so the episode arm in spawn_control_task never fires.
    let (_episode_dummy_tx, episode_dummy_rx) = tokio::sync::watch::channel(0u32);
    // MiDaS obstacle avoidance disabled for real mode (f32::MAX thresholds = never triggered).
    // On real hardware MiDaS readings are too noisy for reactive stopping — the rolling min
    // consistently reads 0.1–0.5m on clear paths, causing constant false stops.
    // US (70 cm slow, 30 cm stop) is the sole reactive obstacle sensor in real mode.
    // MiDaS still feeds the occupancy map / frontier selector via PseudoLidar.
    // obstacle_slow_m=0.0: `nearest >= 0.0` is always true → base_scale=1.0 → MiDaS never stops.
    // US (70 cm slow, 30 cm stop) is the sole reactive obstacle sensor in real mode.
    let ctrl_handle = spawn_control_task(Arc::clone(&bus), control_path_rx, 0.0, 0.0, episode_dummy_rx);

    // ── Motor execution task ──────────────────────────────────────────────────
    // Drains both the safety motor_command channel and the controller cmdvel
    // channel.  Safety commands are prioritised (biased select).
    // CmdVel is dropped unless executive_state ∈ {Exploring, Recovering}.
    // The robot is Idle at startup; arm with SIGUSR1 to begin navigation.
    /// How long without a motor command before the watchdog zeroes the motors.
    const MOTOR_WATCHDOG_MS: u64 = 500;

    let mut motor = motor;
    let mut motor_cmd_rx  = motor_cmd_rx;
    let mut cmdvel_rx     = cmdvel_rx;
    let rx_exec_m         = bus.executive_state.subscribe();
    let rx_safety_m       = bus.safety_state.subscribe();
    let (motor_shutdown_tx, mut motor_shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let motor_handle = tokio::spawn(async move {
        info!("Motor execution task started (disarmed — send SIGUSR1 to arm)");
        let watchdog = tokio::time::Duration::from_millis(MOTOR_WATCHDOG_MS);
        let mut watchdog_logged = false; // suppress repeated watchdog spam when idle
        loop {
            tokio::select! {
                biased;
                // Priority 1: graceful shutdown — zero motors then exit.
                _ = &mut motor_shutdown_rx => {
                    info!("Motor task: shutdown signal — zeroing motors");
                    motor.emergency_stop().await.ok();
                    break;
                }
                // Priority 2: safety-triggered MotorCommand (emergency stop).
                // Executed unconditionally — bypasses armed/safety checks.
                // Do NOT reset watchdog_logged here: safety commands arrive
                // repeatedly during an estop latch and would re-enable the
                // CmdVel-suppressed WARN on every subsequent 10 Hz CmdVel.
                cmd = motor_cmd_rx.recv() => {
                    let Some(cmd) = cmd else { break };
                    if let Err(e) = motor.send_command(cmd).await {
                        error!("Motor error (safety): {e}");
                    }
                }
                // Priority 3: controller CmdVel — only while armed AND not in emergency stop.
                // Checks safety_state directly (not via executive) to avoid the race where
                // executive_state hasn't transitioned to SafetyStopped yet.
                // IMPORTANT: when suppressed we still zero the motors explicitly because
                // CmdVel arriving at 10 Hz preempts the 500 ms watchdog sleep, so the
                // watchdog never fires and motors keep spinning at their last commanded speed.
                Some(cmd_vel) = cmdvel_rx.recv() => {
                    let armed = matches!(
                        *rx_exec_m.borrow(),
                        ExecutiveState::Exploring | ExecutiveState::Recovering
                    );
                    let safe = !matches!(
                        *rx_safety_m.borrow(),
                        SafetyState::EmergencyStop { .. }
                    );
                    if armed && safe {
                        watchdog_logged = false;
                        let cmd = cmdvel_to_motor(cmd_vel, max_motor_duty);
                        if let Err(e) = motor.send_command(cmd).await {
                            error!("Motor error (ctrl): {e}");
                        }
                    } else {
                        // Not armed or safety stop active — zero motors and log once per stretch.
                        if !watchdog_logged {
                            warn!(armed, safe, "Motor task: CmdVel suppressed — zeroing motors");
                            watchdog_logged = true;
                        }
                        if let Err(e) = motor.send_command(MotorCommand::stop(0)).await {
                            error!("Motor suppressed-stop failed: {e}");
                        }
                    }
                }
                // Watchdog: no command for MOTOR_WATCHDOG_MS → zero all motors.
                // Uses send_command (not emergency_stop) — this is normal idle
                // keepalive behaviour, not a safety event.
                // Log only on first occurrence per idle stretch to avoid spam.
                () = tokio::time::sleep(watchdog) => {
                    if !watchdog_logged {
                        // During exploration the brief pause between goal-reached and the
                        // next path arriving is expected — log at debug to avoid noise.
                        let exploring = matches!(
                            *rx_exec_m.borrow(),
                            ExecutiveState::Exploring | ExecutiveState::Recovering
                        );
                        if exploring {
                            tracing::debug!("Motor watchdog: idle gap during exploration");
                        } else {
                            warn!("Motor watchdog: no command for {MOTOR_WATCHDOG_MS}ms — zeroing motors");
                        }
                        watchdog_logged = true;
                    }
                    if let Err(e) = motor.send_command(MotorCommand::stop(0)).await {
                        error!("Motor watchdog stop failed: {e}");
                    }
                }
            }
        }
    });

    // ── Gimbal task ───────────────────────────────────────────────────────────
    // Homes the gimbal at startup, then drives a sinusoidal sweep (±20°, 4 s)
    // with a reactive bias (up to ±10°) toward the more open depth half.
    // Step cap ±5°/frame, total clamped to ±30°.
    let mut gimbal = gimbal;
    let tilt_home = cfg.hal.gimbal.tilt_home_deg;
    let mut rx_depth_g = bus.vision_depth.subscribe();
    let bus_gimbal = Arc::clone(&bus);
    let gimbal_handle = tokio::spawn(async move {
        info!("Gimbal task started");
        if let Err(e) = gimbal.set_angles(0.0, tilt_home).await {
            warn!("Gimbal home failed: {e}");
        }
        let gimbal_t0 = std::time::Instant::now();
        loop {
            let depth = match rx_depth_g.recv().await {
                Ok(d)  => d,
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Closed)    => break,
            };
            // Compute open_bias from MiDaS pixels: higher pixel value = closer.
            // open_bias > 0 means right is more open.  See gimbal_pan_target docs.
            let w = depth.width as usize;
            let h = depth.mask_start_row.min(depth.height) as usize;
            let open_bias = if w >= 2 && h > 0 {
                let mid = w / 2;
                let (mut left_sum, mut right_sum) = (0.0f32, 0.0f32);
                for row in 0..h {
                    for col in 0..mid { left_sum  += depth.data[row * w + col]; }
                    for col in mid..w { right_sum += depth.data[row * w + col]; }
                }
                (left_sum - right_sum) / (h * mid) as f32  // positive = right more open
            } else { 0.0 };

            let t_s = gimbal_t0.elapsed().as_secs_f32();
            let (cur_pan, _) = gimbal.angles();
            let new_pan = gimbal_pan_target(open_bias, t_s, cur_pan);
            if let Err(e) = gimbal.set_pan(new_pan).await {
                warn!("Gimbal pan error: {e}");
            }
            let (cur_pan_out, cur_tilt_out) = gimbal.angles();
            let _ = bus_gimbal.gimbal_pan_deg.send(cur_pan_out);
            let _ = bus_gimbal.gimbal_tilt_deg.send(cur_tilt_out);
        }
    });

    // ── Telemetry task ────────────────────────────────────────────────────────
    let mut rx_pose_t  = bus.slam_pose2d.subscribe();
    let mut rx_lidar_t = bus.vision_pseudo_lidar.subscribe();
    let mut rx_imu_t   = bus.imu_raw.subscribe();
    let telem_handle = tokio::spawn(async move {
        info!("Telemetry task started");
        loop {
            tokio::select! {
                Ok(()) = rx_pose_t.changed() => {
                    let pose = *rx_pose_t.borrow_and_update();
                    telem.write("slam/pose2d", "micro_slam", &pose).await.ok();
                }
                Ok(scan) = rx_lidar_t.recv() => {
                    telem.write("vision/pseudo_lidar", "perception", &*scan).await.ok();
                }
                Ok(imu) = rx_imu_t.recv() => {
                    telem.write("imu/raw", "hal", &imu).await.ok();
                }
                else => break,
            }
        }
    });

    info!("Milestone 2 system running — press Ctrl+C or send SIGTERM to stop");
    // Wait for either SIGINT (Ctrl+C) or SIGTERM (pkill / systemd stop).
    // Both paths must zero the motors before exiting.
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut sig_term = signal(SignalKind::terminate())?;
        tokio::select! {
            _ = tokio::signal::ctrl_c() => { info!("SIGINT received"); }
            _ = sig_term.recv()         => { info!("SIGTERM received"); }
        }
    }
    info!("Shutdown signal received — stopping motors");

    drop(exec_handle);
    drop(cam_handle);
    drop(imu_handle);
    drop(us_handle);
    drop(slam_handle);
    drop(perc_handle);
    drop(map_handle);
    drop(safety_handle);
    drop(plan_handle);
    drop(ctrl_handle);
    drop(gimbal_handle);
    drop(telem_handle);

    // Signal motor task to zero motors and exit, then wait for it to finish.
    // This must happen before process::exit (which bypasses all destructors).
    let _ = motor_shutdown_tx.send(());
    let _ = tokio::time::timeout(
        tokio::time::Duration::from_millis(500),
        motor_handle,
    ).await;

    info!("Shutdown complete");
    // Force-exit to bypass ORT arena teardown which blocks for several seconds.
    std::process::exit(0);
}

// ── CmdVel → MotorCommand ─────────────────────────────────────────────────────

/// Convert a `CmdVel` from pure-pursuit to mecanum wheel duty cycles.
///
/// Normalises vx/vy by `MAX_VX` and omega by `MAX_OMEGA` (matching the
/// constants in `control/src/lib.rs`), then applies standard mecanum kinematics.
/// Preserves wheel ratios when the combined command saturates `max_duty`.
fn cmdvel_to_motor(cmd: CmdVel, max_duty: i8) -> MotorCommand {
    const MAX_VX:    f32 = 0.3;  // m/s — matches PurePursuitController::MAX_VX
    const MAX_OMEGA: f32 = 1.5;  // rad/s — matches PurePursuitController::MAX_OMEGA

    let vx_n    = (cmd.vx    / MAX_VX   ).clamp(-1.0, 1.0);
    let vy_n    = (cmd.vy    / MAX_VX   ).clamp(-1.0, 1.0);
    let omega_n = (cmd.omega / MAX_OMEGA).clamp(-1.0, 1.0);

    // Standard mecanum kinematics (see hal/src/motor.rs for sign convention).
    //   FL = vx + vy − ω    FR = vx − vy + ω
    //   RL = vx − vy − ω    RR = vx + vy + ω
    let fl = vx_n + vy_n - omega_n;
    let fr = vx_n - vy_n + omega_n;
    let rl = vx_n - vy_n - omega_n;
    let rr = vx_n + vy_n + omega_n;

    // Scale so the wheel with the largest magnitude uses the full duty range.
    let max_mag = [fl.abs(), fr.abs(), rl.abs(), rr.abs()]
        .into_iter()
        .fold(1.0_f32, f32::max);
    let scale = max_duty as f32 / max_mag;

    MotorCommand {
        t_ms: cmd.t_ms,
        fl: (fl * scale) as i8,
        fr: (fr * scale) as i8,
        rl: (rl * scale) as i8,
        rr: (rr * scale) as i8,
    }
}

// ── Shared task factories ─────────────────────────────────────────────────────

/// Spawn the occupancy-mapping task.
///
/// Subscribes to `vision_pseudo_lidar` and `slam_pose2d`, updates the shared
/// mapper on each new scan, and publishes grid deltas, frontier lists, and
/// exploration statistics.  Used by both real-robot and sim modes.
fn spawn_mapping_task(
    bus: Arc<bus::Bus>,
    mapper: Arc<RwLock<Mapper>>,
) -> tokio::task::JoinHandle<()> {
    let bus_map       = Arc::clone(&bus);
    let mapper_map    = Arc::clone(&mapper);
    let mut rx_lidar  = bus.vision_pseudo_lidar.subscribe();
    let mut rx_pose_m = bus.slam_pose2d.subscribe();
    tokio::spawn(async move {
        info!("Mapping task started");
        loop {
            match rx_lidar.recv().await {
                Ok(scan) => {
                    let pose = *rx_pose_m.borrow_and_update();
                    let (delta, frontiers, stats) = {
                        let mut m = mapper_map.write().await;
                        m.update(&scan, &pose)
                    };
                    let _ = bus_map.map_grid_delta.send(delta);
                    let _ = bus_map.map_frontiers.send(frontiers);
                    let _ = bus_map.map_explored_stats.send(stats);
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    warn!("Mapping task lagged {n} scans");
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    })
}

/// Spawn the pure-pursuit control task.
///
/// - `obstacle_slow_m`: depth at which forward speed begins to ramp down.
///   Real=1.20 m (large slow zone compensates for 4-5 frame camera lag);
///   Sim=0.70 m (tighter — sim depth is less noisy after floor-contamination fix).
/// - `obstacle_stop_m`: depth at which forward motion stops and the path is
///   abandoned.  Real=0.60 m (raised from 0.40 to account for camera lag and
///   MiDaS noise); Sim=0.05 m (floor noise reaches ~0.10 m so 0.05 avoids livelock).
/// - `episode_rx`: watch channel that fires on episode reset (sim only).
///   For real mode pass a watch receiver whose sender is immediately dropped —
///   `Ok(()) = changed()` in the select! arm never matches on a closed watch,
///   so the arm is effectively disabled.
///
/// Both real and sim use the same behaviour:
///   - 10 Hz pure-pursuit, 0.3 m lookahead.
///   - Angle-weighted speed scaling: side obstacles don't zero forward speed.
///   - Hard stop (vx=0) only inside `obstacle_stop_m`, regardless of angle.
///   - Hysteresis on clear: `obstacle_stopped` only clears after 5 consecutive
///     frames above `obstacle_stop_m * 2.0`, preventing false-clear from noisy
///     MiDaS depth readings causing immediate resume into the obstacle.
fn spawn_control_task(
    bus: Arc<bus::Bus>,
    ctrl_rx: tokio::sync::mpsc::Receiver<core_types::Path>,
    obstacle_slow_m: f32,
    obstacle_stop_m: f32,
    mut episode_rx: tokio::sync::watch::Receiver<u32>,
) -> tokio::task::JoinHandle<()> {
    const CONTROL_HZ: f64 = 10.0;
    const GOAL_TOLERANCE_M: f32 = 0.25;

    let bus_ctrl           = Arc::clone(&bus);
    let mut ctrl_rx        = ctrl_rx;
    let mut rx_pose_c      = bus.slam_pose2d.subscribe();
    let rx_nearest_c       = bus.nearest_obstacle_m.subscribe();
    let rx_nearest_angle_c = bus.nearest_obstacle_angle_rad.subscribe();
    let mut rx_safety_ctrl = bus.safety_state.subscribe();
    let mut rx_us_ctrl     = bus.ultrasonic.subscribe();
    tokio::spawn(async move {
        use std::time::Duration;
        use core_types::Path;
        // US-based speed scaling — independent of MiDaS, always forward-facing.
        // Slow zone: 70 cm → 30 cm linearly; below 30 cm: vx=0.
        // us_scale also applied to omega to prevent sweeping into obstacles while rotating.
        // The US emergency stop still fires at 15 cm (safety task), this is the soft pre-stop.
        const US_SLOW_M: f32 = 0.70;
        const US_STOP_M: f32 = 0.30;
        let mut latest_us_m: f32 = f32::MAX;
        info!("Control task started (pure-pursuit {CONTROL_HZ} Hz, lookahead 0.3 m)");
        let controller = PurePursuitController::with_lookahead(0.3);
        let mut current_path: Option<Path> = None;
        let mut tick = tokio::time::interval(Duration::from_secs_f64(1.0 / CONTROL_HZ));
        let mut obstacle_stopped = false;
        // Hysteresis: only clear obstacle_stopped after the nearest obstacle has
        // been above the clear threshold (2× stop distance) for a minimum wall-
        // clock duration.  Using time rather than tick count avoids counting the
        // same stale borrow() value multiple times when perception updates slower
        // than the control tick rate (~3 Hz MiDaS vs 10 Hz control).
        let clear_hysteresis_m = obstacle_stop_m * 2.0;
        const CLEAR_HOLD_S: f32 = 0.8; // must stay clear for 0.8 s before resuming (~2-3 MiDaS frames)
        let mut clear_since: Option<std::time::Instant> = None;
        loop {
            tokio::select! {
                biased;
                // Episode reset (sim) — clear path and latch flag.
                // For real mode the sender is dropped so changed() always returns
                // Err; the Ok(()) pattern never matches and the arm is skipped.
                Ok(()) = episode_rx.changed() => {
                    episode_rx.borrow_and_update();
                    current_path = None;
                    obstacle_stopped = false;
                    clear_since = None;
                }
                // US range update — track latest for speed scaling in tick arm.
                Ok(reading) = rx_us_ctrl.recv() => {
                    latest_us_m = reading.range_cm / 100.0;
                }
                // Emergency stop from US safety task — clear path on rising edge.
                Ok(()) = rx_safety_ctrl.changed() => {
                    let is_estop = matches!(
                        *rx_safety_ctrl.borrow_and_update(),
                        SafetyState::EmergencyStop { .. }
                    );
                    if is_estop && !obstacle_stopped {
                        warn!("Control: EmergencyStop — clearing path for replan after latch");
                        current_path = None;
                        obstacle_stopped = true;
                        clear_since = None;
                    } else if !is_estop {
                        // EmergStop released — clear immediately so the planner
                        // can assign a new path.  Fix C (toward-angle braking)
                        // prevents re-approach; keeping obstacle_stopped=true
                        // here caused the 5-frame rolling-min to latch and block
                        // the clear timer for > 0.8 s, giving goals = 0.
                        obstacle_stopped = false;
                        clear_since = None;
                    }
                }
                // New path from planner — replace immediately (unless in emergency stop).
                result = ctrl_rx.recv() => {
                    match result {
                        Some(p) => {
                            if obstacle_stopped {
                                debug!(waypoints = p.waypoints.len(), "Control: dropping path — EmergencyStop active");
                            } else {
                                info!(waypoints = p.waypoints.len(), "Control: new path, tracking");
                                current_path = Some(p);
                            }
                        }
                        None => break,
                    }
                }
                // Control tick — re-compute CmdVel from current pose.
                _ = tick.tick() => {
                    // Auto-clear depth stop with time hysteresis — nearest must stay above
                    // clear_hysteresis_m (2× stop distance) continuously for CLEAR_HOLD_S
                    // before resuming.  Any dip below the threshold resets the timer, preventing
                    // noisy MiDaS readings from causing a false clear.  Using wall-clock time
                    // avoids counting the same stale borrow() value across multiple ticks.
                    let nearest = *rx_nearest_c.borrow();
                    if obstacle_stopped {
                        // If US is still in the stop zone, actively reverse to create distance.
                        // Without this the robot parks at 17-29 cm (above 15 cm safety threshold
                        // so escape never fires) and gets stuck indefinitely.
                        if latest_us_m < US_STOP_M {
                            let _ = bus_ctrl.controller_cmd_vel.try_send(CmdVel { t_ms: 0, vx: -0.15, vy: 0.0, omega: 0.0 });
                            clear_since = None; // don't start clear timer while still in stop zone
                            continue;
                        }
                        // MiDaS still sees obstacle and no path to follow — rotate
                        // away from it so the forward view clears and the rolling
                        // minimum can decay to a sane reading before we try to move.
                        if nearest < clear_hysteresis_m && current_path.is_none() {
                            let obs_angle = *rx_nearest_angle_c.borrow();
                            // Turn away: obstacle on right (angle<0) → turn left (+ω)
                            let escape_omega = if obs_angle <= 0.0 { 1.0_f32 } else { -1.0_f32 };
                            let _ = bus_ctrl.controller_cmd_vel.try_send(CmdVel { t_ms: 0, vx: 0.0, vy: 0.0, omega: escape_omega });
                            clear_since = None;
                            continue;
                        }
                        if nearest > clear_hysteresis_m {
                            let since = clear_since.get_or_insert_with(std::time::Instant::now);
                            if since.elapsed().as_secs_f32() >= CLEAR_HOLD_S {
                                info!(clear_hysteresis_m, CLEAR_HOLD_S, "Control: obstacle cleared (hysteresis satisfied)");
                                obstacle_stopped = false;
                                clear_since = None;
                            }
                        } else {
                            clear_since = None; // reset timer on any dip below threshold
                        }
                    }

                    let Some(ref path) = current_path else { continue };
                    let pose = *rx_pose_c.borrow_and_update();

                    // Goal check: within tolerance of the final waypoint → done.
                    if let Some(&[gx, gy]) = path.waypoints.last() {
                        let dx = gx - pose.x_m;
                        let dy = gy - pose.y_m;
                        if dx * dx + dy * dy < GOAL_TOLERANCE_M * GOAL_TOLERANCE_M {
                            info!("Control: goal reached");
                            current_path = None;
                            obstacle_stopped = false;
                            continue;
                        }
                    }

                    // ── Reactive obstacle avoidance ──────────────────────────
                    // Scale vx [0, 1] based on nearest obstacle in camera FOV.
                    // omega is unaffected — the robot can still rotate away.
                    //
                    // Compute controller output first so we can detect "turning
                    // toward obstacle" on the raw (unscaled) omega before we
                    // decide the speed_scale.
                    let mut cmd_vel = controller.compute(&pose, path);
                    let nearest_angle = *rx_nearest_angle_c.borrow();
                    // Is navigation intending to rotate into the detected obstacle?
                    // Use unscaled omega (navigation intent) so US-induced omega
                    // reduction does not mask the dangerous approach condition.
                    let toward = (cmd_vel.omega > 0.05 && nearest_angle > 0.05)
                              || (cmd_vel.omega < -0.05 && nearest_angle < -0.05);
                    let base_scale = if nearest >= obstacle_slow_m {
                        1.0_f32
                    } else if nearest <= obstacle_stop_m {
                        0.0_f32
                    } else {
                        (nearest - obstacle_stop_m) / (obstacle_slow_m - obstacle_stop_m)
                    };
                    let angle_factor = nearest_angle.cos().max(0.0_f32);
                    // When turning toward a detected obstacle treat it as dead-ahead
                    // (effective_factor=1.0) so the robot brakes at full weight.
                    // Without this a 45° side obstacle only applies cos(45°)≈0.7 of
                    // the deceleration — enough to still be moving when the turn
                    // brings the wall into the forward arc and a collision occurs.
                    let effective_factor = if toward { 1.0_f32 } else { angle_factor };
                    // Hard stop in the stop zone; angle weighting only in slowdown zone.
                    let speed_scale = if base_scale <= 0.0 {
                        0.0
                    } else {
                        1.0 - (1.0 - base_scale) * effective_factor
                    };

                    // US-based speed scaling (forward sensor, independent of gimbal).
                    // Takes minimum with MiDaS scale — whichever is more restrictive wins.
                    let us_scale = if latest_us_m >= US_SLOW_M {
                        1.0_f32
                    } else if latest_us_m <= US_STOP_M {
                        0.0_f32
                    } else {
                        (latest_us_m - US_STOP_M) / (US_SLOW_M - US_STOP_M)
                    };
                    let combined_scale = speed_scale.min(us_scale);
                    if us_scale < 1.0 {
                        info!(us_m = latest_us_m, us_scale, combined_scale, "Control: US slowing");
                    }

                    cmd_vel.vx *= combined_scale;
                    // Also scale omega by us_scale — prevents sweeping the US sensor into an
                    // obstacle while rotating (e.g. omega=0.4 sweeps forward-facing US across
                    // a box, causing sudden 49→15 cm reading with no time to brake).
                    cmd_vel.omega *= us_scale;

                    if combined_scale < 1.0 {
                        let obs_deg     = nearest_angle.to_degrees();
                        let heading_deg = pose.theta_rad.to_degrees();
                        let turn_dir    = if cmd_vel.omega.abs() < 0.05 { "straight" }
                                          else if cmd_vel.omega > 0.0   { "turning-left" }
                                          else                           { "turning-right" };
                        let obs_side = if obs_deg > 5.0 { "left" } else if obs_deg < -5.0 { "right" } else { "center" };
                        // `toward` is already computed above from unscaled omega.
                        if combined_scale <= 0.0 {
                            // Hard stop: US at ≤ emergency_stop_cm, OR MiDaS obstacle in forward
                            // arc (≤ 30°), OR navigation is turning toward the detected obstacle.
                            // has_forward / midas_stopped (any-angle stop) are intentionally
                            // excluded: floor-noise MiDaS readings at ~0.10 m (just above the
                            // 0.05 m stop_m) reduce combined_scale to near-zero already, so the
                            // robot crawls, rotation swings the obstacle clear, and no latch fires.
                            let us_stopped     = us_scale <= 0.0;
                            let in_forward_arc = nearest_angle.abs() <= 30f32.to_radians();
                            if us_stopped || in_forward_arc || toward {
                                if !obstacle_stopped {
                                    warn!(
                                        nearest_m   = nearest,
                                        us_m        = latest_us_m,
                                        us_stopped,
                                        obs_deg,
                                        obs_side,
                                        heading_deg,
                                        omega       = cmd_vel.omega,
                                        turn_dir,
                                        turning_toward_obstacle = toward,
                                        "Control: obstacle STOP — replan"
                                    );
                                    obstacle_stopped = true;
                                    clear_since = None;
                                    current_path = None;
                                    // Immediately zero velocity — without this the motor task
                                    // keeps executing the last CmdVel and the robot coasts into
                                    // the obstacle while we wait for the next tick.
                                    let _ = bus_ctrl.controller_cmd_vel.try_send(CmdVel { t_ms: 0, vx: 0.0, vy: 0.0, omega: 0.0 });
                                }
                                continue;
                            }
                            // Side obstacle during rotation, US clear — send omega only.
                        }
                        info!(
                            nearest_m   = nearest,
                            speed_scale,
                            us_scale,
                            combined_scale,
                            obs_deg,
                            obs_side,
                            heading_deg,
                            vx          = cmd_vel.vx,
                            omega       = cmd_vel.omega,
                            turn_dir,
                            turning_toward_obstacle = toward,
                            "Control: slowing"
                        );
                    }

                    tracing::debug!(nearest_m = nearest, vx = cmd_vel.vx, omega = cmd_vel.omega, "Control: cmd_vel");
                    let _ = bus_ctrl.controller_cmd_vel.try_send(cmd_vel);
                }
            }
        }
    })
}

// ── Frontier selection ────────────────────────────────────────────────────────

/// Pick a world-frame goal from the mapper's cached frontier list.
///
/// Frontiers within MIN_GOAL_DIST of the robot (e.g. the robot's own cell
/// whose backward neighbors are unobserved) are excluded so the robot always
/// drives toward a genuinely unexplored area.
fn select_frontier_goal(
    mapper: &Mapper,
    choice: &FrontierChoice,
    pose: &core_types::Pose2D,
    blacklist: &[[f32; 2]],
) -> Option<[f32; 2]> {
    const MIN_GOAL_DIST: f32 = 0.75;    // metres — push robot toward meaningfully distant frontiers
    const BLACKLIST_RADIUS: f32 = 0.60; // metres — skip frontiers near recently-tried goals

    let frontiers = &mapper.last_frontiers;
    if frontiers.is_empty() {
        return None;
    }

    let dist = |f: &core_types::Frontier| -> f32 {
        let dx = f.centroid_x_m - pose.x_m;
        let dy = f.centroid_y_m - pose.y_m;
        (dx * dx + dy * dy).sqrt()
    };

    // Heading-weighted score for Nearest selection.
    // A frontier in the robot's current heading direction gets up to 50% discount,
    // so the robot keeps moving forward rather than turning to a slightly-closer
    // frontier off to the side.
    let heading_score = |f: &core_types::Frontier| -> f32 {
        const HEADING_WEIGHT: f32 = 0.5;
        let dx = f.centroid_x_m - pose.x_m;
        let dy = f.centroid_y_m - pose.y_m;
        let d = (dx * dx + dy * dy).sqrt().max(0.001);
        let angle_to = dy.atan2(dx);
        let alignment = (angle_to - pose.theta_rad).cos().max(0.0);
        d * (1.0 - HEADING_WEIGHT * alignment)
    };

    let blacklisted = |f: &core_types::Frontier| -> bool {
        blacklist.iter().any(|b| {
            let dx = f.centroid_x_m - b[0];
            let dy = f.centroid_y_m - b[1];
            (dx * dx + dy * dy).sqrt() < BLACKLIST_RADIUS
        })
    };

    // Primary: far enough away and not blacklisted.
    // Fallback 1: far enough away (ignoring blacklist).
    // Fallback 2: any frontier (blacklist and distance both waived — stuck situation).
    let candidates: Vec<_> = frontiers.iter()
        .filter(|f| dist(f) >= MIN_GOAL_DIST && !blacklisted(f))
        .collect();
    let fallback1: Vec<_> = frontiers.iter()
        .filter(|f| dist(f) >= MIN_GOAL_DIST)
        .collect();

    let pool: &[&core_types::Frontier] = if !candidates.is_empty() {
        &candidates
    } else if !fallback1.is_empty() {
        &fallback1
    } else {
        // completely stuck — use everything
        return frontiers.iter()
            .min_by(|a, b| dist(a).partial_cmp(&dist(b)).unwrap())
            .map(|f| [f.centroid_x_m, f.centroid_y_m]);
    };

    let chosen: Option<&core_types::Frontier> = match choice {
        FrontierChoice::Nearest | FrontierChoice::RandomValid => {
            pool.iter().copied().min_by(|a, b| heading_score(a).partial_cmp(&heading_score(b)).unwrap())
        }
        FrontierChoice::Largest => {
            pool.iter().copied().max_by_key(|f| f.size_cells)
        }
        FrontierChoice::Leftmost => {
            pool.iter().copied().min_by(|a, b| a.centroid_x_m.partial_cmp(&b.centroid_x_m).unwrap())
        }
        FrontierChoice::Rightmost => {
            pool.iter().copied().max_by(|a, b| a.centroid_x_m.partial_cmp(&b.centroid_x_m).unwrap())
        }
    };
    let goal = chosen.map(|f| [f.centroid_x_m, f.centroid_y_m]);
    if let Some(g) = goal {
        info!(x = g[0], y = g[1], frontiers = frontiers.len(), blacklisted = frontiers.len() - pool.len(), "Planning: frontier goal selected");
    }
    goal
}

// ── Helpers ───────────────────────────────────────────────────────────────────

// ── Safety task (shared between robot-run and sim) ────────────────────────────
//
// Reads from `bus.ultrasonic`, evaluates each reading against `emergency_stop_cm`,
// latches EmergencyStop for 5 s on a trip, and spawns an escape reverse maneuver.
// Used in both `run_robot_mode` (via real US readings) and `run_sim_mode` (via
// synthetic US readings published by `sim_publish`).

fn spawn_safety_task(bus: Arc<bus::Bus>, emergency_stop_cm: f32, enable_watchdog: bool) -> tokio::task::JoinHandle<()> {
    let bus_safety = Arc::clone(&bus);
    let mut rx_us  = bus.ultrasonic.subscribe();
    let escape_reverse_spd: i8 = 35; // duty cycle for escape reverse (matches reverse_speed in config)
    tokio::spawn(async move {
        let mut monitor  = SafetyMonitor::new();
        monitor.stop_threshold_cm = emergency_stop_cm;
        info!(
            threshold_cm = monitor.stop_threshold_cm,
            watchdog_ms  = monitor.watchdog_timeout_ms,
            "Safety task started"
        );
        let watchdog_dur = tokio::time::Duration::from_millis(monitor.watchdog_timeout_ms);
        // Latch: once EmergencyStop fires, hold it for at least this duration even if
        // the US returns a good reading (e.g. blind spot <3 cm → driver returns 400 cm).
        const EMSTOP_LATCH: tokio::time::Duration = tokio::time::Duration::from_secs(5);
        // Escape: after EmergencyStop the robot reverses briefly to clear the obstacle.
        // The escape window suppresses re-stop commands so the reverse isn't interrupted.
        const ESCAPE_DELAY:    tokio::time::Duration = tokio::time::Duration::from_millis(200);
        const ESCAPE_DURATION: tokio::time::Duration = tokio::time::Duration::from_millis(400);  // was 1500 — too far on real HW
        const ESCAPE_ROTATION: tokio::time::Duration = tokio::time::Duration::from_millis(600);  // was 1200
        let mut emstop_until:  Option<tokio::time::Instant> = None;
        let mut escape_until:  Option<tokio::time::Instant> = None;
        loop {
            tokio::select! {
                result = rx_us.recv() => {
                    match result {
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                            warn!("Safety task lagged {n} readings");
                        }
                        Ok(reading) => {
                            // Check whether a previous EmergencyStop latch is still active.
                            let now = tokio::time::Instant::now();
                            if emstop_until.map_or(false, |u| now < u) {
                                // Still latched: keep motors stopped but do NOT re-publish
                                // EmergencyStop on the state bus.  The watch channel already
                                // holds EmergencyStop from the initial send; re-sending on every
                                // US poll tick causes changed() to fire on all receivers every
                                // ~150ms, creating a spam loop in the control task that repeatedly
                                // resets obstacle_stopped.  The motor task uses borrow() so it
                                // still sees EmergencyStop without a repeated notification.
                                if escape_until.map_or(true, |u| now >= u) {
                                    let _ = bus_safety.motor_command.try_send(MotorCommand::stop(reading.t_ms));
                                }
                            } else {
                                let (state, maybe_stop) = monitor.evaluate(&reading);
                                let _ = bus_safety.safety_state.send(state);
                                if let Some(cmd) = maybe_stop {
                                    warn!(range_cm = reading.range_cm, "Safety: EMERGENCY STOP — latching for {}s", EMSTOP_LATCH.as_secs());
                                    emstop_until = Some(now + EMSTOP_LATCH);
                                    escape_until = Some(now + ESCAPE_DELAY + ESCAPE_DURATION + ESCAPE_ROTATION);
                                    // Count each fresh EmergencyStop as a near-miss.
                                    let n = *bus_safety.estop_count.borrow() + 1;
                                    let _ = bus_safety.estop_count.send(n);
                                    if bus_safety.motor_command.try_send(cmd).is_err() {
                                        error!("Safety: motor_command channel full — stop dropped!");
                                    }
                                    // Spawn escape maneuver: wait for stop to take effect,
                                    // reverse to clear the obstacle, then stop again.
                                    let bus_esc = Arc::clone(&bus_safety);
                                    let spd = escape_reverse_spd;
                                    tokio::spawn(async move {
                                        tokio::time::sleep(ESCAPE_DELAY).await;
                                        // Reverse: send repeatedly so sim sustains the command each tick.
                                        info!("Safety: escape — reversing for {}ms", ESCAPE_DURATION.as_millis());
                                        let reverse = MotorCommand { t_ms: 0, fl: -spd, fr: -spd, rl: -spd, rr: -spd };
                                        let deadline = tokio::time::Instant::now() + ESCAPE_DURATION;
                                        while tokio::time::Instant::now() < deadline {
                                            let _ = bus_esc.motor_command.try_send(reverse);
                                            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                                        }
                                        // Rotate to break heading deadlock: turn away from the
                                        // nearest-obstacle angle. Positive angle = obstacle to
                                        // left → rotate right (CW); negative → rotate left (CCW).
                                        let obs_angle = *bus_esc.nearest_obstacle_angle_rad.borrow();
                                        let (fl, fr, rl, rr): (i8, i8, i8, i8) = if obs_angle >= 0.0 {
                                            (spd, -spd, spd, -spd)   // CW: obstacle left → turn right
                                        } else {
                                            (-spd, spd, -spd, spd)   // CCW: obstacle right → turn left
                                        };
                                        info!(obs_angle_rad = obs_angle, cw = (obs_angle >= 0.0),
                                              "Safety: escape — rotating for {}ms", ESCAPE_ROTATION.as_millis());
                                        let rotate = MotorCommand { t_ms: 0, fl, fr, rl, rr };
                                        let deadline = tokio::time::Instant::now() + ESCAPE_ROTATION;
                                        while tokio::time::Instant::now() < deadline {
                                            let _ = bus_esc.motor_command.try_send(rotate);
                                            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                                        }
                                        let _ = bus_esc.motor_command.try_send(MotorCommand::stop(0));
                                        info!("Safety: escape complete");
                                    });
                                }
                            }
                        }
                    }
                }
                _ = tokio::time::sleep(watchdog_dur), if enable_watchdog => {
                    warn!("Safety: watchdog timeout — no US reading");
                    let now = tokio::time::Instant::now();
                    // Only arm a fresh latch; don't extend an existing one so the
                    // latch can expire naturally (e.g. during episode reset).
                    if emstop_until.map_or(true, |u| now >= u) {
                        emstop_until = Some(now + EMSTOP_LATCH);
                    }
                    let (state, cmd) = monitor.evaluate_timeout(0);
                    let _ = bus_safety.safety_state.send(state);
                    let _ = bus_safety.motor_command.try_send(cmd);
                }
            }
        }
    })
}

// ── Sim mode ──────────────────────────────────────────────────────────────────
//
// Runs the full autonomy stack (mapping, planning, control, frontier selection,
// executive) against a `sim_fast::FastSim` instead of real hardware.
//
// The sim driver task replaces all HAL tasks: it reads CmdVel from the same
// bus channel the motor task would use, steps the sim, and publishes synthetic
// sensor data (PseudoLidarScan, Pose2D, UltrasonicReading, nearest_obstacle_m)
// back onto the bus.  Every software task runs unmodified — including the
// reactive obstacle avoidance in the control task.
//
// Usage: `robot sim [seed]`   (seed defaults to 42)
//
// SIGUSR1 arms the robot, same as robot-run mode.
// After a collision the episode auto-resets; the robot stays SafetyStopped
// until SIGUSR1 re-arms it (mirrors real-robot behaviour so the same care
// must be taken in sim as on the real floor).

async fn run_sim_mode(
    bus:              Arc<bus::Bus>,
    cfg:              config::RobotConfig,
    plan_frontier_rx: tokio::sync::mpsc::Receiver<core_types::FrontierChoice>,
    control_path_rx:  tokio::sync::mpsc::Receiver<core_types::Path>,
    motor_cmd_rx:     tokio::sync::mpsc::Receiver<core_types::MotorCommand>,
    cmdvel_rx:        tokio::sync::mpsc::Receiver<core_types::CmdVel>,
) -> anyhow::Result<()> {
    use sim_fast::RoomKind;

    // Usage: `robot sim [explore] [seed]`
    // `explore` keeps the map and maze across collisions (single infinite episode).
    // Without `explore`, each collision/timeout resets the maze and map (training mode).
    let mut args_iter = std::env::args().skip(2);
    let arg2 = args_iter.next();
    let explore_mode = arg2.as_deref() == Some("explore");
    let seed: u64 = if explore_mode {
        args_iter.next().and_then(|s| s.parse().ok()).unwrap_or(cfg.sim.seed)
    } else {
        arg2.and_then(|s| s.parse().ok()).unwrap_or(cfg.sim.seed)
    };
    info!(seed, explore = explore_mode, "Sim mode starting");

    // ── Signal handler — registered FIRST before any spawning ─────────────────
    use tokio::signal::unix::{signal, SignalKind};
    let mut sigusr1_stream = signal(SignalKind::user_defined1())
        .map_err(|e| anyhow::anyhow!("SIGUSR1 handler: {e}"))?;

    // ── Sim HAL objects ───────────────────────────────────────────────────────
    // Each HAL trait impl is backed by SimState: a shared FastSim + watch channels
    // so the sim tick task drives physics while all HAL readers wait on step_rx.
    let room_kind = if cfg.sim.obstacles == 0 {
        RoomKind::Empty
    } else {
        RoomKind::Random(cfg.sim.obstacles)
    };
    let sim_state = SimState::new(seed, room_kind, &cfg.sim);

    let camera: Box<dyn Camera>              = Box::new(SimCameraHal::new(&sim_state, &cfg.hal.camera));
    let imu: Box<dyn Imu>                    = Box::new(sim_state.imu(&cfg.sim));
    let ultrasonic: Box<dyn Ultrasonic>      = Box::new(sim_state.ultrasonic(&cfg.hal.ultrasonic));
    let mut motor: Box<dyn MotorController>  = Box::new(sim_state.motor_controller());
    let mut gimbal: Box<dyn Gimbal>          = Box::new(sim_state.gimbal(&cfg.hal.gimbal));
    let max_motor_duty = cfg.hal.motor.max_speed as i8;
    info!(seed, "Sim HAL initialised");

    // ── Perception pipeline ───────────────────────────────────────────────────
    // Runs real MiDaS on Wolfenstein-rendered frames from SimCamera so the same
    // depth inference code executes in sim and on the real robot.
    let mut depth_infer = DepthInference::new(
        &cfg.perception.midas_model_path,
        cfg.perception.depth_out_width,
        cfg.perception.depth_out_height,
        cfg.perception.depth_mask_rows,
        cfg.perception.num_threads,
    );
    let mut event_gate = EventGate::with_default_threshold();
    let lidar_extractor = PseudoLidarExtractor::new(48, 3.0);
    info!("Sim perception pipeline initialised");

    // ── Mapper ────────────────────────────────────────────────────────────────
    let mapper = Arc::new(tokio::sync::RwLock::new(Mapper::new()));
    info!("Occupancy grid mapper initialised");

    // ── Episode reset notification ────────────────────────────────────────────
    let (episode_reset_tx, episode_reset_rx) =
        tokio::sync::watch::channel(0u32);
    let (force_respawn_tx, force_respawn_rx) = tokio::sync::mpsc::channel::<()>(1);

    // ── Executive task ────────────────────────────────────────────────────────
    let (arm_tx, mut arm_rx)         = tokio::sync::mpsc::channel::<()>(4);
    let (timeout_tx, mut timeout_rx) = tokio::sync::mpsc::channel::<()>(4);
    let mut rx_safety_exec           = bus.safety_state.subscribe();
    let mut rx_bridge_exec           = bus.bridge_cmd.subscribe();
    let bus_exec                     = Arc::clone(&bus);
    let arm_tx_exec                  = arm_tx.clone();
    tokio::spawn(async move {
        let mut exec = executive::Executive::new(Arc::clone(&bus_exec));
        info!("Sim executive task started (Idle) — send SIGUSR1 to arm");
        loop {
            tokio::select! {
                biased;
                _ = arm_rx.recv() => {
                    if !matches!(exec.state(), ExecutiveState::Exploring) {
                        match exec.arm() {
                            Ok(())  => info!("Sim executive: armed → Exploring"),
                            Err(e)  => warn!("Sim executive arm rejected: {e}"),
                        }
                    }
                }
                _ = timeout_rx.recv() => {
                    if matches!(exec.state(), ExecutiveState::Exploring | ExecutiveState::Recovering) {
                        let _ = exec.transition(ExecutiveState::SafetyStopped);
                    }
                }
                Ok(()) = rx_safety_exec.changed() => {
                    let safety = rx_safety_exec.borrow().clone();
                    if matches!(safety, SafetyState::EmergencyStop { .. }) {
                        match exec.state() {
                            ExecutiveState::Exploring | ExecutiveState::Recovering => {
                                let _ = exec.transition(ExecutiveState::SafetyStopped);
                            }
                            _ => {}
                        }
                    } else {
                        if matches!(exec.state(), ExecutiveState::SafetyStopped) {
                            info!("Sim executive: safety cleared — auto-recovering to Exploring");
                            let _ = exec.arm();
                        }
                    }
                }
                Ok(()) = rx_bridge_exec.changed() => {
                    let cmd = rx_bridge_exec.borrow_and_update().clone();
                    match cmd {
                        BridgeCommand::Arm  => { let _ = arm_tx_exec.send(()).await; }
                        BridgeCommand::Stop => { let _ = exec.transition(ExecutiveState::Idle); }
                        BridgeCommand::None => {}
                    }
                }
            }
        }
    });

    // SIGUSR1 → arm (pre-registered stream passed into spawned task)
    let arm_tx_sig = arm_tx.clone();
    tokio::spawn(async move {
        loop {
            sigusr1_stream.recv().await;
            info!("SIGUSR1 received — arming sim robot");
            let _ = arm_tx_sig.send(()).await;
        }
    });

    // ── Frontier selector task ────────────────────────────────────────────────
    exploration_rl::spawn_selector_task(Arc::clone(&bus)).await;

    // ── Safety task (shared with robot-run) ───────────────────────────────────
    spawn_safety_task(Arc::clone(&bus), cfg.agent.safety.emergency_stop_cm, false);

    // ── Mapping task ──────────────────────────────────────────────────────────
    spawn_mapping_task(Arc::clone(&bus), Arc::clone(&mapper));

    // ── Camera task ───────────────────────────────────────────────────────────
    let bus_cam  = Arc::clone(&bus);
    let mut camera = camera;
    tokio::spawn(async move {
        info!("Sim camera task started");
        loop {
            match camera.read_frame().await {
                Ok(frame) => {
                    let arc = Arc::new(frame);
                    let _ = bus_cam.camera_frame_raw.send(Arc::clone(&arc));
                    let gray = rgb_to_gray(&arc);
                    let _ = bus_cam.camera_frame_gray.send(Arc::new(gray));
                }
                Err(e) => { info!("Sim camera task exiting: {e}"); break; }
            }
        }
    });

    // ── IMU task ──────────────────────────────────────────────────────────────
    let bus_imu = Arc::clone(&bus);
    let mut imu = imu;
    tokio::spawn(async move {
        info!("Sim IMU task started");
        loop {
            match imu.read_sample().await {
                Ok(s) => { let _ = bus_imu.imu_raw.send(s); }
                Err(e) => { info!("Sim IMU task exiting: {e}"); break; }
            }
        }
    });

    // ── Crash detection task (IMU-based) ─────────────────────────────────────
    // SimImu injects a ±20 m/s² spike on step.collision — above the 15 m/s²
    // threshold — so this task fires on simulated wall impacts exactly as it
    // would on the real robot.
    {
        const CRASH_ACCEL_THRESHOLD: f32 = 15.0;
        const CRASH_DEBOUNCE_MS: u64 = 2_000;
        let bus_crash  = Arc::clone(&bus);
        let mut rx_imu_c  = bus.imu_raw.subscribe();
        let rx_exec_crash = bus.executive_state.subscribe();
        tokio::spawn(async move {
            let mut last_crash_ms: u64 = 0;
            loop {
                match rx_imu_c.recv().await {
                    Ok(s) => {
                        let armed = matches!(
                            *rx_exec_crash.borrow(),
                            ExecutiveState::Exploring | ExecutiveState::Recovering
                        );
                        if !armed { continue; }
                        let horiz = (s.accel_x * s.accel_x + s.accel_y * s.accel_y).sqrt();
                        if horiz > CRASH_ACCEL_THRESHOLD && s.t_ms > last_crash_ms + CRASH_DEBOUNCE_MS {
                            last_crash_ms = s.t_ms;
                            let n = *bus_crash.collision_count.borrow() + 1;
                            let _ = bus_crash.collision_count.send(n);
                            warn!(horiz_accel_m_s2 = horiz, collision_n = n, "Sim crash detected — IMU impact spike");
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {}
                    Err(tokio::sync::broadcast::error::RecvError::Closed)    => break,
                }
            }
        });
    }

    // ── Ultrasonic task ───────────────────────────────────────────────────────
    let bus_us = Arc::clone(&bus);
    let mut ultrasonic = ultrasonic;
    tokio::spawn(async move {
        info!("Sim ultrasonic task started");
        loop {
            match ultrasonic.read_distance().await {
                Ok(r) => { let _ = bus_us.ultrasonic.send(r); }
                Err(e) => { info!("Sim ultrasonic task exiting: {e}"); break; }
            }
        }
    });

    // ── Perception task (SimCamera frame → MiDaS depth → pseudo-lidar) ───────
    // Identical to real robot path: SimCamera renders Wolfenstein frames at 10 Hz
    // so MiDaS runs on synthetic but geometrically correct first-person views.
    let bus_perc   = Arc::clone(&bus);
    let mut rx_gray = bus.camera_frame_gray.subscribe();
    tokio::spawn(async move {
        info!("Sim perception task started");
        let mut last_infer = std::time::Instant::now();
        const MAX_INFER_INTERVAL: std::time::Duration = std::time::Duration::from_secs(2);
        const NEAR_HIST_LEN: usize = 5;
        let mut near_hist: std::collections::VecDeque<f32> =
            std::collections::VecDeque::with_capacity(NEAR_HIST_LEN);
        // MiDaS depth noise: simulate real-camera sensor noise and MiDaS jitter.
        // σ=0.03 on normalised [0,1] depth is consistent with empirical MiDaS variance
        // on the 1MP USB camera.  Applied per-pixel after inference so the same contrast-
        // based obstacle detection sees realistic noise rather than clean synthetic output.
        const DEPTH_NOISE_SIGMA: f32 = 0.03;
        let mut depth_rng: u64 = seed.wrapping_add(0x1234_5678_9ABC_DEF0);
        loop {
            match rx_gray.recv().await {
                Ok(gray) => {
                    let force_timer = last_infer.elapsed() >= MAX_INFER_INTERVAL;
                    if force_timer { event_gate.force(); }
                    if event_gate.should_infer(&gray) {
                        last_infer = std::time::Instant::now();
                        let rgb_stub = core_types::CameraFrame {
                            t_ms: gray.t_ms, width: gray.width, height: gray.height,
                            data: gray.data.iter().flat_map(|&v| [v, v, v]).collect(),
                        };
                        match tokio::task::block_in_place(|| depth_infer.infer(&rgb_stub)) {
                            Ok(mut depth) => {
                                // Add per-pixel Gaussian noise to simulate real camera+MiDaS jitter.
                                for v in &mut depth.data {
                                    *v = (*v + sim_fast::hal_impls::gaussian(&mut depth_rng, DEPTH_NOISE_SIGMA)).clamp(0.0, 1.0);
                                }
                                let mut scan = lidar_extractor.extract(&depth);
                                let (near_m, near_cam_rad) = nearest_in_fov(&scan);
                                let pan_deg = *bus_perc.gimbal_pan_deg.borrow();
                                let pan_rad = -pan_deg * std::f32::consts::PI / 180.0;
                                for ray in &mut scan.rays { ray.angle_rad += pan_rad; }
                                let near_angle_rad = near_cam_rad + pan_rad;
                                near_hist.push_back(near_m);
                                if near_hist.len() > NEAR_HIST_LEN { near_hist.pop_front(); }
                                let near_m_min = near_hist.iter().cloned().fold(f32::MAX, f32::min);
                                if near_m < 1.5 || near_m_min < 1.5 {
                                    let near_deg = near_angle_rad.to_degrees();
                                    let side = if near_deg > 5.0 { "left" } else if near_deg < -5.0 { "right" } else { "center" };
                                    info!(nearest_m = near_m_min, nearest_raw_m = near_m, angle_deg = near_deg, side, "Sim perception: nearest obstacle");
                                }
                                let _ = bus_perc.vision_depth.send(Arc::new(depth));
                                let _ = bus_perc.vision_pseudo_lidar.send(Arc::new(scan));
                                let _ = bus_perc.nearest_obstacle_m.send(near_m_min);
                                let _ = bus_perc.nearest_obstacle_angle_rad.send(near_angle_rad);
                            }
                            Err(e) => error!("Sim depth inference error: {e}"),
                        }
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    warn!("Sim perception: camera lagged {n} frames");
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    });

    // ── Gimbal task ───────────────────────────────────────────────────────────
    // SimGimbal stores pan in `latest_pan` Arc<Mutex<f32>>.  The sim tick task
    // reads `latest_pan` when calling step_with_pan so camera and physics share
    // the same pan angle without any extra channels.
    let tilt_home = cfg.hal.gimbal.tilt_home_deg;
    let mut rx_depth_g = bus.vision_depth.subscribe();
    let bus_gimbal = Arc::clone(&bus);
    tokio::spawn(async move {
        info!("Sim gimbal task started");
        if let Err(e) = gimbal.set_angles(0.0, tilt_home).await {
            warn!("Sim gimbal home failed: {e}");
        }
        let gimbal_t0 = std::time::Instant::now();
        loop {
            let depth = match rx_depth_g.recv().await {
                Ok(d)  => d,
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Closed)    => break,
            };
            let w = depth.width as usize;
            let h = depth.mask_start_row.min(depth.height) as usize;
            let open_bias = if w >= 2 && h > 0 {
                let mid = w / 2;
                let (mut left_sum, mut right_sum) = (0.0f32, 0.0f32);
                for row in 0..h {
                    for col in 0..mid { left_sum  += depth.data[row * w + col]; }
                    for col in mid..w { right_sum += depth.data[row * w + col]; }
                }
                (left_sum - right_sum) / (h * mid) as f32
            } else { 0.0 };
            let t_s = gimbal_t0.elapsed().as_secs_f32();
            let (cur_pan, _) = gimbal.angles();
            let new_pan = gimbal_pan_target(open_bias, t_s, cur_pan);
            if let Err(e) = gimbal.set_pan(new_pan).await {
                warn!("Sim gimbal pan error: {e}");
            }
            let (cur_pan_out, cur_tilt_out) = gimbal.angles();
            let _ = bus_gimbal.gimbal_pan_deg.send(cur_pan_out);
            let _ = bus_gimbal.gimbal_tilt_deg.send(cur_tilt_out);
        }
    });

    // ── Motor execution task ──────────────────────────────────────────────────
    // Identical logic to real robot: drains safety motor_command and controller
    // CmdVel, honours armed/safety state, watchdog-zeros when idle.
    // SimMotorController stores the latest MotorCommand in Arc<Mutex> so the
    // sim tick task reads it without blocking the async executor.
    let mut motor_cmdvel_rx = cmdvel_rx;
    let mut motor_cmd_rx_m  = motor_cmd_rx;
    let rx_exec_m   = bus.executive_state.subscribe();
    let rx_safety_m = bus.safety_state.subscribe();
    let (motor_shutdown_tx, mut motor_shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let motor_handle = tokio::spawn(async move {
        info!("Sim motor task started (disarmed — send SIGUSR1 to arm)");
        let watchdog = tokio::time::Duration::from_millis(500);
        let mut watchdog_logged = false;
        loop {
            tokio::select! {
                biased;
                _ = &mut motor_shutdown_rx => {
                    motor.emergency_stop().await.ok();
                    break;
                }
                cmd = motor_cmd_rx_m.recv() => {
                    let Some(cmd) = cmd else { break };
                    if let Err(e) = motor.send_command(cmd).await {
                        error!("Sim motor error (safety): {e}");
                    }
                }
                Some(cmd_vel) = motor_cmdvel_rx.recv() => {
                    let armed = matches!(
                        *rx_exec_m.borrow(),
                        ExecutiveState::Exploring | ExecutiveState::Recovering
                    );
                    let safe = !matches!(
                        *rx_safety_m.borrow(),
                        SafetyState::EmergencyStop { .. }
                    );
                    if armed && safe {
                        watchdog_logged = false;
                        let cmd = cmdvel_to_motor(cmd_vel, max_motor_duty);
                        if let Err(e) = motor.send_command(cmd).await {
                            error!("Sim motor error (ctrl): {e}");
                        }
                    } else {
                        if !watchdog_logged {
                            warn!(armed, safe, "Sim motor: CmdVel suppressed — zeroing motors");
                            watchdog_logged = true;
                        }
                        if let Err(e) = motor.send_command(MotorCommand::stop(0)).await {
                            error!("Sim motor suppressed-stop failed: {e}");
                        }
                    }
                }
                () = tokio::time::sleep(watchdog) => {
                    if !watchdog_logged {
                        let exploring = matches!(
                            *rx_exec_m.borrow(),
                            ExecutiveState::Exploring | ExecutiveState::Recovering
                        );
                        if exploring {
                            tracing::debug!("Sim motor watchdog: idle gap during exploration");
                        } else {
                            warn!("Sim motor watchdog: no command — zeroing motors");
                        }
                        watchdog_logged = true;
                    }
                    if let Err(e) = motor.send_command(MotorCommand::stop(0)).await {
                        error!("Sim motor watchdog stop failed: {e}");
                    }
                }
            }
        }
    });

    // ── Planning task ─────────────────────────────────────────────────────────
    let bus_plan      = Arc::clone(&bus);
    let mapper_plan   = Arc::clone(&mapper);
    let mut plan_rx   = plan_frontier_rx;
    let mut rx_pose_p = bus.slam_pose2d.subscribe();
    let mut rx_scan_p = bus.vision_pseudo_lidar.subscribe();
    let mut episode_rx_plan = episode_reset_rx.clone();
    let timeout_tx_plan = timeout_tx.clone();
    let force_respawn_tx_plan = force_respawn_tx.clone();
    tokio::spawn(async move {
        use tokio::time::Instant;
        const GOAL_BLACKLIST_SECS: u64 = 10;
        const MAX_CONSECUTIVE_FAILURES: u32 = 10;
        // After this many fruitless escape cycles (robot isolated from all frontiers),
        // force an episode reset rather than cycling indefinitely until MAX_STEPS.
        const MAX_ESCAPE_FAILURES: u32 = 3;
        info!("Sim planning task started (A*, 4-cell clearance)");
        let planner = AStarPlanner::new();
        let mut goal_blacklist: Vec<([f32; 2], Instant)> = Vec::new();
        let mut last_goal: Option<[f32; 2]> = None;
        let mut last_path_sent_at: Option<Instant> = None;
        let mut consecutive_failures: u32 = 0;
        let mut escape_failures: u32 = 0;
        let mut last_scan: Option<std::sync::Arc<core_types::PseudoLidarScan>> = None;
        loop {
            let choice;
            tokio::select! {
                biased;
                Ok(()) = episode_rx_plan.changed() => {
                    let ep = *episode_rx_plan.borrow_and_update();
                    info!(ep, "Sim planning: episode reset — clearing state");
                    goal_blacklist.clear();
                    consecutive_failures = 0;
                    escape_failures = 0;
                    last_goal = None;
                    last_path_sent_at = None;
                    while plan_rx.try_recv().is_ok() {}
                    continue;
                }
                maybe = plan_rx.recv() => {
                    match maybe {
                        Some(c) => { choice = c; }
                        None => break,
                    }
                }
            }

            // Keep last_scan fresh by draining any queued scan messages.
            while let Ok(s) = rx_scan_p.try_recv() { last_scan = Some(s); }

            let now = Instant::now();
            goal_blacklist.retain(|(_, exp)| *exp > now);
            let bl: Vec<[f32; 2]> = goal_blacklist.iter().map(|(g, _)| *g).collect();

            let pose = *rx_pose_p.borrow_and_update();
            let maybe_goal = {
                let m = mapper_plan.read().await;
                select_frontier_goal(&m, &choice, &pose, &bl)
            };
            if let Some(goal) = maybe_goal {
                const REPLAN_HOLD_S: u64 = 3;
                let goal_matches_last = last_goal.map_or(false, |lg| {
                    let dx = lg[0] - goal[0]; let dy = lg[1] - goal[1];
                    (dx*dx + dy*dy).sqrt() < 0.25
                });
                let path_sent_recently = last_path_sent_at.map_or(false, |t| {
                    now.duration_since(t) < std::time::Duration::from_secs(REPLAN_HOLD_S)
                });
                let path_sent_and_expired = goal_matches_last
                    && last_path_sent_at.map_or(false, |t| {
                        let age = now.duration_since(t).as_secs();
                        age >= REPLAN_HOLD_S && age < REPLAN_HOLD_S * 6
                    });
                if goal_matches_last && path_sent_recently {
                    continue;
                }
                const PATH_COMMIT_S: u64 = 10;
                let path_committed = !goal_matches_last
                    && last_path_sent_at.map_or(false, |t| {
                        now.duration_since(t) < std::time::Duration::from_secs(PATH_COMMIT_S)
                    });
                if path_committed {
                    continue;
                }
                if path_sent_and_expired {
                    warn!(x = goal[0], y = goal[1], "Sim planning: frontier persists after visit — blacklisting");
                    let exp = now + std::time::Duration::from_secs(GOAL_BLACKLIST_SECS * 4);
                    goal_blacklist.push((goal, exp));
                    last_goal = None;
                    last_path_sent_at = None;
                    continue;
                }
                last_goal = Some(goal);

                let planned = {
                    let m = mapper_plan.read().await;
                    planner.plan_with_clearance(&m, &pose, goal, 7)
                        .or_else(|| planner.plan_with_clearance(&m, &pose, goal, 5))
                };
                match planned {
                    Some(path) => {
                        info!(waypoints = path.waypoints.len(), "Sim planning: path found");
                        last_path_sent_at = Some(now);
                        let _ = bus_plan.planner_path.send(path).await;
                    }
                    None => {
                        warn!(?choice, "Sim planning: no path to frontier");
                        let exp = now + std::time::Duration::from_secs(GOAL_BLACKLIST_SECS);
                        goal_blacklist.push((goal, exp));
                        last_goal = None;
                        consecutive_failures += 1;
                        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
                            let mut escaped = false;
                            if let Some(ref scan) = last_scan {
                                if let Some(target) = sim_escape(&bus_plan, &pose, scan, consecutive_failures).await {
                                    let escape_path = {
                                        let m = mapper_plan.read().await;
                                        planner.plan(&m, &pose, target)
                                    };
                                    match escape_path {
                                        Some(path) => {
                                            info!(waypoints = path.waypoints.len(), wx = target[0], wy = target[1], "Sim escape: A* path validated — injecting");
                                            let _ = bus_plan.planner_path.send(path).await;
                                            escaped = true;
                                        }
                                        None => warn!(wx = target[0], wy = target[1], "Sim escape: A* failed on escape target — staying still"),
                                    }
                                }
                            }
                            if !escaped { escape_failures += 1; }
                            goal_blacklist.clear();
                            consecutive_failures = 0;
                            last_goal = None;
                            if escape_failures >= MAX_ESCAPE_FAILURES && !explore_mode {
                                warn!(escape_failures, "Sim planning: robot isolated — forcing episode reset");
                                escape_failures = 0;
                                let _ = timeout_tx_plan.send(()).await;
                            } else if escape_failures >= MAX_ESCAPE_FAILURES {
                                warn!(escape_failures, "Sim planning: robot isolated (explore mode — forcing respawn)");
                                escape_failures = 0;
                                let _ = force_respawn_tx_plan.try_send(());
                            }
                            while plan_rx.try_recv().is_ok() {}
                            tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                            while plan_rx.try_recv().is_ok() {}
                        }
                    }
                }
            }
        }
    });

    // ── Control task ──────────────────────────────────────────────────────────
    spawn_control_task(Arc::clone(&bus), control_path_rx,
        cfg.sim.obstacle_slow_m, cfg.sim.obstacle_stop_m, episode_reset_rx.clone());

    // ── Sim tick task ─────────────────────────────────────────────────────────
    // Physics loop at 10 Hz: reads latest motor command + gimbal pan from the
    // shared Arc<Mutex> slots (written by motor task / gimbal task), calls
    // FastSim::step_with_pan, publishes the step on step_tx so SimCamera,
    // SimImu, and SimUltrasonic all wake up simultaneously.
    spawn_sim_tick_task(
        sim_state,
        Arc::clone(&bus),
        episode_reset_tx,
        arm_tx.clone(),
        timeout_tx.clone(),
        force_respawn_rx,
        explore_mode,
    );

    // ── UI Bridge — same port 9000 WebSocket telemetry as real robot ─────────
    start_ui_bridge(Arc::clone(&bus), UiBridgeConfig::default()).await?;

    info!("Sim mode running — press Ctrl+C or send SIGTERM to stop");
    {
        use tokio::signal::unix::SignalKind;
        let mut sig_term = tokio::signal::unix::signal(SignalKind::terminate())?;
        tokio::select! {
            _ = tokio::signal::ctrl_c() => { info!("SIGINT received"); }
            _ = sig_term.recv()         => { info!("SIGTERM received"); }
        }
    }

    // Zero sim motors before process exit.
    let _ = motor_shutdown_tx.send(());
    let _ = tokio::time::timeout(
        tokio::time::Duration::from_millis(500),
        motor_handle,
    ).await;

    info!("Sim mode: shutting down");
    Ok(())
}

/// Physics loop for sim mode.
///
/// Drives `FastSim::step_with_pan` at 10 Hz and publishes the `SimStep` on
/// `sim_state.step_tx` so all HAL trait objects (SimCamera, SimImu,
/// SimUltrasonic) wake simultaneously.  Motor commands arrive via
/// `sim_state.latest_motor` (written by `SimMotorController` from the motor
/// task) and the gimbal pan via `sim_state.latest_pan` (written by `SimGimbal`
/// from the gimbal task).
fn spawn_sim_tick_task(
    sim_state:            SimState,
    bus:                  Arc<bus::Bus>,
    episode_reset_tx:     tokio::sync::watch::Sender<u32>,
    arm_tx:               tokio::sync::mpsc::Sender<()>,
    timeout_tx:           tokio::sync::mpsc::Sender<()>,
    mut force_respawn_rx: tokio::sync::mpsc::Receiver<()>,
    explore_mode:         bool,
) -> tokio::task::JoinHandle<()> {
    use sim_fast::Action;
    use std::time::Duration;

    tokio::spawn(async move {
        // Publish initial state so HAL readers don't block indefinitely on step_rx.
        let initial_step = {
            let mut sim = sim_state.sim.lock().await;
            sim.reset()
        };
        {
            let sim = sim_state.sim.lock().await;
            let walls: Vec<u8> = sim.wall_grid().iter().map(|&w| w as u8).collect();
            let _ = bus.sim_ground_truth.send(Arc::new(walls));
        }
        let _ = sim_state.step_tx.send(Arc::new(initial_step.clone()));
        let _ = bus.slam_pose2d.send(initial_step.pose);
        info!("Sim tick task started — send SIGUSR1 to arm");

        let mut episode: u32 = 0;
        loop {
            tokio::time::sleep(Duration::from_millis(100)).await;

            // Read latest actuator state.
            let latest_motor = sim_state.latest_motor.lock().unwrap().take();
            let pan_deg      = *sim_state.latest_pan.lock().unwrap();

            // Convert MotorCommand to sim Action.
            // Reverse (fl<=0 && fr<=0) → Stop (no Backward action in FastSim).
            // Rotation inferred from opposite front-wheel signs.
            let action = match &latest_motor {
                Some(cmd) if cmd.fl == 0 && cmd.fr == 0 && cmd.rl == 0 && cmd.rr == 0
                    => Action::Stop,
                Some(cmd) if cmd.fl <= 0 && cmd.fr <= 0 => Action::Stop,
                Some(cmd) if cmd.fl > 0  && cmd.fr < 0  => Action::RotateRight,
                Some(cmd) if cmd.fl < 0  && cmd.fr > 0  => Action::RotateLeft,
                Some(_)                                  => Action::Forward,
                None                                     => Action::Stop,
            };

            let step = {
                let mut sim = sim_state.sim.lock().await;
                sim.step_with_pan(action as u8, pan_deg)
            };

            // Wake SimCamera, SimImu, SimUltrasonic.
            let _ = sim_state.step_tx.send(Arc::new(step.clone()));

            // Ground-truth pose → SLAM channel (no dead-reckoning in sim).
            let _ = bus.slam_pose2d.send(step.pose);

            // ── Force respawn (explore mode, planning stuck) ──────────────────
            if explore_mode && force_respawn_rx.try_recv().is_ok() {
                warn!("Sim: planning stuck — forced respawn (explore mode, map preserved)");
                tokio::time::sleep(Duration::from_millis(500)).await;
                let respawn_step = {
                    let mut sim = sim_state.sim.lock().await;
                    sim.respawn()
                };
                let _ = sim_state.step_tx.send(Arc::new(respawn_step.clone()));
                let _ = bus.slam_pose2d.send(respawn_step.pose);
                let _ = bus.safety_state.send(SafetyState::Ok);
                let _ = arm_tx.send(()).await;
                continue;
            }

            // ── Collision ────────────────────────────────────────────────────
            if step.collision {
                if explore_mode {
                    warn!(
                        robot_x = step.pose.x_m, robot_y = step.pose.y_m,
                        "Sim tick: COLLISION — respawning (explore mode, map preserved)"
                    );
                    tokio::time::sleep(Duration::from_millis(500)).await;
                    let respawn_step = {
                        let mut sim = sim_state.sim.lock().await;
                        sim.respawn()
                    };
                    let _ = sim_state.step_tx.send(Arc::new(respawn_step.clone()));
                    let _ = bus.slam_pose2d.send(respawn_step.pose);
                    let _ = bus.safety_state.send(SafetyState::Ok);
                    let _ = arm_tx.send(()).await;
                } else {
                    episode += 1;
                    warn!(
                        episode, robot_x = step.pose.x_m, robot_y = step.pose.y_m,
                        "Sim tick: COLLISION — resetting episode"
                    );
                    tokio::time::sleep(Duration::from_millis(500)).await;
                    let reset_step = {
                        let mut sim = sim_state.sim.lock().await;
                        sim.reset()
                    };
                    let _ = episode_reset_tx.send(episode);
                    let _ = sim_state.step_tx.send(Arc::new(reset_step.clone()));
                    let _ = bus.slam_pose2d.send(reset_step.pose);
                    {
                        let sim = sim_state.sim.lock().await;
                        let walls: Vec<u8> = sim.wall_grid().iter().map(|&w| w as u8).collect();
                        let _ = bus.sim_ground_truth.send(Arc::new(walls));
                    }
                    let _ = bus.safety_state.send(SafetyState::Ok);
                    info!("Sim: auto-re-arming after episode reset");
                    let _ = arm_tx.send(()).await;
                }
                continue;
            }

            // ── Episode timeout ───────────────────────────────────────────────
            if step.done && !explore_mode {
                episode += 1;
                info!(episode, "Sim tick: episode timeout — resetting");
                let _ = timeout_tx.send(()).await;
                tokio::time::sleep(Duration::from_millis(800)).await;
                let reset_step = {
                    let mut sim = sim_state.sim.lock().await;
                    sim.reset()
                };
                let _ = episode_reset_tx.send(episode);
                let _ = sim_state.step_tx.send(Arc::new(reset_step.clone()));
                let _ = bus.slam_pose2d.send(reset_step.pose);
                {
                    let sim = sim_state.sim.lock().await;
                    let walls: Vec<u8> = sim.wall_grid().iter().map(|&w| w as u8).collect();
                    let _ = bus.sim_ground_truth.send(Arc::new(walls));
                }
                let _ = arm_tx.send(()).await;
                continue;
            }
        }
    })
}

// ── Shared perception / gimbal helpers ───────────────────────────────────────
//
// These functions are called from BOTH the real-robot path and the sim path.
// ⚠  Any change to constants, thresholds, or sign conventions must be
//    verified in BOTH call sites and kept in sync deliberately.

/// Return `(range_m, angle_rad)` of the nearest obstacle within the usable FOV.
///
/// Rays within ±50° of the camera centre are considered.  The outer ±5° band
/// (±50°–±55°) is excluded because lens vignette produces a persistent
/// high-depth artifact at the edges regardless of scene content.
///
/// # Dual-path usage
/// - **Real robot**: called in the perception task after `PseudoLidarExtractor::extract()`,
///   before the pan-offset correction is applied.  The angle returned is in
///   *camera frame*; the caller adds the pan offset to convert to robot frame.
/// - **Sim**: called in `sim_publish()`.  Ray angles are already in robot frame
///   (pan baked in by `FastSim::cast_lidar_pan`), so the returned angle is
///   already in robot frame.
fn nearest_in_fov(scan: &core_types::PseudoLidarScan) -> (f32, f32) {
    scan.rays.iter()
        .filter(|r| r.angle_rad.abs() <= 50f32.to_radians())
        .min_by(|a, b| a.range_m.partial_cmp(&b.range_m).unwrap())
        .map(|r| (r.range_m, r.angle_rad))
        .unwrap_or((f32::MAX, 0.0))
}

/// Compute the target gimbal pan angle (degrees) for one control frame.
///
/// Combines a sinusoidal sweep with a reactive bias toward the more open side:
///
/// | Component | Value |
/// |-----------|-------|
/// | Sweep amplitude | ±20° |
/// | Sweep period    | 4 s  |
/// | Reactive cap    | ±10° |
/// | Step cap        | ±5°/frame |
/// | Total range     | ±30° |
///
/// # Arguments
/// - `open_bias` — positive means the **right** side is more open than the left.
///   How to compute this differs by data source:
///   - **Real robot** (MiDaS pixels, higher = closer):
///     `open_bias = (left_sum - right_sum) / n`
///   - **Sim** (pseudo-lidar range_m, higher = more open):
///     `open_bias = right_avg - left_avg`
///   Both yield the same sign convention: positive → pan right.
/// - `t_s`     — elapsed time in seconds (monotonically increasing per episode).
/// - `cur_pan` — current gimbal pan in degrees.
///
/// # Dual-path usage
/// - **Real robot**: called in the gimbal task, `t_s` from `Instant::elapsed()`.
/// - **Sim**: called in `sim_reactive_pan()`, `t_s = sweep_frame as f32 * 0.1`.
fn gimbal_pan_target(open_bias: f32, t_s: f32, cur_pan: f32) -> f32 {
    const SWEEP_AMP: f32     = 20.0; // degrees
    const SWEEP_PERIOD_S: f32 = 4.0; // seconds
    const REACTIVE_GAIN: f32  = 10.0;
    const REACTIVE_CAP: f32   = 10.0; // degrees
    const STEP_CAP: f32       = 5.0;  // degrees per frame
    const PAN_LIMIT: f32      = 30.0; // degrees

    let sweep    = SWEEP_AMP * (2.0 * std::f32::consts::PI * t_s / SWEEP_PERIOD_S).sin();
    let reactive = (open_bias * REACTIVE_GAIN).clamp(-REACTIVE_CAP, REACTIVE_CAP);
    let target   = (sweep + reactive).clamp(-PAN_LIMIT, PAN_LIMIT);
    let step     = (target - cur_pan).clamp(-STEP_CAP, STEP_CAP);
    cur_pan + step
}

/// Stuck-recovery escape maneuver for sim mode.
///
/// Scans all rays, divides them into 8 sectors, picks the sector with the
/// highest average clear range.  Returns the world-frame escape target
/// `[wx, wy]` for the caller to validate with A* before sending a path.
/// Returns `None` if no scan rays are available.
///
/// Side-effects: pans the gimbal toward the escape direction and sleeps 250 ms
/// so the camera preview settles before the body starts turning.
async fn sim_escape(
    bus: &bus::Bus,
    pose: &core_types::Pose2D,
    scan: &core_types::PseudoLidarScan,
    failures: u32,
) -> Option<[f32; 2]> {
    const N: usize = 8;
    let rays = &scan.rays;
    if rays.is_empty() {
        warn!(failures, "Sim escape: no scan available — skipping");
        return None;
    }

    // Accumulate average range per sector (rays ordered by angle index).
    let mut sum = [0.0f32; N];
    let mut cnt = [0usize; N];
    for (i, r) in rays.iter().enumerate() {
        let s = (i * N / rays.len()).min(N - 1);
        sum[s] += r.range_m;
        cnt[s] += 1;
    }
    let best_sector = (0..N)
        .filter(|&s| cnt[s] > 0)
        .max_by(|&a, &b| {
            let avg_a = sum[a] / cnt[a] as f32;
            let avg_b = sum[b] / cnt[b] as f32;
            avg_a.partial_cmp(&avg_b).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);

    // Best ray in the winning sector.
    let lo = best_sector * rays.len() / N;
    let hi = ((best_sector + 1) * rays.len() / N).min(rays.len());
    let best_ray = rays[lo..hi]
        .iter()
        .max_by(|a, b| a.range_m.partial_cmp(&b.range_m).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();

    let world_angle = pose.theta_rad + best_ray.angle_rad;
    let dist = (best_ray.range_m * 0.5_f32).max(0.4);
    let wx = pose.x_m + dist * world_angle.cos();
    let wy = pose.y_m + dist * world_angle.sin();

    let angle_deg = (best_ray.angle_rad * 180.0 / std::f32::consts::PI) as i32;
    warn!(
        failures,
        angle_deg,
        range_m = best_ray.range_m,
        wx, wy,
        "Sim escape: panning camera then injecting escape path"
    );

    // Step 1: stop the robot immediately so it doesn't drive into a wall while
    // we compute and validate the escape path.  Without this, the robot
    // continues its last CmdVel during the 250 ms gimbal-settle pause and can
    // collide before the new path takes effect (observed: Forward at -161°).
    let _ = bus.controller_cmd_vel.try_send(core_types::CmdVel { t_ms: pose.t_ms, vx: 0.0, vy: 0.0, omega: 0.0 });

    // Step 2: pan the camera to preview the escape direction before moving.
    // Gimbal is clamped to ±30°; directions behind the robot clamp to the
    // nearest edge so the camera at least hints at the intended turn.
    let pan_deg = best_ray.angle_rad.to_degrees().clamp(-30.0, 30.0);
    let _ = bus.gimbal_pan_deg.send(pan_deg);

    // Brief pause so the gimbal visually settles before the body starts turning.
    tokio::time::sleep(tokio::time::Duration::from_millis(250)).await;

    // Return the target — the planning task validates with A* before use.
    Some([wx, wy])
}

// ── Calibrate mode ────────────────────────────────────────────────────────────
//
// Collects IMU samples for 30 seconds while the robot is stationary on a flat
// surface, then prints mean accel/gyro offsets for pasting into robot_config.yaml.

async fn run_calibrate(mut imu: Box<dyn hal::Imu>) -> anyhow::Result<()> {
    use std::time::{Duration, Instant};

    const DURATION_SECS: u64 = 30;
    info!(
        duration_s = DURATION_SECS,
        "IMU bias calibration — keep robot stationary on a flat surface"
    );

    let deadline = Instant::now() + Duration::from_secs(DURATION_SECS);
    let (mut ax, mut ay, mut az) = (0.0f64, 0.0f64, 0.0f64);
    let (mut gx, mut gy, mut gz) = (0.0f64, 0.0f64, 0.0f64);
    let mut count = 0u64;

    while Instant::now() < deadline {
        match imu.read_sample().await {
            Ok(s) => {
                ax += s.accel_x as f64;
                ay += s.accel_y as f64;
                az += s.accel_z as f64;
                gx += s.gyro_x as f64;
                gy += s.gyro_y as f64;
                gz += s.gyro_z as f64;
                count += 1;
                if count % 200 == 0 {
                    let remaining = (deadline - Instant::now()).as_secs();
                    info!(count, remaining_s = remaining, "Calibration in progress");
                }
            }
            Err(e) => error!("IMU read error: {e}"),
        }
    }

    anyhow::ensure!(count > 0, "No IMU samples collected — check IMU wiring (/dev/i2c-6)");

    let n = count as f64;
    let (ax_b, ay_b, az_b) = (ax / n, ay / n, az / n);
    let (gx_b, gy_b, gz_b) = (gx / n, gy / n, gz / n);

    info!(
        samples = count,
        ax_bias = ax_b, ay_bias = ay_b, az_bias = az_b,
        gx_bias = gx_b, gy_bias = gy_b, gz_bias = gz_b,
        "Calibration complete"
    );
    info!("Add these values to robot_config.yaml under imu:");
    info!("  ax_bias: {ax_b:.6}");
    info!("  ay_bias: {ay_b:.6}");
    info!("  az_bias: {az_b:.6}");
    info!("  gx_bias: {gx_b:.6}");
    info!("  gy_bias: {gy_b:.6}");
    info!("  gz_bias: {gz_b:.6}");

    Ok(())
}

// ── SLAM-debug mode ───────────────────────────────────────────────────────────
//
// Runs camera → depth inference → pseudo-lidar → IMU dead-reckoning and logs
// everything to logs/slam_debug.ndjson.  No planning, no motor commands.
// Drive manually (or stay still) while this runs and inspect pose drift.

async fn run_slam_debug(
    camera:  Box<dyn hal::Camera>,
    imu:     Box<dyn hal::Imu>,
    gimbal:  Box<dyn hal::Gimbal>,
    bus:     Arc<Bus>,
    cfg:     &RobotConfig,
) -> anyhow::Result<()> {
    info!("SLAM debug — camera + IMU + telemetry only (no planning / motor control)");

    // Home gimbal so tilt_home_deg takes effect even in slam-debug mode.
    let mut gimbal = gimbal;
    let tilt_home = cfg.hal.gimbal.tilt_home_deg;
    if let Err(e) = gimbal.set_angles(0.0, tilt_home).await {
        warn!("Gimbal home failed: {e}");
    }

    let mut telem = TelemetryWriter::open("logs/slam_debug.ndjson").await?;
    telem.event("runtime", "slam_debug_start").await?;

    let mut depth_infer = DepthInference::new(
        &cfg.perception.midas_model_path,
        cfg.perception.depth_out_width,
        cfg.perception.depth_out_height,
        cfg.perception.depth_mask_rows,
        cfg.perception.num_threads,
    );
    let mut event_gate    = EventGate::with_default_threshold();
    let lidar_extractor   = PseudoLidarExtractor::new(48, 3.0);
    let mut slam          = ImuDeadReckon::with_gz_bias(cfg.imu.gz_bias);

    // Camera task
    let bus_cam     = Arc::clone(&bus);
    let mut camera  = camera;
    let cam_handle  = tokio::spawn(async move {
        loop {
            match camera.read_frame().await {
                Ok(frame) => {
                    let arc = Arc::new(frame);
                    let _ = bus_cam.camera_frame_raw.send(Arc::clone(&arc));
                    let gray = rgb_to_gray(&arc);
                    let _ = bus_cam.camera_frame_gray.send(Arc::new(gray));
                }
                Err(e) => error!("Camera error: {e}"),
            }
        }
    });

    // IMU task
    let bus_imu    = Arc::clone(&bus);
    let mut imu    = imu;
    let imu_handle = tokio::spawn(async move {
        loop {
            match imu.read_sample().await {
                Ok(s) => { let _ = bus_imu.imu_raw.send(s); }
                Err(e) => error!("IMU error: {e}"),
            }
        }
    });

    // SLAM task
    let bus_slam    = Arc::clone(&bus);
    let mut rx_imu  = bus.imu_raw.subscribe();
    let slam_handle = tokio::spawn(async move {
        loop {
            match rx_imu.recv().await {
                Ok(sample) => {
                    let pose = slam.update(&sample);
                    let _ = bus_slam.slam_pose2d.send(pose);
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    warn!("SLAM: IMU lagged {n} samples");
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    });

    // Perception task
    let bus_perc    = Arc::clone(&bus);
    let mut rx_gray = bus.camera_frame_gray.subscribe();
    let perc_handle = tokio::spawn(async move {
        loop {
            match rx_gray.recv().await {
                Ok(gray) => {
                    if event_gate.should_infer(&gray) {
                        let rgb_stub = core_types::CameraFrame {
                            t_ms: gray.t_ms, width: gray.width, height: gray.height,
                            data: gray.data.iter().flat_map(|&v| [v, v, v]).collect(),
                        };
                        match tokio::task::block_in_place(|| depth_infer.infer(&rgb_stub)) {
                            Ok(depth) => {
                                let scan = lidar_extractor.extract(&depth);
                                // Nearest obstacle — camera frame, no pan offset (slam_debug
                                // intentionally skips gimbal correction for raw MiDaS analysis).
                                let (near_m, near_angle_rad) = nearest_in_fov(&scan);
                                if near_m < 1.5 {
                                    let near_deg = near_angle_rad.to_degrees();
                                    let side = if near_deg > 5.0 { "left" } else if near_deg < -5.0 { "right" } else { "center" };
                                    info!(nearest_m = near_m, angle_deg = near_deg, side, "Perception: nearest obstacle");
                                }
                                let _ = bus_perc.vision_depth.send(Arc::new(depth));
                                let _ = bus_perc.vision_pseudo_lidar.send(Arc::new(scan));
                                let _ = bus_perc.nearest_obstacle_m.send(near_m);
                                let _ = bus_perc.nearest_obstacle_angle_rad.send(near_angle_rad);
                            }
                            Err(e) => error!("Depth inference error: {e}"),
                        }
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    warn!("Perception: camera lagged {n} frames");
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    });

    // Telemetry task
    let mut rx_pose_t  = bus.slam_pose2d.subscribe();
    let mut rx_lidar_t = bus.vision_pseudo_lidar.subscribe();
    let mut rx_imu_t   = bus.imu_raw.subscribe();
    let telem_handle = tokio::spawn(async move {
        loop {
            tokio::select! {
                Ok(()) = rx_pose_t.changed() => {
                    let pose = *rx_pose_t.borrow_and_update();
                    telem.write("slam/pose2d", "micro_slam", &pose).await.ok();
                }
                Ok(scan) = rx_lidar_t.recv() => {
                    telem.write("vision/pseudo_lidar", "perception", &*scan).await.ok();
                }
                Ok(imu) = rx_imu_t.recv() => {
                    telem.write("imu/raw", "hal", &imu).await.ok();
                }
                else => break,
            }
        }
    });

    info!("SLAM debug running — press Ctrl+C to stop. Log: logs/slam_debug.ndjson");
    tokio::signal::ctrl_c().await?;
    info!("Shutdown");

    drop(cam_handle);
    drop(imu_handle);
    drop(slam_handle);
    drop(perc_handle);
    drop(telem_handle);

    Ok(())
}

fn rgb_to_gray(frame: &core_types::CameraFrame) -> core_types::GrayFrame {
    let data: Vec<u8> = frame.data
        .chunks_exact(3)
        .map(|rgb| {
            let r = rgb[0] as u32;
            let g = rgb[1] as u32;
            let b = rgb[2] as u32;
            ((r * 77 + g * 150 + b * 29) >> 8) as u8
        })
        .collect();
    core_types::GrayFrame {
        t_ms: frame.t_ms,
        width: frame.width,
        height: frame.height,
        data,
    }
}
