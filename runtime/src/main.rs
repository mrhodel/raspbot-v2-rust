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
use tracing::{error, info, warn};

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
        return run_sim_mode(bus, cfg, plan_frontier_rx, control_path_rx, cmdvel_rx).await;
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
        return run_slam_debug(camera, imu, bus, &cfg).await;
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
                                let scan = lidar_extractor.extract(&depth);
                                // Forward-arc nearest obstacle: only rays within ±30°
                                // of centre.  Side walls don't threaten forward motion.
                                let near_m = scan.rays.iter()
                                    .filter(|r| r.angle_rad.abs() <= std::f32::consts::FRAC_PI_6)
                                    .map(|r| r.range_m)
                                    .fold(f32::MAX, f32::min);
                                let _ = bus_perc.vision_depth.send(Arc::new(depth));
                                let _ = bus_perc.vision_pseudo_lidar.send(Arc::new(scan));
                                let _ = bus_perc.nearest_obstacle_m.send(near_m);
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
    let bus_map        = Arc::clone(&bus);
    let mapper_map     = Arc::clone(&mapper);
    let mut rx_lidar   = bus.vision_pseudo_lidar.subscribe();
    let mut rx_pose_m  = bus.slam_pose2d.subscribe();
    let map_handle = tokio::spawn(async move {
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
    });

    // ── Safety task ───────────────────────────────────────────────────────────
    let bus_safety = Arc::clone(&bus);
    let mut rx_us  = bus.ultrasonic.subscribe();
    let safety_handle = tokio::spawn(async move {
        info!(
            threshold_cm = safety::STOP_THRESHOLD_CM,
            watchdog_ms  = safety::WATCHDOG_TIMEOUT_MS,
            "Safety task started"
        );
        let monitor      = SafetyMonitor::new();
        let watchdog_dur = tokio::time::Duration::from_millis(monitor.watchdog_timeout_ms);
        // Latch: once EmergencyStop fires, hold it for at least this duration even if
        // the US returns a good reading (e.g. blind spot <3 cm → driver returns 400 cm).
        const EMSTOP_LATCH: tokio::time::Duration = tokio::time::Duration::from_secs(5);
        let mut emstop_until: Option<tokio::time::Instant> = None;
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
                                // Still latched: re-publish EmergencyStop regardless of reading.
                                // The motor_command channel may already be draining; try_send is
                                // fire-and-forget (motor watchdog also zeros on its own tick).
                                let state = SafetyState::EmergencyStop {
                                    reason: "EmergencyStop latched (blind-spot protection)".into(),
                                };
                                let _ = bus_safety.safety_state.send(state);
                                let _ = bus_safety.motor_command.try_send(MotorCommand::stop(reading.t_ms));
                            } else {
                                let (state, maybe_stop) = monitor.evaluate(&reading);
                                let _ = bus_safety.safety_state.send(state);
                                if let Some(cmd) = maybe_stop {
                                    warn!(range_cm = reading.range_cm, "Safety: EMERGENCY STOP — latching for {}s", EMSTOP_LATCH.as_secs());
                                    emstop_until = Some(now + EMSTOP_LATCH);
                                    if bus_safety.motor_command.try_send(cmd).is_err() {
                                        error!("Safety: motor_command channel full — stop dropped!");
                                    }
                                }
                            }
                        }
                    }
                }
                _ = tokio::time::sleep(watchdog_dur) => {
                    warn!("Safety: watchdog timeout — no US reading");
                    let now = tokio::time::Instant::now();
                    emstop_until = Some(now + EMSTOP_LATCH);
                    let (state, cmd) = monitor.evaluate_timeout(0);
                    let _ = bus_safety.safety_state.send(state);
                    let _ = bus_safety.motor_command.try_send(cmd);
                }
            }
        }
    });

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
                // Blacklist a goal that's selected twice in a row — stuck loop.
                if last_goal.map_or(false, |lg| {
                    let dx = lg[0] - goal[0]; let dy = lg[1] - goal[1];
                    (dx*dx + dy*dy).sqrt() < 0.10
                }) {
                    warn!(x = goal[0], y = goal[1], "Planning: same goal repeated — blacklisting");
                    let exp = now + std::time::Duration::from_secs(GOAL_BLACKLIST_SECS);
                    goal_blacklist.push((goal, exp));
                    last_goal = None;
                    continue;
                }
                last_goal = Some(goal);

                let m = mapper_plan.read().await;
                match planner.plan(&m, &pose, goal) {
                    Some(path) => {
                        info!(waypoints = path.waypoints.len(), "Planning: path found");
                        let _ = bus_plan.planner_path.send(path).await;
                    }
                    None => {
                        warn!(?choice, "Planning: no path to frontier");
                        let exp = now + std::time::Duration::from_secs(GOAL_BLACKLIST_SECS);
                        goal_blacklist.push((goal, exp));
                        last_goal = None;
                    }
                }
            } else {
                warn!(?choice, "Planning: no frontiers available");
            }
        }
    });

    // ── Control task ──────────────────────────────────────────────────────────
    // Runs pure-pursuit at CONTROL_HZ against the live pose watch.
    // A new Path from the planner replaces the active path immediately.
    // When the robot reaches the final waypoint (within GOAL_TOLERANCE_M),
    // the path is cleared and CmdVel stops — the motor watchdog zeroes motors.
    //
    // Reactive obstacle avoidance: reads nearest_obstacle_m from the pseudo-lidar
    // bus on every tick.  Forward speed scales to zero as the obstacle closes in;
    // at OBSTACLE_STOP_M the path is abandoned and a replan is requested by
    // clearing current_path (the planner will re-run on the next FrontierChoice).
    // This fires well above the 25 cm hardware safety interlock.
    const CONTROL_HZ: f64 = 10.0;
    const GOAL_TOLERANCE_M: f32 = 0.25; // metres — close enough to declare goal reached
    /// Nearest-obstacle distance at which forward speed begins scaling down.
    const OBSTACLE_SLOW_M: f32 = 0.80;
    /// Nearest-obstacle distance at which forward motion stops and path is abandoned.
    /// Must be well above the 25 cm hardware safety interlock.
    const OBSTACLE_STOP_M: f32 = 0.40;

    let bus_ctrl       = Arc::clone(&bus);
    let mut ctrl_rx    = control_path_rx;
    let mut rx_pose_c  = bus.slam_pose2d.subscribe();
    let rx_nearest_c   = bus.nearest_obstacle_m.subscribe();
    let ctrl_handle = tokio::spawn(async move {
        use std::time::Duration;
        use core_types::Path;
        info!("Control task started (pure-pursuit {CONTROL_HZ} Hz, lookahead 0.3 m)");
        let controller = PurePursuitController::new();
        let mut current_path: Option<Path> = None;
        let mut tick = tokio::time::interval(Duration::from_secs_f64(1.0 / CONTROL_HZ));
        // Tracks whether we've already cleared the path due to an obstacle so we
        // don't spam the log and re-clear on every tick while stopped.
        let mut obstacle_stopped = false;
        loop {
            tokio::select! {
                biased;
                // New path from planner — replace immediately.
                result = ctrl_rx.recv() => {
                    match result {
                        Some(p) => {
                            info!(waypoints = p.waypoints.len(), "Control: new path, tracking");
                            current_path = Some(p);
                            obstacle_stopped = false; // fresh path — re-enable avoidance check
                        }
                        None => break,
                    }
                }
                // Control tick — re-compute CmdVel from current pose.
                _ = tick.tick() => {
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
                    let nearest = *rx_nearest_c.borrow();
                    let speed_scale = if nearest >= OBSTACLE_SLOW_M {
                        1.0_f32
                    } else if nearest <= OBSTACLE_STOP_M {
                        0.0_f32
                    } else {
                        (nearest - OBSTACLE_STOP_M) / (OBSTACLE_SLOW_M - OBSTACLE_STOP_M)
                    };

                    if speed_scale <= 0.0 {
                        if !obstacle_stopped {
                            warn!(
                                nearest_m = nearest,
                                stop_threshold_m = OBSTACLE_STOP_M,
                                "Control: obstacle too close — stopping and requesting replan"
                            );
                            obstacle_stopped = true;
                            current_path = None; // one-shot: planner replans on next FrontierChoice
                        }
                        // Motor watchdog will zero motors within 500 ms.
                        continue;
                    }

                    let mut cmd_vel = controller.compute(&pose, path);
                    cmd_vel.vx *= speed_scale;
                    if speed_scale < 0.99 {
                        tracing::debug!(nearest_m = nearest, speed_scale, "Control: slowing for obstacle");
                    }
                    let _ = bus_ctrl.controller_cmd_vel.try_send(cmd_vel);
                }
            }
        }
    });

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
                cmd = motor_cmd_rx.recv() => {
                    let Some(cmd) = cmd else { break };
                    watchdog_logged = false;
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
                        warn!("Motor watchdog: no command for {MOTOR_WATCHDOG_MS}ms — zeroing motors");
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
    // Homes the gimbal at startup, then pans reactively toward the more open
    // half of the depth map (simple left/right free-space heuristic).
    let mut gimbal = gimbal;
    let mut rx_depth_g = bus.vision_depth.subscribe();
    let bus_gimbal = Arc::clone(&bus);
    let gimbal_handle = tokio::spawn(async move {
        info!("Gimbal task started");
        if let Err(e) = gimbal.set_angles(0.0, 0.0).await {
            warn!("Gimbal home failed: {e}");
        }
        loop {
            let depth = match rx_depth_g.recv().await {
                Ok(d)  => d,
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Closed)    => break,
            };
            // Simple reactive pan: compare average depth of left vs right half.
            // MiDaS values: higher = closer.  Pan toward the more open (lower) side.
            let w = depth.width as usize;
            let h = depth.mask_start_row.min(depth.height) as usize;
            if w < 2 || h == 0 { continue; }
            let mid = w / 2;
            let (mut left_sum, mut right_sum) = (0.0f32, 0.0f32);
            for row in 0..h {
                for col in 0..mid    { left_sum  += depth.data[row * w + col]; }
                for col in mid..w    { right_sum += depth.data[row * w + col]; }
            }
            let n = (h * mid) as f32;
            // right_avg > left_avg → right side more blocked → pan left (negative).
            let pan_error = (left_sum - right_sum) / n;
            let (cur_pan, _) = gimbal.angles();
            // Clamp step to ±5° per frame so the gimbal doesn't lurch on noisy
            // MiDaS frames. Deadband of 8° filters per-frame renormalization
            // noise (even a static symmetric scene can produce ~0.3 imbalance).
            let step = (pan_error * 20.0).clamp(-5.0, 5.0);
            let new_pan = (cur_pan + step).clamp(-90.0, 90.0);
            if (new_pan - cur_pan).abs() > 8.0 {
                if let Err(e) = gimbal.set_pan(new_pan).await {
                    warn!("Gimbal pan error: {e}");
                }
            }
            let _ = bus_gimbal.gimbal_pan_deg.send(gimbal.angles().0);
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

    info!("Milestone 2 system running — press Ctrl+C to stop");
    tokio::signal::ctrl_c().await?;
    info!("Shutdown signal received");

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
    const MIN_GOAL_DIST: f32 = 0.30;    // metres — skip frontiers trivially close to robot
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
            pool.iter().copied().min_by(|a, b| dist(a).partial_cmp(&dist(b)).unwrap())
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
    _cfg:             config::RobotConfig,
    plan_frontier_rx: tokio::sync::mpsc::Receiver<core_types::FrontierChoice>,
    control_path_rx:  tokio::sync::mpsc::Receiver<core_types::Path>,
    cmdvel_rx:        tokio::sync::mpsc::Receiver<core_types::CmdVel>,
) -> anyhow::Result<()> {
    use std::sync::Arc;
    use tokio::sync::RwLock;

    let seed: u64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    info!(seed, "Sim mode starting");

    // ── Signal handler — registered FIRST before any spawning ─────────────────
    // Calling signal() here (in the current async context, not inside a spawned
    // task) installs the OS-level handler immediately, so SIGUSR1 cannot kill
    // the process even if it arrives before the spawned tasks are scheduled.
    use tokio::signal::unix::{signal, SignalKind};
    let mut sigusr1_stream = signal(SignalKind::user_defined1())
        .map_err(|e| anyhow::anyhow!("SIGUSR1 handler: {e}"))?;

    // ── Mapper (shared between mapping + planning tasks) ──────────────────────
    let mapper = Arc::new(RwLock::new(Mapper::new()));

    // ── Executive task ────────────────────────────────────────────────────────
    let (arm_tx, mut arm_rx) = tokio::sync::mpsc::channel::<()>(4);
    let mut rx_safety_exec   = bus.safety_state.subscribe();
    let bus_exec             = Arc::clone(&bus);
    tokio::spawn(async move {
        let mut exec = executive::Executive::new(Arc::clone(&bus_exec));
        info!("Sim executive task started (Idle) — send SIGUSR1 to arm");
        loop {
            tokio::select! {
                biased;
                _ = arm_rx.recv() => {
                    match exec.arm() {
                        Ok(())  => info!("Sim executive: armed → Exploring"),
                        Err(e)  => warn!("Sim executive arm rejected: {e}"),
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

    // ── Mapping task ──────────────────────────────────────────────────────────
    let bus_map       = Arc::clone(&bus);
    let mapper_map    = Arc::clone(&mapper);
    let mut rx_lidar  = bus.vision_pseudo_lidar.subscribe();
    let mut rx_pose_m = bus.slam_pose2d.subscribe();
    tokio::spawn(async move {
        info!("Sim mapping task started");
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
                    warn!("Sim mapping task lagged {n} scans");
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    });

    // ── Planning task ─────────────────────────────────────────────────────────
    let bus_plan      = Arc::clone(&bus);
    let mapper_plan   = Arc::clone(&mapper);
    let mut plan_rx   = plan_frontier_rx;
    let mut rx_pose_p = bus.slam_pose2d.subscribe();
    tokio::spawn(async move {
        use tokio::time::Instant;
        const GOAL_BLACKLIST_SECS: u64 = 10;
        info!("Sim planning task started (A*, 4-cell clearance)");
        let planner = AStarPlanner::new();
        let mut goal_blacklist: Vec<([f32; 2], Instant)> = Vec::new();
        let mut last_goal: Option<[f32; 2]> = None;
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
                // Blacklist a goal that's selected twice in a row — stuck loop.
                if last_goal.map_or(false, |lg| {
                    let dx = lg[0] - goal[0]; let dy = lg[1] - goal[1];
                    (dx*dx + dy*dy).sqrt() < 0.10
                }) {
                    warn!(x = goal[0], y = goal[1], "Sim planning: same goal repeated — blacklisting");
                    let exp = now + std::time::Duration::from_secs(GOAL_BLACKLIST_SECS);
                    goal_blacklist.push((goal, exp));
                    last_goal = None;
                    continue;
                }
                last_goal = Some(goal);

                let m = mapper_plan.read().await;
                match planner.plan(&m, &pose, goal) {
                    Some(path) => {
                        info!(waypoints = path.waypoints.len(), "Sim planning: path found");
                        let _ = bus_plan.planner_path.send(path).await;
                    }
                    None => {
                        warn!(?choice, "Sim planning: no path to frontier");
                        let exp = now + std::time::Duration::from_secs(GOAL_BLACKLIST_SECS);
                        goal_blacklist.push((goal, exp));
                        last_goal = None;
                    }
                }
            }
        }
    });

    // ── Control task ──────────────────────────────────────────────────────────
    // Identical to robot-run: reactive obstacle avoidance active.
    const SIM_CONTROL_HZ: f64 = 10.0;
    const SIM_GOAL_TOLERANCE_M: f32 = 0.25;
    const SIM_OBSTACLE_SLOW_M: f32 = 0.80;
    const SIM_OBSTACLE_STOP_M: f32 = 0.40;
    let bus_ctrl       = Arc::clone(&bus);
    let mut ctrl_rx    = control_path_rx;
    let mut rx_pose_c  = bus.slam_pose2d.subscribe();
    let rx_nearest_c   = bus.nearest_obstacle_m.subscribe();
    tokio::spawn(async move {
        use std::time::Duration;
        use core_types::Path;
        info!("Sim control task started (pure-pursuit {SIM_CONTROL_HZ} Hz)");
        let controller = PurePursuitController::new();
        let mut current_path: Option<Path> = None;
        let mut tick = tokio::time::interval(Duration::from_secs_f64(1.0 / SIM_CONTROL_HZ));
        let mut obstacle_stopped = false;
        loop {
            tokio::select! {
                biased;
                result = ctrl_rx.recv() => {
                    match result {
                        Some(p) => {
                            info!(waypoints = p.waypoints.len(), "Sim control: new path, tracking");
                            current_path = Some(p);
                            obstacle_stopped = false;
                        }
                        None => break,
                    }
                }
                _ = tick.tick() => {
                    let Some(ref path) = current_path else { continue };
                    let pose = *rx_pose_c.borrow_and_update();
                    if let Some(&[gx, gy]) = path.waypoints.last() {
                        let dx = gx - pose.x_m;
                        let dy = gy - pose.y_m;
                        if dx * dx + dy * dy < SIM_GOAL_TOLERANCE_M * SIM_GOAL_TOLERANCE_M {
                            info!("Sim control: goal reached");
                            current_path = None;
                            obstacle_stopped = false;
                            continue;
                        }
                    }
                    let nearest = *rx_nearest_c.borrow();
                    let speed_scale = if nearest >= SIM_OBSTACLE_SLOW_M {
                        1.0_f32
                    } else if nearest <= SIM_OBSTACLE_STOP_M {
                        0.0_f32
                    } else {
                        (nearest - SIM_OBSTACLE_STOP_M) / (SIM_OBSTACLE_SLOW_M - SIM_OBSTACLE_STOP_M)
                    };
                    if speed_scale <= 0.0 {
                        if !obstacle_stopped {
                            warn!(nearest_m = nearest, "Sim control: obstacle too close — stopping and requesting replan");
                            obstacle_stopped = true;
                            current_path = None;
                        }
                        continue;
                    }
                    let mut cmd_vel = controller.compute(&pose, path);
                    cmd_vel.vx *= speed_scale;
                    let _ = bus_ctrl.controller_cmd_vel.try_send(cmd_vel);
                }
            }
        }
    });

    // ── Sim driver task ───────────────────────────────────────────────────────
    // Reads CmdVel from the bus, steps FastSim, publishes synthetic sensor data.
    // On collision: publishes EmergencyStop then auto-resets the episode after
    // a 2 s pause (mirrors the 5 s safety latch on the real robot).
    let bus_sim       = Arc::clone(&bus);
    let arm_tx_sim    = arm_tx.clone();
    let mut cmdvel_rx = cmdvel_rx;
    tokio::spawn(async move {
        use sim_fast::{FastSim, Action};
        use std::time::Duration;
        use std::sync::Arc as StdArc;
        use tokio::sync::Mutex;

        let sim = StdArc::new(Mutex::new(FastSim::new(seed)));
        let rx_pan = bus_sim.gimbal_pan_deg.subscribe();

        let initial = sim.lock().await.reset();
        sim_publish(&bus_sim, &initial);
        info!("Sim driver task started — send SIGUSR1 to arm");

        let mut episode = 0u32;
        loop {
            // Drain cmdvel channel — use the last one received.
            let mut latest_cv: Option<core_types::CmdVel> = None;
            while let Ok(cv) = cmdvel_rx.try_recv() {
                latest_cv = Some(cv);
            }

            let action = latest_cv.map(sim_cmdvel_to_action).unwrap_or(Action::Stop);
            let pan_deg = *rx_pan.borrow();

            let step = sim.lock().await.step_with_pan(action as u8, pan_deg);
            sim_publish(&bus_sim, &step);

            if step.collision {
                episode += 1;
                warn!(episode, "Sim: COLLISION — resetting episode in 2 s");
                let _ = bus_sim.safety_state.send(SafetyState::EmergencyStop {
                    reason: format!("Sim collision (episode {episode})"),
                });
                tokio::time::sleep(Duration::from_secs(2)).await;

                // Reset and publish the fresh starting state.
                let reset_step = sim.lock().await.reset();
                sim_publish(&bus_sim, &reset_step);
                // Clear the latch: new episode, new space.
                let _ = bus_sim.safety_state.send(SafetyState::Ok);
                // Re-arm automatically so exploration continues unattended.
                // (Same behaviour as hitting SIGUSR1 after a collision.)
                info!("Sim: auto-re-arming after episode reset");
                let _ = arm_tx_sim.send(()).await;
                continue;
            }

            if step.done {
                episode += 1;
                info!(episode, "Sim: max steps reached — resetting");
                let reset_step = sim.lock().await.reset();
                sim_publish(&bus_sim, &reset_step);
                let _ = arm_tx_sim.send(()).await;
                continue;
            }

            // 10 Hz pacing — the control task also runs at 10 Hz so we stay in sync.
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    });

    info!("Sim mode running — press Ctrl+C to stop");
    tokio::signal::ctrl_c().await?;
    info!("Sim mode: shutting down");
    Ok(())
}

/// Publish one SimStep's worth of sensor data onto the bus.
fn sim_publish(bus: &bus::Bus, step: &sim_fast::SimStep) {
    use core_types::UltrasonicReading;
    use std::sync::Arc;

    // Pose (ground truth).
    let _ = bus.slam_pose2d.send(step.pose);

    // Pseudo-lidar scan → mapping + control tasks.
    let _ = bus.vision_pseudo_lidar.send(Arc::new(step.scan.clone()));

    // Forward-arc nearest obstacle: only rays within ±30° of centre.
    let nearest = step.scan.rays.iter()
        .filter(|r| r.angle_rad.abs() <= std::f32::consts::FRAC_PI_6)
        .map(|r| r.range_m)
        .fold(f32::MAX, f32::min);
    let _ = bus.nearest_obstacle_m.send(nearest);

    // Synthetic ultrasonic: forward ray (angle closest to 0).
    let fwd_range_m = step.scan.rays.iter()
        .min_by(|a, b| a.angle_rad.abs().partial_cmp(&b.angle_rad.abs()).unwrap())
        .map(|r| r.range_m)
        .unwrap_or(3.0);
    let _ = bus.ultrasonic.send(UltrasonicReading {
        t_ms:     step.pose.t_ms,
        range_cm: fwd_range_m * 100.0,
    });

    // IMU.
    let _ = bus.imu_raw.send(step.imu);
}

/// Convert a continuous CmdVel to the nearest discrete sim Action.
/// Dominant component wins; small commands map to Stop.
fn sim_cmdvel_to_action(cmd: core_types::CmdVel) -> sim_fast::Action {
    use sim_fast::Action;
    const THRESH: f32 = 0.05;
    let vx = cmd.vx;
    let vy = cmd.vy;
    let om = cmd.omega;
    let dominant = [
        (vx.abs(),  if vx > 0.0 { Action::Forward     } else { Action::Stop }),
        (vy.abs(),  if vy > 0.0 { Action::StrafeLeft  } else { Action::StrafeRight }),
        (om.abs(),  if om > 0.0 { Action::RotateLeft  } else { Action::RotateRight }),
    ]
    .into_iter()
    .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
    .map(|(mag, act)| if mag >= THRESH { act } else { Action::Stop })
    .unwrap_or(Action::Stop);
    dominant
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
    bus:     Arc<Bus>,
    cfg:     &RobotConfig,
) -> anyhow::Result<()> {
    info!("SLAM debug — camera + IMU + telemetry only (no planning / motor control)");

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
                                // Forward-arc nearest obstacle: only rays within ±30°
                                // of centre.  Side walls don't threaten forward motion.
                                let near_m = scan.rays.iter()
                                    .filter(|r| r.angle_rad.abs() <= std::f32::consts::FRAC_PI_6)
                                    .map(|r| r.range_m)
                                    .fold(f32::MAX, f32::min);
                                let _ = bus_perc.vision_depth.send(Arc::new(depth));
                                let _ = bus_perc.vision_pseudo_lidar.send(Arc::new(scan));
                                let _ = bus_perc.nearest_obstacle_m.send(near_m);
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
