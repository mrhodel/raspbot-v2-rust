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
use core_types::{CmdVel, ExecutiveState, FrontierChoice, MotorCommand, SafetyState};
use executive::Executive;
use hal::{
    Camera, Gimbal, Imu, MotorController, Tof, Ultrasonic,
    StubCamera, StubGimbal, StubImu, StubMotorController, StubTof, StubUltrasonic,
    YahboomGimbal, YahboomMotorController, YahboomUltrasonic,
};
#[cfg(feature = "usb-camera")]
use hal::V4L2Camera;
#[cfg(feature = "mpu6050")]
use hal::Mpu6050Imu;
#[cfg(feature = "vl53l8cx")]
use hal::Vl53l8cxTof;
use mapping::Mapper;
use micro_slam::ImuDeadReckon;
use perception::{DepthInference, EventGate, PseudoLidarExtractor};
use planning::{AStarPlanner, reachable_set};
use telemetry::TelemetryWriter;
use ui_bridge::{UiBridgeConfig, start as start_ui_bridge};
use exploration_rl::spawn_selector_task;
use sim_fast::hal_impls::SimState;
use sim_vision::SimCamera as SimCameraHal;
use bus::BridgeCommand;

mod tasks;
use tasks::{spawn_control_task, spawn_mapping_task, spawn_safety_task};

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

    let tof: Box<dyn Tof> = {
        #[cfg(feature = "vl53l8cx")]
        {
            let tc = &cfg.hal.tof;
            match Vl53l8cxTof::new(tc.i2c_bus, tc.i2c_address, tc.ranging_mode, tc.integration_time_ms) {
                Ok(t)  => { info!("ToF: VL53L8CX on /dev/i2c-{}", tc.i2c_bus); Box::new(t) as Box<dyn Tof> }
                Err(e) => { warn!("ToF: VL53L8CX unavailable ({e:#}), using stub"); Box::new(StubTof::new()) }
            }
        }
        #[cfg(not(feature = "vl53l8cx"))]
        {
            info!("ToF: stub (vl53l8cx feature disabled)");
            Box::new(StubTof::new())
        }
    };

    let max_motor_duty = cfg.hal.motor.max_speed as i8;
    let min_motor_duty = cfg.robot.min_speed as i8;
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
    // Scale factor converts normalised CmdVel (range ±1.0) to physical m/s.
    // max_motor_duty caps the normalised wheel speed; kinematics.forward_speed_m_s
    // is the measured top speed at 100% duty.
    let vel_scale = (max_motor_duty as f32 / 100.0) * cfg.kinematics.forward_speed_m_s;
    let mut slam = ImuDeadReckon::with_gz_bias(cfg.imu.gz_bias);
    info!(gz_bias = cfg.imu.gz_bias, vel_scale, "Micro-SLAM initialised (IMU dead-reckoning + motor feedforward)");

    // ── Mapper (shared: mapping task writes, planning task reads) ─────────────
    let mapper = Arc::new(RwLock::new(Mapper::new()));
    info!("Occupancy grid mapper initialised (5 cm/cell)");

    // ── Executive task ────────────────────────────────────────────────────────
    // Owns the state machine.  Reacts to three event sources:
    //   1. `safety_state` watch — fires SafetyStopped when the US trips.
    //   2. arm_tx channel      — SIGUSR1 or any future arm command.
    //   3. bridge_cmd watch    — MANUAL/AUTO/STOP from the UI.
    //
    // Motor CmdVel is blocked unless ExecutiveState ∈ {Exploring, Recovering}.
    // To arm: `kill -USR1 <pid>` (or send on arm_tx from a future UI).
    let (arm_tx, mut arm_rx) = tokio::sync::mpsc::channel::<()>(4);
    let mut rx_safety_exec   = bus.safety_state.subscribe();
    let mut rx_bridge_exec_r = bus.bridge_cmd.subscribe();
    let bus_exec             = Arc::clone(&bus);
    let arm_tx_exec          = arm_tx.clone();
    let exec_handle = tokio::spawn(async move {
        let arm_tx = arm_tx_exec;
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
                Ok(()) = rx_bridge_exec_r.changed() => {
                    let cmd = rx_bridge_exec_r.borrow_and_update().clone();
                    match cmd {
                        BridgeCommand::Arm  => { let _ = arm_tx.send(()).await; }
                        BridgeCommand::Stop => { let _ = exec.transition(ExecutiveState::Idle); }
                        BridgeCommand::Manual => {
                            // Allow from any state: stop autonomous mode first if needed.
                            if !matches!(exec.state(),
                                ExecutiveState::Idle | ExecutiveState::ManualDrive)
                            {
                                let _ = exec.transition(ExecutiveState::Idle);
                            }
                            if matches!(exec.state(), ExecutiveState::Idle) {
                                match exec.transition(ExecutiveState::ManualDrive) {
                                    Ok(())  => info!("Executive: manual drive activated"),
                                    Err(e)  => warn!("Manual transition rejected: {e}"),
                                }
                            }
                        }
                        BridgeCommand::Auto => {
                            if matches!(exec.state(), ExecutiveState::ManualDrive) {
                                match exec.transition(ExecutiveState::Idle) {
                                    Ok(())  => info!("Executive: manual drive deactivated → Idle"),
                                    Err(e)  => warn!("Auto transition rejected: {e}"),
                                }
                            }
                        }
                        BridgeCommand::None => {}
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

    // ── ToF task ──────────────────────────────────────────────────────────────
    // Reads VL53L8CX 8×8 scans at up to 60 Hz and publishes:
    //   • nearest_obstacle_m / nearest_obstacle_angle_rad — used by the control
    //     task for reactive speed scaling (replaces noisy MiDaS on real hardware).
    //   • vision_pseudo_lidar — same PseudoLidarScan format as MiDaS, so the
    //     mapping task automatically ingests ToF data alongside MiDaS scans.
    let bus_tof = Arc::clone(&bus);
    let mut tof = tof;
    let tof_handle = tokio::spawn(async move {
        info!("ToF task started (VL53L8CX ±19.69° forward arc)");
        loop {
            match tof.read_scan().await {
                Ok(scan) => {
                    // Nearest obstacle across all rays.
                    let (nearest_m, nearest_angle_rad) = scan.rays.iter()
                        .fold((f32::MAX, 0.0_f32), |(min_r, min_a), r| {
                            if r.range_m < min_r { (r.range_m, r.angle_rad) } else { (min_r, min_a) }
                        });
                    let nearest_m = if nearest_m == f32::MAX { 3.0 } else { nearest_m };
                    // Only override nearest_obstacle_m if ToF sees something CLOSER than
                    // the current MiDaS reading.  MiDaS (wide FOV) is the baseline owner;
                    // ToF (narrow ±20°, precise) overrides only when it detects an imminent
                    // forward obstacle.  This prevents the 20 Hz ToF open-space value (3.0m)
                    // from washing out MiDaS detections at wider angles.
                    let current_nearest = *bus_tof.nearest_obstacle_m.borrow();
                    if nearest_m < current_nearest {
                        let _ = bus_tof.nearest_obstacle_m.send(nearest_m);
                        let _ = bus_tof.nearest_obstacle_angle_rad.send(nearest_angle_rad);
                    }
                }
                Err(e) => tracing::error!("ToF error: {e}"),
            }
        }
    });

    // Depth MJPEG server (port+1 = 8081, mirrors sim setup).
    if cfg.hal.camera.stream_enabled {
        let depth_tx = bus.vision_depth.clone();
        let depth_port = cfg.hal.camera.stream_port + 1;
        tokio::spawn(hal::mjpeg::run_depth_server(depth_port, depth_tx));
        info!("Depth MJPEG stream → http://0.0.0.0:{depth_port}/");
    }

    // ── SLAM task (IMU-only, runs independently of depth inference) ──────────
    // Kept separate so that the 147ms depth inference does not starve IMU
    // integration: heading would freeze during every inference frame if both
    // were in the same select! loop.
    let bus_slam        = Arc::clone(&bus);
    let mut rx_imu      = bus.imu_raw.subscribe();
    let rx_eff_vel      = bus.effective_cmd_vel.subscribe();
    let slam_handle = tokio::spawn(async move {
        info!("SLAM task started (IMU dead-reckoning + motor feedforward)");
        loop {
            match rx_imu.recv().await {
                Ok(sample) => {
                    slam.set_velocity(&rx_eff_vel.borrow(), vel_scale);
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
                                let (near_m, near_cam_rad) = nearest_in_fov(&scan, 50f32.to_radians());
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
                                // Publish MiDaS nearest obstacle too: ToF covers ±20° forward
                                // only, MiDaS provides wider FOV coverage for the safety channels.
                                // Rolling-min (near_m_min) prevents single noisy far readings
                                // from causing false clears.
                                let _ = bus_perc.nearest_obstacle_m.send(near_m_min);
                                // Reset angle to 0 when clear so the orange dot doesn't
                                // float at a stale angle when nearest_m returns to 3.0m.
                                let publish_angle = if near_m_min >= 3.0 { 0.0 } else { near_angle_rad };
                                let _ = bus_perc.nearest_obstacle_angle_rad.send(publish_angle);
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
        const GOAL_BLACKLIST_SECS: u64 = 30;
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
        let mut no_frontier_streak: u32 = 0;
        // Counts how many times the "pre-arm guard" (cells < 10 000) has cleared the
        // blacklist.  If this fires repeatedly without the robot making progress, the
        // robot is physically isolated from all remaining frontiers (tight passages,
        // US livelock).  After MAX_STUCK_RESETS, give up and stop navigation.
        const MAX_STUCK_RESETS: u32 = 3;
        let mut stuck_resets: u32 = 0;
        loop {
            let Some(choice) = plan_rx.recv().await else { break };
            let now = Instant::now();
            goal_blacklist.retain(|(_, exp)| *exp > now);
            let bl: Vec<[f32; 2]> = goal_blacklist.iter().map(|(g, _)| *g).collect();

            let pose = *rx_pose_p.borrow_and_update();
            let reach = {
                let m = mapper_plan.read().await;
                let start_cell = m.world_to_cell(pose.x_m, pose.y_m);
                reachable_set(&m, start_cell, 5, 50_000)
            };
            let raw_frontiers: Vec<core_types::Frontier> = {
                let m = mapper_plan.read().await;
                m.last_frontiers.clone()
            };
            let annotated = annotate_frontiers(&raw_frontiers, &reach, &bl, &[], last_goal, &pose);
            let _ = bus_plan.map_frontier_annotations.send(annotated);
            let maybe_goal = {
                let m = mapper_plan.read().await;
                select_frontier_goal(&m, &choice, &pose, &bl, Some(&reach), last_goal)
            };
            if let Some(goal) = maybe_goal {
                no_frontier_streak = 0;
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
                // 2 s is enough to protect a path in flight; goals are reached in
                // 2-7 s so the window expires shortly after arrival.
                // Reduced from 3 s: the 3-second "frontier update delay" observed in the UI
                // was this window — the goal annotation doesn't change until PATH_COMMIT_S
                // expires after the previous goal was reached.
                const PATH_COMMIT_S: u64 = 2;
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
                        .or_else(|| planner.plan_with_clearance(&m, &pose, goal, 3))
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
                no_frontier_streak += 1;
                const EXPLORE_DONE_STREAK: u32 = 5;
                if no_frontier_streak >= EXPLORE_DONE_STREAK {
                    let cells = mapper_plan.read().await.grid.len();
                    const MIN_CELLS_FOR_DONE: usize = 10_000;
                    if cells >= MIN_CELLS_FOR_DONE {
                        info!(no_frontier_streak, cells, "Exploration complete — no reachable frontiers remaining");
                        break; // motor watchdog stops the robot
                    }
                    stuck_resets += 1;
                    if stuck_resets >= MAX_STUCK_RESETS {
                        warn!(stuck_resets, cells,
                            "Planning: robot isolated from all frontiers — stopping navigation");
                        break; // motor watchdog stops the robot
                    }
                    warn!(no_frontier_streak, cells, stuck_resets,
                        "Planning: frontier streak reset (cells < {MIN_CELLS_FOR_DONE}, likely pre-arm)");
                    no_frontier_streak = 0;
                    goal_blacklist.clear();
                }
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
    // ToF reactive avoidance: 0.70 m slow zone, 0.25 m hard stop.
    // The ToF task publishes to nearest_obstacle_m at up to 60 Hz (much faster than
    // MiDaS at ~3 Hz), giving the control task clean metric readings for speed scaling.
    // US (70 cm slow, 30 cm stop) provides an independent forward-facing interlock.
    let min_vx = (min_motor_duty as f32 / max_motor_duty as f32) * control::MAX_VX;
    let ctrl_handle = spawn_control_task(Arc::clone(&bus), control_path_rx, 0.70, 0.25, episode_dummy_rx, min_vx);

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
    let mut manual_rx_m   = bus.manual_cmd_vel.subscribe();
    let bus_motor         = Arc::clone(&bus);
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
                    let _ = bus_motor.effective_cmd_vel.send(CmdVel::default());
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
                        let _ = bus_motor.effective_cmd_vel.send(cmd_vel);
                        let cmd = cmdvel_to_motor(cmd_vel, max_motor_duty, min_motor_duty);
                        if let Err(e) = motor.send_command(cmd).await {
                            error!("Motor error (ctrl): {e}");
                        }
                    } else if matches!(*rx_exec_m.borrow(), ExecutiveState::ManualDrive) {
                        // ManualDrive: control task CmdVels are stale autonomous commands.
                        // Discard silently — zeroing here would clobber the active manual command.
                    } else {
                        // Not armed or safety stop active — zero motors and log once per stretch.
                        if !watchdog_logged {
                            warn!(armed, safe, "Motor task: CmdVel suppressed — zeroing motors");
                            watchdog_logged = true;
                        }
                        let _ = bus_motor.effective_cmd_vel.send(CmdVel::default());
                        if let Err(e) = motor.send_command(MotorCommand::stop(0)).await {
                            error!("Motor suppressed-stop failed: {e}");
                        }
                    }
                }
                // Priority 4: manual drive velocity from UI bridge.
                Ok(()) = manual_rx_m.changed() => {
                    if matches!(*rx_exec_m.borrow(), ExecutiveState::ManualDrive) {
                        let vel = *manual_rx_m.borrow_and_update();
                        watchdog_logged = false;
                        let _ = bus_motor.effective_cmd_vel.send(vel);
                        let cmd = cmdvel_to_motor(vel, max_motor_duty, min_motor_duty);
                        if let Err(e) = motor.send_command(cmd).await {
                            error!("Motor error (manual): {e}");
                        }
                    }
                }
                // Watchdog: no command for MOTOR_WATCHDOG_MS → zero all motors.
                // Uses send_command (not emergency_stop) — this is normal idle
                // keepalive behaviour, not a safety event.
                // Log only on first occurrence per idle stretch to avoid spam.
                () = tokio::time::sleep(watchdog) => {
                    let state = rx_exec_m.borrow().clone();
                    if matches!(state, ExecutiveState::ManualDrive) {
                        // Re-send held velocity so keys feel responsive.
                        let vel = *manual_rx_m.borrow();
                        let _ = bus_motor.effective_cmd_vel.send(vel);
                        let cmd = cmdvel_to_motor(vel, max_motor_duty, min_motor_duty);
                        if let Err(e) = motor.send_command(cmd).await {
                            error!("Motor watchdog manual error: {e}");
                        }
                    } else {
                        if !watchdog_logged {
                            let exploring = matches!(
                                state,
                                ExecutiveState::Exploring | ExecutiveState::Recovering
                            );
                            if exploring {
                                tracing::debug!("Motor watchdog: idle gap during exploration");
                            } else {
                                warn!("Motor watchdog: no command for {MOTOR_WATCHDOG_MS}ms — zeroing motors");
                            }
                            watchdog_logged = true;
                        }
                        let _ = bus_motor.effective_cmd_vel.send(CmdVel::default());
                        if let Err(e) = motor.send_command(MotorCommand::stop(0)).await {
                            error!("Motor watchdog stop failed: {e}");
                        }
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
    let rx_exec_g = bus_gimbal.executive_state.subscribe();
    let rx_nearest_g     = bus_gimbal.nearest_obstacle_m.subscribe();
    let rx_near_angle_g  = bus_gimbal.nearest_obstacle_angle_rad.subscribe();
    let gimbal_handle = tokio::spawn(async move {
        info!("Gimbal task started");
        if let Err(e) = gimbal.set_angles(0.0, tilt_home).await {
            warn!("Gimbal home failed: {e}");
        }
        let gimbal_t0 = std::time::Instant::now();
        let mut last_bias = 0.0f32;
        let mut interval = tokio::time::interval(std::time::Duration::from_millis(100));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        // Threshold below which we track the obstacle instead of sweeping open.
        const GIMBAL_DANGER_M: f32 = 0.60; // m — track obstacles within this distance
        loop {
            interval.tick().await;
            let mut closed = false;
            loop {
                match rx_depth_g.try_recv() {
                    Ok(d) => { last_bias = depth_open_bias(&d); }
                    Err(tokio::sync::broadcast::error::TryRecvError::Empty)      => break,
                    Err(tokio::sync::broadcast::error::TryRecvError::Lagged(_))  => continue,
                    Err(tokio::sync::broadcast::error::TryRecvError::Closed)     => { closed = true; break; }
                }
            }
            if closed { break; }
            // When a nearby obstacle is within GIMBAL_DANGER_M, point camera toward
            // it for an accurate reading instead of sweeping the open side.
            let nearest_m   = *rx_nearest_g.borrow();
            let near_ang_deg = rx_near_angle_g.borrow().to_degrees();
            // When a nearby obstacle is within GIMBAL_DANGER_M, point ToF toward it so it
            // gets an accurate metric reading.  Exception: if the obstacle is primarily
            // to the side (> 30°), look forward instead — the forward wall is invisible
            // to ToF when the pan tracks a side obstacle, and pseudo-lidar covers ±55°
            // anyway.  This prevents crashes when obstacles are on both sides at corners.
            let danger_angle = if nearest_m < GIMBAL_DANGER_M {
                if near_ang_deg.abs() > 30.0 { Some(0.0_f32) } else { Some(near_ang_deg) }
            } else {
                None
            };
            // In manual mode lock pan to 0 so the camera points straight ahead.
            let new_pan = if matches!(*rx_exec_g.borrow(), ExecutiveState::ManualDrive) {
                0.0
            } else {
                let t_s = gimbal_t0.elapsed().as_secs_f32();
                let (cur_pan, _) = gimbal.angles();
                gimbal_pan_target(last_bias, t_s, cur_pan, danger_angle)
            };
            if let Err(e) = gimbal.set_pan(new_pan).await {
                warn!("Gimbal pan error: {e}");
            }
            // Re-send tilt every tick — servo can drift under gravity if not refreshed.
            if let Err(e) = gimbal.set_tilt(tilt_home).await {
                warn!("Gimbal tilt error: {e}");
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
    drop(tof_handle);
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
///
/// Dead-band: wheels commanded to a non-zero duty below `min_duty` are snapped
/// up to `min_duty` (preserving sign).  Below this threshold the physical motors
/// stall; snapping ensures the robot either moves at minimum speed or stops —
/// matching real hardware behaviour in both real and sim modes.
fn cmdvel_to_motor(cmd: CmdVel, max_duty: i8, min_duty: i8) -> MotorCommand {
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

    // Apply dead-band: snap non-zero sub-threshold duties to ±min_duty.
    let db = |d: f32| -> i8 {
        let v = (d * scale) as i8;
        if v == 0 { 0 } else if v.abs() < min_duty { min_duty * v.signum() } else { v }
    };

    MotorCommand {
        t_ms: cmd.t_ms,
        fl: db(fl),
        fr: db(fr),
        rl: db(rl),
        rr: db(rr),
    }
}

// ── Spawn helper functions (for selected tasks; most spawned from `tasks` module) ───


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
    reach: Option<&std::collections::HashSet<(i32, i32)>>,
    committed_goal: Option<[f32; 2]>,
) -> Option<[f32; 2]> {
    const MIN_GOAL_DIST: f32 = 0.75;    // metres — push robot toward meaningfully distant frontiers
    const BLACKLIST_RADIUS: f32 = 0.60; // metres — skip frontiers near recently-tried goals

    let frontiers = &mapper.last_frontiers;
    if frontiers.is_empty() {
        return None;
    }

    // Sticky-goal: if we already have an active goal and it still exists as a
    // valid, reachable, non-blacklisted frontier AND the robot hasn't yet reached
    // the goal area, return it immediately.
    // This prevents heading-score instability from flipping the selected frontier
    // every planning tick as the robot rotates during pure-pursuit navigation.
    // We only pick a new goal when the current one has been explored (frontier
    // disappeared), blacklisted, or the robot is now within MIN_GOAL_DIST of it.
    if let Some(cg) = committed_goal {
        // If robot has already reached the committed goal area, don't stick.
        let robot_dist_to_cg = {
            let dx = cg[0] - pose.x_m;
            let dy = cg[1] - pose.y_m;
            (dx * dx + dy * dy).sqrt()
        };
        if robot_dist_to_cg >= MIN_GOAL_DIST {
            let still_valid = frontiers.iter().any(|f| {
                let dx = f.centroid_x_m - cg[0];
                let dy = f.centroid_y_m - cg[1];
                let d_to_committed = (dx * dx + dy * dy).sqrt();
                // Accept if within 1 m of the committed position (frontier centroids
                // shift slightly as more cells are observed; 1 m is generous enough
                // to track the same physical frontier without accidentally sticking
                // to a different one).
                d_to_committed < 1.0
                    && f.size_cells >= 5
                    && !blacklist.iter().any(|b| {
                        let bx = f.centroid_x_m - b[0];
                        let by = f.centroid_y_m - b[1];
                        (bx * bx + by * by).sqrt() < BLACKLIST_RADIUS
                    })
                    && {
                        // Reachability check (same as main filter below).
                        let (cx, cy) = mapper.world_to_cell(f.centroid_x_m, f.centroid_y_m);
                        mapper.grid.is_free(cx, cy)
                    }
                    && {
                        let Some(r) = reach else { return true; };
                        let (fcx, fcy) = mapper.world_to_cell(f.centroid_x_m, f.centroid_y_m);
                        (-8_i32..=8).any(|dx2| (-8_i32..=8).any(|dy2| r.contains(&(fcx + dx2, fcy + dy2))))
                    }
            });
            if still_valid {
                return Some(cg);
            }
        }
    }

    let dist = |f: &core_types::Frontier| -> f32 {
        let dx = f.centroid_x_m - pose.x_m;
        let dy = f.centroid_y_m - pose.y_m;
        (dx * dx + dy * dy).sqrt()
    };

    // Heading-weighted score for Nearest selection.
    // A frontier in the robot's current heading direction gets up to 50% discount,
    // so the robot keeps moving forward rather than turning to a slightly-closer
    // frontier off to the side.  Dividing by sqrt(size_cells) makes larger frontiers
    // look effectively closer, biasing exploration away from small wall slivers.
    let heading_score = |f: &core_types::Frontier| -> f32 {
        const HEADING_WEIGHT: f32 = 0.5;
        let dx = f.centroid_x_m - pose.x_m;
        let dy = f.centroid_y_m - pose.y_m;
        let d = (dx * dx + dy * dy).sqrt().max(0.001);
        let angle_to = dy.atan2(dx);
        let alignment = (angle_to - pose.theta_rad).cos().max(0.0);
        let size_weight = (f.size_cells as f32).sqrt().max(1.0);
        d * (1.0 - HEADING_WEIGHT * alignment) / size_weight
    };

    let blacklisted = |f: &core_types::Frontier| -> bool {
        blacklist.iter().any(|b| {
            let dx = f.centroid_x_m - b[0];
            let dy = f.centroid_y_m - b[1];
            (dx * dx + dy * dy).sqrt() < BLACKLIST_RADIUS
        })
    };

    // Reject frontiers whose centroid falls in unknown or obstacle space.
    // This filters edge frontiers whose centroids land beyond the arena wall
    // (the centroid averages free + unknown cells, pushing it outside the map).
    // A* would exhaust MAX_NODES on every such goal — they are never reachable.
    let reachable_centroid = |f: &core_types::Frontier| -> bool {
        let (cx, cy) = mapper.world_to_cell(f.centroid_x_m, f.centroid_y_m);
        mapper.grid.is_free(cx, cy)
    };

    // BFS reachability pre-screen: reject frontiers that are topologically
    // disconnected from the robot (on the far side of a wall, in a narrow gap the
    // clearance zone blocks, etc.).  Any frontier whose centroid and an 8-cell
    // radius around it has no cell in `reach` is skipped — no A* call is made.
    // When `reach` is None (real-robot path), the check is skipped.
    let reachable_frontier = |f: &core_types::Frontier| -> bool {
        let Some(r) = reach else { return true; };
        let (fcx, fcy) = mapper.world_to_cell(f.centroid_x_m, f.centroid_y_m);
        for dx in -8_i32..=8 {
            for dy in -8_i32..=8 {
                if r.contains(&(fcx + dx, fcy + dy)) {
                    return true;
                }
            }
        }
        false
    };

    // Ignore tiny frontier clusters (wall corner slivers the robot can't fit into).
    // Clusters this small are unreachable at any clearance level; attempting them
    // burns A* budget and triggers the escape/blacklist cycle unnecessarily.
    const MIN_FRONTIER_SIZE: u32 = 5;
    let large_enough = |f: &&core_types::Frontier| f.size_cells >= MIN_FRONTIER_SIZE;

    // Primary: far enough away, not blacklisted, centroid in free space, BFS-reachable.
    // Fallback 1: far enough away and centroid in free space (ignoring blacklist).
    // Fallback 2: any large-enough frontier (blacklist and distance both waived).
    let candidates: Vec<_> = frontiers.iter()
        .filter(|f| large_enough(f) && dist(f) >= MIN_GOAL_DIST && !blacklisted(f) && reachable_centroid(f) && reachable_frontier(f))
        .collect();
    let fallback1: Vec<_> = frontiers.iter()
        .filter(|f| large_enough(f) && dist(f) >= MIN_GOAL_DIST && !blacklisted(f) && reachable_centroid(f) && reachable_frontier(f))
        .collect();

    let pool: &[&core_types::Frontier] = if !candidates.is_empty() {
        &candidates
    } else if !fallback1.is_empty() {
        &fallback1
    } else {
        // No non-blacklisted, distance-filtered frontier found.
        // Still respect the blacklist — returning None here lets no_frontier_streak
        // accumulate and triggers the exploration-complete / escape path.  The old
        // blacklist-ignoring fallback was causing the robot to re-select already-
        // failed wall-adjacent frontiers indefinitely instead of moving on.
        let fallback2: Vec<_> = frontiers.iter()
            .filter(|f| large_enough(f) && !blacklisted(f) && reachable_centroid(f) && reachable_frontier(f))
            .collect();
        return fallback2.iter().copied()
            .min_by(|a, b| dist(a).partial_cmp(&dist(b)).unwrap())
            .map(|f| [f.centroid_x_m, f.centroid_y_m])
            .or_else(|| frontiers.iter()
                .filter(|f| large_enough(f) && !blacklisted(f))
                .min_by(|a, b| dist(a).partial_cmp(&dist(b)).unwrap())
                .map(|f| [f.centroid_x_m, f.centroid_y_m]));
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


/// Annotate a frontier list with planner-state status codes for the UI display.
///
/// Called by both the real-robot and sim planning tasks.  Accepts separate soft
/// and hard blacklist slices so callers can pass `&[]` for absent lists.
fn annotate_frontiers(
    raw_frontiers: &[core_types::Frontier],
    reach: &std::collections::HashSet<(i32, i32)>,
    soft_bl: &[[f32; 2]],
    hard_bl: &[[f32; 2]],
    last_goal: Option<[f32; 2]>,
    pose: &core_types::Pose2D,
) -> Vec<core_types::Frontier> {
    use core_types::frontier_status as fs;
    const MIN_FRONTIER_SIZE: u32 = 5;
    const MIN_GOAL_DIST:     f32 = 0.75;
    const BLACKLIST_RADIUS:  f32 = 0.60;
    const RES:               f32 = 0.05;

    raw_frontiers.iter().map(|f| {
        let dist = {
            let dx = f.centroid_x_m - pose.x_m;
            let dy = f.centroid_y_m - pose.y_m;
            (dx * dx + dy * dy).sqrt()
        };
        let is_goal = last_goal.map_or(false, |g| {
            (f.centroid_x_m - g[0]).abs() < 0.15
                && (f.centroid_y_m - g[1]).abs() < 0.15
        });
        let bl_min = |bl: &[[f32; 2]]| -> f32 {
            bl.iter().map(|b| {
                let bx = f.centroid_x_m - b[0];
                let by = f.centroid_y_m - b[1];
                (bx * bx + by * by).sqrt()
            }).fold(f32::INFINITY, f32::min)
        };
        let fcx = (f.centroid_x_m / RES).floor() as i32;
        let fcy = (f.centroid_y_m / RES).floor() as i32;
        let reachable = (-8_i32..=8).any(|ddx| {
            (-8_i32..=8).any(|ddy| reach.contains(&(fcx + ddx, fcy + ddy)))
        });
        let status = if is_goal {
            fs::CURRENT_GOAL
        } else if f.size_cells < MIN_FRONTIER_SIZE {
            fs::TOO_SMALL
        } else if !reachable {
            fs::UNREACHABLE
        } else if bl_min(hard_bl) < BLACKLIST_RADIUS {
            fs::HARD_BLACKLISTED
        } else if bl_min(soft_bl) < BLACKLIST_RADIUS {
            fs::SOFT_BLACKLISTED
        } else if dist < MIN_GOAL_DIST {
            fs::TOO_CLOSE
        } else {
            fs::NORMAL
        };
        core_types::Frontier { status, ..*f }
    }).collect()
}

//
// Reads from `bus.ultrasonic`, evaluates each reading against `emergency_stop_cm`,
// latches EmergencyStop for 5 s on a trip, and spawns an escape reverse maneuver.
// Used in both `run_robot_mode` (via real US readings) and `run_sim_mode` (via
// synthetic US readings published by `sim_publish`).


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

// ── Sim run statistics ────────────────────────────────────────────────────────

struct RoomRecord {
    reason:     &'static str,   // "complete" | "timeout" | "cascade" | "isolated" | "reset"
    duration_s: f64,
    cells:      u32,            // explored_cells at room end
    crashes:    u32,            // collisions during this room
}

struct SimRunStats {
    run_start:       std::time::Instant,
    rooms:           Vec<RoomRecord>,
    astar_ok:        u32,
    astar_fail:      u32,
    cascades:        u32,
    hard_bl_pushes:  u32,
    /// Set by planning task before sending explore_done so the sim tick task
    /// can record the correct room exit reason.
    next_room_reason: Option<&'static str>,
}

fn print_sim_summary(stats: &SimRunStats, bus: &bus::Bus) {
    let elapsed = stats.run_start.elapsed();
    let h = elapsed.as_secs() / 3600;
    let m = (elapsed.as_secs() % 3600) / 60;
    let s = elapsed.as_secs() % 60;

    let total_crashes   = *bus.collision_count.borrow();
    let total_near_miss = *bus.estop_count.borrow();
    let total_episodes  = *bus.episode_count.borrow();
    let hours           = elapsed.as_secs_f64() / 3600.0;

    let rooms       = &stats.rooms;
    let n_complete  = rooms.iter().filter(|r| r.reason == "complete").count();
    let n_timeout   = rooms.iter().filter(|r| r.reason == "timeout").count();
    let n_cascade   = rooms.iter().filter(|r| r.reason == "cascade").count();
    let n_isolated  = rooms.iter().filter(|r| r.reason == "isolated").count();
    let n_bl_stuck  = rooms.iter().filter(|r| r.reason == "hard_bl_isolated").count();
    let n_reset     = rooms.iter().filter(|r| r.reason == "reset").count();

    let complete_times: Vec<f64> = rooms.iter()
        .filter(|r| r.reason == "complete").map(|r| r.duration_s).collect();
    let avg_clear    = if complete_times.is_empty() { 0.0 }
        else { complete_times.iter().sum::<f64>() / complete_times.len() as f64 };
    let min_clear    = complete_times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_clear    = complete_times.iter().cloned().fold(0.0_f64, f64::max);

    let total_cells  = rooms.iter().map(|r| r.cells).sum::<u32>();
    let avg_cells    = if rooms.is_empty() { 0 } else { total_cells / rooms.len() as u32 };
    let max_cells    = rooms.iter().map(|r| r.cells).max().unwrap_or(0);

    let clean_rooms  = rooms.iter().filter(|r| r.crashes == 0).count();
    let worst_crash  = rooms.iter().map(|r| r.crashes).max().unwrap_or(0);
    let avg_crashes  = if rooms.is_empty() { 0.0 }
        else { total_crashes as f64 / rooms.len() as f64 };

    let astar_total  = stats.astar_ok + stats.astar_fail;
    let astar_pct    = if astar_total == 0 { 100.0 }
        else { 100.0 * stats.astar_ok as f64 / astar_total as f64 };

    // Box is W=52 chars wide between the two border chars (total line = 54).
    // `line(s)` pads `s` to exactly W chars so borders always align.
    const W: usize = 52;
    let sep  = "═".repeat(W);
    let line = |s: String| { eprintln!("║{:<W$}║", s, W = W); };
    let sect = |s: &str|   { eprintln!("╠{sep}╣"); line(format!("  {s}")); };
    let kv   = |label: &str, val: String| {
        line(format!("    {:<26}{:>20}", format!("{label} :"), val));
    };

    eprintln!("\n╔{sep}╗");
    line(format!("{:^W$}", "SIM RUN SUMMARY", W = W));
    eprintln!("╠{sep}╣");
    line(format!("  {:<26}{:>20}", "Run time :", format!("{}h {:02}m {:02}s", h, m, s)));
    line(format!("  {:<26}{:>20}", "Rooms entered :", total_episodes));

    sect("ROOM OUTCOMES");
    kv("Explored (complete)", format!("{}", n_complete));
    kv("Timed out",           format!("{}", n_timeout));
    kv("Phys. blocked exit",  format!("{}", n_bl_stuck));
    kv("Cascade crash exit",  format!("{}", n_cascade));
    kv("Isolated (A* stuck)", format!("{}", n_isolated));
    kv("Episode reset",       format!("{}", n_reset));

    sect("EXPLORATION TIMING  (complete rooms only)");
    if !complete_times.is_empty() {
        kv("Avg clear time", format!("{:.1}s", avg_clear));
        kv("Fastest",        format!("{:.1}s", min_clear));
        kv("Slowest",        format!("{:.1}s", max_clear));
    } else {
        line("    (no rooms fully explored)".to_string());
    }

    sect("MAPPING");
    kv("Total cells explored", format!("{}", total_cells));
    kv("Avg cells / room",     format!("{}", avg_cells));
    kv("Best room (cells)",    format!("{}", max_cells));

    sect("SAFETY");
    kv("Collisions (crashes)", format!("{}  ({:.1}/hr)", total_crashes,
        if hours > 0.0 { total_crashes as f64 / hours } else { 0.0 }));
    kv("Near-misses (US)",     format!("{}  ({:.1}/hr)", total_near_miss,
        if hours > 0.0 { total_near_miss as f64 / hours } else { 0.0 }));
    kv("Avg crashes / room",   format!("{:.1}", avg_crashes));
    kv("Worst room (crashes)", format!("{}", worst_crash));
    kv("Clean rooms (0 crash)",format!("{}", clean_rooms));

    sect("PLANNING");
    kv("A* paths found",      format!("{}  ({:.1}%)", stats.astar_ok, astar_pct));
    kv("A* failures",         format!("{}", stats.astar_fail));
    kv("Cascade escapes",     format!("{}", stats.cascades));
    kv("Hard-blacklist hits", format!("{}", stats.hard_bl_pushes));

    eprintln!("╚{sep}╝\n");
}

async fn run_sim_mode(
    bus:              Arc<bus::Bus>,
    cfg:              config::RobotConfig,
    plan_frontier_rx: tokio::sync::mpsc::Receiver<core_types::FrontierChoice>,
    control_path_rx:  tokio::sync::mpsc::Receiver<core_types::Path>,
    motor_cmd_rx:     tokio::sync::mpsc::Receiver<core_types::MotorCommand>,
    cmdvel_rx:        tokio::sync::mpsc::Receiver<core_types::CmdVel>,
) -> anyhow::Result<()> {
    use sim_fast::RoomKind;

    let sim_stats = std::sync::Arc::new(std::sync::Mutex::new(SimRunStats {
        run_start:        std::time::Instant::now(),
        rooms:            Vec::new(),
        astar_ok:         0,
        astar_fail:       0,
        cascades:         0,
        hard_bl_pushes:   0,
        next_room_reason: None,
    }));

    // Usage: `robot sim [explore] [single-box] [seed]`
    // `explore` keeps the map and maze across collisions (single infinite episode).
    // `single-box` uses a single centered 1m×1m obstacle instead of random maze.
    // Without `explore`, each collision/timeout resets the maze and map (training mode).
    let args: Vec<String> = std::env::args().skip(2).collect();
    let explore_mode  = args.iter().any(|a| a == "explore");
    let single_box    = args.iter().any(|a| a == "single-box");
    let use_tof       = args.iter().any(|a| a == "tof");
    let seed: u64 = args.iter()
        .filter_map(|a| a.parse().ok())
        .next()
        .unwrap_or(cfg.sim.seed);
    info!(seed, explore = explore_mode, single_box, use_tof, "Sim mode starting");

    // ── Signal handler — registered FIRST before any spawning ─────────────────
    use tokio::signal::unix::{signal, SignalKind};
    let mut sigusr1_stream = signal(SignalKind::user_defined1())
        .map_err(|e| anyhow::anyhow!("SIGUSR1 handler: {e}"))?;

    // ── Sim HAL objects ───────────────────────────────────────────────────────
    // Each HAL trait impl is backed by SimState: a shared FastSim + watch channels
    // so the sim tick task drives physics while all HAL readers wait on step_rx.
    let room_kind = if single_box {
        RoomKind::SingleBox { side_m: 1.0 }
    } else if cfg.sim.obstacles == 0 {
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
    let min_motor_duty = cfg.robot.min_speed as i8;
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
    info!("Sim perception pipeline initialised");

    // ── Mapper ────────────────────────────────────────────────────────────────
    // Pre-seed room boundary walls so frontiers never form at room edges and
    // the robot doesn't drive into unknown walls during frontier exploration.
    // The mapper is seeded with the correct wall grid inside spawn_sim_tick_task,
    // immediately after sim.reset() generates the final maze.  Seeding here
    // (before reset) would use a different maze (FastSim::new also calls
    // generate_maze, advancing the RNG, so reset() produces a different layout).
    let mapper = Arc::new(tokio::sync::RwLock::new(Mapper::new()));
    info!("Occupancy grid mapper initialised (will be seeded after sim reset)");

    // ── Episode reset notification ────────────────────────────────────────────
    let (episode_reset_tx, episode_reset_rx) =
        tokio::sync::watch::channel(0u32);
    let (force_respawn_tx, force_respawn_rx) = tokio::sync::mpsc::channel::<()>(1);
    // Fired by the planning task when exploration is complete (all frontiers exhausted).
    // The sim tick task receives this and starts a new room.
    let (explore_done_tx, explore_done_rx)   = tokio::sync::mpsc::channel::<()>(1);

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
                        BridgeCommand::Arm    => { let _ = arm_tx_exec.send(()).await; }
                        BridgeCommand::Stop   => { let _ = exec.transition(ExecutiveState::Idle); }
                        BridgeCommand::Manual => {
                            // Allow from any state: stop autonomous mode first if needed.
                            if !matches!(exec.state(),
                                ExecutiveState::Idle | ExecutiveState::ManualDrive)
                            {
                                let _ = exec.transition(ExecutiveState::Idle);
                            }
                            if matches!(exec.state(), ExecutiveState::Idle) {
                                match exec.transition(ExecutiveState::ManualDrive) {
                                    Ok(())  => info!("Sim executive: manual drive activated"),
                                    Err(e)  => warn!("Manual transition rejected: {e}"),
                                }
                            }
                        }
                        BridgeCommand::Auto => {
                            if matches!(exec.state(), ExecutiveState::ManualDrive) {
                                match exec.transition(ExecutiveState::Idle) {
                                    Ok(())  => info!("Sim executive: manual drive deactivated → Idle"),
                                    Err(e)  => warn!("Auto transition rejected: {e}"),
                                }
                            }
                        }
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

    // ── Sim MJPEG server ──────────────────────────────────────────────────────
    if cfg.hal.camera.stream_enabled {
        let port      = cfg.hal.camera.stream_port;
        let frame_tx  = bus.camera_frame_raw.clone();
        tokio::spawn(hal::mjpeg::run_server(port, frame_tx));
        info!("Sim MJPEG stream → http://0.0.0.0:{port}/");
        // Depth map stream on port+1 (32×32 f32, scaled to 256×256 JPEG).
        let depth_tx = bus.vision_depth.clone();
        tokio::spawn(hal::mjpeg::run_depth_server(port + 1, depth_tx));
        info!("Depth MJPEG stream → http://0.0.0.0:{}/", port + 1);
    }

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
                                // Only publish depth for UI/gimbal pan sweep.
                                // vision_pseudo_lidar and nearest_obstacle_m come from the sim
                                // tick task (ground-truth geometry) so thresholds are metric.
                                let _ = bus_perc.vision_depth.send(Arc::new(depth));
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
    let rx_exec_g    = bus_gimbal.executive_state.subscribe();
    let rx_nearest_g    = bus_gimbal.nearest_obstacle_m.subscribe();
    let rx_near_angle_g = bus_gimbal.nearest_obstacle_angle_rad.subscribe();
    let sim_danger_m = cfg.sim.obstacle_stop_m * 4.0; // 0.60 m
    tokio::spawn(async move {
        info!("Sim gimbal task started");
        if let Err(e) = gimbal.set_angles(0.0, tilt_home).await {
            warn!("Sim gimbal home failed: {e}");
        }
        let gimbal_t0 = std::time::Instant::now();
        let mut last_bias = 0.0f32;
        let mut interval = tokio::time::interval(std::time::Duration::from_millis(100));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            interval.tick().await;
            let mut closed = false;
            loop {
                match rx_depth_g.try_recv() {
                    Ok(d) => { last_bias = depth_open_bias(&d); }
                    Err(tokio::sync::broadcast::error::TryRecvError::Empty)      => break,
                    Err(tokio::sync::broadcast::error::TryRecvError::Lagged(_))  => continue,
                    Err(tokio::sync::broadcast::error::TryRecvError::Closed)     => { closed = true; break; }
                }
            }
            if closed { break; }
            let nearest_m    = *rx_nearest_g.borrow();
            let near_ang_deg = rx_near_angle_g.borrow().to_degrees();
            // Side obstacles (> 30°) → look forward instead of tracking them.
            // Pseudo-lidar covers ±55°; forward clearance is the blind spot when
            // pan tracks a side wall at a corner.
            let danger_angle = if nearest_m < sim_danger_m {
                if near_ang_deg.abs() > 30.0 { Some(0.0_f32) } else { Some(near_ang_deg) }
            } else {
                None
            };
            // In manual mode lock pan to 0 so the camera points straight ahead.
            let new_pan = if matches!(*rx_exec_g.borrow(), ExecutiveState::ManualDrive) {
                0.0
            } else {
                let t_s = gimbal_t0.elapsed().as_secs_f32();
                let (cur_pan, _) = gimbal.angles();
                gimbal_pan_target(last_bias, t_s, cur_pan, danger_angle)
            };
            if let Err(e) = gimbal.set_pan(new_pan).await {
                warn!("Sim gimbal pan error: {e}");
            }
            // Re-send tilt every tick — keeps it at home if servo drifts.
            if let Err(e) = gimbal.set_tilt(tilt_home).await {
                warn!("Sim gimbal tilt error: {e}");
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
    let mut motor_cmdvel_rx  = cmdvel_rx;
    let mut motor_cmd_rx_m   = motor_cmd_rx;
    let rx_exec_m    = bus.executive_state.subscribe();
    let rx_safety_m  = bus.safety_state.subscribe();
    let mut manual_rx = bus.manual_cmd_vel.subscribe();
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
                        let cmd = cmdvel_to_motor(cmd_vel, max_motor_duty, min_motor_duty);
                        if let Err(e) = motor.send_command(cmd).await {
                            error!("Sim motor error (ctrl): {e}");
                        }
                    } else if matches!(*rx_exec_m.borrow(), ExecutiveState::ManualDrive) {
                        // ManualDrive: control task CmdVels are stale autonomous commands.
                        // Discard silently — zeroing here would clobber the manual command
                        // that was just written to latest_motor this same 100 ms tick.
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
                Ok(()) = manual_rx.changed() => {
                    if matches!(*rx_exec_m.borrow(), ExecutiveState::ManualDrive) {
                        let vel = *manual_rx.borrow_and_update();
                        watchdog_logged = false;
                        let cmd = cmdvel_to_motor(vel, max_motor_duty, min_motor_duty);
                        if let Err(e) = motor.send_command(cmd).await {
                            error!("Sim motor error (manual): {e}");
                        }
                    }
                }
                () = tokio::time::sleep(watchdog) => {
                    let state = rx_exec_m.borrow().clone();
                    if matches!(state, ExecutiveState::ManualDrive) {
                        // In manual drive: re-send held velocity so keys feel responsive.
                        let vel = *manual_rx.borrow();
                        let cmd = cmdvel_to_motor(vel, max_motor_duty, min_motor_duty);
                        if let Err(e) = motor.send_command(cmd).await {
                            error!("Sim motor watchdog manual error: {e}");
                        }
                    } else {
                        if !watchdog_logged {
                            let exploring = matches!(
                                state,
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
    let _force_respawn_tx_plan = force_respawn_tx.clone();
    let explore_done_tx_plan = explore_done_tx.clone();
    let stats_plan = std::sync::Arc::clone(&sim_stats);
    tokio::spawn(async move {
        use tokio::time::Instant;
        const GOAL_BLACKLIST_SECS: u64 = 30;
        const MAX_CONSECUTIVE_FAILURES: u32 = 10;
        // After this many fruitless escape cycles (robot isolated from all frontiers),
        // force an episode reset rather than cycling indefinitely until MAX_STEPS.
        const MAX_ESCAPE_FAILURES: u32 = 3;
        info!("Sim planning task started (A*, 4-cell clearance)");
        let planner = AStarPlanner::new();
        let mut goal_blacklist: Vec<([f32; 2], Instant)> = Vec::new();
        // Hard blacklist: no-progress and collision-based entries.
        // NOT cleared on A*-saturation so the robot doesn't immediately retry
        // corners it physically cannot navigate through (crash 9 pattern: same
        // frontier re-selected 12+ times after each saturation clear).
        let mut hard_blacklist: Vec<([f32; 2], Instant)> = Vec::new();
        let mut last_goal: Option<[f32; 2]> = None;
        let mut last_path_sent_at: Option<Instant> = None;
        // Time we FIRST sent a path to the current goal (not reset on re-sends).
        // Used to detect goals the robot navigates toward indefinitely without
        // reaching — e.g. a frontier at the edge of a tight space that slows the
        // robot to near-zero before it arrives.  last_path_sent_at resets every
        // 1 s on re-plans, so it can never accumulate enough age to trip the
        // path_sent_and_expired check.  goal_first_sent_at only advances when the
        // goal changes, so it correctly measures total time spent on this goal.
        let mut goal_first_sent_at: Option<Instant> = None;
        let mut consecutive_failures: u32 = 0;
        let mut escape_failures: u32 = 0;
        // Counts how many times the escape trigger has fired (MAX_CONSECUTIVE_FAILURES
        // reached) regardless of whether escape succeeded.  When this saturates in
        // explore mode it means frontiers pass BFS pre-screen but A* cannot reach
        // any of them within MAX_NODES — the arena is effectively fully explored.
        let mut escape_cycles: u32 = 0;
        let mut no_frontier_streak: u32 = 0;
        let mut last_scan: Option<std::sync::Arc<core_types::PseudoLidarScan>> = None;
        let mut last_crash_plan: u32 = 0;
        // No-progress detector: if the robot hasn't moved 0.2 m in 6 s while pursuing
        // a goal, blacklist it immediately.  Catches late-exploration deadlock cycles
        // where the deadlock-backup spins the robot but the planner keeps re-sending
        // the same blocked frontier (wall-adjacent, US-stops the robot every approach).
        // The 6 s window is shorter than REPLAN_HOLD_S=15 s to break the spin cycle
        // quickly; it's longer than typical planning ticks (~1 s) to avoid false fires.
        let mut no_progress_pos: Option<([f32; 2], Instant)> = None;
        loop {
            let choice;
            tokio::select! {
                biased;
                Ok(()) = episode_rx_plan.changed() => {
                    let ep = *episode_rx_plan.borrow_and_update();
                    info!(ep, "Sim planning: episode reset — clearing state");
                    goal_blacklist.clear();
                    hard_blacklist.clear();
                    consecutive_failures = 0;
                    escape_failures = 0;
                    escape_cycles = 0;
                    no_frontier_streak = 0;
                    last_goal = None;
                    last_path_sent_at = None;
                    goal_first_sent_at = None;
                    // Sync to current collision_count so the crash detector doesn't
                    // fire spuriously on the first planning tick of the new episode
                    // (collision_count persists across rooms; resetting to 0 would
                    // trigger a false "crash detected" the first time the loop runs).
                    last_crash_plan = *bus_plan.collision_count.borrow();
                    no_progress_pos = None;
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
            hard_blacklist.retain(|(_, exp)| *exp > now);

            // Collision-based blacklist: if a crash occurred since the last planning cycle,
            // blacklist the goal we were pursuing so the new path routes around the crash area.
            {
                let crash_n = *bus_plan.collision_count.borrow();
                if crash_n > last_crash_plan {
                    last_crash_plan = crash_n;
                    if let Some(goal) = last_goal {
                        let exp = now + std::time::Duration::from_secs(GOAL_BLACKLIST_SECS * 3);
                        warn!(gx = goal[0], gy = goal[1], "Sim planning: crash — hard-blacklisting goal");
                        hard_blacklist.push((goal, exp));
                        { let mut s = stats_plan.lock().unwrap(); s.hard_bl_pushes += 1; }
                        last_goal = None;
                        last_path_sent_at = None;
                        goal_first_sent_at = None;
                        no_progress_pos = None;
                    }
                    // Also blacklist the robot's current position so the planner avoids
                    // routing back into the same physical crash area via a nearby frontier.
                    // BLACKLIST_RADIUS (0.60 m) provides a safe exclusion bubble.
                    let p = *rx_pose_p.borrow();
                    let crash_pos = [p.x_m, p.y_m];
                    let exp = now + std::time::Duration::from_secs(GOAL_BLACKLIST_SECS * 6);
                    warn!(rx = crash_pos[0], ry = crash_pos[1], "Sim planning: crash — hard-blacklisting robot position");
                    hard_blacklist.push((crash_pos, exp));
                    { let mut s = stats_plan.lock().unwrap(); s.hard_bl_pushes += 1; }

                    // Cascade detection: if a recent hard_blacklist entry is within
                    // CASCADE_RADIUS of this crash position, the robot is physically
                    // trapped and will crash indefinitely (deadlock backup can't escape).
                    // Force a new room immediately rather than burning 100s of crash cycles.
                    const CASCADE_RADIUS: f32 = 0.30;
                    let is_cascade = explore_mode && hard_blacklist.iter()
                        .filter(|(b, _)| *b != crash_pos)
                        .any(|(b, _)| {
                            let dx = crash_pos[0] - b[0];
                            let dy = crash_pos[1] - b[1];
                            (dx * dx + dy * dy).sqrt() < CASCADE_RADIUS
                        });
                    if is_cascade {
                        warn!(rx = crash_pos[0], ry = crash_pos[1], "Sim planning: crash cascade detected — forcing new room");
                        { let mut s = stats_plan.lock().unwrap(); s.cascades += 1; s.next_room_reason = Some("cascade"); }
                        let _ = explore_done_tx_plan.send(()).await;
                        while plan_rx.try_recv().is_ok() {}
                        let _ = episode_rx_plan.borrow_and_update();
                        let _ = episode_rx_plan.changed().await;
                        let _ = episode_rx_plan.borrow_and_update();
                        goal_blacklist.clear();
                        hard_blacklist.clear();
                        consecutive_failures = 0;
                        escape_cycles = 0;
                        escape_failures = 0;
                        no_frontier_streak = 0;
                        last_goal = None;
                        last_path_sent_at = None;
                        goal_first_sent_at = None;
                        no_progress_pos = None;
                        last_crash_plan = *bus_plan.collision_count.borrow();
                        while plan_rx.try_recv().is_ok() {}
                        continue;
                    }
                }
            }

            // BFS reachability pre-screen: compute the robot's reachable cell set once
            // per planning trigger (not per retry).  Used by select_frontier_goal to
            // skip frontiers that A* can never reach, avoiding MAX_NODES exhaustion.
            // clearance=5 matches the single A* level we attempt, so the BFS pre-screen
            // rejects exactly the frontiers A* will also reject at clearance=5.
            let pose_pre = *rx_pose_p.borrow_and_update();
            let reach = {
                let m = mapper_plan.read().await;
                let start_cell = m.world_to_cell(pose_pre.x_m, pose_pre.y_m);
                reachable_set(&m, start_cell, 5, 50_000)
            };

            // Annotate frontiers with planner state for display.
            // Clone frontier list quickly then release the lock, then annotate lock-free.
            {
                let soft_bl: Vec<[f32; 2]> = goal_blacklist.iter().map(|(g, _)| *g).collect();
                let hard_bl: Vec<[f32; 2]> = hard_blacklist.iter().map(|(g, _)| *g).collect();
                let raw_frontiers: Vec<core_types::Frontier> = {
                    let m = mapper_plan.read().await;
                    m.last_frontiers.clone()
                };
                let annotated = annotate_frontiers(&raw_frontiers, &reach, &soft_bl, &hard_bl, last_goal, &pose_pre);
                let _ = bus_plan.map_frontier_annotations.send(annotated);
            }

            // Goal-reached blacklist: if the robot has closed within MIN_GOAL_DIST of
            // last_goal, the control task already cleared its path on arrival.  Blacklist
            // the goal so the planner picks a fresh frontier instead of issuing a
            // trivially short (2-waypoint) path to the same shifting frontier centroid.
            {
                const MIN_GOAL_DIST: f32 = 0.75;
                if let Some(lg) = last_goal {
                    let dx = lg[0] - pose_pre.x_m;
                    let dy = lg[1] - pose_pre.y_m;
                    if (dx * dx + dy * dy).sqrt() < MIN_GOAL_DIST {
                        info!(gx = lg[0], gy = lg[1], "Sim planning: robot reached goal area — blacklisting");
                        let exp = now + std::time::Duration::from_secs(GOAL_BLACKLIST_SECS);
                        goal_blacklist.push((lg, exp));
                        last_goal = None;
                        last_path_sent_at = None;
                        goal_first_sent_at = None;
                        no_progress_pos = None;
                    }
                }
            }

            // Inner retry loop: on A* failure, immediately try the next-best
            // frontier (updated blacklist, same FrontierChoice) rather than
            // waiting ~1 s for the next plan_rx.recv() cycle.
            const MAX_INLINE_RETRIES: usize = 5;
            'planning: for _retry in 0..=MAX_INLINE_RETRIES {
                let bl: Vec<[f32; 2]> = goal_blacklist.iter().chain(hard_blacklist.iter()).map(|(g, _)| *g).collect();
                let pose = *rx_pose_p.borrow_and_update();
                let maybe_goal = {
                    let m = mapper_plan.read().await;
                    select_frontier_goal(&m, &choice, &pose, &bl, Some(&reach), last_goal)
                };
                let Some(goal) = maybe_goal else {
                    // No large-enough frontier available.  Count consecutive
                    // planning triggers with no valid goal; once stable, the
                    // arena is fully explored (only unreachable corner slivers remain).
                    no_frontier_streak += 1;
                    const EXPLORE_DONE_STREAK: u32 = 5;
                    if explore_mode && no_frontier_streak >= EXPLORE_DONE_STREAK {
                        let cells = mapper_plan.read().await.grid.len();
                        // Guard against pre-arm blacklist saturation: the planning task
                        // runs before the robot is armed, can exhaust small frontier sets
                        // from the spawn scan alone, and hit the streak before the robot
                        // has moved at all.  Require ≥10 000 observed cells before
                        // declaring exploration complete.  Below that threshold, clear the
                        // blacklist and reset the streak so post-arm exploration proceeds.
                        const MIN_CELLS_FOR_DONE: usize = 10_000;
                        if cells >= MIN_CELLS_FOR_DONE {
                            info!(no_frontier_streak, cells, "Exploration complete — no reachable frontiers remaining");
                            // Signal the sim tick task to start a new room.
                            let _ = explore_done_tx_plan.send(()).await;
                            // Drain any queued FrontierChoice so we don't plan on the old map.
                            while plan_rx.try_recv().is_ok() {}
                            // Mark episode_rx as seen, then wait for the new room's reset signal.
                            let _ = episode_rx_plan.borrow_and_update();
                            let _ = episode_rx_plan.changed().await;
                            let _ = episode_rx_plan.borrow_and_update();
                            // Reset all planning state for the new room.
                            goal_blacklist.clear();
                            hard_blacklist.clear();
                            consecutive_failures = 0;
                            escape_failures = 0;
                            escape_cycles = 0;
                            no_frontier_streak = 0;
                            last_goal = None;
                            last_path_sent_at = None;
                            goal_first_sent_at = None;
                            no_progress_pos = None;
                            last_crash_plan = *bus_plan.collision_count.borrow();
                            while plan_rx.try_recv().is_ok() {}
                            continue; // back to the outer select! loop
                        }
                        // Two cases:
                        // A) Pre-arm: robot hasn't moved, soft-blacklisted the spawn
                        //    scan frontiers.  Clear soft blacklist only.
                        // B) Physically stuck: A* reaches the frontiers but the robot
                        //    can't traverse them (US livelock, tight corridor).  Hard
                        //    entries must NOT be cleared — that causes the endless
                        //    re-blacklist loop seen when both reachable frontiers are
                        //    in tight spots.  Instead, if hard_blacklist is non-empty
                        //    after clearing soft, declare the robot stuck and force a
                        //    new room after one more streak cycle (streak will hit 5
                        //    again immediately since all remaining frontiers are hard-BL).
                        goal_blacklist.clear();
                        no_frontier_streak = 0;
                        if !hard_blacklist.is_empty() {
                            warn!(no_frontier_streak, cells, hard_bl = hard_blacklist.len(),
                                "Planning: no reachable frontiers after hard-BL — robot isolated, forcing new room");
                            { let mut s = stats_plan.lock().unwrap(); s.next_room_reason = Some("hard_bl_isolated"); }
                            let _ = explore_done_tx_plan.send(()).await;
                            while plan_rx.try_recv().is_ok() {}
                            let _ = episode_rx_plan.borrow_and_update();
                            let _ = episode_rx_plan.changed().await;
                            let _ = episode_rx_plan.borrow_and_update();
                            goal_blacklist.clear();
                            hard_blacklist.clear();
                            consecutive_failures = 0;
                            escape_failures = 0;
                            escape_cycles = 0;
                            no_frontier_streak = 0;
                            last_goal = None;
                            last_path_sent_at = None;
                            goal_first_sent_at = None;
                            no_progress_pos = None;
                            last_crash_plan = *bus_plan.collision_count.borrow();
                            while plan_rx.try_recv().is_ok() {}
                            continue;
                        }
                        warn!(no_frontier_streak, cells,
                            "Planning: frontier streak reset (cells < {MIN_CELLS_FOR_DONE}, pre-arm)");
                    }
                    break 'planning;
                };
                no_frontier_streak = 0;

                // REPLAN_HOLD_S: how long a frontier must persist after we sent a path
                // to it before it's considered stuck (robot couldn't make progress).
                // 15 s gives enough time for the robot to reach frontiers 3-5 m away
                // at 0.3 m/s.  Premature blacklisting was causing 91%+ blacklist ratios.
                const REPLAN_HOLD_S: u64 = 15;
                // RESEND_HOLD_S: minimum time before resending the same path.
                // 1 s keeps path fresh from current pose; paths refreshed every second
                // also reset last_path_sent_at, preventing path_sent_and_expired from
                // firing prematurely while the robot is still navigating toward a goal.
                const RESEND_HOLD_S: u64 = 1;
                let goal_matches_last = last_goal.map_or(false, |lg| {
                    let dx = lg[0] - goal[0]; let dy = lg[1] - goal[1];
                    (dx*dx + dy*dy).sqrt() < 0.25
                });
                let path_sent_recently = last_path_sent_at.map_or(false, |t| {
                    now.duration_since(t) < std::time::Duration::from_secs(RESEND_HOLD_S)
                });
                // Detect goals the robot has been navigating toward without reaching:
                // last_path_sent_at resets every re-plan (~1 s) so it can never age
                // past REPLAN_HOLD_S.  goal_first_sent_at is set only when the goal
                // first changes, so it correctly measures total elapsed time on this goal.
                let path_sent_and_expired = goal_matches_last
                    && goal_first_sent_at.map_or(false, |t| {
                        now.duration_since(t).as_secs() >= REPLAN_HOLD_S
                    });
                // No-progress blacklist: robot hasn't moved ≥ 0.2 m in 6 s while
                // pursuing this goal → deadlock/spin cycle near a blocked frontier.
                // Blacklist sooner than REPLAN_HOLD_S to break the spin cycle fast.
                let pose_now = *rx_pose_p.borrow();
                let no_progress_timeout = if goal_matches_last {
                    let (ref_pos, ref_t) = no_progress_pos.get_or_insert_with(|| {
                        ([pose_now.x_m, pose_now.y_m], now)
                    });
                    // Measure progress toward the goal, not raw displacement.
                    // Deadlock backups move the robot ~0.36 m backward (away from
                    // the goal), which used to reset the timer and prevent no_progress
                    // from ever firing during repeated backup/retry cycles.
                    // Only reset if the robot got ≥ 0.20 m closer to the goal.
                    let prev_dist = ((ref_pos[0] - goal[0]).powi(2) + (ref_pos[1] - goal[1]).powi(2)).sqrt();
                    let curr_dist = ((pose_now.x_m - goal[0]).powi(2) + (pose_now.y_m - goal[1]).powi(2)).sqrt();
                    if prev_dist - curr_dist >= 0.20 {
                        // Robot advanced ≥ 0.2 m toward goal — genuine progress.
                        *ref_pos = [pose_now.x_m, pose_now.y_m];
                        *ref_t   = now;
                        false
                    } else {
                        now.duration_since(*ref_t).as_secs() >= 6
                    }
                } else {
                    // New goal — reset checkpoint.
                    no_progress_pos = Some(([pose_now.x_m, pose_now.y_m], now));
                    false
                };
                if goal_matches_last && path_sent_recently {
                    break 'planning;
                }
                // Reduced from 5 s → 2 s: the "3-second frontier delay" in the UI was this
                // window — the goal annotation won't change until PATH_COMMIT_S expires.
                // 2 s still protects active paths (robot takes 2-7 s per goal at 0.3 m/s);
                // shorter commit means the next frontier is selected faster after goal reach.
                const PATH_COMMIT_S: u64 = 2;
                let path_committed = !goal_matches_last
                    && last_path_sent_at.map_or(false, |t| {
                        now.duration_since(t) < std::time::Duration::from_secs(PATH_COMMIT_S)
                    });
                if path_committed {
                    break 'planning;
                }
                if path_sent_and_expired || no_progress_timeout {
                    let age = goal_first_sent_at.map_or(0, |t| now.duration_since(t).as_secs());
                    let reason = if no_progress_timeout && !path_sent_and_expired {
                        "no-progress (stuck/spinning)"
                    } else {
                        "timeout"
                    };
                    warn!(x = goal[0], y = goal[1], age_s = age, reason,
                          "Sim planning: goal unreached — hard-blacklisting");
                    let exp = now + std::time::Duration::from_secs(GOAL_BLACKLIST_SECS * 4);
                    hard_blacklist.push((goal, exp));
                    last_goal = None;
                    last_path_sent_at = None;
                    goal_first_sent_at = None;
                    no_progress_pos = None;
                    continue 'planning;
                }
                // Update goal tracking: only set goal_first_sent_at on goal change.
                if !goal_matches_last {
                    goal_first_sent_at = Some(now);
                }
                last_goal = Some(goal);

                // Plan at clearance=7 (35 cm from walls) first. With robot radius 0.15 m
                // this gives 20 cm of body clearance, preventing corner-clip crashes where
                // the obstacle is just outside the ±55° sensor FOV. Falls back to clearance=5
                // (25 cm, 10 cm body clearance) for tight corridors where clearance=7 finds
                // no path. Clearance=3 is no longer tried — it allowed the robot body to
                // touch walls outside the FOV, causing physics collisions.
                // Tight-corridor escape still uses [7,5,3,2,1,0] separately below.
                let mut planned = None;
                let mut total_failure = false;
                {
                    let m = mapper_plan.read().await;
                    if let Some(path) = planner.plan_with_clearance(&m, &pose, goal, 7)
                        .or_else(|| planner.plan_with_clearance(&m, &pose, goal, 5)) {
                        planned = Some(path);
                    } else {
                        total_failure = true;
                    }
                }
                match planned {
                    Some(path) => {
                        info!(waypoints = path.waypoints.len(), "Sim planning: path found");
                        { let mut s = stats_plan.lock().unwrap(); s.astar_ok += 1; }
                        last_path_sent_at = Some(now);
                        let _ = bus_plan.planner_path.send(path).await;
                        consecutive_failures = 0;
                        escape_cycles = 0;
                        break 'planning;
                    }
                    None => {
                        { let mut s = stats_plan.lock().unwrap(); s.astar_fail += 1; }
                        warn!(_retry, ?choice, total_failure, "Sim planning: no path to frontier");
                        // Goals that fail at ALL clearance levels are structurally unreachable
                        // (e.g. frontier centroid beyond arena wall).  Use a 10× longer
                        // blacklist so they don't dominate retry cycles.
                        let bl_secs = if total_failure { GOAL_BLACKLIST_SECS * 10 } else { GOAL_BLACKLIST_SECS };
                        let exp = now + std::time::Duration::from_secs(bl_secs);
                        goal_blacklist.push((goal, exp));
                        last_goal = None;
                        consecutive_failures += 1;
                        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
                            let mut escaped = false;
                            if let Some(ref scan) = last_scan {
                                if let Some(target) = sim_escape(&bus_plan, &pose, scan, consecutive_failures).await {
                                    // Release lock between each level so mapping can write.
                                    let mut escape_path = None;
                                    'esc: for clearance in [7i32, 5, 3, 2, 1, 0] {
                                        let m = mapper_plan.read().await;
                                        if let Some(p) = planner.plan_with_clearance(&m, &pose, target, clearance) {
                                            escape_path = Some(p);
                                            break 'esc;
                                        }
                                        drop(m);
                                    }
                                    match escape_path {
                                        Some(path) => {
                                            info!(waypoints = path.waypoints.len(), wx = target[0], wy = target[1], "Sim escape: A* path validated — injecting");
                                            let _ = bus_plan.planner_path.send(path).await;
                                            escaped = true;
                                        }
                                        None => warn!(wx = target[0], wy = target[1], "Sim escape: A* failed on escape target at all clearances — staying still"),
                                    }
                                }
                            }
                            if !escaped { escape_failures += 1; }
                            // Track total escape trigger count (whether or not escape succeeded).
                            // A successful escape that doesn't restore planning means the
                            // remaining frontiers are A*-unreachable despite passing BFS
                            // pre-screen (path exists but exceeds MAX_NODES budget).
                            escape_cycles += 1;
                            const MAX_ESCAPE_CYCLES: u32 = 2;
                            goal_blacklist.clear();
                            consecutive_failures = 0;
                            last_goal = None;
                            if explore_mode && escape_cycles >= MAX_ESCAPE_CYCLES {
                                warn!(escape_cycles, "Sim planning: A*-unreachable frontiers after {} escape cycles — starting new room", escape_cycles);
                                { let mut s = stats_plan.lock().unwrap(); s.next_room_reason = Some("isolated"); }
                                escape_cycles = 0;
                                escape_failures = 0;
                                let _ = explore_done_tx_plan.send(()).await;
                                while plan_rx.try_recv().is_ok() {}
                                let _ = episode_rx_plan.borrow_and_update();
                                let _ = episode_rx_plan.changed().await;
                                let _ = episode_rx_plan.borrow_and_update();
                                goal_blacklist.clear();
                                hard_blacklist.clear();
                                consecutive_failures = 0;
                                no_frontier_streak = 0;
                                last_goal = None;
                                last_path_sent_at = None;
                                goal_first_sent_at = None;
                                no_progress_pos = None;
                                last_crash_plan = *bus_plan.collision_count.borrow();
                                while plan_rx.try_recv().is_ok() {}
                                continue;
                            }
                            if escape_failures >= MAX_ESCAPE_FAILURES && !explore_mode {
                                warn!(escape_failures, "Sim planning: robot isolated — forcing episode reset");
                                escape_failures = 0;
                                let _ = timeout_tx_plan.send(()).await;
                            } else if escape_failures >= MAX_ESCAPE_FAILURES {
                                // In explore mode, robot is truly stuck — start a new room.
                                warn!(escape_failures, "Sim planning: robot isolated (explore mode) — starting new room");
                                { let mut s = stats_plan.lock().unwrap(); s.next_room_reason = Some("isolated"); }
                                escape_failures = 0;
                                let _ = explore_done_tx_plan.send(()).await;
                                while plan_rx.try_recv().is_ok() {}
                                let _ = episode_rx_plan.borrow_and_update();
                                let _ = episode_rx_plan.changed().await;
                                let _ = episode_rx_plan.borrow_and_update();
                                goal_blacklist.clear();
                                hard_blacklist.clear();
                                consecutive_failures = 0;
                                escape_cycles = 0;
                                no_frontier_streak = 0;
                                last_goal = None;
                                last_path_sent_at = None;
                                goal_first_sent_at = None;
                                no_progress_pos = None;
                                last_crash_plan = *bus_plan.collision_count.borrow();
                                while plan_rx.try_recv().is_ok() {}
                                continue;
                            }
                            while plan_rx.try_recv().is_ok() {}
                            tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                            while plan_rx.try_recv().is_ok() {}
                            break 'planning;
                        }
                        // A* failed but not at escape threshold — continue
                        // 'planning to try the next frontier immediately.
                    }
                }
            }
        }
    });

    // ── Control task ──────────────────────────────────────────────────────────
    let min_vx = (min_motor_duty as f32 / max_motor_duty as f32) * control::MAX_VX;
    spawn_control_task(Arc::clone(&bus), control_path_rx,
        cfg.sim.obstacle_slow_m, cfg.sim.obstacle_stop_m, episode_reset_rx.clone(), min_vx);

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
        explore_done_rx,
        Arc::clone(&mapper),
        explore_mode,
        use_tof,
        cfg.sim.range_dropout,
        cfg.sim.range_noise_m,
        seed,
        max_motor_duty,
        std::sync::Arc::clone(&sim_stats),
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

    // Force immediate exit — do not wait for Tokio to drain tasks.
    // The planning task runs A* synchronously (blocking a runtime thread),
    // so the runtime would block for up to ~8 s per A* search before the
    // worker thread yields.  `process::exit` terminates immediately after
    // motors are zeroed, which is all the cleanup we need.
    info!("Sim mode: shutting down");
    {
        let stats = sim_stats.lock().unwrap();
        print_sim_summary(&stats, &bus);
    }
    std::process::exit(0);
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
    sim_state:              SimState,
    bus:                    Arc<bus::Bus>,
    episode_reset_tx:       tokio::sync::watch::Sender<u32>,
    arm_tx:                 tokio::sync::mpsc::Sender<()>,
    timeout_tx:             tokio::sync::mpsc::Sender<()>,
    mut force_respawn_rx:   tokio::sync::mpsc::Receiver<()>,
    mut explore_done_rx:    tokio::sync::mpsc::Receiver<()>,
    mapper:                 Arc<tokio::sync::RwLock<mapping::Mapper>>,
    explore_mode:           bool,
    use_tof:                bool,
    range_dropout:          f32,
    range_noise_m:          f32,
    seed:                   u64,
    max_motor_duty:         i8,
    sim_stats:              std::sync::Arc<std::sync::Mutex<SimRunStats>>,
) -> tokio::task::JoinHandle<()> {
    use std::time::Duration;
    use std::time::Instant;

    tokio::spawn(async move {
        // Publish initial state so HAL readers don't block indefinitely on step_rx.
        // IMPORTANT: reset() calls generate_maze() again (RNG has advanced since
        // FastSim::new), producing a different maze than what was seeded into the
        // mapper at startup.  Reseed immediately after reset so mapper and sim are
        // always consistent.
        let initial_step = {
            let mut sim = sim_state.sim.lock().await;
            sim.reset()
        };
        {
            let sim = sim_state.sim.lock().await;
            let walls: Vec<u8> = sim.wall_grid().iter().map(|&w| w as u8).collect();
            let _ = bus.sim_ground_truth.send(Arc::new(walls));
            // Reseed mapper with the wall grid from the maze actually in use.
            let mut m = mapper.write().await;
            m.seed_walls(sim.wall_grid(), 200);
            let _ = bus.map_grid_delta.send(m.initial_delta());
        }
        let _ = sim_state.step_tx.send(Arc::new(initial_step.clone()));
        let _ = bus.slam_pose2d.send(initial_step.pose);
        info!("Sim tick task started — send SIGUSR1 to arm");

        let mut episode: u32 = 0;
        let mut episode_timeout_count: u32 = 0;
        let mut scan_rng: u64 = seed.wrapping_add(0xFEED_BEEF_CAFE_BABE);
        let mut last_collision_time = Instant::now() - Duration::from_secs(1);  // Allow first collision
        // Suppression covers full safety escape cycle: delay(200) + reverse(600) + margin = 1s.
        // No rotation in escape — planner re-routes after backup.
        let collision_suppression_period = Duration::from_millis(1000);
        // Per-room wall-clock deadline for explore mode.  If the robot hasn't
        // completed exploration within this window, start a fresh room.
        // Separate from MAX_STEPS (used for RL training episode length).
        const EXPLORE_ROOM_TIMEOUT_S: u64 = 900; // 15 minutes
        let mut room_start_time = Instant::now();
        let mut rx_explored_st = bus.map_explored_stats.subscribe();
        let mut latest_cells: u32 = 0;
        let mut crashes_at_room_start: u32 = *bus.collision_count.borrow();
        loop {
            tokio::time::sleep(Duration::from_millis(100)).await;

            // Drain explored_stats to keep latest_cells current (non-blocking).
            while let Ok(es) = rx_explored_st.try_recv() { latest_cells = es.explored_cells; }

            // Read latest actuator state.
            let latest_motor = sim_state.latest_motor.lock().unwrap().take();
            let pan_deg      = *sim_state.latest_pan.lock().unwrap();

            // Decode MotorCommand to continuous (vx, vy, omega) in sim velocity
            // space using inverse mecanum kinematics.  This preserves blended
            // forward+rotation commands from the pure-pursuit controller; the old
            // discrete-action dispatch lost the rotation component when the FR
            // integer duty cycle truncated to 0 (robot drove straight into walls).
            let (target_vx, target_vy, target_omega) = match &latest_motor {
                Some(cmd) => sim_fast::motor_cmd_to_vel(
                    cmd.fl, cmd.fr, cmd.rl, cmd.rr, max_motor_duty),
                None => (0.0_f32, 0.0_f32, 0.0_f32),
            };

            let step = {
                let mut sim = sim_state.sim.lock().await;
                sim.step_continuous(target_vx, target_vy, target_omega, pan_deg)
            };

            // Wake SimCamera, SimImu, SimUltrasonic.
            let _ = sim_state.step_tx.send(Arc::new(step.clone()));

            // Ground-truth pose → SLAM channel (no dead-reckoning in sim).
            let _ = bus.slam_pose2d.send(step.pose);

            // Ground-truth pseudo-lidar: geometric ray distances with optional noise.
            // Bypasses MiDaS domain-gap so obstacle_slow_m / obstacle_stop_m are
            // metric-accurate and match real-mode threshold values (0.70 / 0.25 m).
            //
            // ToF mode: the 48-ray lidar scan still feeds vision_pseudo_lidar for
            // mapping (wide FOV needed).  The VL53L8CX 8-ray scan separately drives
            // nearest_obstacle_m / nearest_obstacle_angle_rad with metric-accurate,
            // no-dropout depth — matching the real hardware role of the sensor.
            {
                let mut scan = step.scan.clone();
                for ray in &mut scan.rays {
                    // Always consume both RNG values to keep the sequence deterministic.
                    let noise   = sim_fast::hal_impls::gaussian(&mut scan_rng, range_noise_m);
                    let dropped = (sim_fast::hal_impls::xorshift(&mut scan_rng) as f64
                        / u64::MAX as f64) < range_dropout as f64;
                    if dropped {
                        ray.range_m = 3.0; // dropped ray → max range (no obstacle detected)
                    } else if ray.range_m < 3.0 {
                        // Only add range noise to rays that actually hit a wall.
                        // Max-range rays (open space, no wall hit) stay at exactly 3.0
                        // so the mapper's `>= MAX_RANGE_M` check skips them correctly.
                        // Without this guard, `3.0 - epsilon` noise creates a false
                        // obstacle arch at ~3 m in every scan direction.
                        ray.range_m = (ray.range_m + noise).clamp(0.0, 2.99);
                    }
                }
                let _ = bus.vision_pseudo_lidar.send(Arc::new(scan));

                // Nearest obstacle: start from the wide-FOV lidar for full coverage,
                // then override with ToF where it can see (ToF is metric-accurate, no dropout).
                // Taking min(tof, lidar) is conservative: blocks any direction either sensor
                // sees an obstacle, preventing blind-side crashes in the ToF narrow cone.
                let lidar_nearest = nearest_in_fov(&step.scan, 55f32.to_radians());
                let (near_m, near_angle_rad) = if use_tof {
                    let sim = sim_state.sim.lock().await;
                    let mut tof = sim.cast_tof(pan_deg);
                    drop(sim);
                    // 1.5 cm noise, no dropout (VL53L8CX characteristics).
                    for ray in &mut tof.rays {
                        let noise = sim_fast::hal_impls::gaussian(&mut scan_rng, 0.015);
                        if ray.range_m < 3.0 {
                            ray.range_m = (ray.range_m + noise).clamp(0.0, 2.99);
                        }
                    }
                    let tof_nearest = nearest_in_fov(&tof, 55f32.to_radians());
                    // Use ToF reading when it's closer (more accurate at short range);
                    // fall back to lidar for obstacles outside the ±20° ToF cone.
                    if tof_nearest.0 <= lidar_nearest.0 { tof_nearest } else { lidar_nearest }
                } else {
                    lidar_nearest
                };
                if near_m < 1.5 {
                    let side = if near_angle_rad.to_degrees() > 5.0 { "left" }
                               else if near_angle_rad.to_degrees() < -5.0 { "right" }
                               else { "center" };
                    tracing::debug!(nearest_m = near_m, angle_deg = near_angle_rad.to_degrees(), side,
                           "Sim tick: nearest obstacle (geometric)");
                }
                let _ = bus.nearest_obstacle_m.send(near_m);
                let _ = bus.nearest_obstacle_angle_rad.send(near_angle_rad);
            }

            // ── Stuck planning recovery (explore mode) ───────────────────────────
            // The planner sends this when all frontiers fail A* at all clearance levels.
            // Nothing to do in the sim tick — the planning task already handles recovery
            // by trying lower clearances and escape paths.  Drain the channel so it
            // does not fill up and block future sends.
            if explore_mode { let _ = force_respawn_rx.try_recv(); }

            // ── Collision ────────────────────────────────────────────────────
            if step.collision {
                // Suppress duplicate collision reports within 500ms to prevent recovery cascades
                let now = Instant::now();
                let time_since_last = now.duration_since(last_collision_time);
                if time_since_last < collision_suppression_period {
                    // Collision detected but suppressed (likely robot still touching wall during recovery)
                    let _ = sim_state.step_tx.send(Arc::new(step.clone()));
                    continue;  // Skip collision handling, let recovery continue
                }
                last_collision_time = now;  // Update last collision time
                if explore_mode {
                    // Collision in explore mode: DON'T respawn, let unified collision
                    // recovery in control task handle it (back up, then replan)
                    warn!(
                        robot_x = step.pose.x_m, robot_y = step.pose.y_m,
                        "Sim tick: COLLISION — collision_count incremented, control task will recover"
                    );
                    // Increment collision count so control task detects and backs up
                    let n = *bus.collision_count.borrow() + 1;
                    let _ = bus.collision_count.send(n);
                    // Don't respawn — keep robot at collision point, map preserved
                    // Send current step (robot at collision point) so poses stay consistent
                    let _ = sim_state.step_tx.send(Arc::new(step.clone()));
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

            // ── New room trigger: episode timeout (explore mode stuck) ───────
            // In explore mode use a wall-clock timer (15 min) rather than
            // MAX_STEPS so RL training episode length is unaffected.
            let explore_timeout = explore_mode
                && room_start_time.elapsed().as_secs() >= EXPLORE_ROOM_TIMEOUT_S;
            // Episode timeout in training mode (non-explore): same reset path.
            let training_timeout = step.done && !explore_mode;
            // Exploration complete: planning task drained all frontiers and sent
            // an explore_done signal.
            let explore_complete = explore_mode && explore_done_rx.try_recv().is_ok();

            if explore_complete || explore_timeout || training_timeout {
                episode += 1;
                let is_timeout = explore_timeout || training_timeout;
                // Record room stats before resetting state.
                {
                    let mut s = sim_stats.lock().unwrap();
                    let reason = s.next_room_reason.take().unwrap_or_else(||
                        if is_timeout { "timeout" } else { "complete" }
                    );
                    s.rooms.push(RoomRecord {
                        reason,
                        duration_s: room_start_time.elapsed().as_secs_f64(),
                        cells:      latest_cells,
                        crashes:    bus.collision_count.borrow().saturating_sub(crashes_at_room_start),
                    });
                }
                if is_timeout {
                    episode_timeout_count += 1;
                    let _ = bus.episode_timeout_count.send(episode_timeout_count);
                    warn!(episode, episode_timeout_count,
                        "Sim tick: room timeout — starting new room");
                } else {
                    info!(episode, "Sim tick: exploration complete — starting new room");
                }
                let _ = bus.episode_count.send(episode);
                if training_timeout { let _ = timeout_tx.send(()).await; }
                // Brief pause so the UI can display the completion state.
                tokio::time::sleep(Duration::from_millis(if is_timeout { 800 } else { 2000 })).await;
                // Generate new maze (FastSim::reset advances the internal RNG).
                let reset_step = {
                    let mut sim = sim_state.sim.lock().await;
                    sim.reset()
                };
                // Clear the occupancy map for the new room.
                {
                    let sim = sim_state.sim.lock().await;
                    let walls: Vec<u8> = sim.wall_grid().iter().map(|&w| w as u8).collect();
                    let _ = bus.sim_ground_truth.send(Arc::new(walls));
                    let mut m = mapper.write().await;
                    *m = mapping::Mapper::new();
                    m.seed_walls(sim.wall_grid(), 200);
                    let _ = bus.map_grid_delta.send(m.initial_delta());
                }
                let _ = episode_reset_tx.send(episode);
                let _ = sim_state.step_tx.send(Arc::new(reset_step.clone()));
                let _ = bus.slam_pose2d.send(reset_step.pose);
                let _ = bus.safety_state.send(SafetyState::Ok);
                room_start_time = Instant::now();
                latest_cells = 0;
                crashes_at_room_start = *bus.collision_count.borrow();
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

/// Return `(range_m, angle_rad)` of the nearest obstacle within `max_fov_rad` of centre.
///
/// # Dual-path usage
/// - **Real robot**: pass `50°`.  The outer ±5° band (±50°–±55°) is excluded because
///   lens vignette produces a persistent false-close artifact at the frame edges.
/// - **Sim**: pass `55°`.  No vignette exists; using the full lidar FOV gives 5° more
///   coverage on each side, reducing blind-side crash risk.
fn nearest_in_fov(scan: &core_types::PseudoLidarScan, max_fov_rad: f32) -> (f32, f32) {
    scan.rays.iter()
        .filter(|r| r.angle_rad.abs() <= max_fov_rad)
        .min_by(|a, b| a.range_m.partial_cmp(&b.range_m).unwrap())
        .map(|r| (r.range_m, r.angle_rad))
        .unwrap_or((f32::MAX, 0.0))
}

/// Compute the open-space bias from a MiDaS depth map.
///
/// Returns a value where **positive means the right side is more open**.
/// Higher pixel values in the depth map indicate closer objects.
fn depth_open_bias(depth: &core_types::DepthMap) -> f32 {
    let w = depth.width as usize;
    let h = depth.mask_start_row.min(depth.height) as usize;
    if w < 2 || h == 0 { return 0.0; }
    let mid = w / 2;
    let (mut left_sum, mut right_sum) = (0.0f32, 0.0f32);
    for row in 0..h {
        for col in 0..mid { left_sum  += depth.data[row * w + col]; }
        for col in mid..w { right_sum += depth.data[row * w + col]; }
    }
    (left_sum - right_sum) / (h * mid) as f32
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
/// `danger_angle_deg`: when `Some(angle)`, the nearest obstacle is close and the
/// pan should track it so ToF can measure it accurately.  Overrides the normal
/// sweep + reactive bias — pointing away from a nearby obstacle (the default
/// open-side bias) causes ToF to miss it, producing a blind spot in the forward
/// cone just when the robot turns into the wall.
fn gimbal_pan_target(open_bias: f32, t_s: f32, cur_pan: f32,
                     danger_angle_deg: Option<f32>) -> f32 {
    const SWEEP_AMP: f32     = 20.0; // degrees
    const SWEEP_PERIOD_S: f32 = 4.0; // seconds
    const REACTIVE_GAIN: f32  = 10.0;
    const REACTIVE_CAP: f32   = 10.0; // degrees
    const STEP_CAP: f32       = 5.0;  // degrees per frame
    const PAN_LIMIT: f32      = 30.0; // degrees

    let target = if let Some(angle) = danger_angle_deg {
        // Track toward the nearby obstacle so ToF locks onto it instead of
        // sweeping away.  Clamp to pan range — if the obstacle is at 54° but
        // PAN_LIMIT=30°, we point as far toward it as the hardware allows.
        angle.clamp(-PAN_LIMIT, PAN_LIMIT)
    } else {
        let sweep    = SWEEP_AMP * (2.0 * std::f32::consts::PI * t_s / SWEEP_PERIOD_S).sin();
        let reactive = (open_bias * REACTIVE_GAIN).clamp(-REACTIVE_CAP, REACTIVE_CAP);
        (sweep + reactive).clamp(-PAN_LIMIT, PAN_LIMIT)
    };
    let step = (target - cur_pan).clamp(-STEP_CAP, STEP_CAP);
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
    // Clamp target to arena interior so A* never gets an out-of-bounds goal.
    const MARGIN: f32 = 0.5;
    let wx = (pose.x_m + dist * world_angle.cos())
        .clamp(MARGIN, sim_fast::ARENA_M - MARGIN);
    let wy = (pose.y_m + dist * world_angle.sin())
        .clamp(MARGIN, sim_fast::ARENA_M - MARGIN);

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
                                let (near_m, near_angle_rad) = nearest_in_fov(&scan, 50f32.to_radians());
                                if near_m < 1.5 {
                                    let near_deg = near_angle_rad.to_degrees();
                                    let side = if near_deg > 5.0 { "left" } else if near_deg < -5.0 { "right" } else { "center" };
                                    info!(nearest_m = near_m, angle_deg = near_deg, side, "Perception: nearest obstacle");
                                }
                                let _ = bus_perc.vision_depth.send(Arc::new(depth));
                                let _ = bus_perc.vision_pseudo_lidar.send(Arc::new(scan));
                                // nearest_obstacle_m/angle_rad are owned by the ToF task (~20 Hz).
                                // Do not overwrite from MiDaS (~3 Hz, noisier).
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
