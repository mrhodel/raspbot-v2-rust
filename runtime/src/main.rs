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
//!   robot hw-test      — hardware component tests
//!   robot robot-run    — autonomous operation (requires trained policy)
//!   robot robot-debug  — autonomous with heavy telemetry
//!   robot slam-debug   — camera/IMU localisation validation
//!   (default)          — full sensor + mapping loop

use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

use bus::Bus;
use config::RobotConfig;
use control::PurePursuitController;
use core_types::FrontierChoice;
use executive::Executive;
use hal::{
    Camera, Imu, Ultrasonic,
    StubCamera, StubGimbal, StubImu, StubMotorController, StubUltrasonic,
};
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

    // ── Bus ───────────────────────────────────────────────────────────────────
    let (bus, rx, _watch_rx) = Bus::new(bus::CAP);
    // Extract mpsc receivers for planning/control tasks; keep rest alive.
    let plan_frontier_rx = rx.decision_frontier;
    let control_path_rx  = rx.planner_path;
    let _motor_cmd_rx    = rx.motor_command;
    let _cmdvel_rx       = rx.controller_cmd_vel;
    let _telem_event_rx  = rx.telemetry_event;
    info!("Message bus created");

    // ── Telemetry ─────────────────────────────────────────────────────────────
    let mut telem = TelemetryWriter::open("logs/robot.ndjson").await?;
    telem.event("runtime", "system_boot").await?;
    info!("Telemetry writer opened → logs/robot.ndjson");

    // ── HAL stubs ─────────────────────────────────────────────────────────────
    let camera:     Box<dyn Camera>     = Box::new(StubCamera::new(&cfg.hal.camera));
    let imu:        Box<dyn Imu>        = Box::new(StubImu::new());
    let ultrasonic: Box<dyn Ultrasonic> = Box::new(StubUltrasonic::new());
    let _motor  = StubMotorController;
    let _gimbal = StubGimbal::new(cfg.hal.gimbal.tilt_neutral);
    info!("HAL stubs initialised");

    // ── Perception ────────────────────────────────────────────────────────────
    let depth_infer    = DepthInference::new(
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
    let mut slam = ImuDeadReckon::new();
    info!("Micro-SLAM initialised (IMU dead-reckoning)");

    // ── Mapper (shared: mapping task writes, planning task reads) ─────────────
    let mapper = Arc::new(RwLock::new(Mapper::new()));
    info!("Occupancy grid mapper initialised (5 cm/cell)");

    // ── Executive ─────────────────────────────────────────────────────────────
    let _exec = Executive::new(Arc::clone(&bus));
    info!("Executive initialised (Idle)");

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

    // ── Perception + SLAM task ────────────────────────────────────────────────
    let bus_perc   = Arc::clone(&bus);
    let mut rx_gray = bus.camera_frame_gray.subscribe();
    let mut rx_imu  = bus.imu_raw.subscribe();
    let perc_handle = tokio::spawn(async move {
        info!("Perception/SLAM task started");
        loop {
            tokio::select! {
                Ok(gray) = rx_gray.recv() => {
                    if event_gate.should_infer(&gray) {
                        let rgb_stub = core_types::CameraFrame {
                            t_ms: gray.t_ms, width: gray.width, height: gray.height,
                            data: gray.data.iter().flat_map(|&v| [v, v, v]).collect(),
                        };
                        match depth_infer.infer(&rgb_stub) {
                            Ok(depth) => {
                                let scan = lidar_extractor.extract(&depth);
                                let _ = bus_perc.vision_depth.send(Arc::new(depth));
                                let _ = bus_perc.vision_pseudo_lidar.send(Arc::new(scan));
                            }
                            Err(e) => error!("Depth inference error: {e}"),
                        }
                    }
                }
                Ok(sample) = rx_imu.recv() => {
                    let pose = slam.update(&sample);
                    let _ = bus_perc.slam_pose2d.send(pose);
                }
                else => break,
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
        let monitor     = SafetyMonitor::new();
        let timeout_dur = tokio::time::Duration::from_millis(monitor.watchdog_timeout_ms);
        loop {
            match tokio::time::timeout(timeout_dur, rx_us.recv()).await {
                Err(_elapsed) => {
                    let (state, cmd) = monitor.evaluate_timeout(0);
                    let _ = bus_safety.safety_state.send(state);
                    let _ = bus_safety.motor_command.try_send(cmd);
                }
                Ok(Err(tokio::sync::broadcast::error::RecvError::Closed)) => break,
                Ok(Err(tokio::sync::broadcast::error::RecvError::Lagged(n))) => {
                    warn!("Safety task lagged {n} readings");
                }
                Ok(Ok(reading)) => {
                    let (state, maybe_stop) = monitor.evaluate(&reading);
                    let _ = bus_safety.safety_state.send(state);
                    if let Some(cmd) = maybe_stop {
                        let _ = bus_safety.motor_command.try_send(cmd);
                    }
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
        info!("Planning task started (A*, 2-cell clearance)");
        let planner = AStarPlanner::new();
        loop {
            let Some(choice) = plan_rx.recv().await else { break };
            let pose = *rx_pose_p.borrow_and_update();
            let maybe_goal = {
                let m = mapper_plan.read().await;
                select_frontier_goal(&m, &choice)
            };
            if let Some(goal) = maybe_goal {
                let m = mapper_plan.read().await;
                match planner.plan(&m, &pose, goal) {
                    Some(path) => {
                        info!(waypoints = path.waypoints.len(), "Planning: path found");
                        let _ = bus_plan.planner_path.send(path).await;
                    }
                    None => warn!(?choice, "Planning: no path to frontier"),
                }
            } else {
                warn!(?choice, "Planning: no frontiers available");
            }
        }
    });

    // ── Control task ──────────────────────────────────────────────────────────
    // Triggered by new Path from planning. Runs pure-pursuit once per path;
    // real-time re-evaluation against the pose watch is Phase 11 work.
    let bus_ctrl       = Arc::clone(&bus);
    let mut ctrl_rx    = control_path_rx;
    let mut rx_pose_c  = bus.slam_pose2d.subscribe();
    let ctrl_handle = tokio::spawn(async move {
        info!("Control task started (pure-pursuit, lookahead 0.3 m)");
        let controller = PurePursuitController::new();
        loop {
            let Some(path) = ctrl_rx.recv().await else { break };
            let pose = *rx_pose_c.borrow_and_update();
            let cmd_vel = controller.compute(&pose, &path);
            let _ = bus_ctrl.controller_cmd_vel.try_send(cmd_vel);
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

    drop(cam_handle);
    drop(imu_handle);
    drop(us_handle);
    drop(perc_handle);
    drop(map_handle);
    drop(safety_handle);
    drop(plan_handle);
    drop(ctrl_handle);
    drop(telem_handle);

    info!("Shutdown complete");
    Ok(())
}

// ── Frontier selection ────────────────────────────────────────────────────────

/// Pick a world-frame goal from the mapper's cached frontier list.
fn select_frontier_goal(mapper: &Mapper, choice: &FrontierChoice) -> Option<[f32; 2]> {
    let frontiers = &mapper.last_frontiers;
    if frontiers.is_empty() {
        return None;
    }
    let chosen = match choice {
        FrontierChoice::Nearest | FrontierChoice::RandomValid => frontiers.first(),
        FrontierChoice::Largest => frontiers.iter().max_by_key(|f| f.size_cells),
        FrontierChoice::Leftmost => frontiers
            .iter()
            .min_by(|a, b| a.centroid_x_m.partial_cmp(&b.centroid_x_m).unwrap()),
        FrontierChoice::Rightmost => frontiers
            .iter()
            .max_by(|a, b| a.centroid_x_m.partial_cmp(&b.centroid_x_m).unwrap()),
    };
    chosen.map(|f| [f.centroid_x_m, f.centroid_y_m])
}

// ── Helpers ───────────────────────────────────────────────────────────────────

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
