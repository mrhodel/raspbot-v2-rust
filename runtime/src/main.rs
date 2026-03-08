//! Robot runtime — Milestone 1 (debug-capable foundation).
//!
//! Boots the message bus, stubs for all hardware, and the perception /
//! micro-SLAM pipeline. Logs all key messages to NDJSON telemetry.
//!
//! Run modes are selected via the first CLI argument:
//!   robot hw-test      — hardware component tests (motors, gimbal, etc.)
//!   robot robot-run    — autonomous operation (requires trained policy)
//!   robot robot-debug  — autonomous with heavy telemetry
//!   robot slam-debug   — camera/IMU localisation validation
//!   (default)          — Milestone 1 sensor loop with telemetry

use std::sync::Arc;
use tracing::{error, info};

use bus::Bus;
use config::RobotConfig;
use executive::Executive;
use hal::{Camera, Imu, StubCamera, StubGimbal, StubImu, StubMotorController, StubUltrasonic};
use micro_slam::ImuDeadReckon;
use perception::{DepthInference, EventGate, PseudoLidarExtractor};
use telemetry::TelemetryWriter;
use ui_bridge::{UiBridgeConfig, start as start_ui_bridge};

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
    let (bus, _rx, _watch_rx) = Bus::new(bus::CAP);
    // Bus::new() already returns Arc<Bus>; do not wrap again.
    info!("Message bus created");

    // ── Telemetry ─────────────────────────────────────────────────────────────
    let mut telem = TelemetryWriter::open("logs/robot.ndjson").await?;
    telem.event("runtime", "system_boot").await?;
    info!("Telemetry writer opened → logs/robot.ndjson");

    // ── HAL stubs ─────────────────────────────────────────────────────────────
    let camera: Box<dyn Camera> = Box::new(StubCamera::new(&cfg.hal.camera));
    let imu: Box<dyn Imu> = Box::new(StubImu::new());
    let _motor = StubMotorController;
    let _ultrasonic = StubUltrasonic::new();
    let _gimbal = StubGimbal::new(cfg.hal.gimbal.tilt_neutral);
    info!("HAL stubs initialised");

    // ── Perception ────────────────────────────────────────────────────────────
    // Depth inference at 192×192; switch to 256×256 after benchmarking (Phase 6).
    let depth_infer = DepthInference::new(192, 192);
    let mut event_gate = EventGate::with_default_threshold();
    let lidar_extractor = PseudoLidarExtractor::new(48, 3.0);
    info!("Perception pipeline initialised (stub depth inference)");

    // ── Micro-SLAM ────────────────────────────────────────────────────────────
    let mut slam = ImuDeadReckon::new();
    info!("Micro-SLAM initialised (IMU dead-reckoning stub)");

    // ── Executive ─────────────────────────────────────────────────────────────
    let _exec = Executive::new(Arc::clone(&bus));
    info!("Executive initialised (state: Idle)");

    // ── UI Bridge ─────────────────────────────────────────────────────────────
    start_ui_bridge(Arc::clone(&bus), UiBridgeConfig::default()).await?;

    // ── Spawn camera task ─────────────────────────────────────────────────────
    let bus_cam = Arc::clone(&bus);
    let mut camera = camera;
    let cam_handle = tokio::spawn(async move {
        info!("Camera task started");
        loop {
            match camera.read_frame().await {
                Ok(frame) => {
                    let arc = Arc::new(frame);
                    let _ = bus_cam.camera_frame_raw.send(Arc::clone(&arc));

                    // Convert to grayscale and publish.
                    let gray = rgb_to_gray(&arc);
                    let _ = bus_cam.camera_frame_gray.send(Arc::new(gray));
                }
                Err(e) => error!("Camera error: {e}"),
            }
        }
    });

    // ── Spawn IMU task ────────────────────────────────────────────────────────
    let bus_imu = Arc::clone(&bus);
    let mut imu = imu;
    let imu_handle = tokio::spawn(async move {
        info!("IMU task started");
        loop {
            match imu.read_sample().await {
                Ok(sample) => {
                    let _ = bus_imu.imu_raw.send(sample);
                }
                Err(e) => error!("IMU error: {e}"),
            }
        }
    });

    // ── Spawn perception + SLAM task ──────────────────────────────────────────
    let bus_perc = Arc::clone(&bus);
    let mut rx_gray  = bus.camera_frame_gray.subscribe();
    let mut rx_imu   = bus.imu_raw.subscribe();
    let perc_handle = tokio::spawn(async move {
        info!("Perception/SLAM task started");
        loop {
            tokio::select! {
                Ok(gray) = rx_gray.recv() => {
                    if event_gate.should_infer(&gray) {
                        // Convert gray frame to a synthetic CameraFrame for the
                        // depth stub (real: pass RGB directly to MiDaS).
                        let rgb_stub = core_types::CameraFrame {
                            t_ms: gray.t_ms,
                            width: gray.width,
                            height: gray.height,
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

    // ── Spawn telemetry logger task ───────────────────────────────────────────
    // slam_pose2d is a watch channel (latest state); use changed() + borrow.
    // Lidar and IMU are broadcast channels; use recv() as normal.
    let mut rx_pose  = bus.slam_pose2d.subscribe();   // watch::Receiver
    let mut rx_lidar = bus.vision_pseudo_lidar.subscribe();
    let mut rx_imu2  = bus.imu_raw.subscribe();
    let telem_handle = tokio::spawn(async move {
        info!("Telemetry task started");
        loop {
            tokio::select! {
                Ok(()) = rx_pose.changed() => {
                    let pose = *rx_pose.borrow_and_update();
                    telem.write("slam/pose2d", "micro_slam", &pose).await.ok();
                }
                Ok(scan) = rx_lidar.recv() => {
                    telem.write("vision/pseudo_lidar", "perception", &*scan).await.ok();
                }
                Ok(imu) = rx_imu2.recv() => {
                    telem.write("imu/raw", "hal", &imu).await.ok();
                }
                else => break,
            }
        }
    });

    info!("Milestone 1 system running — press Ctrl+C to stop");
    tokio::signal::ctrl_c().await?;
    info!("Shutdown signal received");

    // Tasks will be dropped; channels close naturally.
    drop(cam_handle);
    drop(imu_handle);
    drop(perc_handle);
    drop(telem_handle);

    info!("Shutdown complete");
    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn rgb_to_gray(frame: &core_types::CameraFrame) -> core_types::GrayFrame {
    let data: Vec<u8> = frame.data
        .chunks_exact(3)
        .map(|rgb| {
            let r = rgb[0] as u32;
            let g = rgb[1] as u32;
            let b = rgb[2] as u32;
            // BT.601 luma coefficients (×256 integer approximation).
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
