//! Camera bring-up and body-mask calibration tool.
//!
//! Requires the `usb-camera` feature (default on Pi):
//!   cargo build --release --example camera_test
//!
//! Modes:
//!   (default)  capture N frames, report actual FPS, then exit
//!   --stream   continuous capture + MJPEG stream; open browser to print URL
//!   --save-frame <path.png>  save one frame as PNG for body-mask calibration
//!
//! Usage (on the Pi):
//!   sudo ./target/release/examples/camera_test
//!   sudo ./target/release/examples/camera_test -- --stream
//!   sudo ./target/release/examples/camera_test -- --save-frame /tmp/body_mask_ref.png
//!
//! Body-mask calibration workflow:
//!   1. Run with --save-frame, gimbal at neutral (pan=0, tilt=0).
//!   2. Open the PNG on your laptop and count how many rows from the bottom
//!      show the robot's own chassis / frame (typically 10-30 rows).
//!   3. Set  body_mask_bottom_rows: <count>  in robot_config.yaml.

use std::{
    env,
    path::PathBuf,
    time::Instant,
};

use anyhow::Result;
use hal::V4L2Camera;
use hal::camera::Camera;
use tracing::info;

// ── Default capture settings ──────────────────────────────────────────────────
const DEVICE: &str = "/dev/video0";
const WIDTH:  u32  = 320;
const HEIGHT: u32  = 240;
const FPS:    u32  = 15;
const STREAM_PORT: u16 = 8080;

// Number of frames to capture in default benchmark mode.
const BENCH_FRAMES: u32 = 60;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn save_png(frame: &core_types::CameraFrame, path: &PathBuf) -> Result<()> {
    use image::{ImageFormat, RgbImage};
    use std::io::Cursor;

    let img = RgbImage::from_raw(frame.width, frame.height, frame.data.clone())
        .ok_or_else(|| anyhow::anyhow!("frame data size mismatch"))?;

    let mut buf = Cursor::new(Vec::new());
    img.write_to(&mut buf, ImageFormat::Png)?;

    std::fs::write(path, buf.into_inner())?;
    println!("Saved: {}", path.display());
    println!(
        "Inspect the PNG and count how many rows from the bottom show the robot body."
    );
    println!("Then set  body_mask_bottom_rows: <count>  in robot_config.yaml.");
    Ok(())
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("hal=debug".parse().unwrap()),
        )
        .init();

    let args: Vec<String> = env::args().collect();
    let stream_mode   = args.iter().any(|a| a == "--stream");
    let save_frame    = args.iter().position(|a| a == "--save-frame")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from);

    // Determine whether to start the MJPEG server.
    let stream_port = if stream_mode { Some(STREAM_PORT) } else { None };

    println!("Camera test  {DEVICE}  {WIDTH}×{HEIGHT} @ {FPS} fps");
    if let Some(port) = stream_port {
        println!("MJPEG stream → open  http://raspbot:{port}/  in a browser");
    }

    let mut cam = V4L2Camera::new(DEVICE, WIDTH, HEIGHT, FPS, stream_port)?;
    let (actual_w, actual_h) = cam.resolution();
    println!("Actual resolution: {actual_w}×{actual_h}");

    if let Some(ref path) = save_frame {
        // Grab one frame and save it for body-mask visual inspection.
        println!("Capturing one frame for body-mask calibration...");
        let frame = cam.read_frame().await?;
        save_png(&frame, path)?;
        return Ok(());
    }

    if stream_mode {
        // Stream mode: capture continuously, print FPS every 5 s.
        println!("Streaming (Ctrl-C to stop)...");
        let mut count  = 0u64;
        let mut t_tick = Instant::now();
        loop {
            let frame = cam.read_frame().await?;
            count += 1;
            if t_tick.elapsed().as_secs() >= 5 {
                let fps = count as f64 / t_tick.elapsed().as_secs_f64();
                println!("  {fps:.1} fps  last frame t={}ms", frame.t_ms);
                count  = 0;
                t_tick = Instant::now();
            }
        }
    } else {
        // Benchmark mode: capture BENCH_FRAMES frames, report FPS.
        println!("Capturing {BENCH_FRAMES} frames...");
        let t0 = Instant::now();
        let mut last_t_ms = 0u64;
        for i in 1..=BENCH_FRAMES {
            let frame = cam.read_frame().await?;
            last_t_ms = frame.t_ms;
            if i % 15 == 0 {
                println!("  frame {i:3}  t={}ms  size={}B", frame.t_ms, frame.data.len());
            }
        }
        let elapsed = t0.elapsed().as_secs_f64();
        let fps     = BENCH_FRAMES as f64 / elapsed;
        println!(
            "\nCaptured {BENCH_FRAMES} frames in {elapsed:.2}s  ({fps:.1} fps)\n\
             Last frame timestamp: {last_t_ms}ms"
        );
    }

    Ok(())
}
