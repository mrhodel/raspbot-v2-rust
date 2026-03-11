//! MPU-6050 bring-up and characterisation tool.
//!
//! Requires the `mpu6050` feature (default on Pi):
//!   cargo build --release --example imu_test
//!
//! Modes:
//!   (default)  print 100 samples at ~100 Hz, report noise floor, then exit
//!   --stream   continuous print until Ctrl-C
//!   --gyro-z   print only gyro_z in °/s — useful for verifying CW/CCW sign
//!
//! Usage:
//!   sudo ./target/release/examples/imu_test
//!   sudo ./target/release/examples/imu_test -- --stream
//!   sudo ./target/release/examples/imu_test -- --gyro-z

use std::{env, time::Instant};

use anyhow::Result;
use hal::imu::Imu;
use hal::Mpu6050Imu;

const I2C_BUS: u8 = 6;
const BENCH_SAMPLES: u32 = 100;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("hal=debug".parse().unwrap()),
        )
        .init();

    let args: Vec<String> = env::args().collect();
    let stream_mode = args.iter().any(|a| a == "--stream");
    let gyro_z_mode = args.iter().any(|a| a == "--gyro-z");

    let mut imu = Mpu6050Imu::new(I2C_BUS)?;
    println!("MPU-6050 on /dev/i2c-{I2C_BUS}  ready.");

    if stream_mode {
        println!("Streaming (Ctrl-C to stop)...");
        println!("{:>8}  {:>8} {:>8} {:>8}  {:>8} {:>8} {:>8}",
            "t_ms", "ax", "ay", "az", "gx", "gy", "gz");
        loop {
            let s = imu.read_sample().await?;
            println!(
                "{:>8}  {:>8.3} {:>8.3} {:>8.3}  {:>8.4} {:>8.4} {:>8.4}",
                s.t_ms,
                s.accel_x, s.accel_y, s.accel_z,
                s.gyro_x,  s.gyro_y,  s.gyro_z,
            );
        }
    } else if gyro_z_mode {
        println!("gyro_z (°/s) — rotate robot CW to confirm negative sign:");
        loop {
            let s = imu.read_sample().await?;
            let gz_deg = s.gyro_z * 180.0 / std::f32::consts::PI;
            println!("{:>8}ms  gz = {:>8.3} °/s", s.t_ms, gz_deg);
        }
    } else {
        // Benchmark: collect BENCH_SAMPLES, report rate and noise floor.
        println!("Collecting {BENCH_SAMPLES} samples...");
        let t0 = Instant::now();
        let mut samples = Vec::with_capacity(BENCH_SAMPLES as usize);
        for _ in 0..BENCH_SAMPLES {
            samples.push(imu.read_sample().await?);
        }
        let elapsed = t0.elapsed().as_secs_f64();
        let rate    = BENCH_SAMPLES as f64 / elapsed;

        // Noise floor: std-dev of gyro_z over still samples.
        let gz_vals: Vec<f32> = samples.iter().map(|s| s.gyro_z).collect();
        let mean = gz_vals.iter().sum::<f32>() / gz_vals.len() as f32;
        let var  = gz_vals.iter().map(|&v| (v - mean).powi(2)).sum::<f32>()
            / gz_vals.len() as f32;
        let std_gz = var.sqrt();

        let az_vals: Vec<f32> = samples.iter().map(|s| s.accel_z).collect();
        let az_mean = az_vals.iter().sum::<f32>() / az_vals.len() as f32;

        println!(
            "\n{BENCH_SAMPLES} samples in {elapsed:.2}s  ({rate:.0} Hz)\n\
             gyro_z  mean={mean:.5} rad/s  std={std_gz:.5} rad/s\n\
             accel_z mean={az_mean:.4} m/s²  (expect ~9.81)"
        );

        // Print first 5 and last 5.
        println!("\nFirst 5:");
        for s in samples.iter().take(5) {
            println!(
                "  t={:>6}ms  a=({:>7.3},{:>7.3},{:>7.3})  g=({:>7.4},{:>7.4},{:>7.4})",
                s.t_ms, s.accel_x, s.accel_y, s.accel_z,
                s.gyro_x, s.gyro_y, s.gyro_z,
            );
        }
        println!("Last 5:");
        for s in samples.iter().rev().take(5).rev() {
            println!(
                "  t={:>6}ms  a=({:>7.3},{:>7.3},{:>7.3})  g=({:>7.4},{:>7.4},{:>7.4})",
                s.t_ms, s.accel_x, s.accel_y, s.accel_z,
                s.gyro_x, s.gyro_y, s.gyro_z,
            );
        }
    }

    Ok(())
}
