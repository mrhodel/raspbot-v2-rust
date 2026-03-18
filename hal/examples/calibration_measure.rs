//! Calibration measurement harness: run motors at fixed duty, integrate IMU, report distance/rotation.
//!
//! Usage:
//!   calibration_measure forward 6        [run all 4 motors forward for 6 sec, measure distance via accel]
//!   calibration_measure rotate 6         [run diagonal motors for rotation, measure via gyro]

use std::time::{Duration, Instant};
use std::io::Write;
use anyhow::{Context, Result};
use rppal::i2c::I2c;

// Motor and IMU I2C constants (from motor_test.rs)
const MOTOR_BUS:  u8  = 1;
const MOTOR_ADDR: u16 = 0x2B;
const REG_MOTOR:  u8  = 0x01;

const MPU_BUS:          u8  = 6;
const MPU_ADDR:         u16 = 0x68;
const REG_PWR_MGMT_1:   u8  = 0x6B;
const REG_GYRO_CONFIG:  u8  = 0x1B;
const REG_ACCEL_CONFIG: u8  = 0x1C;
const REG_ACCEL_XOUT_H: u8  = 0x3B;

const ACCEL_SCALE: f32 = 9.81 / 4096.0;  // ±8 g → m/s²
const GYRO_SCALE:  f32 = std::f32::consts::PI / 180.0 / 65.5;  // ±500 °/s → rad/s

const SAMPLE_MS:   u64 = 50; // 20 Hz sampling

#[derive(Debug, Clone, Copy)]
struct ImuSample {
    ax: f32, ay: f32, az: f32,
    gx: f32, gy: f32, gz: f32,
}

fn mpu_init(i2c: &mut I2c) -> Result<()> {
    i2c.block_write(REG_PWR_MGMT_1,  &[0x00]).context("MPU wake-up")?;
    std::thread::sleep(Duration::from_millis(100));  // Critical: wait for MPU to wake
    i2c.block_write(REG_GYRO_CONFIG,  &[0x08]).context("MPU gyro ±500 °/s")?;
    i2c.block_write(REG_ACCEL_CONFIG, &[0x10]).context("MPU accel ±8 g")?;
    Ok(())
}

fn mpu_read(i2c: &I2c) -> Result<ImuSample> {
    let mut regs = [0u8; 14];
    i2c.block_read(REG_ACCEL_XOUT_H, &mut regs)
        .context("MPU read accel+gyro")?;

    let ax = i16::from_be_bytes([regs[0], regs[1]]) as f32 * ACCEL_SCALE;
    let ay = i16::from_be_bytes([regs[2], regs[3]]) as f32 * ACCEL_SCALE;
    let az = i16::from_be_bytes([regs[4], regs[5]]) as f32 * ACCEL_SCALE;
    let gx = i16::from_be_bytes([regs[8], regs[9]]) as f32 * GYRO_SCALE;
    let gy = i16::from_be_bytes([regs[10], regs[11]]) as f32 * GYRO_SCALE;
    let gz = i16::from_be_bytes([regs[12], regs[13]]) as f32 * GYRO_SCALE;

    Ok(ImuSample { ax, ay, az, gx, gy, gz })
}

fn motor_set(i2c: &mut I2c, motor_id: u8, duty: i8) -> Result<()> {
    let (dir, speed) = if duty >= 0 {
        (0u8, (duty as u16 * 255 / 100) as u8)
    } else {
        (1u8, ((-duty as i16) as u16 * 255 / 100) as u8)
    };
    i2c.block_write(REG_MOTOR, &[motor_id, dir, speed])
        .context("Motor I2C write failed")?;
    std::thread::sleep(Duration::from_millis(10));
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: calibration_measure <forward|rotate> <duration_sec>");
        std::process::exit(1);
    }

    let mode = &args[1];
    let duration_sec: u64 = args[2].parse()?;
    let duration = Duration::from_secs(duration_sec);

    println!("Calibration Measurement: {} for {} seconds", mode, duration_sec);
    println!("================================================");

    let mut motor_i2c = I2c::with_bus(1).context("Open I2C-1 for motors")?;
    motor_i2c.set_slave_address(MOTOR_ADDR).context("Set motor I2C address")?;

    let mut mpu_i2c = I2c::with_bus(6).context("Open I2C-6 for IMU")?;
    mpu_i2c.set_slave_address(MPU_ADDR).context("Set MPU I2C address")?;

    mpu_init(&mut mpu_i2c)?;

    // Run test and always stop motors, even on error
    let result = execute_test(&mut motor_i2c, &mut mpu_i2c, mode, duration, duration_sec);

    // CRITICAL: Stop motors multiple times to ensure they stop
    eprintln!("\n[SAFETY] Stopping motors...");
    for attempt in 1..=5 {
        for id in 0..4 {
            let _ = motor_i2c.block_write(REG_MOTOR, &[id, 0, 0]);
        }
        std::thread::sleep(Duration::from_millis(50));
        eprintln!("  Motors stop command #{}", attempt);
    }
    eprintln!("[SAFETY] Motors should be stopped.\n");

    result
}

fn execute_test(motor_i2c: &mut I2c, mpu_i2c: &I2c, mode: &str, duration: Duration, duration_sec: u64) -> Result<()> {

    // Start motors based on mode
    let duty: i8 = 30; // 30% duty
    println!("Starting motors at {}% duty...", duty);

    match mode {
        "forward" => {
            // All 4 motors forward
            motor_set(motor_i2c, 0, duty)?;   // FL
            motor_set(motor_i2c, 1, duty)?;   // RL
            motor_set(motor_i2c, 2, duty)?;   // FR
            motor_set(motor_i2c, 3, duty)?;   // RR
        }
        "rotate" => {
            // Correct CW rotation from motor_test.rs: FL(+), RL(+), FR(-), RR(-)
            // Motor IDs: 0=FL, 1=RL, 2=FR, 3=RR
            motor_set(motor_i2c, 0, duty)?;   // FL = +
            motor_set(motor_i2c, 1, duty)?;   // RL = +
            motor_set(motor_i2c, 2, -duty)?;  // FR = -
            motor_set(motor_i2c, 3, -duty)?;  // RR = -
        }
        _ => {
            eprintln!("Invalid mode: {}. Use 'forward' or 'rotate'", mode);
            std::process::exit(1);
        }
    }

    // Collect IMU samples
    let start = Instant::now();
    let mut samples: Vec<(f64, ImuSample)> = Vec::new();

    println!("Collecting IMU samples for {} sec...", duration_sec);
    while start.elapsed() < duration {
        let t_sec = start.elapsed().as_secs_f64();
        let sample = mpu_read(&mpu_i2c)?;
        samples.push((t_sec, sample));

        print!(".");
        std::io::stdout().flush()?;
        std::thread::sleep(Duration::from_millis(SAMPLE_MS));
    }
    println!("\nCollected {} samples", samples.len());

    // Stop motors (send twice to be safe)
    for _ in 0..2 {
        motor_set(motor_i2c, 0, 0)?;
        motor_set(motor_i2c, 1, 0)?;
        motor_set(motor_i2c, 2, 0)?;
        motor_set(motor_i2c, 3, 0)?;
        std::thread::sleep(Duration::from_millis(50));
    }
    println!("Motors stopped.");

    // Analysis
    println!("\n================================================");
    println!("Analysis Results:");
    println!("================================================");

    match mode {
        "forward" => {
            // Integrate accel_x (forward direction) to get speed
            let mut velocity = 0.0f64;
            let mut distance = 0.0f64;

            for i in 0..samples.len() {
                let (_, sample) = samples[i];
                let ax_f64 = sample.ax as f64;

                let dt = if i > 0 {
                    (samples[i].0 - samples[i - 1].0) as f64
                } else {
                    0.05
                };

                // Integrate twice: v = ∫a dt, x = ∫v dt
                velocity += ax_f64 * dt;
                distance += velocity * dt;
            }

            let speed = distance / duration_sec as f64;
            println!("Forward Speed Measurement:");
            println!("  Total distance (integrated): {:.3} m", distance);
            println!("  Duration: {} sec", duration_sec);
            println!("  Computed speed: {:.3} m/s", speed);
            println!("  Sim constant: 0.30 m/s");
            println!("  Error: {:.1}%", ((speed - 0.30) / 0.30 * 100.0).abs());
        }
        "rotate" => {
            // Integrate gyro_z to get total rotation
            let mut rotation_rad = 0.0f64;

            for i in 0..samples.len() {
                let (_, sample) = samples[i];
                let gz_rad = sample.gz as f64;

                let dt = if i > 0 {
                    (samples[i].0 - samples[i - 1].0) as f64
                } else {
                    0.05
                };

                rotation_rad += gz_rad * dt;
            }

            let omega = rotation_rad / duration_sec as f64;

            println!("Rotation Rate Measurement:");
            println!("  Total rotation (integrated): {:.1}° ({:.3} rad)",
                     rotation_rad * 180.0 / std::f32::consts::PI as f64, rotation_rad);
            println!("  Duration: {} sec", duration_sec);
            println!("  Computed omega: {:.3} rad/s", omega);
            println!("  Sim constant: 1.0 rad/s");
            println!("  Error: {:.1}%", ((omega - 1.0) / 1.0 * 100.0).abs());
        }
        _ => {}
    }

    println!("\n================================================");
    println!("Raw IMU samples (first 5 and last 5):");
    println!("  t(s)   ax(m/s²)  ay(m/s²)  az(m/s²)  gx(r/s)  gy(r/s)  gz(r/s)");

    for i in 0..samples.len().min(5) {
        let (t, s) = samples[i];
        println!("  {:.2}   {:7.2}   {:7.2}   {:7.2}   {:6.3}  {:6.3}  {:6.3}",
                 t, s.ax, s.ay, s.az, s.gx, s.gy, s.gz);
    }
    if samples.len() > 10 {
        println!("  ...");
    }
    for i in samples.len().saturating_sub(5)..samples.len() {
        let (t, s) = samples[i];
        println!("  {:.2}   {:7.2}   {:7.2}   {:7.2}   {:6.3}  {:6.3}  {:6.3}",
                 t, s.ax, s.ay, s.az, s.gx, s.gy, s.gz);
    }

    Ok(())
}

