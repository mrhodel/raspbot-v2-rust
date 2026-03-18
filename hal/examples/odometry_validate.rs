//! Phase 14.2.5: Validate odometry on real robot with calibrated kinematics
//!
//! Test: Drive forward 2m → rotate 360° → measure final pose error via IMU
//! Uses IMU dead reckoning to evaluate calibration accuracy on actual floor
//!
//! Usage:
//!   odometry_validate <duration_sec>
//! Example:
//!   odometry_validate 30

use std::time::{Duration, Instant};
use std::io::Write;
use anyhow::{Context, Result};
use rppal::i2c::I2c;

// Motor and IMU I2C constants
const MOTOR_ADDR: u16 = 0x2B;
const REG_MOTOR: u8 = 0x01;
const MPU_ADDR: u16 = 0x68;
const REG_PWR_MGMT_1: u8 = 0x6B;
const REG_GYRO_CONFIG: u8 = 0x1B;
const REG_ACCEL_CONFIG: u8 = 0x1C;
const REG_ACCEL_XOUT_H: u8 = 0x3B;

const ACCEL_SCALE: f32 = 9.81 / 4096.0;  // ±8g → m/s²
const GYRO_SCALE: f32 = std::f32::consts::PI / 180.0 / 65.5;  // ±500°/s → rad/s

const SAMPLE_MS: u64 = 50;  // 20 Hz sampling

// Calibrated kinematics (Phase 14.2.4: wheels-on measurement)
const FORWARD_SPEED_M_S: f32 = 1.61;
const ROTATION_RATE_RAD_S: f32 = 13.7;  // Updated from 14.3 (wheels-off) to 13.7 (wheels-on floor)

#[derive(Debug, Clone, Copy)]
struct ImuSample {
    ax: f32, ay: f32, az: f32,
    gx: f32, gy: f32, gz: f32,
}

#[derive(Debug, Clone, Copy)]
struct Pose {
    x_m: f32,
    y_m: f32,
    theta_rad: f32,
}

fn mpu_init(i2c: &mut I2c) -> Result<()> {
    i2c.block_write(REG_PWR_MGMT_1, &[0x00]).context("MPU wake-up")?;
    std::thread::sleep(Duration::from_millis(100));  // Critical: wait for MPU to wake
    i2c.block_write(REG_GYRO_CONFIG, &[0x08]).context("MPU gyro ±500 °/s")?;
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
    i2c.block_write(REG_MOTOR, &[motor_id, dir, speed])?;
    std::thread::sleep(Duration::from_millis(10));
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: odometry_validate <duration_sec>");
        std::process::exit(1);
    }

    let duration_sec: u64 = args[1].parse()?;
    let duration = Duration::from_secs(duration_sec);

    println!("════════════════════════════════════════════════════════════");
    println!("Phase 14.2.5: Odometry Validation (Real Robot IMU Test)");
    println!("════════════════════════════════════════════════════════════\n");

    println!("Test: Drive forward 2m → Rotate 360° → measure final pose error");
    println!("Kinematics: {} m/s forward, {} rad/s rotation\n", FORWARD_SPEED_M_S, ROTATION_RATE_RAD_S);

    // Initialize I2C
    let mut motor_i2c = I2c::with_bus(1).context("Open I2C-1 for motors")?;
    motor_i2c.set_slave_address(MOTOR_ADDR).context("Set motor I2C address")?;

    let mut mpu_i2c = I2c::with_bus(6).context("Open I2C-6 for IMU")?;
    mpu_i2c.set_slave_address(MPU_ADDR).context("Set MPU I2C address")?;

    mpu_init(&mut mpu_i2c)?;

    // Run test and always stop motors
    let result = run_odometry_test(&mut motor_i2c, &mut mpu_i2c, duration, duration_sec);

    // CRITICAL: Stop motors
    eprintln!("\n[SAFETY] Stopping motors...");
    for attempt in 1..=5 {
        for id in 0..4 {
            let _ = motor_i2c.block_write(REG_MOTOR, &[id, 0, 0]);
        }
        std::thread::sleep(Duration::from_millis(50));
        eprintln!("  Motors stop command #{}", attempt);
    }
    eprintln!("[SAFETY] Motors stopped.\n");

    result
}

fn run_odometry_test(motor_i2c: &mut I2c, mpu_i2c: &I2c, duration: Duration, duration_sec: u64) -> Result<()> {
    let duty: i8 = 35;  // Safe duty for loaded wheels on floor
    println!("Starting motors at {}% duty...", duty);

    // Collect IMU samples during motion
    let start = Instant::now();
    let mut samples: Vec<(f64, ImuSample)> = Vec::new();
    let mut phase = "forward";
    let mut phase_start = Instant::now();
    const FORWARD_DURATION_S: f64 = 2.0 / FORWARD_SPEED_M_S as f64;  // Time to travel 2m
    const ROTATION_DURATION_S: f64 = (2.0 * std::f32::consts::PI / ROTATION_RATE_RAD_S) as f64;  // Time for 360°

    // Phase 1: Forward
    println!("Phase 1: Drive forward 2m ({:.1}s)...", FORWARD_DURATION_S);
    motor_set(motor_i2c, 0, duty)?;   // FL
    motor_set(motor_i2c, 1, duty)?;   // RL
    motor_set(motor_i2c, 2, duty)?;   // FR
    motor_set(motor_i2c, 3, duty)?;   // RR

    while start.elapsed() < Duration::from_secs_f64(FORWARD_DURATION_S) {
        let t_sec = start.elapsed().as_secs_f64();
        let sample = mpu_read(&mpu_i2c)?;
        samples.push((t_sec, sample));
        print!(".");
        std::io::stdout().flush()?;
        std::thread::sleep(Duration::from_millis(SAMPLE_MS));
    }
    println!(" done");

    // Phase 2: Rotate 360°
    phase_start = Instant::now();
    phase = "rotate";
    println!("Phase 2: Rotate 360° ({:.1}s)...", ROTATION_DURATION_S);
    motor_set(motor_i2c, 0, duty)?;   // FL
    motor_set(motor_i2c, 1, duty)?;   // RL
    motor_set(motor_i2c, 2, -duty)?;  // FR
    motor_set(motor_i2c, 3, -duty)?;  // RR

    let rotation_start = start.elapsed().as_secs_f64();
    while start.elapsed().as_secs_f64() - rotation_start < ROTATION_DURATION_S {
        let t_sec = start.elapsed().as_secs_f64();
        let sample = mpu_read(&mpu_i2c)?;
        samples.push((t_sec, sample));
        print!(".");
        std::io::stdout().flush()?;
        std::thread::sleep(Duration::from_millis(SAMPLE_MS));
    }
    println!(" done");

    // Phase 3: Drive backward 2m (return to start)
    phase_start = Instant::now();
    phase = "return";
    println!("Phase 3: Drive backward 2m ({:.1}s)...", FORWARD_DURATION_S);
    motor_set(motor_i2c, 0, -duty)?;   // FL reverse
    motor_set(motor_i2c, 1, -duty)?;   // RL reverse
    motor_set(motor_i2c, 2, -duty)?;   // FR reverse
    motor_set(motor_i2c, 3, -duty)?;   // RR reverse

    let return_start = start.elapsed().as_secs_f64();
    while start.elapsed().as_secs_f64() - return_start < FORWARD_DURATION_S {
        let t_sec = start.elapsed().as_secs_f64();
        let sample = mpu_read(&mpu_i2c)?;
        samples.push((t_sec, sample));
        print!(".");
        std::io::stdout().flush()?;
        std::thread::sleep(Duration::from_millis(SAMPLE_MS));
    }
    println!(" done");

    println!("\nCollected {} IMU samples\n", samples.len());

    // Analyze: Integrate IMU to compute final pose
    let mut pose = Pose { x_m: 0.0, y_m: 0.0, theta_rad: 0.0 };
    let mut vx = 0.0f32;
    let mut vy = 0.0f32;

    // IMU biases (from robot_config.yaml, last calibrated 2026-03-13)
    const AX_BIAS: f32 = 0.364051;
    const AY_BIAS: f32 = -0.063712;
    const AZ_BIAS: f32 = 10.001171;
    const GX_BIAS: f32 = -0.012430;
    const GY_BIAS: f32 = 0.031440;
    const GZ_BIAS: f32 = -0.016137;

    println!("════════════════════════════════════════════════════════════");
    println!("Odometry Integration Results");
    println!("════════════════════════════════════════════════════════════\n");

    for i in 0..samples.len() {
        let (_, sample) = samples[i];
        let dt = if i > 0 {
            samples[i].0 - samples[i - 1].0
        } else {
            SAMPLE_MS as f64 / 1000.0
        } as f32;

        // Subtract bias from raw IMU values
        let ax_biased = sample.ax - AX_BIAS;
        let ay_biased = sample.ay - AY_BIAS;
        let az_biased = sample.az - AZ_BIAS;
        let gz_biased = sample.gz - GZ_BIAS;

        // Rotate acceleration to global frame (body frame to world frame)
        let cos_theta = pose.theta_rad.cos();
        let sin_theta = pose.theta_rad.sin();
        let ax_global = ax_biased * cos_theta - ay_biased * sin_theta;
        let ay_global = ax_biased * sin_theta + ay_biased * cos_theta;

        // Integrate acceleration to velocity
        vx += ax_global * dt;
        vy += ay_global * dt;

        // Integrate velocity to position
        pose.x_m += vx * dt;
        pose.y_m += vy * dt;

        // Integrate angular velocity (already bias-subtracted)
        pose.theta_rad += gz_biased * dt;
    }

    // Normalize heading to [-π, π]
    while pose.theta_rad > std::f32::consts::PI {
        pose.theta_rad -= 2.0 * std::f32::consts::PI;
    }
    while pose.theta_rad < -std::f32::consts::PI {
        pose.theta_rad += 2.0 * std::f32::consts::PI;
    }

    println!("Final pose (after forward/rotate/return):");
    println!("  x = {:.3} m (expected: ~0 m)", pose.x_m);
    println!("  y = {:.3} m (expected: ~0 m)", pose.y_m);
    println!("  θ = {:.2}° (expected: ~0°)\n", pose.theta_rad.to_degrees());

    let position_error = (pose.x_m.powi(2) + pose.y_m.powi(2)).sqrt();
    let heading_error = pose.theta_rad.abs();

    println!("════════════════════════════════════════════════════════════");
    println!("ODOMETRY ERROR SUMMARY");
    println!("════════════════════════════════════════════════════════════");
    println!("Position error: {:.3} m (distance from start)", position_error);
    println!("Heading error: {:.2}° (rotation from start)", heading_error.to_degrees());
    println!();
    println!("⚠ Note: IMU dead reckoning accumulates error over time.");
    println!("Perfect = 0, Good = <0.1m, Acceptable = <0.3m\n");

    Ok(())
}
