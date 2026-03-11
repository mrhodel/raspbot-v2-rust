//! Gimbal bring-up and direction verification test.
//!
//! Moves the pan and tilt servos through key positions so you can verify
//! that the direction mapping is correct for this particular servo mounting.
//!
//! Usage (on the Pi):
//!   sudo ./target/release/examples/gimbal_test
//!   sudo ./target/release/examples/gimbal_test -- --sweep      # continuous sweep
//!   sudo ./target/release/examples/gimbal_test -- --raw 90 30  # raw servo angles

use std::{io::{stdin, stdout, Write}, thread::sleep, time::Duration};

use rppal::i2c::I2c;

// ── Yahboom board constants ───────────────────────────────────────────────────
const I2C_BUS:       u8  = 1;
const I2C_ADDR:      u16 = 0x2B;
const REG_SERVO:     u8  = 0x02;
const SERVO_PAN:     u8  = 1;
const SERVO_TILT:    u8  = 2;
const PAN_CENTER:    u8  = 90;
const TILT_NEUTRAL:  u8  = 30;   // raw degrees that level the camera
const STEP_MS:       u64 = 30;   // ms between incremental steps

fn write_servo(i2c: &mut I2c, id: u8, raw: u8) -> anyhow::Result<()> {
    i2c.block_write(REG_SERVO, &[id, raw])?;
    sleep(Duration::from_millis(20));
    Ok(())
}

/// Move servo smoothly from current to target in 1° steps.
fn move_to(i2c: &mut I2c, id: u8, from_raw: u8, to_raw: u8) -> anyhow::Result<()> {
    let (start, end) = (from_raw as i16, to_raw as i16);
    let step: i16 = if end >= start { 1 } else { -1 };
    let mut pos = start;
    while pos != end {
        write_servo(i2c, id, pos as u8)?;
        sleep(Duration::from_millis(STEP_MS));
        pos += step;
    }
    write_servo(i2c, id, to_raw)?;
    Ok(())
}

fn pause(msg: &str) {
    print!("{msg}  [Enter] ");
    stdout().flush().unwrap();
    let mut s = String::new();
    stdin().read_line(&mut s).unwrap();
}

fn run_verification(i2c: &mut I2c) -> anyhow::Result<()> {
    println!("\n── Gimbal direction verification ────────────────────────────");
    println!("Watch the camera and confirm each movement matches the label.\n");

    // ── Center ────────────────────────────────────────────────────────────────
    println!("Moving to CENTER (pan=0°, tilt=0°/level)...");
    move_to(i2c, SERVO_PAN,  i2c_read_dummy(), PAN_CENTER)?;
    move_to(i2c, SERVO_TILT, i2c_read_dummy(), TILT_NEUTRAL)?;
    pause("Camera should be pointing straight ahead, level.");

    // ── Pan left ──────────────────────────────────────────────────────────────
    println!("Pan LEFT  (pan_deg = -90 → raw=0°)...");
    move_to(i2c, SERVO_PAN, PAN_CENTER, 0)?;
    pause("Camera should be turned to the LEFT.");

    // ── Pan right ─────────────────────────────────────────────────────────────
    println!("Pan RIGHT (pan_deg = +90 → raw=180°)...");
    move_to(i2c, SERVO_PAN, 0, 180)?;
    pause("Camera should be turned to the RIGHT.");

    // ── Back to center ────────────────────────────────────────────────────────
    println!("Pan back to CENTER...");
    move_to(i2c, SERVO_PAN, 180, PAN_CENTER)?;

    // ── Tilt up ───────────────────────────────────────────────────────────────
    // tilt_deg=+30 → raw = TILT_NEUTRAL - 30 = 0
    println!("Tilt UP   (tilt_deg = +30 → raw=0°)...");
    move_to(i2c, SERVO_TILT, TILT_NEUTRAL, 0)?;
    pause("Camera should be tilted UP (looking at ceiling).");

    // ── Tilt down ─────────────────────────────────────────────────────────────
    // tilt_deg=-45 → raw = TILT_NEUTRAL + 45 = 75
    println!("Tilt DOWN (tilt_deg = -45 → raw=75°)...");
    move_to(i2c, SERVO_TILT, 0, 75)?;
    pause("Camera should be tilted DOWN (looking at floor).");

    // ── Return to neutral ─────────────────────────────────────────────────────
    println!("Returning to neutral (pan=0, tilt=level)...");
    move_to(i2c, SERVO_TILT, 75, TILT_NEUTRAL)?;
    move_to(i2c, SERVO_PAN,  PAN_CENTER, PAN_CENTER)?;

    println!("\nIf any direction was REVERSED, report which axis");
    println!("and we will flip the sign in the driver.\n");
    Ok(())
}

fn run_sweep(i2c: &mut I2c) -> anyhow::Result<()> {
    println!("\nSweep mode — Ctrl-C to stop");
    loop {
        // Pan sweep
        for raw in (0u8..=180).step_by(2) {
            write_servo(i2c, SERVO_PAN, raw)?;
            sleep(Duration::from_millis(STEP_MS));
        }
        for raw in (0u8..=180).rev().step_by(2) {
            write_servo(i2c, SERVO_PAN, raw)?;
            sleep(Duration::from_millis(STEP_MS));
        }
        write_servo(i2c, SERVO_PAN, PAN_CENTER)?;

        // Tilt sweep
        for raw in (0u8..=75).step_by(3) {
            write_servo(i2c, SERVO_TILT, raw)?;
            sleep(Duration::from_millis(STEP_MS));
        }
        for raw in (0u8..=75).rev().step_by(3) {
            write_servo(i2c, SERVO_TILT, raw)?;
            sleep(Duration::from_millis(STEP_MS));
        }
        write_servo(i2c, SERVO_TILT, TILT_NEUTRAL)?;
    }
}

fn run_raw(i2c: &mut I2c, pan_raw: u8, tilt_raw: u8) -> anyhow::Result<()> {
    println!("Raw command: pan servo1={pan_raw}°  tilt servo2={tilt_raw}°");
    write_servo(i2c, SERVO_PAN, pan_raw)?;
    write_servo(i2c, SERVO_TILT, tilt_raw)?;
    println!("Done.");
    Ok(())
}

// move_to needs a starting raw — since we can't read back servo position,
// start from center on first call.
fn i2c_read_dummy() -> u8 { 90 }

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    println!("Gimbal test  I2C-{I2C_BUS} @ {I2C_ADDR:#04x}");
    println!("Protocol: block_write(0x02, [servo_id, angle])");
    println!("  PAN  servo1: raw = 90 + pan_deg   (0°=left, 90°=center, 180°=right)");
    println!("  TILT servo2: raw = {TILT_NEUTRAL} - tilt_deg   (0°=up, {TILT_NEUTRAL}°=level, 75°=down)\n");

    let mut i2c = I2c::with_bus(I2C_BUS)?;
    i2c.set_slave_address(I2C_ADDR)?;

    if args.iter().any(|a| a == "--sweep") {
        run_sweep(&mut i2c)
    } else if args.iter().any(|a| a == "--raw") {
        let pan_raw  = args.iter().skip_while(|a| *a != "--raw").nth(1)
            .and_then(|s| s.parse().ok()).unwrap_or(PAN_CENTER);
        let tilt_raw = args.iter().skip_while(|a| *a != "--raw").nth(2)
            .and_then(|s| s.parse().ok()).unwrap_or(TILT_NEUTRAL);
        run_raw(&mut i2c, pan_raw, tilt_raw)
    } else {
        run_verification(&mut i2c)
    }
}
