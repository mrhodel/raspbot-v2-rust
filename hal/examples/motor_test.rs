//! Yahboom motor hardware test.
//!
//! Tests each wheel individually, then tests combined motions.
//! SAFETY: Run with the robot elevated — wheels must be off the ground.
//!
//! Usage (on Pi):
//!   cargo run --example motor_test -p hal
//!
//! The test prints what it is doing before each move. Watch the wheels and
//! verify direction. Any wheel spinning backward when "forward" is expected
//! indicates a wiring or ID mapping issue — note it and report.

use std::time::Duration;
use anyhow::Result;
use rppal::i2c::I2c;

// ── Yahboom board constants (same as motor.rs) ────────────────────────────────

const I2C_BUS:  u8  = 1;
const I2C_ADDR: u16 = 0x2B;
const REG_MOTOR: u8 = 0x01;

// Motor IDs on the Yahboom board
const M_FL: u8 = 0;
const M_RL: u8 = 1;
const M_FR: u8 = 2;
const M_RR: u8 = 3;

// Speed used for all tests (duty cycle out of 100)
const TEST_SPEED: i8 = 20;
// How long each move runs
const MOVE_S: u64 = 1;
// Pause between tests
const PAUSE_S: u64 = 1;

// ── I2C helpers ───────────────────────────────────────────────────────────────

fn write_motor(i2c: &mut I2c, motor_id: u8, duty: i8) -> Result<()> {
    let (dir, speed) = if duty >= 0 {
        (0u8, (duty as u16 * 255 / 100) as u8)
    } else {
        (1u8, ((-duty as i16) as u16 * 255 / 100) as u8)
    };
    i2c.block_write(REG_MOTOR, &[motor_id, dir, speed])?;
    std::thread::sleep(Duration::from_millis(10));
    Ok(())
}

fn set_motors(i2c: &mut I2c, fl: i8, fr: i8, rl: i8, rr: i8) -> Result<()> {
    // Board order: FL(0), RL(1), FR(2), RR(3)
    for (id, duty) in [(M_FL, fl), (M_RL, rl), (M_FR, fr), (M_RR, rr)] {
        write_motor(i2c, id, duty)?;
    }
    Ok(())
}

fn stop_all(i2c: &mut I2c) -> Result<()> {
    set_motors(i2c, 0, 0, 0, 0)
}

// ── Test runner ───────────────────────────────────────────────────────────────

struct Test<'a> {
    name:    &'a str,
    fl:      i8,
    fr:      i8,
    rl:      i8,
    rr:      i8,
    note:    &'a str,
}

impl<'a> Test<'a> {
    fn run(&self, i2c: &mut I2c) -> Result<()> {
        println!("\n[TEST] {}", self.name);
        if !self.note.is_empty() {
            println!("       {}", self.note);
        }
        println!("       FL={:+3}  FR={:+3}  RL={:+3}  RR={:+3}",
                 self.fl, self.fr, self.rl, self.rr);

        set_motors(i2c, self.fl, self.fr, self.rl, self.rr)?;
        std::thread::sleep(Duration::from_secs(MOVE_S));

        println!("       -- stop --");
        stop_all(i2c)?;
        std::thread::sleep(Duration::from_secs(PAUSE_S));
        Ok(())
    }
}

// ── main ─────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    println!("=============================================================");
    println!(" Yahboom Raspbot V2  —  Motor Hardware Test");
    println!("=============================================================");
    println!();
    println!(" SAFETY CHECK: wheels must be OFF THE GROUND before proceeding.");
    println!(" Each motor runs for {} second(s) at {}% duty cycle.", MOVE_S, TEST_SPEED);
    println!();
    println!(" Motor layout (top view, front = top):");
    println!("    FL --- FR");
    println!("    |       |");
    println!("    RL --- RR");
    println!();

    for i in (1..=3).rev() {
        println!("Starting in {}...", i);
        std::thread::sleep(Duration::from_secs(1));
    }
    println!("GO!\n");

    // Open I2C
    let mut i2c = I2c::with_bus(I2C_BUS)?;
    i2c.set_slave_address(I2C_ADDR)?;
    println!("I2C opened: /dev/i2c-{I2C_BUS} @ 0x{I2C_ADDR:02X}");

    // Ensure stopped
    stop_all(&mut i2c)?;
    std::thread::sleep(Duration::from_millis(200));

    let s = TEST_SPEED;

    let tests: &[Test] = &[
        // ── Individual wheels ──────────────────────────────────────────────
        Test { name: "FL only (Front-Left)",  fl: s, fr:0, rl:0, rr:0,
               note: "Only the front-left wheel should spin." },
        Test { name: "FR only (Front-Right)", fl:0, fr: s, rl:0, rr:0,
               note: "Only the front-right wheel should spin." },
        Test { name: "RL only (Rear-Left)",   fl:0, fr:0, rl: s, rr:0,
               note: "Only the rear-left wheel should spin." },
        Test { name: "RR only (Rear-Right)",  fl:0, fr:0, rl:0, rr: s,
               note: "Only the rear-right wheel should spin." },

        // ── Combined motions ───────────────────────────────────────────────
        Test { name: "Forward",     fl: s, fr: s, rl: s, rr: s,
               note: "All wheels forward — robot would move toward front." },
        Test { name: "Backward",    fl:-s, fr:-s, rl:-s, rr:-s,
               note: "All wheels backward — robot would move toward rear." },
        Test { name: "Rotate CW",   fl: s, fr:-s, rl: s, rr:-s,
               note: "Left wheels fwd, right wheels rev — robot would spin clockwise." },
        Test { name: "Rotate CCW",  fl:-s, fr: s, rl:-s, rr: s,
               note: "Left wheels rev, right wheels fwd — robot would spin counter-clockwise." },
        Test { name: "Strafe Right",fl: s, fr:-s, rl:-s, rr: s,
               note: "Robot would slide right (no rotation)." },
        Test { name: "Strafe Left", fl:-s, fr: s, rl: s, rr:-s,
               note: "Robot would slide left (no rotation)." },
    ];

    let mut failed = false;
    for test in tests {
        if let Err(e) = test.run(&mut i2c) {
            eprintln!("[ERROR] {}: {e}", test.name);
            failed = true;
            // Still try to stop before aborting
            let _ = stop_all(&mut i2c);
            break;
        }
    }

    // Final stop
    let _ = stop_all(&mut i2c);

    println!();
    if failed {
        println!("Test FAILED — check errors above.");
        std::process::exit(1);
    } else {
        println!("=============================================================");
        println!(" All tests complete.");
        println!(" Review wheel directions above and note any that spun");
        println!(" opposite to the expected direction.");
        println!("=============================================================");
    }
    Ok(())
}
