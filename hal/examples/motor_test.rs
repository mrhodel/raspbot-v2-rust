//! Yahboom motor hardware test — interactive.
//!
//! Runs each test, stops the motors, then waits for:
//!   [P] Pass   [F] Fail   [R] Repeat
//!
//! Results are summarised at the end.
//!
//! SAFETY: Run with the robot elevated — wheels must be off the ground.
//!
//! Usage (on Pi):
//!   cargo run --example motor_test -p hal

use std::io::{self, BufRead, Write};
use std::time::Duration;
use anyhow::Result;
use rppal::i2c::I2c;

// ── Constants ─────────────────────────────────────────────────────────────────

const I2C_BUS:   u8  = 1;
const I2C_ADDR:  u16 = 0x2B;
const REG_MOTOR: u8  = 0x01;

const M_FL: u8 = 0;
const M_RL: u8 = 1;
const M_FR: u8 = 2;
const M_RR: u8 = 3;

/// min_speed from robot_config.yaml — stall threshold
const TEST_SPEED: i8 = 20;
const MOVE_S:     u64 = 1;

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
    for (id, duty) in [(M_FL, fl), (M_RL, rl), (M_FR, fr), (M_RR, rr)] {
        write_motor(i2c, id, duty)?;
    }
    Ok(())
}

fn stop_all(i2c: &mut I2c) -> Result<()> {
    set_motors(i2c, 0, 0, 0, 0)
}

// ── Interactive prompt ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
enum Verdict { Pass, Fail }

fn prompt() -> Verdict {
    let stdin = io::stdin();
    loop {
        print!("  >> [P]ass  [F]ail  [R]epeat : ");
        io::stdout().flush().unwrap();
        let mut line = String::new();
        stdin.lock().read_line(&mut line).unwrap();
        match line.trim().to_uppercase().as_str() {
            "P" | "" => return Verdict::Pass,
            "F"      => return Verdict::Fail,
            "R"      => return Verdict::Pass, // caller checks separately
            _        => println!("     Enter P, F, or R."),
        }
    }
}

/// Run a test, stop, then prompt. Returns (Verdict, repeated).
/// On 'R', repeats the move once and re-prompts.
fn run_test(
    i2c:  &mut I2c,
    name: &str,
    note: &str,
    fl: i8, fr: i8, rl: i8, rr: i8,
) -> Result<Verdict> {
    loop {
        println!();
        println!("  [TEST] {name}");
        if !note.is_empty() {
            println!("         {note}");
        }
        println!("         FL={fl:+3}  FR={fr:+3}  RL={rl:+3}  RR={rr:+3}  ({MOVE_S}s)");

        set_motors(i2c, fl, fr, rl, rr)?;
        std::thread::sleep(Duration::from_secs(MOVE_S));
        stop_all(i2c)?;
        std::thread::sleep(Duration::from_millis(300));

        // Re-read stdin char to handle buffered newlines
        print!("  >> [P]ass  [F]ail  [R]epeat : ");
        io::stdout().flush().unwrap();
        let stdin = io::stdin();
        let mut line = String::new();
        stdin.lock().read_line(&mut line).unwrap();
        match line.trim().to_uppercase().as_str() {
            "P" | "" => return Ok(Verdict::Pass),
            "F"      => return Ok(Verdict::Fail),
            "R"      => continue,          // loop → repeat
            _        => {
                println!("     Unrecognised — treating as Pass.");
                return Ok(Verdict::Pass);
            }
        }
    }
}

// ── main ─────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    println!("================================================================");
    println!(" Yahboom Raspbot V2  —  Motor Hardware Test");
    println!(" Speed: {TEST_SPEED}% duty cycle (min_speed / stall threshold)");
    println!("================================================================");
    println!();
    println!(" SAFETY: wheels must be OFF THE GROUND.");
    println!();
    println!(" Motor layout (top view, front = top):");
    println!("    FL --- FR");
    println!("    |       |");
    println!("    RL --- RR");
    println!();

    for i in (1..=3).rev() {
        print!(" Starting in {i}...  \r");
        io::stdout().flush().unwrap();
        std::thread::sleep(Duration::from_secs(1));
    }
    println!(" GO!                  ");
    println!();

    let mut i2c = I2c::with_bus(I2C_BUS)?;
    i2c.set_slave_address(I2C_ADDR)?;
    println!(" I2C: /dev/i2c-{I2C_BUS} @ 0x{I2C_ADDR:02X}  OK");
    stop_all(&mut i2c)?;
    std::thread::sleep(Duration::from_millis(200));

    let s = TEST_SPEED;

    let tests: &[(&str, &str, i8, i8, i8, i8)] = &[
        // name,          note,                                       fl  fr  rl  rr
        ("FL (Front-Left)",  "Only front-left wheel should spin.",    s,  0,  0,  0 ),
        ("FR (Front-Right)", "Only front-right wheel should spin.",   0,  s,  0,  0 ),
        ("RL (Rear-Left)",   "Only rear-left wheel should spin.",     0,  0,  s,  0 ),
        ("RR (Rear-Right)",  "Only rear-right wheel should spin.",    0,  0,  0,  s ),
        ("Forward",          "All wheels fwd — robot moves forward.", s,  s,  s,  s ),
        ("Backward",         "All wheels rev — robot moves back.",   -s, -s, -s, -s ),
        ("Rotate CW",        "Left fwd / right rev — spins CW.",      s, -s,  s, -s ),
        ("Rotate CCW",       "Left rev / right fwd — spins CCW.",    -s,  s, -s,  s ),
        ("Strafe Right",     "Robot slides right (no rotation).",     s, -s, -s,  s ),
        ("Strafe Left",      "Robot slides left (no rotation).",     -s,  s,  s, -s ),
    ];

    let mut results: Vec<(&str, Verdict)> = Vec::new();

    for &(name, note, fl, fr, rl, rr) in tests {
        let verdict = run_test(&mut i2c, name, note, fl, fr, rl, rr)?;
        results.push((name, verdict));
    }

    // Final stop (belt-and-suspenders)
    let _ = stop_all(&mut i2c);

    // Summary
    println!();
    println!("================================================================");
    println!(" RESULTS");
    println!("================================================================");
    let mut any_fail = false;
    for (name, verdict) in &results {
        let tag = if *verdict == Verdict::Pass { "PASS" } else { "FAIL" };
        println!("  [{tag}]  {name}");
        if *verdict == Verdict::Fail { any_fail = true; }
    }
    println!("================================================================");
    if any_fail {
        println!(" Some tests FAILED — review motor IDs or wiring.");
        std::process::exit(1);
    } else {
        println!(" All tests passed.");
    }

    Ok(())
}
