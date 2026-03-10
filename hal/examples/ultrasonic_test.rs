//! Ultrasonic sensor bring-up test.
//!
//! Reads the HC-SR04 via the Yahboom expansion board (I2C bus 1, addr 0x2B) and
//! prints distance readings at ~3 Hz.  Hold objects at known distances to verify
//! calibration.
//!
//! Wire protocol (from Raspbot_Lib.py):
//!   Enable:  block_write(0x07, [1])
//!   Wait:    ~100 ms
//!   Read hi: block_read(0x1b, 1)
//!   Read lo: block_read(0x1a, 1)
//!   dist_mm: (hi << 8) | lo
//!
//! Usage (on the Pi):
//!   cargo run --example ultrasonic_test
//!   cargo run --example ultrasonic_test -- --samples 5    # heavier median filter
//!   cargo run --example ultrasonic_test -- --raw          # show raw bytes

use std::time::Instant;

use rppal::i2c::I2c;

// ── Yahboom board constants ────────────────────────────────────────────────────
const I2C_BUS:       u8  = 1;
const I2C_ADDR:      u16 = 0x2B;
const REG_ENABLE:    u8  = 0x07;
const REG_DIST_H:    u8  = 0x1b;
const REG_DIST_L:    u8  = 0x1a;
const SETTLE_MS:     u64 = 100;
const INTER_SAMPLE:  u64 = 20;
const LOOP_PERIOD:   u64 = 400;  // ms between printed readings (~2.5 Hz)

fn parse_args() -> (u8, bool) {
    let args: Vec<String> = std::env::args().collect();
    let mut samples = 3u8;
    let mut raw = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--samples" | "-s" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    samples = v.parse().unwrap_or(3);
                }
            }
            "--raw" | "-r" => raw = true,
            _ => {}
        }
        i += 1;
    }
    (samples, raw)
}

fn read_once(i2c: &mut I2c, show_raw: bool) -> anyhow::Result<f32> {
    let mut hi = [0u8; 1];
    let mut lo = [0u8; 1];
    i2c.block_read(REG_DIST_H, &mut hi)?;
    i2c.block_read(REG_DIST_L, &mut lo)?;

    if show_raw {
        print!("  hi={:#04x} lo={:#04x}", hi[0], lo[0]);
    }

    let dist_mm = ((hi[0] as u32) << 8) | lo[0] as u32;
    if dist_mm == 0 {
        return Ok(2.0); // blind spot
    }
    Ok((dist_mm as f32 / 10.0).clamp(2.0, 400.0))
}

fn main() -> anyhow::Result<()> {
    let (n_samples, show_raw) = parse_args();

    println!("Ultrasonic bring-up  I2C-{I2C_BUS} @ {I2C_ADDR:#04x}  {n_samples} sample(s)");
    println!("Protocol: enable reg=0x07, read hi=0x1b lo=0x1a");
    println!("Hold objects at known distances; Ctrl-C to exit.\n");

    let mut i2c = I2c::with_bus(I2C_BUS)?;
    i2c.set_slave_address(I2C_ADDR)?;

    let t0 = Instant::now();
    let mut iteration = 0u32;

    loop {
        let iter_start = std::time::Instant::now();

        // Enable sensor and wait for it to settle.
        i2c.block_write(REG_ENABLE, &[1])?;
        std::thread::sleep(std::time::Duration::from_millis(SETTLE_MS));

        let mut readings: Vec<f32> = Vec::with_capacity(n_samples as usize);
        for s in 0..n_samples {
            match read_once(&mut i2c, show_raw && s == 0) {
                Ok(cm) => readings.push(cm),
                Err(e) => eprintln!("  sample {s} error: {e}"),
            }
            if s + 1 < n_samples {
                std::thread::sleep(std::time::Duration::from_millis(INTER_SAMPLE));
            }
        }

        // Disable sensor between readings.
        let _ = i2c.block_write(REG_ENABLE, &[0]);

        if readings.is_empty() {
            eprintln!("All samples failed — check wiring.");
            std::thread::sleep(std::time::Duration::from_millis(500));
            continue;
        }

        readings.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = readings[readings.len() / 2];
        let t_s = t0.elapsed().as_secs_f32();

        if show_raw { println!(); }

        println!(
            "[{t_s:7.2}s #{iteration:04}]  {median:6.1} cm{}",
            if median < 15.0 {
                "  *** CLOSE ***"
            } else if median > 350.0 {
                "  (no echo / max range)"
            } else {
                ""
            }
        );
        iteration += 1;

        let elapsed_ms = iter_start.elapsed().as_millis() as u64;
        if elapsed_ms < LOOP_PERIOD {
            std::thread::sleep(std::time::Duration::from_millis(LOOP_PERIOD - elapsed_ms));
        }
    }
}
