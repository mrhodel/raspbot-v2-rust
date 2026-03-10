//! Ultrasonic sensor bring-up test.
//!
//! Reads the HC-SR04 via the Yahboom expansion board (I2C bus 1, addr 0x2B) and
//! prints distance readings at ~3 Hz.  Hold objects at known distances to verify
//! calibration.
//!
//! Usage (on the Pi):
//!   cargo run --example ultrasonic_test
//!   cargo run --example ultrasonic_test -- --samples 5    # heavier median filter
//!   cargo run --example ultrasonic_test -- --raw          # print raw bytes too

use std::time::Instant;

use rppal::i2c::I2c;

// ── Yahboom board constants ────────────────────────────────────────────────────
const I2C_BUS:       u8  = 1;
const I2C_ADDR:      u16 = 0x2B;
const REG_US:        u8  = 0x02;
const TRIGGER_MS:    u64 = 60;
const INTER_SAMPLE:  u64 = 20;   // ms gap between consecutive samples
const LOOP_PERIOD:   u64 = 350;  // ms between printed readings (~3 Hz)

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

fn trigger_and_read(i2c: &mut I2c, raw: bool) -> anyhow::Result<f32> {
    // Trigger: send register byte 0x02.
    i2c.write(&[REG_US])?;
    std::thread::sleep(std::time::Duration::from_millis(TRIGGER_MS));

    // Read 3 bytes (24-bit big-endian mm).
    let mut buf = [0u8; 3];
    i2c.write_read(&[REG_US], &mut buf)?;

    if raw {
        print!("  raw=[{:#04x} {:#04x} {:#04x}]", buf[0], buf[1], buf[2]);
    }

    let dist_mm = ((buf[0] as u32) << 16) | ((buf[1] as u32) << 8) | buf[2] as u32;
    if dist_mm == 0 {
        // HC-SR04 blind spot (target < ~2 cm)
        return Ok(2.0);
    }
    Ok((dist_mm as f32 / 10.0).clamp(2.0, 400.0))
}

fn median_cm(samples: &mut Vec<f32>) -> f32 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn main() -> anyhow::Result<()> {
    let (n_samples, show_raw) = parse_args();

    println!("Ultrasonic bring-up  I2C-{I2C_BUS} @ {I2C_ADDR:#04x}  {n_samples} sample(s)");
    println!("Hold objects at known distances; Ctrl-C to exit.\n");

    let mut i2c = I2c::with_bus(I2C_BUS)?;
    i2c.set_slave_address(I2C_ADDR)?;

    let t0 = Instant::now();
    let mut iteration = 0u32;

    loop {
        let iter_start = std::time::Instant::now();
        let mut readings: Vec<f32> = Vec::with_capacity(n_samples as usize);

        for s in 0..n_samples {
            match trigger_and_read(&mut i2c, show_raw && s == 0) {
                Ok(cm) => readings.push(cm),
                Err(e) => eprintln!("  sample {s} error: {e}"),
            }
            if s + 1 < n_samples {
                std::thread::sleep(std::time::Duration::from_millis(INTER_SAMPLE));
            }
        }

        if readings.is_empty() {
            eprintln!("All samples failed — check wiring.");
            std::thread::sleep(std::time::Duration::from_millis(500));
            continue;
        }

        let med = median_cm(&mut readings);
        let t_s = t0.elapsed().as_secs_f32();

        if show_raw {
            println!();
        }
        println!(
            "[{t_s:7.2}s #{iteration:04}]  {med:6.1} cm{}",
            if med < 15.0 {
                "  *** CLOSE ***"
            } else if med > 350.0 {
                "  (no echo / max range)"
            } else {
                ""
            }
        );
        iteration += 1;

        // Sleep the remainder of LOOP_PERIOD
        let elapsed_ms = iter_start.elapsed().as_millis() as u64;
        if elapsed_ms < LOOP_PERIOD {
            std::thread::sleep(std::time::Duration::from_millis(LOOP_PERIOD - elapsed_ms));
        }
    }
}
