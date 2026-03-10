//! Ultrasonic sensor bring-up, verification, and calibration.
//!
//! Modes:
//!   (default)  continuous distance display + optional CSV log
//!   --calib    guided calibration at known distances; computes scale factor
//!              and offers to write it back to robot_config.yaml
//!
//! Usage (on the Pi):
//!   sudo cargo run --release --example ultrasonic_test
//!   sudo cargo run --release --example ultrasonic_test -- --log runs/us_log.csv
//!   sudo cargo run --release --example ultrasonic_test -- --calib
//!   sudo cargo run --release --example ultrasonic_test -- --calib --log runs/us_calib.csv
//!   sudo cargo run --release --example ultrasonic_test -- --samples 5 --log runs/us_log.csv

use std::{
    fs::File,
    io::{BufWriter, Write},
    time::Instant,
};

use rppal::i2c::I2c;

// ── Yahboom board constants ────────────────────────────────────────────────────
const I2C_BUS:      u8  = 1;
const I2C_ADDR:     u16 = 0x2B;
const REG_ENABLE:   u8  = 0x07;
const REG_DIST_H:   u8  = 0x1b;
const REG_DIST_L:   u8  = 0x1a;
const SETTLE_MS:    u64 = 100;
const INTER_SAMPLE: u64 = 20;   // ms between median-filter samples
const LOOP_PERIOD:  u64 = 350;  // ms between printed readings

// Calibration distances (cm). User places an object at each point in order.
const CALIB_POINTS_CM: &[f32] = &[5.0, 15.0, 30.0, 100.0, 200.0];
const CALIB_SAMPLES:   u32    = 20;  // readings per calibration point (~7 s each)
const CALIB_SAMPLE_MS: u64    = 350;

// ── Args ──────────────────────────────────────────────────────────────────────
struct Args {
    n_samples: u8,
    log_path:  Option<String>,
    calib:     bool,
    raw:       bool,
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let mut a = Args { n_samples: 3, log_path: None, calib: false, raw: false };
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--samples" | "-s" => {
                i += 1;
                if let Some(v) = argv.get(i) { a.n_samples = v.parse().unwrap_or(3); }
            }
            "--log" | "-l" => {
                i += 1;
                if let Some(v) = argv.get(i) { a.log_path = Some(v.clone()); }
            }
            "--calib" | "-c" => a.calib  = true,
            "--raw"   | "-r" => a.raw    = true,
            _ => {}
        }
        i += 1;
    }
    a
}

// ── I2C helpers ───────────────────────────────────────────────────────────────

/// Enable sensor, settle, read hi/lo N times, return (median_cm, raw bytes of first sample).
fn read_median(i2c: &mut I2c, n: u8) -> anyhow::Result<(f32, u8, u8)> {
    i2c.block_write(REG_ENABLE, &[1])?;
    std::thread::sleep(std::time::Duration::from_millis(SETTLE_MS));

    let mut samples: Vec<(f32, u8, u8)> = Vec::with_capacity(n as usize);
    for idx in 0..n {
        let mut hi = [0u8; 1];
        let mut lo = [0u8; 1];
        if i2c.block_read(REG_DIST_H, &mut hi).and_then(|_| i2c.block_read(REG_DIST_L, &mut lo)).is_ok() {
            let mm = ((hi[0] as u32) << 8) | lo[0] as u32;
            let cm = if mm == 0 { 2.0f32 } else { (mm as f32 / 10.0).clamp(2.0, 500.0) };
            samples.push((cm, hi[0], lo[0]));
        }
        if idx + 1 < n {
            std::thread::sleep(std::time::Duration::from_millis(INTER_SAMPLE));
        }
    }
    let _ = i2c.block_write(REG_ENABLE, &[0]);

    if samples.is_empty() {
        anyhow::bail!("all samples failed");
    }
    samples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mid = &samples[samples.len() / 2];
    Ok(*mid)
}

// ── Continuous mode ───────────────────────────────────────────────────────────

fn run_continuous(i2c: &mut I2c, args: &Args) -> anyhow::Result<()> {
    let log_file = args.log_path.as_ref().map(|p| {
        let f = BufWriter::new(File::create(p).expect("cannot create log file"));
        println!("Logging to: {p}");
        f
    });
    let mut log = log_file;
    if let Some(ref mut w) = log {
        writeln!(w, "t_ms,hi,lo,dist_mm,dist_cm")?;
    }

    let t0 = Instant::now();
    let mut n = 0u32;

    println!("\nReading every ~{LOOP_PERIOD}ms  ({} sample(s) per reading)\n", args.n_samples);

    loop {
        let iter_start = Instant::now();
        let t_ms = t0.elapsed().as_millis() as u64;

        match read_median(i2c, args.n_samples) {
            Ok((cm, hi, lo)) => {
                let mm = ((hi as u32) << 8) | lo as u32;
                let flag = if cm < 15.0 { "  *** CLOSE ***" } else if cm > 390.0 { "  (no echo)" } else { "" };

                if args.raw {
                    println!("[{t_ms:7}ms #{n:04}]  hi={hi:#04x} lo={lo:#04x}  {cm:6.1} cm{flag}");
                } else {
                    println!("[{t_ms:7}ms #{n:04}]  {cm:6.1} cm{flag}");
                }

                if let Some(ref mut w) = log {
                    writeln!(w, "{t_ms},{hi:#04x},{lo:#04x},{mm},{cm:.1}")?;
                    w.flush()?;
                }
            }
            Err(e) => eprintln!("[{t_ms:7}ms #{n:04}]  ERROR: {e}"),
        }
        n += 1;

        let elapsed = iter_start.elapsed().as_millis() as u64;
        if elapsed < LOOP_PERIOD {
            std::thread::sleep(std::time::Duration::from_millis(LOOP_PERIOD - elapsed));
        }
    }
}

// ── Calibration mode ──────────────────────────────────────────────────────────

fn run_calib(i2c: &mut I2c, args: &Args) -> anyhow::Result<()> {
    let log_file = args.log_path.as_ref().map(|p| {
        let f = BufWriter::new(File::create(p).expect("cannot create log file"));
        println!("Logging to: {p}");
        f
    });
    let mut log = log_file;
    if let Some(ref mut w) = log {
        writeln!(w, "actual_cm,t_ms,hi,lo,dist_mm,dist_cm")?;
    }

    println!("\nCalibration mode — {} points", CALIB_POINTS_CM.len());
    println!("For each point: place an object at the stated distance from the");
    println!("sensor face, press Enter, then hold still for ~{} s.\n",
             CALIB_SAMPLES * CALIB_SAMPLE_MS as u32 / 1000);

    let t0 = Instant::now();
    let mut point_results: Vec<(f32, f32)> = Vec::new(); // (actual, mean_measured)

    for &actual_cm in CALIB_POINTS_CM {
        print!("Place object at {actual_cm:.0} cm, then press Enter... ");
        std::io::stdout().flush()?;
        let mut line = String::new();
        std::io::stdin().read_line(&mut line)?;

        println!("  Measuring ({CALIB_SAMPLES} samples)...");
        let mut readings: Vec<f32> = Vec::with_capacity(CALIB_SAMPLES as usize);

        for s in 0..CALIB_SAMPLES {
            let t_ms = t0.elapsed().as_millis() as u64;
            match read_median(i2c, 3) {
                Ok((cm, hi, lo)) => {
                    let mm = ((hi as u32) << 8) | lo as u32;
                    readings.push(cm);
                    print!("  [{s:02}] {cm:6.1} cm\r");
                    std::io::stdout().flush()?;
                    if let Some(ref mut w) = log {
                        writeln!(w, "{actual_cm},{t_ms},{hi:#04x},{lo:#04x},{mm},{cm:.1}")?;
                        w.flush()?;
                    }
                }
                Err(e) => eprintln!("  [{s:02}] error: {e}"),
            }
            std::thread::sleep(std::time::Duration::from_millis(
                CALIB_SAMPLE_MS.saturating_sub(SETTLE_MS + 3 * INTER_SAMPLE),
            ));
        }
        println!();

        if readings.is_empty() {
            println!("  No valid readings — skipping point.");
            continue;
        }
        readings.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = readings[readings.len() / 2];
        let mean: f32 = readings.iter().sum::<f32>() / readings.len() as f32;
        let variance: f32 = readings.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
            / readings.len() as f32;
        let std_dev = variance.sqrt();
        let error = median - actual_cm;
        let error_pct = error / actual_cm * 100.0;
        println!(
            "  actual={actual_cm:.0} cm  median={median:.1} cm  \
             mean={mean:.1} cm  std={std_dev:.2} cm  \
             error={error:+.1} cm ({error_pct:+.1}%)"
        );
        point_results.push((actual_cm, median));
    }

    // ── Summary ───────────────────────────────────────────────────────────────
    println!("\n── Calibration Summary ─────────────────────────────────────");
    println!("{:<12} {:>12} {:>12} {:>12}", "Actual (cm)", "Measured (cm)", "Error (cm)", "Error (%)");
    println!("{}", "─".repeat(52));
    let mut scale_sum = 0.0f32;
    let mut scale_n = 0u32;
    for &(actual, measured) in &point_results {
        let err = measured - actual;
        let err_pct = err / actual * 100.0;
        let scale = actual / measured;
        println!("{:<12.0} {:>12.1} {:>+12.1} {:>+11.1}%", actual, measured, err, err_pct);
        if actual >= 10.0 { // exclude blind-spot point from scale calculation
            scale_sum += scale;
            scale_n += 1;
        }
    }
    println!("{}", "─".repeat(52));

    if scale_n > 0 {
        let mean_scale = scale_sum / scale_n as f32;
        println!("\nMean scale factor (actual/measured): {mean_scale:.4}");
        println!("  Current factor: 1.0000 (dist_cm = raw_mm / 10.0)");
        if (mean_scale - 1.0).abs() > 0.02 {
            println!("  Deviation > 2% — calibration recommended.");
        } else {
            println!("  Deviation < 2% — factory calibration adequate for safety interlock.");
        }

        // Offer to write scale factor to robot_config.yaml.
        print!("\nWrite calibration scale factor to robot_config.yaml? [y/N]: ");
        std::io::stdout().flush()?;
        let mut choice = String::new();
        std::io::stdin().read_line(&mut choice)?;
        if choice.trim().eq_ignore_ascii_case("y") {
            write_scale_to_config(mean_scale)?;
        } else {
            println!("Skipped. To apply manually, set us_scale_factor: {mean_scale:.4} in robot_config.yaml.");
        }
    }

    Ok(())
}

fn write_scale_to_config(scale: f32) -> anyhow::Result<()> {
    // Find the robot_config.yaml relative to the binary location.
    let config_paths = [
        "Documents/robot_config.yaml",
        "../Documents/robot_config.yaml",
        "../../Documents/robot_config.yaml",
    ];
    let path = config_paths.iter()
        .find(|p| std::path::Path::new(p).exists())
        .copied()
        .unwrap_or("Documents/robot_config.yaml");

    let content = std::fs::read_to_string(path)?;

    let new_line = format!("    scale_factor: {scale:.4}   # empirical calibration (actual/measured)");
    let updated = if content.contains("scale_factor:") {
        // Replace existing line.
        content
            .lines()
            .map(|l| if l.trim_start().starts_with("scale_factor:") { new_line.clone() } else { l.to_string() })
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        // Insert after samples_per_reading.
        content
            .lines()
            .flat_map(|l| {
                let mut v = vec![l.to_string()];
                if l.trim_start().starts_with("samples_per_reading:") {
                    v.push(new_line.clone());
                }
                v
            })
            .collect::<Vec<_>>()
            .join("\n")
    };

    std::fs::write(path, updated + "\n")?;
    println!("Wrote scale_factor: {scale:.4}  →  {path}");
    Ok(())
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    let args = parse_args();

    println!("Ultrasonic sensor test  I2C-{I2C_BUS} @ {I2C_ADDR:#04x}");
    println!("Protocol: enable=0x07, hi=0x1b, lo=0x1a");

    let mut i2c = I2c::with_bus(I2C_BUS)?;
    i2c.set_slave_address(I2C_ADDR)?;

    if args.calib {
        run_calib(&mut i2c, &args)
    } else {
        println!("Ctrl-C to exit.");
        run_continuous(&mut i2c, &args)
    }
}
