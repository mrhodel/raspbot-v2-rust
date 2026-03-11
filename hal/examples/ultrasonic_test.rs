//! Ultrasonic sensor bring-up, verification, and calibration.
//!
//! Modes:
//!   (default)  continuous distance display + optional CSV log
//!   --calib    guided calibration; shows live reading while you position,
//!              press Enter to capture samples at current distance
//!
//! Usage (on the Pi):
//!   sudo ./target/release/examples/ultrasonic_test
//!   sudo ./target/release/examples/ultrasonic_test -- --log runs/us_log.csv
//!   sudo ./target/release/examples/ultrasonic_test -- --calib --log runs/us_calib.csv
//!   sudo ./target/release/examples/ultrasonic_test -- --samples 5 --raw

use std::{
    fs::File,
    io::{BufWriter, Write, stdin, stdout},
    sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}},
    time::Instant,
};

use rppal::i2c::I2c;

// ── Yahboom board constants ────────────────────────────────────────────────────
const I2C_BUS:        u8  = 1;
const I2C_ADDR:       u16 = 0x2B;
const REG_ENABLE:     u8  = 0x07;
const REG_DIST_H:     u8  = 0x1b;
const REG_DIST_L:     u8  = 0x1a;
const SETTLE_MS:      u64 = 100;
const INTER_SAMPLE:   u64 = 20;
const LOOP_PERIOD:    u64 = 350;   // ms between continuous-mode lines

// Calibration: test points (description, nominal_cm; 0.0 = observation only)
const CALIB_POINTS: &[(&str, f32)] = &[
    // Edge cases
    ("~ 350-400 cm  — move an object back until reading starts dropping or jumps",  0.0),
    ("  open space  — nothing in front, sensor aimed at far wall or ceiling (no echo)", 0.0),
    ("< 1 cm        — touch a flat object to the sensor face",                      0.0),
    // Calibration ladder (nominal_cm > 0 → included in scale factor)
    ("5 cm   (near minimum range)",                    5.0),
    ("15 cm  (safety-stop threshold — most important)", 15.0),
    ("30 cm  (typical close obstacle)",               30.0),
    ("100 cm",                                       100.0),
    ("200 cm",                                       200.0),
    // Surface-type
    ("~50 cm, surface tilted ~45° to the sensor axis", 0.0),
];
const CALIB_SAMPLES:   u32 = 20;   // readings per calibration point
const CALIB_SAMPLE_MS: u64 = 350;  // ms between calibration samples

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
            "--calib" | "-c" => a.calib = true,
            "--raw"   | "-r" => a.raw   = true,
            _ => {}
        }
        i += 1;
    }
    a
}

// ── I2C primitives ────────────────────────────────────────────────────────────

/// Enable sensor, wait for settle, read one (hi, lo) pair, disable.
fn read_single(i2c: &mut I2c) -> anyhow::Result<(f32, u8, u8)> {
    i2c.block_write(REG_ENABLE, &[1])?;
    std::thread::sleep(std::time::Duration::from_millis(SETTLE_MS));
    let mut hi = [0u8; 1];
    let mut lo = [0u8; 1];
    i2c.block_read(REG_DIST_H, &mut hi)?;
    i2c.block_read(REG_DIST_L, &mut lo)?;
    let _ = i2c.block_write(REG_ENABLE, &[0]);
    let mm = ((hi[0] as u32) << 8) | lo[0] as u32;
    // dist_mm == 0 → no echo (open space) or blind spot — both treated as max range.
    let cm = if mm == 0 { 400.0f32 } else { (mm as f32 / 10.0).clamp(2.0, 500.0) };
    Ok((cm, hi[0], lo[0]))
}

/// Median-filtered read: enable once, take n samples, disable.
fn read_median(i2c: &mut I2c, n: u8) -> anyhow::Result<(f32, u8, u8)> {
    i2c.block_write(REG_ENABLE, &[1])?;
    std::thread::sleep(std::time::Duration::from_millis(SETTLE_MS));
    let mut samples: Vec<(f32, u8, u8)> = Vec::with_capacity(n as usize);
    for idx in 0..n {
        let mut hi = [0u8; 1];
        let mut lo = [0u8; 1];
        if i2c.block_read(REG_DIST_H, &mut hi).and_then(|_| i2c.block_read(REG_DIST_L, &mut lo)).is_ok() {
            let mm = ((hi[0] as u32) << 8) | lo[0] as u32;
            let cm = if mm == 0 { 400.0f32 } else { (mm as f32 / 10.0).clamp(2.0, 500.0) };
            samples.push((cm, hi[0], lo[0]));
        }
        if idx + 1 < n {
            std::thread::sleep(std::time::Duration::from_millis(INTER_SAMPLE));
        }
    }
    let _ = i2c.block_write(REG_ENABLE, &[0]);
    if samples.is_empty() { anyhow::bail!("all samples failed"); }
    samples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    Ok(samples[samples.len() / 2])
}

/// Format a distance value for display.
/// `raw_zero` = true when the register read 0x0000 (no-echo or blind-spot).
fn fmt_dist(cm: f32) -> String {
    if cm >= 399.0 {
        // We set dist_mm==0 → max_range in the driver, so this is the no-echo path.
        format!("{cm:6.1} cm  [NO ECHO — open space or blind spot < 2 cm]")
    } else if cm < 15.0 {
        format!("{cm:6.1} cm  *** CLOSE ***")
    } else {
        format!("{cm:6.1} cm")
    }
}

// ── Continuous mode ───────────────────────────────────────────────────────────

fn run_continuous(i2c: Arc<Mutex<I2c>>, args: &Args) -> anyhow::Result<()> {
    let log_file = open_log(args.log_path.as_deref())?;
    let mut log = log_file;
    if let Some(ref mut w) = log {
        writeln!(w, "t_ms,hi,lo,dist_mm,dist_cm")?;
    }

    let t0 = Instant::now();
    let mut n = 0u32;
    println!("\nReading every ~{LOOP_PERIOD}ms  ({} sample median)  Ctrl-C to exit\n",
             args.n_samples);

    loop {
        let iter_start = Instant::now();
        let t_ms = t0.elapsed().as_millis() as u64;
        let result = {
            let mut guard = i2c.lock().unwrap();
            read_median(&mut *guard, args.n_samples)
        };
        match result {
            Ok((cm, hi, lo)) => {
                let mm = ((hi as u32) << 8) | lo as u32;
                if args.raw {
                    println!("[{t_ms:7}ms #{n:04}]  hi={hi:#04x} lo={lo:#04x}  {}", fmt_dist(cm));
                } else {
                    println!("[{t_ms:7}ms #{n:04}]  {}", fmt_dist(cm));
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

/// Show a live distance reading on one line (overwriting with \r) until the
/// user presses Enter, then return the last measured value.
fn live_until_enter(i2c: Arc<Mutex<I2c>>) -> f32 {
    let stop = Arc::new(AtomicBool::new(false));
    let stop2 = stop.clone();
    let last_cm = Arc::new(Mutex::new(0.0f32));
    let last2 = last_cm.clone();

    let handle = std::thread::spawn(move || {
        while !stop2.load(Ordering::Relaxed) {
            let result = {
                let mut guard = i2c.lock().unwrap();
                read_single(&mut *guard)
            };
            if let Ok((cm, _, _)) = result {
                *last2.lock().unwrap() = cm;
                print!("  Live: {}  (press Enter to capture) \r", fmt_dist(cm));
                let _ = stdout().flush();
            }
            // Don't sleep — read_single already takes ~100ms
        }
    });

    let mut line = String::new();
    let _ = stdin().read_line(&mut line);

    stop.store(true, Ordering::Relaxed);
    let _ = handle.join();
    println!(); // newline after the \r line
    let val = *last_cm.lock().unwrap();
    val
}

fn run_calib(i2c: Arc<Mutex<I2c>>, args: &Args) -> anyhow::Result<()> {
    let log_file = open_log(args.log_path.as_deref())?;
    let mut log = log_file;
    if let Some(ref mut w) = log {
        writeln!(w, "point,nominal_cm,sample,t_ms,hi,lo,dist_mm,dist_cm")?;
    }

    println!("\nCalibration mode — {} points, {} samples each (~{} s/point)",
             CALIB_POINTS.len(), CALIB_SAMPLES,
             CALIB_SAMPLES * CALIB_SAMPLE_MS as u32 / 1000);
    println!("Watch the live reading to position your object, then press Enter.\n");

    let t0 = Instant::now();
    let mut point_results: Vec<(f32, f32, f32, f32)> = Vec::new(); // (nominal, measured_median, mean, std)

    for (pt_idx, &(description, nominal_cm)) in CALIB_POINTS.iter().enumerate() {
        println!("── Point {} / {} ─────────────────────────────────────", pt_idx + 1, CALIB_POINTS.len());
        if nominal_cm > 0.0 {
            println!("Target: {description}");
        } else {
            println!("Test:   {description}");
        }

        live_until_enter(i2c.clone());

        println!("  Capturing {} samples...", CALIB_SAMPLES);
        let mut readings: Vec<f32> = Vec::with_capacity(CALIB_SAMPLES as usize);

        for s in 0..CALIB_SAMPLES {
            let t_ms = t0.elapsed().as_millis() as u64;
            let result = {
                let mut guard = i2c.lock().unwrap();
                read_single(&mut *guard)
            };
            match result {
                Ok((cm, hi, lo)) => {
                    let mm = ((hi as u32) << 8) | lo as u32;
                    readings.push(cm);
                    print!("  [{s:02}/{CALIB_SAMPLES}] {}\r", fmt_dist(cm));
                    let _ = stdout().flush();
                    if let Some(ref mut w) = log {
                        writeln!(w, "{pt_idx},{nominal_cm},{s},{t_ms},{hi:#04x},{lo:#04x},{mm},{cm:.1}")?;
                        w.flush()?;
                    }
                }
                Err(e) => eprintln!("  [{s:02}] error: {e}"),
            }
            let elapsed = t0.elapsed().as_millis() as u64 - t_ms;
            let gap = CALIB_SAMPLE_MS.saturating_sub(elapsed);
            if gap > 0 {
                std::thread::sleep(std::time::Duration::from_millis(gap));
            }
        }
        println!();

        if readings.is_empty() {
            println!("  No valid readings — skipping.\n");
            continue;
        }

        readings.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = readings[readings.len() / 2];
        let mean: f32 = readings.iter().sum::<f32>() / readings.len() as f32;
        let std_dev = (readings.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
            / readings.len() as f32).sqrt();
        let min = readings[0];
        let max = readings[readings.len() - 1];

        if nominal_cm > 0.0 {
            let err = median - nominal_cm;
            let err_pct = err / nominal_cm * 100.0;
            println!(
                "  nominal={nominal_cm:.0} cm  median={median:.1}  mean={mean:.1}  \
                 std={std_dev:.2}  min={min:.1}  max={max:.1}  \
                 error={err:+.1} cm ({err_pct:+.1}%)"
            );
            point_results.push((nominal_cm, median, mean, std_dev));
        } else {
            println!(
                "  (no nominal)  median={median:.1}  mean={mean:.1}  std={std_dev:.2}  \
                 min={min:.1}  max={max:.1}"
            );
        }
        println!();
    }

    // ── Summary ───────────────────────────────────────────────────────────────
    if !point_results.is_empty() {
        println!("── Calibration Summary ──────────────────────────────────────");
        println!("{:<12} {:>14} {:>12} {:>12}", "Nominal (cm)", "Measured (cm)", "Error (cm)", "Error (%)");
        println!("{}", "─".repeat(54));
        let mut scale_sum = 0.0f32;
        let mut scale_n   = 0u32;
        for &(nominal, median, _mean, std_dev) in &point_results {
            let err     = median - nominal;
            let err_pct = err / nominal * 100.0;
            println!("{:<12.0} {:>14.1} {:>+12.1} {:>+11.1}%  (std {std_dev:.2})",
                     nominal, median, err, err_pct);
            if nominal >= 10.0 {
                scale_sum += nominal / median;
                scale_n += 1;
            }
        }
        println!("{}", "─".repeat(54));

        if scale_n > 0 {
            let mean_scale = scale_sum / scale_n as f32;
            println!("\nMean scale factor (nominal/measured, points ≥10 cm): {mean_scale:.4}");
            if (mean_scale - 1.0).abs() > 0.02 {
                println!("Deviation > 2% — calibration factor recommended.");
                println!("  dist_cm_corrected = raw_dist_cm × {mean_scale:.4}");
            } else {
                println!("Deviation < 2% — factory calibration adequate for safety interlock.");
            }

            print!("\nWrite scale_factor: {mean_scale:.4} to robot_config.yaml? [y/N]: ");
            stdout().flush()?;
            let mut choice = String::new();
            stdin().read_line(&mut choice)?;
            if choice.trim().eq_ignore_ascii_case("y") {
                write_scale_to_config(mean_scale)?;
            } else {
                println!("Skipped.");
            }
        }
    }
    Ok(())
}

fn open_log(path: Option<&str>) -> anyhow::Result<Option<BufWriter<File>>> {
    match path {
        Some(p) => {
            if let Some(dir) = std::path::Path::new(p).parent() {
                std::fs::create_dir_all(dir)?;
            }
            println!("Logging to: {p}");
            Ok(Some(BufWriter::new(File::create(p)?)))
        }
        None => Ok(None),
    }
}

fn write_scale_to_config(scale: f32) -> anyhow::Result<()> {
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
    let new_line = format!("    scale_factor: {scale:.4}   # empirical calibration (nominal/measured)");
    let updated = if content.contains("scale_factor:") {
        content.lines()
            .map(|l| if l.trim_start().starts_with("scale_factor:") { new_line.clone() } else { l.to_string() })
            .collect::<Vec<_>>().join("\n")
    } else {
        content.lines()
            .flat_map(|l| {
                let mut v = vec![l.to_string()];
                if l.trim_start().starts_with("samples_per_reading:") { v.push(new_line.clone()); }
                v
            })
            .collect::<Vec<_>>().join("\n")
    };
    std::fs::write(path, updated + "\n")?;
    println!("Wrote scale_factor: {scale:.4}  →  {path}");
    Ok(())
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    let args = parse_args();

    println!("Ultrasonic sensor test  I2C-{I2C_BUS} @ {I2C_ADDR:#04x}");
    println!("Protocol: enable=0x07  hi=0x1b  lo=0x1a  dist_cm = mm/10");
    if let Some(ref p) = args.log_path { println!("Log: {p}"); }

    let i2c = {
        let mut raw = I2c::with_bus(I2C_BUS)?;
        raw.set_slave_address(I2C_ADDR)?;
        Arc::new(Mutex::new(raw))
    };

    if args.calib {
        run_calib(i2c, &args)
    } else {
        run_continuous(i2c, &args)
    }
}
