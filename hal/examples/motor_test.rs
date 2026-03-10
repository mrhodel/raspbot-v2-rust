//! Yahboom motor hardware test — interactive or automated, with IMU telemetry.
//!
//! Modes
//! ─────
//!   (default)  Interactive: P/F/R prompt after each test.
//!   --auto     Automated:   IMU determines pass/fail; no prompts between tests.
//!              Pauses only between surfaces so you can reposition.
//!   --sweep    Stall sweep: ramps each motor DOWN from --speed to find stall speed.
//!
//! Options
//! ───────
//!   --speed N           Duty cycle % (default 40).
//!   --surfaces A,B,...  Comma-separated surface names (default: prompts once).
//!   --auto              Automated pass/fail from IMU (use after axis calibration).
//!   --sweep             Stall-speed characterisation instead of motion tests.
//!
//! IMU thresholds (tune after first manual run on hardwood):
//!   MOTION_ACCEL_THRESH   min peak accel (m/s²) to count as "moving"
//!   MOTION_GYRO_THRESH    min peak gyro_z (rad/s) to count as "rotating"
//!
//! Results saved to  runs/motor_test_<surface>_<ts>.csv
//!
//! SAFETY: wheels must be OFF THE GROUND for individual-wheel tests.
//!
//! Usage (on Pi):
//!   cargo run --example motor_test -p hal
//!   cargo run --example motor_test -p hal -- --speed 40 --auto --surfaces hardwood,concrete,rug
//!   cargo run --example motor_test -p hal -- --sweep --speed 50 --surfaces hardwood

use std::fs::OpenOptions;
use std::io::{self, BufRead, Write as IoWrite};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use anyhow::{Context, Result};
use rppal::i2c::I2c;

// ── Motor constants ────────────────────────────────────────────────────────────

const MOTOR_BUS:  u8  = 1;
const MOTOR_ADDR: u16 = 0x2B;
const REG_MOTOR:  u8  = 0x01;

const M_FL: u8 = 0;
const M_RL: u8 = 1;
const M_FR: u8 = 2;
const M_RR: u8 = 3;

const DEFAULT_SPEED: i8 = 40;
const MOVE_S:        u64 = 2;  // motion test duration (more = better IMU data)
const SWEEP_S_MS:    u64 = 600; // stall-sweep duration per speed step

// ── MPU-6050 constants ─────────────────────────────────────────────────────────

const MPU_BUS:          u8  = 6;
const MPU_ADDR:         u16 = 0x68;
const REG_PWR_MGMT_1:   u8  = 0x6B;
const REG_GYRO_CONFIG:  u8  = 0x1B;
const REG_ACCEL_CONFIG: u8  = 0x1C;
const REG_ACCEL_XOUT_H: u8  = 0x3B;

const ACCEL_SCALE: f32 = 9.81 / 8192.0;                        // ±4 g → m/s²
const GYRO_SCALE:  f32 = std::f32::consts::PI / 180.0 / 65.5;  // ±500 °/s → rad/s
const SAMPLE_MS:   u64 = 10; // 100 Hz

// ── Auto-assessment thresholds ────────────────────────────────────────────────
// Calibrate these after the first manual hardwood run.
// Raise if false-positives (noise flagged as motion).
// Lower if false-negatives (real motion not detected).

/// Peak translational accel (m/s²) required to declare a motor "moving".
const MOTION_ACCEL_THRESH: f32 = 0.20;
/// Peak |gyro_z| (rad/s) required to declare a rotation test passing.
const MOTION_GYRO_THRESH: f32 = 0.05;
/// Accel threshold for single-wheel tests (lower — robot is mostly anchored).
const SINGLE_WHEEL_THRESH: f32 = 0.10;

// ── IMU types ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
struct ImuSample {
    t_ms: u64,
    ax: f32, ay: f32, az: f32,
    gx: f32, gy: f32, gz: f32,
}

#[derive(Clone, Copy)]
struct Bias { ax: f32, ay: f32, az: f32, gx: f32, gy: f32, gz: f32 }

impl Bias {
    fn zero() -> Self { Self { ax: 0., ay: 0., az: 0., gx: 0., gy: 0., gz: 0. } }
}

// ── MPU helpers ───────────────────────────────────────────────────────────────

fn mpu_init(i2c: &mut I2c) -> Result<()> {
    i2c.block_write(REG_PWR_MGMT_1,  &[0x00]).context("MPU wake-up")?;
    std::thread::sleep(Duration::from_millis(100));
    i2c.block_write(REG_GYRO_CONFIG,  &[0x08]).context("gyro ±500°/s")?;
    i2c.block_write(REG_ACCEL_CONFIG, &[0x08]).context("accel ±4g")?;
    Ok(())
}

fn read_mpu(i2c: &mut I2c, t_ms: u64, bias: Bias) -> Result<ImuSample> {
    let mut buf = [0u8; 14];
    i2c.block_read(REG_ACCEL_XOUT_H, &mut buf)?;
    let raw = |hi: usize| i16::from_be_bytes([buf[hi], buf[hi + 1]]) as f32;
    Ok(ImuSample {
        t_ms,
        ax: raw(0)  * ACCEL_SCALE - bias.ax,
        ay: raw(2)  * ACCEL_SCALE - bias.ay,
        az: raw(4)  * ACCEL_SCALE - bias.az,
        gx: raw(8)  * GYRO_SCALE  - bias.gx,
        gy: raw(10) * GYRO_SCALE  - bias.gy,
        gz: raw(12) * GYRO_SCALE  - bias.gz,
    })
}

fn collect_bias(i2c: &mut I2c) -> Result<Bias> {
    println!("  Collecting 1 s static baseline — hold still...");
    let zero = Bias::zero();
    let t0 = Instant::now();
    let mut s = [0f64; 6];
    let mut n = 0u32;
    while t0.elapsed().as_millis() < 1000 {
        if let Ok(m) = read_mpu(i2c, 0, zero) {
            s[0] += m.ax as f64; s[1] += m.ay as f64; s[2] += m.az as f64;
            s[3] += m.gx as f64; s[4] += m.gy as f64; s[5] += m.gz as f64;
            n += 1;
        }
        std::thread::sleep(Duration::from_millis(SAMPLE_MS));
    }
    if n == 0 { return Err(anyhow::anyhow!("no MPU samples during baseline")); }
    let f = n as f64;
    Ok(Bias {
        ax: (s[0]/f) as f32, ay: (s[1]/f) as f32, az: (s[2]/f) as f32,
        gx: (s[3]/f) as f32, gy: (s[4]/f) as f32, gz: (s[5]/f) as f32,
    })
}

// ── Sampling thread ────────────────────────────────────────────────────────────

struct Sampler {
    stop:    Arc<AtomicBool>,
    join:    std::thread::JoinHandle<()>,
    samples: Arc<Mutex<Vec<ImuSample>>>,
}

impl Sampler {
    fn start(bias: Bias) -> Self {
        let stop    = Arc::new(AtomicBool::new(false));
        let samples = Arc::new(Mutex::new(Vec::<ImuSample>::new()));
        let stop2    = stop.clone();
        let samples2 = samples.clone();

        let join = std::thread::spawn(move || {
            let mut i2c = match I2c::with_bus(MPU_BUS) {
                Ok(mut i) => { let _ = i.set_slave_address(MPU_ADDR); i }
                Err(e)    => { eprintln!("  [WARN] MPU sampler: {e}"); return; }
            };
            let t0 = Instant::now();
            while !stop2.load(Ordering::Relaxed) {
                let t_ms = t0.elapsed().as_millis() as u64;
                if let Ok(s) = read_mpu(&mut i2c, t_ms, bias) {
                    samples2.lock().unwrap().push(s);
                }
                std::thread::sleep(Duration::from_millis(SAMPLE_MS));
            }
        });

        Self { stop, join, samples }
    }

    fn finish(self) -> Vec<ImuSample> {
        self.stop.store(true, Ordering::Relaxed);
        let _ = self.join.join();
        Arc::try_unwrap(self.samples).unwrap().into_inner().unwrap()
    }
}

// ── Metrics ────────────────────────────────────────────────────────────────────

struct Metrics {
    n:       usize,
    dur_s:   f32,
    mean_ax: f32,
    mean_ay: f32,
    mean_gz: f32,
    peak_ax: f32,
    peak_ay: f32,
    peak_gz: f32,
    peak_mag: f32, // peak |accel_xy|
}

fn compute_metrics(samples: &[ImuSample]) -> Metrics {
    let n = samples.len();
    if n == 0 {
        return Metrics { n: 0, dur_s: 0., mean_ax: 0., mean_ay: 0., mean_gz: 0.,
                         peak_ax: 0., peak_ay: 0., peak_gz: 0., peak_mag: 0. };
    }
    let f = n as f32;
    let mean_ax  = samples.iter().map(|s| s.ax).sum::<f32>() / f;
    let mean_ay  = samples.iter().map(|s| s.ay).sum::<f32>() / f;
    let mean_gz  = samples.iter().map(|s| s.gz).sum::<f32>() / f;
    let peak_ax  = samples.iter().map(|s| s.ax.abs()).fold(0f32, f32::max);
    let peak_ay  = samples.iter().map(|s| s.ay.abs()).fold(0f32, f32::max);
    let peak_gz  = samples.iter().map(|s| s.gz.abs()).fold(0f32, f32::max);
    let peak_mag = samples.iter()
        .map(|s| (s.ax * s.ax + s.ay * s.ay).sqrt())
        .fold(0f32, f32::max);
    let dur_s = samples.last().map_or(0, |s| s.t_ms) as f32 / 1000.0;
    Metrics { n, dur_s, mean_ax, mean_ay, mean_gz, peak_ax, peak_ay, peak_gz, peak_mag }
}

fn print_metrics(m: &Metrics) {
    if m.n == 0 { println!("  (no IMU data)"); return; }
    println!("  IMU ({:.1} s, {} samples):", m.dur_s, m.n);
    println!("    accel_x  mean={:+.3} m/s²  peak={:.3}", m.mean_ax, m.peak_ax);
    println!("    accel_y  mean={:+.3} m/s²  peak={:.3}", m.mean_ay, m.peak_ay);
    println!("    gyro_z   mean={:+.4} rad/s peak={:.4}", m.mean_gz, m.peak_gz);
    let lat = m.mean_ay.abs();
    let fwd = m.mean_ax.abs();
    if lat + fwd > 0.05 {
        println!("    Strafe purity  {:.0}% lateral  {:.0}% forward creep",
            lat / (lat + fwd) * 100.0, fwd / (lat + fwd) * 100.0);
    }
}

// ── Auto-assessment ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
enum Verdict { Pass, Fail }

fn auto_assess(test_name: &str, m: &Metrics) -> (Verdict, String) {
    let is_rotate = test_name.contains("Rotate");
    let is_single = ["FL", "FR", "RL", "RR"].iter().any(|w| test_name.starts_with(w));

    if is_rotate {
        if m.peak_gz > MOTION_GYRO_THRESH {
            (Verdict::Pass, format!("rotation detected peak_gz={:.4} rad/s", m.peak_gz))
        } else {
            (Verdict::Fail, format!("no rotation: peak_gz={:.4} < {:.4}", m.peak_gz, MOTION_GYRO_THRESH))
        }
    } else if is_single {
        if m.peak_mag > SINGLE_WHEEL_THRESH {
            (Verdict::Pass, format!("motion detected peak_mag={:.3} m/s²", m.peak_mag))
        } else {
            (Verdict::Fail, format!("no motion: peak_mag={:.3} < {:.3}", m.peak_mag, SINGLE_WHEEL_THRESH))
        }
    } else {
        if m.peak_mag > MOTION_ACCEL_THRESH {
            (Verdict::Pass, format!("motion detected peak_mag={:.3} m/s²", m.peak_mag))
        } else {
            (Verdict::Fail, format!("no motion: peak_mag={:.3} < {:.3}", m.peak_mag, MOTION_ACCEL_THRESH))
        }
    }
}

// ── CSV helpers ────────────────────────────────────────────────────────────────

fn csv_path(surface: &str, ts: u64) -> String {
    format!("runs/motor_test_{}_{}.csv", surface, ts)
}

fn append_csv(path: &str, surface: &str, test: &str, samples: &[ImuSample]) -> Result<()> {
    let new_file = !std::path::Path::new(path).exists();
    let mut f = OpenOptions::new().create(true).append(true).open(path)
        .with_context(|| format!("open {path}"))?;
    if new_file {
        writeln!(f, "surface,test,t_ms,ax,ay,az,gx,gy,gz")?;
    }
    for s in samples {
        writeln!(f, "{},{},{},{:.4},{:.4},{:.4},{:.5},{:.5},{:.5}",
            surface, test, s.t_ms, s.ax, s.ay, s.az, s.gx, s.gy, s.gz)?;
    }
    Ok(())
}

fn append_sweep_csv(path: &str, surface: &str, motor: &str, speed: i8, m: &Metrics) -> Result<()> {
    let new_file = !std::path::Path::new(path).exists();
    let mut f = OpenOptions::new().create(true).append(true).open(path)
        .with_context(|| format!("open {path}"))?;
    if new_file {
        writeln!(f, "surface,motor,speed_pct,peak_mag,peak_gz,mean_ax,mean_ay,moving")?;
    }
    let moving = m.peak_mag > SINGLE_WHEEL_THRESH;
    writeln!(f, "{},{},{},{:.4},{:.5},{:.4},{:.4},{}",
        surface, motor, speed, m.peak_mag, m.peak_gz, m.mean_ax, m.mean_ay,
        if moving { 1 } else { 0 })?;
    Ok(())
}

// ── I2C motor helpers ─────────────────────────────────────────────────────────

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

fn stop_all(i2c: &mut I2c) -> Result<()> { set_motors(i2c, 0, 0, 0, 0) }

fn single_motor(i2c: &mut I2c, label: &str, speed: i8) -> Result<()> {
    match label {
        "FL" => set_motors(i2c, speed, 0, 0, 0),
        "FR" => set_motors(i2c, 0, speed, 0, 0),
        "RL" => set_motors(i2c, 0, 0, speed, 0),
        "RR" => set_motors(i2c, 0, 0, 0, speed),
        _    => Ok(()),
    }
}

// ── Motion tests ──────────────────────────────────────────────────────────────

fn run_test(
    motor_i2c: &mut I2c,
    bias:    Bias,
    csv:     &str,
    surface: &str,
    auto:    bool,
    name:    &str,
    note:    &str,
    fl: i8, fr: i8, rl: i8, rr: i8,
) -> Result<Verdict> {
    loop {
        println!();
        println!("  [TEST] {name}");
        if !note.is_empty() { println!("         {note}"); }
        println!("         FL={fl:+3}  FR={fr:+3}  RL={rl:+3}  RR={rr:+3}  ({MOVE_S}s)");

        let sampler = Sampler::start(bias);
        set_motors(motor_i2c, fl, fr, rl, rr)?;
        std::thread::sleep(Duration::from_secs(MOVE_S));
        stop_all(motor_i2c)?;
        let imu = sampler.finish();

        let m = compute_metrics(&imu);
        print_metrics(&m);
        if let Err(e) = append_csv(csv, surface, name, &imu) {
            eprintln!("  [WARN] CSV: {e}");
        }
        std::thread::sleep(Duration::from_millis(300));

        if auto {
            let (verdict, reason) = auto_assess(name, &m);
            let tag = if verdict == Verdict::Pass { "PASS" } else { "FAIL" };
            println!("  [AUTO {tag}]  {reason}");
            return Ok(verdict);
        }

        print!("  >> [P]ass  [F]ail  [R]epeat : ");
        io::stdout().flush().unwrap();
        let mut line = String::new();
        io::stdin().lock().read_line(&mut line).unwrap();
        match line.trim().to_uppercase().as_str() {
            "P" | "" => return Ok(Verdict::Pass),
            "F"      => return Ok(Verdict::Fail),
            "R"      => continue,
            _        => { println!("     Unrecognised — treating as Pass."); return Ok(Verdict::Pass); }
        }
    }
}

// ── Stall sweep ────────────────────────────────────────────────────────────────

fn run_sweep(
    motor_i2c: &mut I2c,
    bias:      Bias,
    csv:       &str,
    surface:   &str,
    speed_max: i8,
) -> Result<()> {
    const STEP: i8 = 5;
    const MIN_SPEED: i8 = 5;

    println!();
    println!("  === Stall Speed Sweep — {surface} ===");
    println!("  Stepping {speed_max}% → {MIN_SPEED}% in {STEP}% steps ({SWEEP_S_MS}ms each)");
    println!("  Motion threshold: {SINGLE_WHEEL_THRESH:.2} m/s²  (tune if needed)");

    for motor in ["FL", "FR", "RL", "RR"] {
        println!();
        println!("  ── Motor {motor} ──");
        let mut last_moving: Option<i8> = None;
        let mut speed = speed_max;
        while speed >= MIN_SPEED {
            let sampler = Sampler::start(bias);
            single_motor(motor_i2c, motor, speed)?;
            std::thread::sleep(Duration::from_millis(SWEEP_S_MS));
            stop_all(motor_i2c)?;
            let imu = sampler.finish();
            std::thread::sleep(Duration::from_millis(200));

            let m = compute_metrics(&imu);
            let moving = m.peak_mag > SINGLE_WHEEL_THRESH;
            let tag = if moving { "MOVING " } else { "STALLED" };
            println!("    {motor} @ {:3}%  peak_mag={:.3} m/s²  peak_gz={:.4} rad/s  [{tag}]",
                speed, m.peak_mag, m.peak_gz);

            if let Err(e) = append_sweep_csv(csv, surface, motor, speed, &m) {
                eprintln!("    [WARN] CSV: {e}");
            }

            if moving { last_moving = Some(speed); }
            speed -= STEP;
        }

        match last_moving {
            Some(s) => println!("  {motor} minimum reliable speed: {s}%"),
            None    => println!("  {motor} stalled at all tested speeds — threshold may need lowering"),
        }
    }
    Ok(())
}

// ── Arg parsing ───────────────────────────────────────────────────────────────

struct Args {
    speed:    i8,
    auto:     bool,
    sweep:    bool,
    surfaces: Vec<String>,
}

fn parse_args() -> Result<Args> {
    let argv: Vec<String> = std::env::args().collect();
    let mut speed    = DEFAULT_SPEED;
    let mut auto     = false;
    let mut sweep    = false;
    let mut surfaces = Vec::new();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--speed" => {
                i += 1;
                speed = argv.get(i)
                    .ok_or_else(|| anyhow::anyhow!("--speed requires a value"))?
                    .parse::<i8>().context("--speed must be 1–100")?;
            }
            "--auto"  => { auto  = true; }
            "--sweep" => { sweep = true; }
            "--surfaces" => {
                i += 1;
                let s = argv.get(i).ok_or_else(|| anyhow::anyhow!("--surfaces requires a value"))?;
                surfaces = s.split(',').map(|x| x.trim().to_string()).collect();
            }
            other => eprintln!("  [WARN] Unknown arg: {other}"),
        }
        i += 1;
    }
    Ok(Args { speed, auto, sweep, surfaces })
}

// ── main ─────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = parse_args()?;

    let mode = if args.sweep { "Stall Sweep" } else if args.auto { "Auto" } else { "Interactive" };
    println!("================================================================");
    println!(" Yahboom Raspbot V2  —  Motor Test + IMU  [{mode}]");
    println!(" Speed: {}%  Duration: {}s", args.speed, if args.sweep { SWEEP_S_MS as f32 / 1000.0 } else { MOVE_S as f32 });
    println!("================================================================");
    println!();
    if !args.sweep {
        println!(" SAFETY: wheels must be OFF THE GROUND for single-wheel tests.");
        println!();
    }

    // Determine surfaces
    let surfaces: Vec<String> = if !args.surfaces.is_empty() {
        args.surfaces.clone()
    } else {
        print!(" Surface [hardwood / concrete / rug / ...]: ");
        io::stdout().flush().unwrap();
        let mut s = String::new();
        io::stdin().lock().read_line(&mut s).unwrap();
        let s = s.trim();
        vec![if s.is_empty() { "hardwood".to_string() } else { s.to_string() }]
    };

    // Hardware init
    let mut motor_i2c = I2c::with_bus(MOTOR_BUS)?;
    motor_i2c.set_slave_address(MOTOR_ADDR)?;
    println!(" Motors:   /dev/i2c-{MOTOR_BUS} @ 0x{MOTOR_ADDR:02X}  OK");
    stop_all(&mut motor_i2c)?;

    let mut mpu_i2c = I2c::with_bus(MPU_BUS)?;
    mpu_i2c.set_slave_address(MPU_ADDR)?;
    mpu_init(&mut mpu_i2c)?;
    println!(" MPU-6050: /dev/i2c-{MPU_BUS} @ 0x{MPU_ADDR:02X}  OK");
    println!();

    let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    std::fs::create_dir_all("runs").ok();

    let s = args.speed;
    let motion_tests: &[(&str, &str, i8, i8, i8, i8)] = &[
        ("FL (Front-Left)",  "Only front-left wheel should spin.",    s,  0,  0,  0),
        ("FR (Front-Right)", "Only front-right wheel should spin.",   0,  s,  0,  0),
        ("RL (Rear-Left)",   "Only rear-left wheel should spin.",     0,  0,  s,  0),
        ("RR (Rear-Right)",  "Only rear-right wheel should spin.",    0,  0,  0,  s),
        ("Forward",          "All wheels fwd — robot moves forward.", s,  s,  s,  s),
        ("Backward",         "All wheels rev — robot moves back.",   -s, -s, -s, -s),
        ("Rotate CW",        "Left fwd / right rev — spins CW.",      s, -s,  s, -s),
        ("Rotate CCW",       "Left rev / right fwd — spins CCW.",    -s,  s, -s,  s),
        ("Strafe Right",     "Robot slides right (no rotation).",     s, -s, -s,  s),
        ("Strafe Left",      "Robot slides left (no rotation).",     -s,  s,  s, -s),
    ];

    // Collect summary across surfaces for comparison printout
    let mut surface_results: Vec<(String, Vec<(&str, Verdict)>)> = Vec::new();

    for (idx, surface) in surfaces.iter().enumerate() {
        if idx > 0 {
            println!();
            println!("================================================================");
            println!(" Next surface: {surface}");
            println!(" Reposition the robot, then press Enter to continue...");
            println!("================================================================");
            io::stdout().flush().unwrap();
            let mut dummy = String::new();
            io::stdin().lock().read_line(&mut dummy).unwrap();
        }

        println!(" Surface: {surface}");
        println!();

        let bias = collect_bias(&mut mpu_i2c)?;
        println!("  Bias  ax={:.3}  ay={:.3}  az={:.3}  gx={:.4}  gy={:.4}  gz={:.4}",
            bias.ax, bias.ay, bias.az, bias.gx, bias.gy, bias.gz);

        let path = csv_path(surface, ts + idx as u64);
        println!(" Logging to: {path}");
        println!();

        if args.sweep {
            run_sweep(&mut motor_i2c, bias, &path, surface, args.speed)?;
        } else {
            let mut results: Vec<(&str, Verdict)> = Vec::new();
            for &(name, note, fl, fr, rl, rr) in motion_tests {
                let v = run_test(
                    &mut motor_i2c, bias, &path, surface,
                    args.auto, name, note, fl, fr, rl, rr,
                )?;
                results.push((name, v));
            }
            surface_results.push((surface.clone(), results));
        }
    }

    let _ = stop_all(&mut motor_i2c);

    // Final summary (motion tests only)
    if !args.sweep && !surface_results.is_empty() {
        println!();
        println!("================================================================");
        if surface_results.len() == 1 {
            println!(" RESULTS — {}", surface_results[0].0);
            println!("================================================================");
            let mut any_fail = false;
            for (name, v) in &surface_results[0].1 {
                let tag = if *v == Verdict::Pass { "PASS" } else { "FAIL" };
                println!("  [{tag}]  {name}");
                if *v == Verdict::Fail { any_fail = true; }
            }
            println!("================================================================");
            if any_fail {
                println!(" Some tests FAILED.");
                std::process::exit(1);
            } else {
                println!(" All tests passed.");
            }
        } else {
            // Multi-surface comparison table
            println!(" RESULTS — surface comparison");
            println!("================================================================");
            let names: Vec<&str> = surface_results[0].1.iter().map(|(n, _)| *n).collect();
            let col = 10usize;
            print!("  {:<28}", "Test");
            for (surf, _) in &surface_results { print!(" {:>col$}", surf); }
            println!();
            println!("  {}", "-".repeat(28 + (col + 1) * surface_results.len()));
            for name in &names {
                print!("  {:<28}", name);
                for (_, results) in &surface_results {
                    let tag = results.iter()
                        .find(|(n, _)| n == name)
                        .map(|(_, v)| if *v == Verdict::Pass { "PASS" } else { "FAIL" })
                        .unwrap_or("----");
                    print!(" {:>col$}", tag);
                }
                println!();
            }
            println!("================================================================");
        }
    }

    Ok(())
}
