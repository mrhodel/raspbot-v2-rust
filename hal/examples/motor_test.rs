//! Yahboom motor hardware test — interactive with IMU telemetry.
//!
//! Samples the MPU-6050 at 100 Hz during each motion and reports
//! accel_x (forward), accel_y (lateral), gyro_z (yaw) metrics.
//! Results saved to  runs/motor_test_<surface>_<unix_ts>.csv
//!
//! SAFETY: Run with the robot elevated — wheels must be off the ground.
//!
//! Usage (on Pi):
//!   cargo run --example motor_test -p hal

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

const TEST_SPEED: i8 = 40;
const MOVE_S:     u64 = 2; // longer = more IMU data

// ── MPU-6050 constants ─────────────────────────────────────────────────────────

const MPU_BUS:          u8  = 6;
const MPU_ADDR:         u16 = 0x68;
const REG_PWR_MGMT_1:   u8  = 0x6B;
const REG_GYRO_CONFIG:  u8  = 0x1B;
const REG_ACCEL_CONFIG: u8  = 0x1C;
const REG_ACCEL_XOUT_H: u8  = 0x3B;

const ACCEL_SCALE: f32 = 9.81 / 16384.0;                        // ±2 g → m/s²
const GYRO_SCALE:  f32 = std::f32::consts::PI / 180.0 / 131.0;  // ±250 °/s → rad/s
const SAMPLE_MS:   u64 = 10;                                      // 100 Hz

// ── IMU types ─────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
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
    i2c.block_write(REG_GYRO_CONFIG,  &[0x00]).context("gyro ±250°/s")?;
    i2c.block_write(REG_ACCEL_CONFIG, &[0x00]).context("accel ±2g")?;
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
        // buf[6..=7] = temperature — skip
        gx: raw(8)  * GYRO_SCALE  - bias.gx,
        gy: raw(10) * GYRO_SCALE  - bias.gy,
        gz: raw(12) * GYRO_SCALE  - bias.gz,
    })
}

/// Collect ~1 s of static samples and return the mean as a bias correction.
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
            // Each test spawns its own I2c handle — I2c is not Sync.
            let mut i2c = match I2c::with_bus(MPU_BUS) {
                Ok(mut i) => { let _ = i.set_slave_address(MPU_ADDR); i }
                Err(e) => { eprintln!("  [WARN] MPU sampler: {e}"); return; }
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

// ── Metrics display ────────────────────────────────────────────────────────────

fn print_metrics(samples: &[ImuSample]) {
    if samples.is_empty() { println!("  (no IMU data)"); return; }
    let n = samples.len() as f32;
    let mean = |f: fn(&ImuSample) -> f32| samples.iter().map(f).sum::<f32>() / n;
    let peak = |f: fn(&ImuSample) -> f32| samples.iter().map(f).fold(0f32, f32::max);

    let avg_ax = mean(|s| s.ax);
    let avg_ay = mean(|s| s.ay);
    let avg_gz = mean(|s| s.gz);
    let max_ax = peak(|s| s.ax.abs());
    let max_ay = peak(|s| s.ay.abs());
    let max_gz = peak(|s| s.gz.abs());

    let dur_s = samples.last().map_or(0, |s| s.t_ms) as f32 / 1000.0;
    println!("  IMU ({:.1} s, {} samples):", dur_s, samples.len());
    println!("    accel_x  mean={:+.3} m/s²  peak={:.3}", avg_ax, max_ax);
    println!("    accel_y  mean={:+.3} m/s²  peak={:.3}", avg_ay, max_ay);
    println!("    gyro_z   mean={:+.4} rad/s peak={:.4}", avg_gz, max_gz);

    // Strafe purity: fraction of horizontal motion that is lateral vs forward creep
    let lat = avg_ay.abs();
    let fwd = avg_ax.abs();
    if lat + fwd > 0.05 {
        println!("    Strafe purity  {:.0}% lateral  {:.0}% forward creep",
            lat / (lat + fwd) * 100.0,
            fwd / (lat + fwd) * 100.0);
    }
}

// ── CSV logging ────────────────────────────────────────────────────────────────

fn append_csv(path: &str, surface: &str, test: &str, samples: &[ImuSample]) -> Result<()> {
    let new_file = !std::path::Path::new(path).exists();
    let mut f = OpenOptions::new().create(true).append(true).open(path)
        .with_context(|| format!("open {path}"))?;
    if new_file {
        writeln!(f, "surface,test,t_ms,ax,ay,az,gx,gy,gz")?;
    }
    for s in samples {
        writeln!(f, "{},{},{},{:.4},{:.4},{:.4},{:.5},{:.5},{:.5}",
            surface, test, s.t_ms,
            s.ax, s.ay, s.az, s.gx, s.gy, s.gz)?;
    }
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

// ── Interactive test ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
enum Verdict { Pass, Fail }

fn run_test(
    motor_i2c: &mut I2c,
    bias:    Bias,
    csv:     &str,
    surface: &str,
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
        print_metrics(&imu);
        if let Err(e) = append_csv(csv, surface, name, &imu) {
            eprintln!("  [WARN] CSV: {e}");
        }

        std::thread::sleep(Duration::from_millis(300));
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

// ── main ─────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    println!("================================================================");
    println!(" Yahboom Raspbot V2  —  Motor Test + IMU Telemetry");
    println!(" Speed: {TEST_SPEED}%  Duration: {MOVE_S}s per test");
    println!("================================================================");
    println!();
    println!(" SAFETY: wheels must be OFF THE GROUND.");
    println!();

    print!(" Surface [hardwood / concrete / rug / ...]: ");
    io::stdout().flush().unwrap();
    let mut surface = String::new();
    io::stdin().lock().read_line(&mut surface).unwrap();
    let surface = {
        let s = surface.trim();
        if s.is_empty() { "hardwood".to_string() } else { s.to_string() }
    };
    println!(" Surface: {surface}");
    println!();

    let mut motor_i2c = I2c::with_bus(MOTOR_BUS)?;
    motor_i2c.set_slave_address(MOTOR_ADDR)?;
    println!(" Motors:   /dev/i2c-{MOTOR_BUS} @ 0x{MOTOR_ADDR:02X}  OK");
    stop_all(&mut motor_i2c)?;

    let mut mpu_i2c = I2c::with_bus(MPU_BUS)?;
    mpu_i2c.set_slave_address(MPU_ADDR)?;
    mpu_init(&mut mpu_i2c)?;
    println!(" MPU-6050: /dev/i2c-{MPU_BUS} @ 0x{MPU_ADDR:02X}  OK");
    println!();

    let bias = collect_bias(&mut mpu_i2c)?;
    println!("  Bias  ax={:.3}  ay={:.3}  az={:.3}  gx={:.4}  gy={:.4}  gz={:.4}",
        bias.ax, bias.ay, bias.az, bias.gx, bias.gy, bias.gz);

    let ts  = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    std::fs::create_dir_all("runs").ok();
    let csv = format!("runs/motor_test_{}_{}.csv", surface, ts);
    println!(" Logging to: {csv}");
    println!();

    let s = TEST_SPEED;
    let tests: &[(&str, &str, i8, i8, i8, i8)] = &[
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

    let mut results: Vec<(&str, Verdict)> = Vec::new();
    for &(name, note, fl, fr, rl, rr) in tests {
        let v = run_test(&mut motor_i2c, bias, &csv, &surface, name, note, fl, fr, rl, rr)?;
        results.push((name, v));
    }

    let _ = stop_all(&mut motor_i2c);

    println!();
    println!("================================================================");
    println!(" RESULTS");
    println!("================================================================");
    let mut any_fail = false;
    for (name, v) in &results {
        let tag = if *v == Verdict::Pass { "PASS" } else { "FAIL" };
        println!("  [{tag}]  {name}");
        if *v == Verdict::Fail { any_fail = true; }
    }
    println!("================================================================");
    println!(" IMU data: {csv}");
    if any_fail {
        println!(" Some tests FAILED — review motor IDs or wiring.");
        std::process::exit(1);
    } else {
        println!(" All tests passed.");
    }

    Ok(())
}
