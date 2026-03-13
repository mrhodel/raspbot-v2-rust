//! Sim visualiser — runs FastSim and streams state to the UI bridge.
//!
//! Opens a browser at `http://localhost:9000/map` to watch the robot explore.
//!
//! # Usage
//! ```
//! cargo run --release -p exploration_rl --example sim_viz -- [seed] [steps_per_sec]
//! ```
//!
//! Defaults: seed = 42, steps_per_sec = 10.
//! The sim resets automatically on collision or episode end.
//!
//! Press Ctrl-C to stop.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use mapping::Mapper;
use sim_fast::{FastSim, Action};
use tracing::info;
use ui_bridge::{UiBridgeConfig, start as start_bridge};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let mut args = std::env::args().skip(1);
    let seed: u64  = args.next().and_then(|s| s.parse().ok()).unwrap_or(42);
    let hz:   f64  = args.next().and_then(|s| s.parse().ok()).unwrap_or(10.0);

    let step_dur = Duration::from_secs_f64(1.0 / hz.max(0.1).min(200.0));

    info!(seed, hz, "sim_viz starting — open http://localhost:9000/map");

    // ── Build bus ──────────────────────────────────────────────────────────
    let (bus, _rx, _watch_rx) = bus::Bus::new(bus::CAP);
    let bus = Arc::new(bus);

    // ── Start UI bridge ────────────────────────────────────────────────────
    start_bridge(Arc::clone(&bus), UiBridgeConfig::default()).await?;

    // ── Set executive state to Exploring ──────────────────────────────────
    use core_types::ExecutiveState;
    let _ = bus.executive_state.send(ExecutiveState::Exploring);

    // ── Run sim loop ───────────────────────────────────────────────────────
    let mut sim       = FastSim::new(seed);
    let mut mapper    = Mapper::new();
    let mut rng       = SimpleRng::new(seed ^ 0xdeadbeef);
    let mut heuristic = Heuristic::new();
    let mut pan_deg:  f32 = 0.0;

    let mut obs = sim.reset();
    let mut episode = 0u32;

    loop {
        let t0 = Instant::now();

        // Reactive gimbal pan — mirrors the real runtime gimbal task.
        // Computed from the scan we just received, applied to the next step.
        pan_deg = reactive_pan(&obs.scan, pan_deg);
        let _ = bus.gimbal_pan_deg.send(pan_deg);

        // Publish pose and lidar from this step.
        let _ = bus.slam_pose2d.send(obs.pose);
        let _ = bus.vision_pseudo_lidar.send(Arc::new(obs.scan.clone()));

        // Update occupancy map and publish delta + frontiers.
        let (delta, frontiers, stats) = mapper.update(&obs.scan, &obs.pose);
        let _ = bus.map_grid_delta.send(delta);
        let _ = bus.map_frontiers.send(frontiers);
        let _ = bus.map_explored_stats.send(stats);

        if obs.done {
            episode += 1;
            let crashed = obs.collision;
            let reason = if crashed { "CRASH" } else { "TIMEOUT" };
            info!(episode, reason, "Episode ended — resetting");

            // Signal the UI: red for crash, orange for timeout.
            let end_state = if crashed {
                ExecutiveState::Fault { reason: "CRASH".to_string() }
            } else {
                ExecutiveState::SafetyStopped
            };
            let _ = bus.executive_state.send(end_state);

            tokio::time::sleep(Duration::from_millis(600)).await;
            heuristic = Heuristic::new();
            pan_deg   = 0.0;
            mapper    = Mapper::new();
            obs       = sim.reset();
            let _ = bus.executive_state.send(ExecutiveState::Exploring);
            continue;
        }

        // Navigation always uses a centered (pan=0) scan so the heuristic sees
        // the full forward corridor regardless of where the visual gimbal points.
        let nav = sim.nav_scan();
        let action = heuristic.act(&nav, &mut rng);
        obs = sim.step_with_pan(action, pan_deg);

        // Pace to target Hz.
        let elapsed = t0.elapsed();
        if elapsed < step_dur {
            tokio::time::sleep(step_dur - elapsed).await;
        }
    }
}

// ── Stateful wall-avoidance heuristic ─────────────────────────────────────────
//
// When the forward sector is within 1.5 m, commit to a spin toward the clearer
// side for enough steps to actually clear the wall.  A single-step turn at
// 1 rad/s · 0.1 s = 5.7° — far too little.  We spin for 8–33 steps (46–189°)
// depending on how close the wall is.

struct Heuristic {
    spin_dir:     u8,
    spin_remaining: i32,
    /// True while we're committed to clearing a wall.  Direction is not
    /// re-evaluated until forward clears — prevents ±30° oscillation in corners.
    turning:      bool,
}

impl Heuristic {
    fn new() -> Self {
        Self { spin_dir: Action::RotateLeft as u8, spin_remaining: 0, turning: false }
    }

    fn act(&mut self, scan: &core_types::PseudoLidarScan, rng: &mut SimpleRng) -> u8 {
        const TURN_THRESH: f32 = 0.6;  // start turning at 0.6 m (±28.6° forward sector)
        const WIDE_GUARD:  f32 = 0.35; // emergency stop if anything within 0.35 m in ±55° FOV

        // Primary forward check (±0.5 rad = ±28.6°).
        let forward = min_sector(scan, -0.5, 0.5);
        // Wide-angle guard: walls at 28.6°–55° that the forward check misses can
        // still cause a collision (crash geometry shows walls < 0.17 m at 55° are fatal).
        // 0.35 m gives a 2× safety margin over the 0.17 m crash threshold.
        let wide_min = min_sector(scan, -1.0, 1.0);
        // If wide_min < WIDE_GUARD, treat the effective forward as blocked so the
        // spin logic fires; but don't clip above TURN_THRESH (avoid spurious triggers).
        let effective_fwd = if wide_min < WIDE_GUARD { wide_min } else { forward };

        // Continue an in-progress spin burst.
        if self.spin_remaining > 0 {
            self.spin_remaining -= 1;
            return self.spin_dir;
        }

        if effective_fwd < TURN_THRESH {
            // Pick a direction only on the first encounter; keep it until clear.
            if !self.turning {
                let left  = avg_sector(scan,  0.5,  f32::MAX);
                let right = avg_sector(scan, f32::MIN, -0.5);
                self.spin_dir = if left >= right {
                    Action::RotateLeft as u8
                } else {
                    Action::RotateRight as u8
                };
                self.turning = true;
            }
            // Issue another burst in the committed direction.
            // At TURN_THRESH: 8 steps (~46°).  At 0.0 m: 33 steps (~189°).
            let spin_steps = ((TURN_THRESH - effective_fwd) / TURN_THRESH * 25.0 + 8.0) as i32;
            self.spin_remaining = spin_steps - 1;
            return self.spin_dir;
        }

        // Both forward sector and wide guard clear — release turn commitment.
        self.turning = false;

        // Random exploration deviation (5% chance per step).
        match rng.next_u32() % 20 {
            0 => Action::RotateLeft  as u8,
            1 => Action::RotateRight as u8,
            _ => Action::Forward     as u8,
        }
    }
}

// ── Reactive pan (mirrors runtime gimbal task) ────────────────────────────────
//
// Splits the scan at its midpoint (camera centre) into left and right halves.
// Lidar range: higher = more open.  Pan toward the more open side.
// pan_deg > 0 = camera looking right (hardware convention).
// Deadband 8°, step cap ±5°/frame — matches the real gimbal task.

fn reactive_pan(scan: &core_types::PseudoLidarScan, cur_pan: f32) -> f32 {
    let n = scan.rays.len();
    if n < 2 { return cur_pan; }
    let mid = n / 2;
    // rays[0..mid]  = camera-right half (lower / more-negative angles)
    // rays[mid..]   = camera-left  half (higher / more-positive angles)
    let right_avg = scan.rays[..mid].iter().map(|r| r.range_m).sum::<f32>() / mid as f32;
    let left_avg  = scan.rays[mid..].iter().map(|r| r.range_m).sum::<f32>() / (n - mid) as f32;
    // right more open → pan right (positive step); left more open → pan left (negative step).
    let pan_error = right_avg - left_avg;
    let step = (pan_error * 20.0).clamp(-5.0, 5.0);
    // Cap at ±30°: at 30° pan the forward sector still has 54° of 57° coverage.
    // Beyond ±45° the forward sector starts going blind (see cast_lidar_pan geometry).
    let new_pan = (cur_pan + step).clamp(-30.0, 30.0);
    if (new_pan - cur_pan).abs() > 8.0 { new_pan } else { cur_pan }
}

// ── Sector range helpers ───────────────────────────────────────────────────────

/// Minimum range of rays whose angle is in [min_a, max_a].
/// Returns 0.0 (treat as blocked) when fewer than 3 rays fall in the sector —
/// this handles the case where pan has shifted the scan outside the queried range.
fn min_sector(scan: &core_types::PseudoLidarScan, min_a: f32, max_a: f32) -> f32 {
    let rays: Vec<f32> = scan.rays.iter()
        .filter(|r| r.angle_rad >= min_a && r.angle_rad <= max_a)
        .map(|r| r.range_m)
        .collect();
    if rays.len() < 3 { return 0.0; }
    rays.iter().cloned().fold(f32::MAX, f32::min).min(3.0)
}

/// Average range of rays whose angle is in [min_a, max_a].
fn avg_sector(scan: &core_types::PseudoLidarScan, min_a: f32, max_a: f32) -> f32 {
    let (sum, n) = scan.rays.iter()
        .filter(|r| r.angle_rad >= min_a && r.angle_rad <= max_a)
        .fold((0.0f32, 0usize), |(s, c), r| (s + r.range_m, c + 1));
    if n == 0 { 3.0 } else { sum / n as f32 }
}

// ── Minimal PRNG (XorShift32) ──────────────────────────────────────────────

struct SimpleRng(u32);

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self((seed as u32).max(1))
    }
    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }
}
