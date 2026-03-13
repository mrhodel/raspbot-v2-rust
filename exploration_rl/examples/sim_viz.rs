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
    let mut sim   = FastSim::new(seed);
    let mut mapper = Mapper::new();
    let mut rng   = SimpleRng::new(seed ^ 0xdeadbeef);

    let mut obs = sim.reset();
    let mut episode = 0u32;

    loop {
        let t0 = Instant::now();

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
            info!(episode, "Episode ended — resetting");
            let _ = bus.executive_state.send(ExecutiveState::SafetyStopped);
            tokio::time::sleep(Duration::from_millis(400)).await;
            mapper = Mapper::new();
            obs = sim.reset();
            let _ = bus.executive_state.send(ExecutiveState::Exploring);
            continue;
        }

        // Simple heuristic action: bias forward, occasionally rotate to explore.
        let action = heuristic_action(&obs.scan, &mut rng);
        obs = sim.step(action);

        // Pace to target Hz.
        let elapsed = t0.elapsed();
        if elapsed < step_dur {
            tokio::time::sleep(step_dur - elapsed).await;
        }
    }
}

// ── Heuristic action selection ─────────────────────────────────────────────
// Uses the lidar scan to avoid walls: if the forward ray is short, rotate;
// otherwise go forward with occasional random turns to aid exploration.

fn heuristic_action(scan: &core_types::PseudoLidarScan, rng: &mut SimpleRng) -> u8 {
    // Find the forward ray (closest to angle 0).
    let forward_range = scan.rays.iter()
        .min_by(|a, b| a.angle_rad.abs().partial_cmp(&b.angle_rad.abs()).unwrap())
        .map(|r| r.range_m)
        .unwrap_or(3.0);

    // Obstacle close ahead — pick the clearer side.
    if forward_range < 0.5 {
        let left_avg: f32 = scan.rays.iter()
            .filter(|r| r.angle_rad > 0.2)
            .map(|r| r.range_m)
            .sum::<f32>()
            / scan.rays.iter().filter(|r| r.angle_rad > 0.2).count().max(1) as f32;
        let right_avg: f32 = scan.rays.iter()
            .filter(|r| r.angle_rad < -0.2)
            .map(|r| r.range_m)
            .sum::<f32>()
            / scan.rays.iter().filter(|r| r.angle_rad < -0.2).count().max(1) as f32;

        return if left_avg >= right_avg {
            Action::RotateLeft as u8
        } else {
            Action::RotateRight as u8
        };
    }

    // Randomly deviate to explore (5% chance per step).
    match rng.next_u32() % 20 {
        0 => Action::RotateLeft  as u8,
        1 => Action::RotateRight as u8,
        _ => Action::Forward     as u8,
    }
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
