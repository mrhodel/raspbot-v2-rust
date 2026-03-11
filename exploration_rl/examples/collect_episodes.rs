//! Collect frontier-selector training transitions from `sim_fast`.
//!
//! Each macro-step:
//!   1. Read current frontiers from the occupancy grid.
//!   2. Select a frontier (epsilon-greedy: classical heuristic + random).
//!   3. Execute NAV_STEPS motor steps steering toward that frontier.
//!   4. Record `(state, action, reward, done)` as one NDJSON line.
//!
//! Output: `checkpoints/transitions.ndjson` — one JSON object per line:
//!   {"state":[...28 f32...],"action":0,"reward":0.12,"done":false}
//!
//! Action IDs:
//!   0=Nearest  1=Largest  2=Leftmost  3=Rightmost  4=RandomValid
//!
//! Usage:
//!   cargo run --release -p exploration_rl --example collect_episodes [episodes] [out.ndjson]

use std::{env, fs, io::Write as IoWrite, path::PathBuf};

use anyhow::Result;
use core_types::{Frontier, Pose2D};
use exploration_rl::build_state_vector;
use mapping::Mapper;
use sim_fast::{Action, FastSim};

// ── Hyperparameters ───────────────────────────────────────────────────────────

/// Probability of choosing a random action (vs. classical heuristic).
const EPSILON: f64 = 0.35;

/// Motor steps executed toward the chosen frontier per macro-step.
const NAV_STEPS: u32 = 30;

/// Max frontier-selection steps per episode.
const MAX_MACRO: u32 = 80;

/// Initial steps at episode start to seed the occupancy map.
const SEED_STEPS: u32 = 15;

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let num_episodes: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(500);
    let out_path = PathBuf::from(
        args.get(2).map(|s| s.as_str()).unwrap_or("checkpoints/transitions.ndjson"),
    );

    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::File::create(&out_path)?;

    let mut rng              = Rng::new(0xdeadbeef_cafebabe_u64);
    let mut total_trans      = 0u64;
    let mut total_collisions = 0u64;
    let mut total_explored   = 0f64;

    for ep in 0..num_episodes {
        let seed = rng.next();
        let mut sim    = FastSim::new(seed);
        let mut mapper = Mapper::new();

        let mut obs = sim.reset();

        // Seed the map with a few forward steps before selecting frontiers.
        for _ in 0..SEED_STEPS {
            mapper.update(&obs.scan, &obs.pose);
            obs = sim.step(Action::Forward as u8);
            if obs.done { break; }
        }

        for _ in 0..MAX_MACRO {
            if obs.done { break; }

            let (_, frontiers, stats_before) = mapper.update(&obs.scan, &obs.pose);
            if frontiers.is_empty() { break; }

            let state   = build_state_vector(&frontiers, &obs.pose);
            let action  = select_action(&frontiers, &obs.pose, &mut rng);
            let target  = frontier_centroid(&frontiers, action, &obs.pose);

            let explored_before = stats_before.explored_cells;
            let mut collision   = false;

            // Navigate toward target for NAV_STEPS.
            for _ in 0..NAV_STEPS {
                let motor = steer_toward(&obs.pose, target);
                obs = sim.step(motor as u8);
                mapper.update(&obs.scan, &obs.pose);
                if obs.collision { collision = true; }
                if obs.done { break; }
            }

            let (_, _, stats_after) = mapper.update(&obs.scan, &obs.pose);
            let explored_gain =
                stats_after.explored_cells.saturating_sub(explored_before) as f32;

            let reward = explored_gain * 0.01
                - if collision { 2.0 } else { 0.0 }
                - 0.01;  // per-step time cost

            // Serialise transition.
            let line = serde_json::json!({
                "state":  state,
                "action": action,
                "reward": reward,
                "done":   obs.done,
            });
            writeln!(file, "{line}")?;

            total_trans      += 1;
            total_explored   += explored_gain as f64;
            if collision { total_collisions += 1; }

            if obs.done { break; }
        }

        if (ep + 1) % 50 == 0 {
            let avg_explored = if total_trans > 0 {
                total_explored / total_trans as f64
            } else { 0.0 };
            eprintln!(
                "ep={:4}  transitions={total_trans}  collisions={total_collisions}  \
                 explored_per_step={avg_explored:.1}",
                ep + 1,
            );
        }
    }

    eprintln!(
        "\nDone. {num_episodes} episodes → {total_trans} transitions → {}",
        out_path.display()
    );
    Ok(())
}

// ── Action selection ─────────────────────────────────────────────────────────

/// Epsilon-greedy: classical heuristic with probability 1-EPSILON,
/// uniform random over 5 actions otherwise.
fn select_action(frontiers: &[Frontier], pose: &Pose2D, rng: &mut Rng) -> u8 {
    if rng.next_f64() < EPSILON {
        rng.next_usize(5) as u8
    } else {
        classical_action(frontiers, pose)
    }
}

/// Classical heuristic:
///   - nearest frontier if it's within 2 m
///   - largest frontier otherwise (avoid getting stuck near start)
fn classical_action(frontiers: &[Frontier], pose: &Pose2D) -> u8 {
    let large: Vec<&Frontier> = frontiers.iter().filter(|f| f.size_cells >= 4).collect();
    if large.is_empty() { return 0; }

    let nearest = large.iter()
        .min_by(|a, b| dist_sq(pose, a).partial_cmp(&dist_sq(pose, b)).unwrap())
        .unwrap();
    let near_dist_sq = dist_sq(pose, nearest);

    if near_dist_sq <= 2.0 * 2.0 {
        0 // Nearest
    } else {
        1 // Largest — bias toward unexplored when nearest is far away
    }
}

// ── Navigation helpers ───────────────────────────────────────────────────────

/// World-frame centroid of the frontier selected by `action`.
fn frontier_centroid(frontiers: &[Frontier], action: u8, pose: &Pose2D) -> [f32; 2] {
    let chosen = match action {
        0 => frontiers.iter()
                .min_by(|a, b| dist_sq(pose, a).partial_cmp(&dist_sq(pose, b)).unwrap()),
        1 => frontiers.iter()
                .max_by_key(|f| f.size_cells),
        2 => frontiers.iter()
                .min_by(|a, b| a.centroid_x_m.partial_cmp(&b.centroid_x_m).unwrap()),
        3 => frontiers.iter()
                .max_by(|a, b| a.centroid_x_m.partial_cmp(&b.centroid_x_m).unwrap()),
        _ => frontiers.iter()
                .min_by(|a, b| dist_sq(pose, a).partial_cmp(&dist_sq(pose, b)).unwrap()),
    };
    chosen.map_or([pose.x_m, pose.y_m], |f| [f.centroid_x_m, f.centroid_y_m])
}

/// One-step motor action to steer toward `target` from `pose`.
fn steer_toward(pose: &Pose2D, target: [f32; 2]) -> Action {
    let dx = target[0] - pose.x_m;
    let dy = target[1] - pose.y_m;
    // Already close enough.
    if dx * dx + dy * dy < 0.1 * 0.1 { return Action::Forward; }
    let world_angle = dy.atan2(dx);
    let local_angle = wrap_angle(world_angle - pose.theta_rad);
    if local_angle.abs() < 0.35 {
        Action::Forward
    } else if local_angle > 0.0 {
        Action::RotateLeft
    } else {
        Action::RotateRight
    }
}

fn dist_sq(pose: &Pose2D, f: &Frontier) -> f32 {
    let dx = f.centroid_x_m - pose.x_m;
    let dy = f.centroid_y_m - pose.y_m;
    dx * dx + dy * dy
}

fn wrap_angle(a: f32) -> f32 {
    use std::f32::consts::PI;
    let mut a = a % (2.0 * PI);
    if a >  PI { a -= 2.0 * PI; }
    if a < -PI { a += 2.0 * PI; }
    a
}

// ── XorShift64 PRNG ──────────────────────────────────────────────────────────

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self { Self(seed) }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn next_usize(&mut self, n: usize) -> usize { (self.next() as usize) % n }
    fn next_f64(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
}
