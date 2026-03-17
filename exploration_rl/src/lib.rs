//! RL-based frontier selection.
//!
//! The RL agent selects *which frontier to target* — it does NOT control
//! motors. Motor control is handled by the classical planning + pure-pursuit
//! stack. This separation keeps the RL action space tiny (5 discrete choices)
//! and lets the robot work reliably before any model is trained.
//!
//! # Layered fallback
//! 1. **ONNX policy** — loaded from `models/frontier_selector.onnx` if present.
//! 2. **Classical heuristic** — always-available fallback; used if no model
//!    file exists or inference fails.
//!
//! # Training
//! Training runs in Python (PyTorch) against episodes collected from `sim_fast`.
//! The `TrainingRunner` in this crate provides a Rust-side episode collector:
//! call `run_episode()` to get a sequence of `(state, action, reward)` tuples
//! suitable for serialisation to disk and consumption by the Python trainer.
//!
//! After training, export the policy network to ONNX and drop the `.onnx` file
//! into `models/`. The `FrontierSelector` will pick it up on next launch.
//!
//! # Bus integration
//! `spawn_selector_task()` integrates with the message bus: it subscribes to
//! `map_frontiers` and `slam_pose2d`, runs the selector, and sends
//! `FrontierChoice` to the planning task via `bus.decision_frontier`.

use std::sync::Arc;
use tracing::{debug, info, warn};

use core_types::{Frontier, FrontierChoice, Pose2D};
use ort::session::Session;
use ort::value::Tensor;

// ── Classical heuristics ──────────────────────────────────────────────────────

/// Select a `FrontierChoice` from the current frontier list using classical
/// (non-learned) rules. Always available; used as fallback before RL trains.
pub fn classical_select(
    frontiers: &[Frontier],
    robot_pose: &Pose2D,
) -> Option<FrontierChoice> {
    if frontiers.is_empty() {
        return None;
    }

    // Filter tiny slivers; fall back to all frontiers if none qualify.
    const MIN_SIZE_CELLS: u32 = 10;
    let large: Vec<&Frontier> = frontiers.iter().filter(|f| f.size_cells >= MIN_SIZE_CELLS).collect();
    let pool: Vec<&Frontier>  = if large.is_empty() { frontiers.iter().collect() } else { large };

    // Score each frontier by size / (1 + dist) — linear distance penalty so
    // far large frontiers (e.g. perimeter walls) beat nearby small slivers.
    let best = pool.iter().copied().max_by(|a, b| {
        let sa = a.size_cells as f32 / (1.0 + dist_sq(robot_pose, a).sqrt());
        let sb = b.size_cells as f32 / (1.0 + dist_sq(robot_pose, b).sqrt());
        sa.partial_cmp(&sb).unwrap()
    });

    best.map(|b| {
        // If the best-scoring frontier is also the nearest, use Nearest so
        // select_frontier_goal can apply its heading-alignment discount.
        let nearest = pool.iter().copied()
            .min_by(|a, c| dist_sq(robot_pose, a).partial_cmp(&dist_sq(robot_pose, c)).unwrap());
        match nearest {
            Some(n) if (n.centroid_x_m - b.centroid_x_m).abs() < 0.01
                    && (n.centroid_y_m - b.centroid_y_m).abs() < 0.01 => FrontierChoice::Nearest,
            _ => FrontierChoice::Largest,
        }
    })
}

fn dist_sq(pose: &Pose2D, f: &Frontier) -> f32 {
    let dx = f.centroid_x_m - pose.x_m;
    let dy = f.centroid_y_m - pose.y_m;
    dx * dx + dy * dy
}

// ── FrontierSelector ──────────────────────────────────────────────────────────

/// Policy wrapper: tries ONNX inference, falls back to classical heuristic.
pub struct FrontierSelector {
    /// Loaded ONNX session (None if no model file found or load failed).
    session: Option<Session>,
    /// Replan cooldown: skip if fewer than this many map updates since last plan.
    min_updates_between_plans: u32,
    updates_since_last_plan: u32,
}

impl FrontierSelector {
    pub fn new(model_path: impl AsRef<std::path::Path>) -> Self {
        let path = model_path.as_ref();
        let session = Self::try_load_model(path);
        if session.is_some() {
            info!("FrontierSelector: ONNX model loaded from {}", path.display());
        } else {
            info!("FrontierSelector: no ONNX model — using classical heuristic");
        }
        Self {
            session,
            min_updates_between_plans: 5,
            updates_since_last_plan: u32::MAX, // trigger immediately
        }
    }

    /// Select a frontier strategy given the current frontiers and robot pose.
    ///
    /// Returns `None` if no frontiers are available or cooldown hasn't elapsed.
    pub fn select(
        &mut self,
        frontiers: &[Frontier],
        robot_pose: &Pose2D,
    ) -> Option<FrontierChoice> {
        self.updates_since_last_plan = self.updates_since_last_plan.saturating_add(1);
        if self.updates_since_last_plan < self.min_updates_between_plans {
            return None;
        }

        let choice = if self.session.is_some() {
            self.onnx_infer(frontiers, robot_pose)
                .or_else(|| classical_select(frontiers, robot_pose))
        } else {
            classical_select(frontiers, robot_pose)
        };

        if choice.is_some() {
            self.updates_since_last_plan = 0;
            debug!(?choice, frontiers = frontiers.len(), "Frontier selected");
        }
        choice
    }

    /// Attempt to load the ONNX model. Returns `Some(Session)` on success.
    fn try_load_model(path: &std::path::Path) -> Option<Session> {
        if !path.exists() {
            return None;
        }
        fn load(path: &std::path::Path) -> ort::Result<Session> {
            Session::builder()?.commit_from_file(path)
        }
        match load(path) {
            Ok(s) => Some(s),
            Err(e) => {
                warn!("Failed to load ONNX model from {}: {e}", path.display());
                None
            }
        }
    }

    /// Run ONNX policy inference. Returns `None` on any error (classical
    /// fallback will be used instead).
    fn onnx_infer(
        &mut self,
        frontiers: &[Frontier],
        robot_pose: &Pose2D,
    ) -> Option<FrontierChoice> {
        let session = self.session.as_mut()?;

        let state = build_state_vector(frontiers, robot_pose);
        // Tensor::from_array((shape, owned_vec)) — no ndarray needed.
        let tensor = Tensor::<f32>::from_array(([1usize, STATE_LEN], state)).ok()?;

        let outputs = session
            .run(ort::inputs!["state" => tensor])
            .ok()?;

        // try_extract_tensor returns (&Shape, &[T]) in ort rc.12.
        let (_, logits) = outputs["logits"].try_extract_tensor::<f32>().ok()?;
        let action = logits
            .iter()
            .copied()
            .enumerate()
            .max_by(|(_, a): &(usize, f32), (_, b): &(usize, f32)| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i as u8)?;

        Some(u8_to_frontier_choice(action))
    }
}

impl Default for FrontierSelector {
    fn default() -> Self {
        Self::new("models/frontier_selector.onnx")
    }
}

// ── State vector builder (for training / inference) ───────────────────────────

/// Maximum number of frontiers encoded in the state vector.
const MAX_FRONTIERS: usize = 8;

/// Total length of the state vector: 4 robot fields + 3 per frontier slot.
const STATE_LEN: usize = 4 + MAX_FRONTIERS * 3; // 28

/// Build a flat state vector for the policy network.
///
/// Layout (fixed size, zero-padded):
///   [robot_x, robot_y, cos(theta), sin(theta),           — 4 floats
///    frontier_0_dx, frontier_0_dy, frontier_0_size_norm,  — 3 × MAX_FRONTIERS
///    ...
///    frontier_N_dx, frontier_N_dy, frontier_N_size_norm]
///
/// Frontier offsets (dx, dy) are relative to robot position and normalised
/// to the arena half-width (5.0 m). Size is normalised by 1000 cells.
pub fn build_state_vector(frontiers: &[Frontier], robot_pose: &Pose2D) -> Vec<f32> {
    let mut v = Vec::with_capacity(STATE_LEN);
    v.push(robot_pose.x_m / 10.0);
    v.push(robot_pose.y_m / 10.0);
    v.push(robot_pose.theta_rad.cos());
    v.push(robot_pose.theta_rad.sin());

    // Sort frontiers by distance (nearest first) before encoding.
    let mut sorted: Vec<&Frontier> = frontiers.iter().collect();
    sorted.sort_by(|a, b| {
        dist_sq(robot_pose, a)
            .partial_cmp(&dist_sq(robot_pose, b))
            .unwrap()
    });

    for i in 0..MAX_FRONTIERS {
        if let Some(f) = sorted.get(i) {
            v.push((f.centroid_x_m - robot_pose.x_m) / 5.0);
            v.push((f.centroid_y_m - robot_pose.y_m) / 5.0);
            v.push(f.size_cells as f32 / 1000.0);
        } else {
            v.extend_from_slice(&[0.0, 0.0, 0.0]); // padding
        }
    }
    debug_assert_eq!(v.len(), STATE_LEN);
    v
}

fn u8_to_frontier_choice(action: u8) -> FrontierChoice {
    match action {
        0 => FrontierChoice::Nearest,
        1 => FrontierChoice::Largest,
        2 => FrontierChoice::Leftmost,
        3 => FrontierChoice::Rightmost,
        _ => FrontierChoice::RandomValid,
    }
}

fn frontier_choice_to_u8(c: &FrontierChoice) -> u8 {
    match c {
        FrontierChoice::Nearest    => 0,
        FrontierChoice::Largest    => 1,
        FrontierChoice::Leftmost   => 2,
        FrontierChoice::Rightmost  => 3,
        FrontierChoice::RandomValid => 4,
    }
}

// ── Training runner ───────────────────────────────────────────────────────────

/// One recorded transition for offline RL training.
#[derive(Debug, Clone)]
pub struct Transition {
    pub state:  Vec<f32>,
    pub action: u8,
    pub reward: f32,
    pub done:   bool,
}

/// Reward signal for frontier selection.
///
/// Call after each planning step completes (robot reached frontier or failed).
pub fn compute_reward(
    new_explored_cells: u32,
    prev_explored_cells: u32,
    collision: bool,
    reached_goal: bool,
) -> f32 {
    let exploration_bonus = (new_explored_cells.saturating_sub(prev_explored_cells)) as f32 * 0.01;
    let collision_penalty = if collision { -5.0 } else { 0.0 };
    let goal_bonus        = if reached_goal { 2.0 } else { 0.0 };
    exploration_bonus + collision_penalty + goal_bonus
}

/// Runs complete episodes against `sim_fast::FastSim` and collects transitions.
///
/// The collected rollouts are suitable for serialisation (e.g., to MessagePack
/// or JSON) and consumption by the Python PPO trainer.
pub struct TrainingRunner {
    pub selector: FrontierSelector,
}

impl TrainingRunner {
    pub fn new() -> Self {
        Self { selector: FrontierSelector::default() }
    }

    /// Run one training episode.
    ///
    /// Returns the collected transitions; caller is responsible for batching
    /// and serialising to disk for the Python training script.
    ///
    /// The mapper and planner are run inline so the training loop is fully
    /// self-contained without a Tokio runtime.
    pub fn run_episode(
        &mut self,
        sim: &mut sim_fast::FastSim,
        max_steps: u32,
    ) -> Vec<Transition> {
        use mapping::Mapper;

        let mut mapper  = Mapper::new();
        let mut transitions = Vec::new();
        let mut prev_explored: u32 = 0;

        let mut obs = sim.reset();
        for _ in 0..max_steps {
            // Update the occupancy grid from the simulated scan.
            let (_, frontiers, stats) = mapper.update(&obs.scan, &obs.pose);
            let state = build_state_vector(&frontiers, &obs.pose);

            // Policy: pick an action (encoded as FrontierChoice index).
            let choice = self.selector
                .select(&frontiers, &obs.pose)
                .unwrap_or(FrontierChoice::Nearest);
            let action = frontier_choice_to_u8(&choice);

            // Step the sim with a motor action derived from the frontier choice.
            // For training we use Forward as the default motor action; full
            // planning integration comes after model export.
            let motor_action = sim_fast::Action::Forward as u8;
            obs = sim.step(motor_action);

            let reward = compute_reward(
                stats.explored_cells,
                prev_explored,
                obs.collision,
                false,
            );
            prev_explored = stats.explored_cells;

            transitions.push(Transition {
                state,
                action,
                reward,
                done: obs.done,
            });

            if obs.done {
                break;
            }
        }
        transitions
    }
}

impl Default for TrainingRunner {
    fn default() -> Self {
        Self::new()
    }
}

// ── Bus integration ───────────────────────────────────────────────────────────

/// Spawn the frontier-selector bus task.
///
/// Subscribes to `map_frontiers` and `slam_pose2d`; sends `FrontierChoice`
/// to the planning task via `bus.decision_frontier`. Call from the runtime.
pub async fn spawn_selector_task(bus: Arc<bus::Bus>) {
    let rx_frontiers = bus.map_frontiers.subscribe();
    let rx_pose      = bus.slam_pose2d.subscribe();

    // Spawn immediately so the caller (run_sim_mode / main) is not blocked.
    // Model loading happens inside the task with a timeout; if ORT hangs (e.g.
    // dev machine without ORT_DYLIB_PATH), only this task stalls — the rest of
    // the runtime keeps running with classical frontier selection as fallback.
    tokio::spawn(async move {
        let mut rx_frontiers = rx_frontiers;
        let mut rx_pose      = rx_pose;

        // Load model on a blocking thread with a 5 s timeout.
        // Fallback constructs a classical-only selector via a guaranteed-absent path
        // so try_load_model() short-circuits on path.exists() without touching ORT.
        let classical = || FrontierSelector::new("/.__no_onnx_model__");
        let mut selector = match tokio::time::timeout(
            std::time::Duration::from_secs(5),
            tokio::task::spawn_blocking(FrontierSelector::default),
        ).await {
            Ok(Ok(s))  => s,
            Ok(Err(_)) => { warn!("FrontierSelector init panicked — using classical"); classical() }
            Err(_)     => { warn!("FrontierSelector init timed out (ORT unavailable?) — using classical"); classical() }
        };

        info!("Frontier selector task started");
        loop {
            match rx_frontiers.recv().await {
                Ok(frontiers) => {
                    let pose = *rx_pose.borrow_and_update();
                    if let Some(choice) = selector.select(&frontiers, &pose) {
                        // Non-blocking send: if the planning task is busy, skip.
                        match bus.decision_frontier.try_send(choice) {
                            Ok(()) => {}
                            Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {
                                debug!("Planning queue full — skipping frontier choice");
                            }
                            Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => break,
                        }
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    warn!("Frontier selector lagged {n} map updates");
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    });
}
