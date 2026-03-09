//! RL-based frontier selection.
//!
//! The RL agent selects *which frontier to target* — it does NOT control
//! motors. Motor control is handled by the classical planning + pure-pursuit
//! stack. This separation keeps the RL action space tiny (5 discrete choices)
//! and lets the robot work reliably before any model is trained.
//!
//! # Layered fallback
//! 1. **ONNX policy** — loaded from `models/frontier_selector.onnx` if present.
//!    (Stub: returns random valid action until a model is exported.)
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

    // Prefer the largest frontier with at least MIN_SIZE_CELLS cells.
    const MIN_SIZE_CELLS: u32 = 4;
    let large: Vec<&Frontier> = frontiers
        .iter()
        .filter(|f| f.size_cells >= MIN_SIZE_CELLS)
        .collect();

    if large.is_empty() {
        return Some(FrontierChoice::Nearest);
    }

    // Among large frontiers, choose the nearest to avoid long detours early.
    let nearest = large.iter().min_by(|a, b| {
        let da = dist_sq(robot_pose, a);
        let db = dist_sq(robot_pose, b);
        da.partial_cmp(&db).unwrap()
    });

    nearest.map(|_| FrontierChoice::Nearest)
}

fn dist_sq(pose: &Pose2D, f: &Frontier) -> f32 {
    let dx = f.centroid_x_m - pose.x_m;
    let dy = f.centroid_y_m - pose.y_m;
    dx * dx + dy * dy
}

// ── FrontierSelector ──────────────────────────────────────────────────────────

/// Policy wrapper: tries ONNX inference, falls back to classical heuristic.
pub struct FrontierSelector {
    /// Path to the ONNX model file (may not exist yet).
    model_path: String,
    /// True if the ONNX model has been successfully loaded.
    model_loaded: bool,
    /// Replan cooldown: skip if fewer than this many map updates since last plan.
    min_updates_between_plans: u32,
    updates_since_last_plan: u32,
}

impl FrontierSelector {
    pub fn new(model_path: impl Into<String>) -> Self {
        let model_path = model_path.into();
        let model_loaded = Self::try_load_model(&model_path);
        if model_loaded {
            info!("FrontierSelector: ONNX model loaded from {model_path}");
        } else {
            info!("FrontierSelector: no ONNX model — using classical heuristic");
        }
        Self {
            model_path,
            model_loaded,
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

        let choice = if self.model_loaded {
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

    /// Attempt to load the ONNX model. Returns true if successful.
    fn try_load_model(path: &str) -> bool {
        // TODO (Phase 10): load with `ort::Session::builder().commit_from_file(path)`
        // For now, check if the file exists and log; actual inference is stubbed.
        std::path::Path::new(path).exists()
    }

    /// Run ONNX policy inference.
    ///
    /// TODO (Phase 10): build state vector (frontiers + pose), run session,
    /// decode argmax of output logits into FrontierChoice.
    fn onnx_infer(
        &self,
        frontiers: &[Frontier],
        robot_pose: &Pose2D,
    ) -> Option<FrontierChoice> {
        // TODO (Phase 10): load ort::Session from self.model_path, build state
        // vector with build_state_vector(), run inference, decode argmax.
        let _ = (&self.model_path, frontiers, robot_pose);
        warn!("ONNX inference not yet implemented — using classical fallback");
        None
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
    let mut v = Vec::with_capacity(4 + MAX_FRONTIERS * 3);
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
    v
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

fn frontier_choice_to_u8(c: &FrontierChoice) -> u8 {
    match c {
        FrontierChoice::Nearest    => 0,
        FrontierChoice::Largest    => 1,
        FrontierChoice::Leftmost   => 2,
        FrontierChoice::Rightmost  => 3,
        FrontierChoice::RandomValid => 4,
    }
}

// ── Bus integration ───────────────────────────────────────────────────────────

/// Spawn the frontier-selector bus task.
///
/// Subscribes to `map_frontiers` and `slam_pose2d`; sends `FrontierChoice`
/// to the planning task via `bus.decision_frontier`. Call from the runtime.
pub async fn spawn_selector_task(bus: Arc<bus::Bus>) {
    let mut rx_frontiers = bus.map_frontiers.subscribe();
    let mut rx_pose      = bus.slam_pose2d.subscribe();
    let mut selector     = FrontierSelector::default();

    tokio::spawn(async move {
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
