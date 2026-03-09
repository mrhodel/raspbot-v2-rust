//! Pure-pursuit motion controller.
//!
//! `PurePursuitController::compute()` takes the robot's current `Pose2D` and
//! a `Path` of waypoints and produces a `CmdVel`.
//!
//! # Algorithm
//! 1. Find the first waypoint that is at least `lookahead_m` ahead of the robot.
//!    If none exists (robot is near the end), use the final waypoint.
//! 2. Compute the heading error to that lookahead point in the robot frame.
//! 3. Scale forward speed by heading alignment (slow down for sharp turns).
//! 4. Proportional angular velocity toward the target heading.
//!
//! `vy` is always 0 in this implementation — mecanum strafing is reserved for
//! a later revision when lateral control is needed.

use core_types::{CmdVel, Path, Pose2D};

/// Default lookahead distance (metres).
const DEFAULT_LOOKAHEAD_M: f32 = 0.3;
/// Maximum forward speed equivalent (will be scaled to duty-cycle by HAL).
const MAX_VX: f32 = 0.3;
/// Maximum angular rate (rad/s equivalent).
const MAX_OMEGA: f32 = 1.5;

pub struct PurePursuitController {
    lookahead_m: f32,
}

impl PurePursuitController {
    pub fn new() -> Self {
        Self { lookahead_m: DEFAULT_LOOKAHEAD_M }
    }

    pub fn with_lookahead(lookahead_m: f32) -> Self {
        Self { lookahead_m }
    }

    /// Compute a `CmdVel` steering the robot toward the next lookahead point.
    ///
    /// Returns a zero command if the path is empty.
    pub fn compute(&self, pose: &Pose2D, path: &Path) -> CmdVel {
        if path.waypoints.is_empty() {
            return CmdVel { t_ms: pose.t_ms, vx: 0.0, vy: 0.0, omega: 0.0 };
        }

        // Lookahead point: first waypoint beyond lookahead distance, else last.
        let target = path
            .waypoints
            .iter()
            .find(|&&[wx, wy]| {
                let dx = wx - pose.x_m;
                let dy = wy - pose.y_m;
                dx * dx + dy * dy >= self.lookahead_m * self.lookahead_m
            })
            .copied()
            .unwrap_or_else(|| *path.waypoints.last().unwrap());

        let dx = target[0] - pose.x_m;
        let dy = target[1] - pose.y_m;
        let dist = (dx * dx + dy * dy).sqrt();
        let angle_world = dy.atan2(dx);
        let heading_err = wrap_angle(angle_world - pose.theta_rad);

        // Forward speed: reduce as heading error grows, scale by distance.
        let vx = MAX_VX
            * heading_err.cos().max(0.0)
            * (dist / self.lookahead_m).min(1.0);

        // Angular rate: proportional to heading error.
        let omega = (MAX_OMEGA * heading_err).clamp(-MAX_OMEGA, MAX_OMEGA);

        CmdVel { t_ms: pose.t_ms, vx, vy: 0.0, omega }
    }
}

impl Default for PurePursuitController {
    fn default() -> Self {
        Self::new()
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn wrap_angle(a: f32) -> f32 {
    use std::f32::consts::PI;
    let mut a = a % (2.0 * PI);
    if a >  PI { a -= 2.0 * PI; }
    if a < -PI { a += 2.0 * PI; }
    a
}
