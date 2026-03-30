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
pub const MAX_VX: f32 = 0.3;
/// Maximum angular rate (rad/s equivalent) — used as cap for spin-in-place.
const MAX_OMEGA: f32 = 4.0;  // rad/s — real robot: 13.7 rad/s @ 100% → 4.8 rad/s @ 35% duty
/// Proportional gain for heading correction (rad/s per radian of heading error).
/// Kept lower than MAX_OMEGA to avoid oscillation on small corrections.
const OMEGA_GAIN: f32 = 2.0;  // rad/s per rad — tune up if tracking is too sluggish

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

        // Find which waypoint we're closest to — this is our position on the
        // path.  Searching from there forward prevents re-targeting already-
        // passed waypoints that are still ≥ lookahead_m behind the robot.
        let closest_idx = path.waypoints
            .iter()
            .enumerate()
            .min_by(|(_, &[ax, ay]), (_, &[bx, by])| {
                let da2 = (ax - pose.x_m).powi(2) + (ay - pose.y_m).powi(2);
                let db2 = (bx - pose.x_m).powi(2) + (by - pose.y_m).powi(2);
                da2.partial_cmp(&db2).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Lookahead point: first waypoint from closest onward that is beyond
        // lookahead distance.  Falls back to the final waypoint when all
        // remaining waypoints are within the lookahead circle.
        let target = path.waypoints[closest_idx..]
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

        // Forward speed: smoothly scaled over ±120° heading error.
        // Old formula (cos.max(0)) zeroed vx at ±90°, causing stop-and-pivot
        // at every A* path corner.  New formula maps [0°→120°] → [MAX_VX→0],
        // so the robot keeps ~33% forward speed through 90° turns (arc motion)
        // and only fully stops when the waypoint is >120° behind.
        const FORWARD_ARC_COS: f32 = -0.5; // cos(120°)
        let vx_factor = ((heading_err.cos() - FORWARD_ARC_COS) / (1.0 - FORWARD_ARC_COS))
            .clamp(0.0, 1.0);
        let vx = MAX_VX * vx_factor * (dist / self.lookahead_m).min(1.0);

        // Angular rate: proportional to heading error, but capped so the
        // turning radius never falls below MIN_RADIUS_M while moving forward.
        // When stopped (vx≈0) the full MAX_OMEGA is allowed so the robot can
        // spin to face a waypoint that is behind it.
        // The cap must engage at any non-zero vx — not just above 0.15 m/s.
        // At low vx (e.g. 0.08 m/s near-zone) the old 0.15 threshold let
        // omega reach MAX_OMEGA=4.0, causing the robot to spin 10× faster
        // than it moved forward and waggle left/right visually.
        const MIN_RADIUS_M: f32 = 0.35;
        let omega_cap = if vx > 0.01 { (vx / MIN_RADIUS_M).min(MAX_OMEGA) } else { MAX_OMEGA };
        let omega = (OMEGA_GAIN * heading_err).clamp(-omega_cap, omega_cap);

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
