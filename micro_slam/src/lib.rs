//! Micro-SLAM / pose estimation.
//!
//! Milestone 1 — IMU-only dead reckoning stub:
//!   Integrates gyro_z from IMU samples to maintain a heading estimate and
//!   publishes a Pose2D on the bus. This satisfies the milestone requirement
//!   "produce a basic fused pose estimate" while the real visual-inertial
//!   pipeline is developed.
//!
//! TODO (Phase 7): implement full pipeline from requirements_v3.md §6:
//!   - FAST/ORB feature detection
//!   - Lucas–Kanade optical flow tracking
//!   - Frame-to-frame visual motion estimation
//!   - IMU propagation and fusion
//!   - Keyframe buffer for local drift reduction
//!
//! See requirements_v3.md §6 for interface contracts (VisualDelta, Pose2D,
//! KeyframeEvent, confidence metric).

use core_types::{CmdVel, ImuSample, Pose2D};
use std::f32::consts::TAU;

/// Integrates IMU gyro_z at a fixed rate to estimate heading.
/// Also integrates motor feedforward velocity (set via `set_velocity`) into x,y.
pub struct ImuDeadReckon {
    pose: Pose2D,
    t0: std::time::Instant,
    last_t_ms: Option<u64>,
    gz_bias: f32,
    vel_fwd_m_s: f32,
    vel_lat_m_s: f32,
}

impl ImuDeadReckon {
    pub fn new() -> Self {
        Self::with_gz_bias(0.0)
    }

    pub fn with_gz_bias(gz_bias: f32) -> Self {
        Self {
            pose: Pose2D::default(),
            t0: std::time::Instant::now(),
            last_t_ms: None,
            gz_bias,
            vel_fwd_m_s: 0.0,
            vel_lat_m_s: 0.0,
        }
    }

    /// Set the current motor feedforward velocity for x,y integration.
    ///
    /// `scale_m_s` = `(max_motor_duty / 100) × kinematics.forward_speed_m_s`.
    /// Called by the runtime each IMU tick with the velocity last applied to
    /// the motors, so x,y is integrated consistently with actual wheel motion.
    pub fn set_velocity(&mut self, cmd: &CmdVel, scale_m_s: f32) {
        const MAX_VX: f32 = 0.3; // matches cmdvel_to_motor / PurePursuitController
        self.vel_fwd_m_s = (cmd.vx / MAX_VX).clamp(-1.0, 1.0) * scale_m_s;
        self.vel_lat_m_s = (cmd.vy / MAX_VX).clamp(-1.0, 1.0) * scale_m_s;
    }

    /// Update pose from one IMU sample. Returns the new Pose2D.
    pub fn update(&mut self, sample: &ImuSample) -> Pose2D {
        let now_ms = self.t0.elapsed().as_millis() as u64;

        if let Some(prev_t) = self.last_t_ms {
            let dt_s = (sample.t_ms.saturating_sub(prev_t)) as f32 / 1000.0;
            self.pose.theta_rad += (sample.gyro_z - self.gz_bias) * dt_s;
            // Normalise to (-π, π].
            self.pose.theta_rad = wrap_angle(self.pose.theta_rad);
            // Integrate motor feedforward velocity into x,y (robot frame → world frame).
            let (sin_h, cos_h) = self.pose.theta_rad.sin_cos();
            self.pose.x_m += (self.vel_fwd_m_s * cos_h - self.vel_lat_m_s * sin_h) * dt_s;
            self.pose.y_m += (self.vel_fwd_m_s * sin_h + self.vel_lat_m_s * cos_h) * dt_s;
        }

        self.last_t_ms = Some(sample.t_ms);
        self.pose.t_ms = now_ms;
        // Low confidence: pure IMU integration drifts over time.
        self.pose.confidence = 0.3;
        self.pose
    }

    pub fn current_pose(&self) -> Pose2D {
        self.pose
    }
}

impl Default for ImuDeadReckon {
    fn default() -> Self { Self::new() }
}

fn wrap_angle(a: f32) -> f32 {
    let a = a % TAU;
    if a > std::f32::consts::PI { a - TAU }
    else if a < -std::f32::consts::PI { a + TAU }
    else { a }
}
