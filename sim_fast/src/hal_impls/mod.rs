//! Sim HAL implementations backed by FastSim physics.
//!
//! One `SimState` is created in `main()` and acts as a factory for all
//! five sim HAL objects (motor, IMU, ultrasonic, gimbal — camera is in sim_vision).
//! All five share `step_rx` to synchronise on the sim tick task.

pub mod gimbal;
pub mod imu;
pub mod motor;
pub mod ultrasonic;

pub use gimbal::SimGimbal;
pub use imu::SimImu;
pub use motor::SimMotorController;
pub use ultrasonic::SimUltrasonic;

use std::sync::Arc;
use tokio::sync::watch;
use core_types::{ImuSample, MotorCommand, Pose2D, PseudoLidarScan};
use config::{GimbalConfig, SimConfig, UltrasonicConfig};
use crate::{FastSim, RoomKind, SimStep};

/// Shared state between the sim tick task and the five sim HAL implementations.
pub struct SimState {
    /// The physics engine, locked for each step.
    pub sim: Arc<tokio::sync::Mutex<FastSim>>,
    /// Latest motor command written by `SimMotorController`, consumed by sim tick task.
    pub latest_motor: Arc<std::sync::Mutex<Option<MotorCommand>>>,
    /// Latest gimbal pan angle written by `SimGimbal`, consumed by sim tick task.
    pub latest_pan: Arc<std::sync::Mutex<f32>>,
    /// Sim tick task publishes each new `SimStep` here; HAL readers watch it.
    pub step_tx: watch::Sender<Arc<SimStep>>,
    /// Clone this to give each HAL reader its own receiver.
    pub step_rx: watch::Receiver<Arc<SimStep>>,
    /// Seed used for this SimState (forwarded to HAL impls for noise RNGs).
    pub seed: u64,
}

impl SimState {
    pub fn new(seed: u64, room_kind: RoomKind, _cfg: &SimConfig) -> Self {
        let sim = FastSim::new(seed, room_kind);
        // Create an initial SimStep to seed the watch channel.
        let initial = SimStep {
            scan:      PseudoLidarScan { t_ms: 0, rays: Vec::new() },
            imu:       ImuSample {
                t_ms: 0, gyro_x: 0.0, gyro_y: 0.0, gyro_z: 0.0,
                accel_x: 0.0, accel_y: 0.0, accel_z: 9.81,
            },
            pose:      Pose2D::default(),
            collision: false,
            done:      false,
        };
        let (step_tx, step_rx) = watch::channel(Arc::new(initial));
        Self {
            sim:          Arc::new(tokio::sync::Mutex::new(sim)),
            latest_motor: Arc::new(std::sync::Mutex::new(None)),
            latest_pan:   Arc::new(std::sync::Mutex::new(0.0)),
            step_tx,
            step_rx,
            seed,
        }
    }

    pub fn motor_controller(&self) -> SimMotorController {
        SimMotorController::new(Arc::clone(&self.latest_motor))
    }

    pub fn imu(&self, cfg: &SimConfig) -> SimImu {
        SimImu::new(
            self.step_rx.clone(),
            self.seed ^ 0xABCD_ABCD,
            cfg.imu_noise_gyro_rad_s,
            cfg.imu_noise_accel_m_s2,
        )
    }

    pub fn ultrasonic(&self, us_cfg: &UltrasonicConfig) -> SimUltrasonic {
        SimUltrasonic::new(
            self.step_rx.clone(),
            us_cfg.max_range_cm,
            us_cfg.min_range_cm,
        )
    }

    pub fn gimbal(&self, gimbal_cfg: &GimbalConfig) -> SimGimbal {
        SimGimbal::new(
            Arc::clone(&self.latest_pan),
            gimbal_cfg.pan_range,
            gimbal_cfg.tilt_range,
        )
    }
}

// ── Box-Muller Gaussian noise ─────────────────────────────────────────────────

/// Generate one Gaussian sample with the given sigma using Box-Muller transform.
pub fn gaussian(rng: &mut u64, sigma: f32) -> f32 {
    let u1 = (xorshift(rng) as f32 / u64::MAX as f32).max(1e-10_f32);
    let u2 = xorshift(rng) as f32 / u64::MAX as f32;
    let magnitude = (-2.0_f32 * u1.ln()).sqrt() * sigma;
    magnitude * (2.0_f32 * std::f32::consts::PI * u2).cos()
}

#[inline]
pub fn xorshift(state: &mut u64) -> u64 {
    let mut x = if *state == 0 { 0xdeadbeef_cafebabe } else { *state };
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}
