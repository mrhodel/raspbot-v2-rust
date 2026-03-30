//! SimImu — reads the latest SimStep and adds configurable sensor noise.
//!
//! On collision steps a large accel_x spike is injected so the real
//! crash-detection task (threshold > 15 m/s²) fires in sim.

use std::sync::Arc;
use anyhow::Result;
use async_trait::async_trait;
use core_types::ImuSample;
use hal::Imu;
use tokio::sync::watch;
use crate::SimStep;
use super::{gaussian, xorshift};

pub struct SimImu {
    step_rx:                watch::Receiver<Arc<SimStep>>,
    rng:                    u64,
    noise_gyro_rad_s:       f32,
    noise_accel_m_s2:       f32,
    crash_spike_accel_m_s2: f32,
}

impl SimImu {
    pub fn new(
        step_rx:                watch::Receiver<Arc<SimStep>>,
        seed:                   u64,
        noise_gyro_rad_s:       f32,
        noise_accel_m_s2:       f32,
        crash_spike_accel_m_s2: f32,
    ) -> Self {
        Self { step_rx, rng: seed, noise_gyro_rad_s, noise_accel_m_s2, crash_spike_accel_m_s2 }
    }
}

#[async_trait]
impl Imu for SimImu {
    async fn read_sample(&mut self) -> Result<ImuSample> {
        self.step_rx.changed().await
            .map_err(|_| anyhow::anyhow!("SimImu: sim tick task exited"))?;
        let step = Arc::clone(&*self.step_rx.borrow());
        let gs  = self.noise_gyro_rad_s;
        let as_ = self.noise_accel_m_s2;
        let rng = &mut self.rng;

        let mut sample = ImuSample {
            t_ms:    step.imu.t_ms,
            gyro_x:  step.imu.gyro_x  + gaussian(rng, gs),
            gyro_y:  step.imu.gyro_y  + gaussian(rng, gs),
            gyro_z:  step.imu.gyro_z  + gaussian(rng, gs),
            accel_x: step.imu.accel_x + gaussian(rng, as_),
            accel_y: step.imu.accel_y + gaussian(rng, as_),
            accel_z: step.imu.accel_z,
        };

        // Inject crash spike so the crash-detection task fires in sim.
        // Magnitude is from config (crash_spike_accel_m_s2 > agent.crash.accel_threshold_m_s2).
        if step.collision {
            let sign = if xorshift(rng) & 1 == 0 { 1.0_f32 } else { -1.0_f32 };
            sample.accel_x += sign * self.crash_spike_accel_m_s2;
        }

        Ok(sample)
    }
}
