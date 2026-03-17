//! SimUltrasonic — reads the latest SimStep forward ray distance.
//!
//! Models the HC-SR04 blind-spot: readings below `min_range_cm` are
//! reported as `max_range_cm` (same convention as YahboomUltrasonic).

use std::sync::Arc;
use anyhow::Result;
use async_trait::async_trait;
use core_types::UltrasonicReading;
use hal::Ultrasonic;
use tokio::sync::watch;
use crate::SimStep;
use super::gaussian;

pub struct SimUltrasonic {
    step_rx:      watch::Receiver<Arc<SimStep>>,
    max_range_cm: f32,
    min_range_cm: f32,
    rng:          u64,
}

impl SimUltrasonic {
    pub fn new(
        step_rx:      watch::Receiver<Arc<SimStep>>,
        max_range_cm: f32,
        min_range_cm: f32,
    ) -> Self {
        // Distinct seed so noise is uncorrelated with SimImu.
        Self { step_rx, max_range_cm, min_range_cm, rng: 0xFEED_FACE_CAFE_BABE }
    }
}

#[async_trait]
impl Ultrasonic for SimUltrasonic {
    async fn read_distance(&mut self) -> Result<UltrasonicReading> {
        self.step_rx.changed().await
            .map_err(|_| anyhow::anyhow!("SimUltrasonic: sim tick task exited"))?;
        let step = Arc::clone(&*self.step_rx.borrow());

        // Pick the ray with angle_rad closest to 0 (straight forward).
        let forward_range_m = step
            .scan
            .rays
            .iter()
            .min_by(|a, b| a.angle_rad.abs().partial_cmp(&b.angle_rad.abs()).unwrap())
            .map(|r| r.range_m)
            .unwrap_or(self.max_range_cm / 100.0);

        let mut range_cm = forward_range_m * 100.0 + gaussian(&mut self.rng, 1.5);

        // Blind-spot model: very close objects → no reliable echo → max range.
        if range_cm < self.min_range_cm {
            range_cm = self.max_range_cm;
        }
        let range_cm = range_cm.clamp(self.min_range_cm, self.max_range_cm);

        Ok(UltrasonicReading {
            t_ms:     step.imu.t_ms,
            range_cm,
        })
    }
}
