//! SimGimbal — clamps and stores pan/tilt angles.
//!
//! `latest_pan` is the same Arc<Mutex<f32>> as SimState.latest_pan, so the
//! sim tick task and SimCamera always see the current pan angle.

use std::sync::Arc;
use anyhow::Result;
use async_trait::async_trait;
use hal::Gimbal;

pub struct SimGimbal {
    pan:        Arc<std::sync::Mutex<f32>>,
    tilt:       f32,
    pan_range:  [f32; 2],
    tilt_range: [f32; 2],
}

impl SimGimbal {
    pub fn new(
        pan:        Arc<std::sync::Mutex<f32>>,
        pan_range:  [f32; 2],
        tilt_range: [f32; 2],
    ) -> Self {
        Self { pan, tilt: 0.0, pan_range, tilt_range }
    }
}

#[async_trait]
impl Gimbal for SimGimbal {
    async fn set_pan(&mut self, deg: f32) -> Result<()> {
        let clamped = deg.clamp(self.pan_range[0], self.pan_range[1]);
        *self.pan.lock().unwrap() = clamped;
        Ok(())
    }

    async fn set_tilt(&mut self, deg: f32) -> Result<()> {
        self.tilt = deg.clamp(self.tilt_range[0], self.tilt_range[1]);
        Ok(())
    }

    async fn set_angles(&mut self, pan_deg: f32, tilt_deg: f32) -> Result<()> {
        self.set_pan(pan_deg).await?;
        self.set_tilt(tilt_deg).await
    }

    fn angles(&self) -> (f32, f32) {
        (*self.pan.lock().unwrap(), self.tilt)
    }
}
