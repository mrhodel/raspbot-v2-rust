//! Gimbal HAL trait and stub.
//!
//! Pan/tilt servos are driven by the Yahboom expansion board via I2C.
//! TODO (Phase 2): implement real I2C servo commands via rppal.

use anyhow::Result;
use async_trait::async_trait;
use tracing::debug;

#[async_trait]
pub trait Gimbal: Send + Sync {
    /// Set pan angle in degrees. Clamped to configured pan_range.
    async fn set_pan(&mut self, deg: f32) -> Result<()>;
    /// Set tilt angle in degrees. Clamped to configured tilt_range.
    async fn set_tilt(&mut self, deg: f32) -> Result<()>;
    /// Return current (pan_deg, tilt_deg).
    fn angles(&self) -> (f32, f32);
}

// ── Stub ──────────────────────────────────────────────────────────────────────

pub struct StubGimbal {
    pan: f32,
    tilt: f32,
}

impl StubGimbal {
    pub fn new(initial_tilt: f32) -> Self {
        Self { pan: 0.0, tilt: initial_tilt }
    }
}

#[async_trait]
impl Gimbal for StubGimbal {
    async fn set_pan(&mut self, deg: f32) -> Result<()> {
        self.pan = deg.clamp(-90.0, 90.0);
        debug!(pan = self.pan, "StubGimbal: pan");
        Ok(())
    }

    async fn set_tilt(&mut self, deg: f32) -> Result<()> {
        self.tilt = deg.clamp(-45.0, 30.0);
        debug!(tilt = self.tilt, "StubGimbal: tilt");
        Ok(())
    }

    fn angles(&self) -> (f32, f32) {
        (self.pan, self.tilt)
    }
}
