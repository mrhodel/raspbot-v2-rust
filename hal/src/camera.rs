//! Camera HAL trait and stub.

use anyhow::Result;
use async_trait::async_trait;
use core_types::CameraFrame;

#[async_trait]
pub trait Camera: Send + Sync {
    /// Capture one frame. Blocks until a frame is available.
    async fn read_frame(&mut self) -> Result<CameraFrame>;
    /// Return configured (width, height) in pixels.
    fn resolution(&self) -> (u32, u32);
}

// ── Stub ──────────────────────────────────────────────────────────────────────

/// Produces synthetic solid-colour frames. Used in tests and before V4L2
/// integration is complete.
pub struct StubCamera {
    width: u32,
    height: u32,
    t0: std::time::Instant,
    frame_count: u64,
    interval_ms: u64,
}

impl StubCamera {
    pub fn new(cfg: &config::CameraConfig) -> Self {
        Self {
            width: cfg.width,
            height: cfg.height,
            t0: std::time::Instant::now(),
            frame_count: 0,
            interval_ms: 1000 / cfg.fps.max(1) as u64,
        }
    }
}

#[async_trait]
impl Camera for StubCamera {
    async fn read_frame(&mut self) -> Result<CameraFrame> {
        tokio::time::sleep(tokio::time::Duration::from_millis(self.interval_ms)).await;
        self.frame_count += 1;
        // Alternating grey frames to simulate slight scene variation.
        let grey = ((self.frame_count % 32) * 4 + 64) as u8;
        Ok(CameraFrame {
            t_ms: self.t0.elapsed().as_millis() as u64,
            width: self.width,
            height: self.height,
            data: vec![grey; (self.width * self.height * 3) as usize],
        })
    }

    fn resolution(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}
