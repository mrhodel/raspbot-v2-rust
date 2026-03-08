//! Event-driven vision gate.
//!
//! Decides whether a new depth inference should run based on frame difference
//! and robot state. When the scene has not changed significantly the gate
//! returns `false` and the caller reuses the previous depth map.

use core_types::GrayFrame;

/// Threshold for mean absolute pixel difference triggering re-inference.
const DEFAULT_DIFF_THRESHOLD: f32 = 8.0;

pub struct EventGate {
    diff_threshold: f32,
    prev_frame: Option<GrayFrame>,
    /// Force re-inference on the next call regardless of frame difference.
    force_next: bool,
}

impl EventGate {
    pub fn new(diff_threshold: f32) -> Self {
        Self { diff_threshold, prev_frame: None, force_next: true }
    }

    pub fn with_default_threshold() -> Self {
        Self::new(DEFAULT_DIFF_THRESHOLD)
    }

    /// Force depth inference on the next `should_infer()` call.
    /// Call this when the robot rotates, the gimbal moves, or exploration
    /// state changes.
    pub fn force(&mut self) {
        self.force_next = true;
    }

    /// Returns `true` if depth inference should run for this frame.
    /// Consumes the frame as the new "previous frame" for comparison.
    pub fn should_infer(&mut self, frame: &GrayFrame) -> bool {
        if self.force_next {
            self.force_next = false;
            self.prev_frame = Some(frame.clone());
            return true;
        }

        let Some(prev) = &self.prev_frame else {
            self.prev_frame = Some(frame.clone());
            return true;
        };

        // Sanity-check dimensions.
        if prev.width != frame.width || prev.height != frame.height {
            self.prev_frame = Some(frame.clone());
            return true;
        }

        let diff = mean_abs_diff(&prev.data, &frame.data);
        self.prev_frame = Some(frame.clone());
        diff >= self.diff_threshold
    }
}

fn mean_abs_diff(a: &[u8], b: &[u8]) -> f32 {
    if a.is_empty() { return 0.0; }
    let sum: u32 = a.iter().zip(b.iter())
        .map(|(&x, &y)| x.abs_diff(y) as u32)
        .sum();
    sum as f32 / a.len() as f32
}
