//! MiDaS depth inference.
//!
//! TODO (Phase 6): load `models/midas_small.onnx` via the `ort` crate and run
//! real inference on the GPU (RTX 3050 on dev machine) or CPU (Pi 5 deploy).
//!
//! For now the stub returns a synthetic depth gradient so downstream code
//! (pseudo-lidar, slam, mapping) can be developed and tested independently.

use anyhow::Result;
use core_types::{CameraFrame, DepthMap};

/// Body-mask: bottom 20% of rows are zeroed (robot chassis occlusion).
/// Calibrate during Phase 5 hw-test and update here + robot_config.yaml.
const BODY_MASK_FRACTION: f32 = 0.20;

pub struct DepthInference {
    /// Output resolution for the depth map (resized from camera frame).
    out_width: u32,
    out_height: u32,
}

impl DepthInference {
    pub fn new(out_width: u32, out_height: u32) -> Self {
        Self { out_width, out_height }
    }

    /// Run inference and return a depth map.
    ///
    /// Real: resize frame → run MiDaS ONNX → normalise → apply mask.
    /// Stub: return a linear gradient (near at bottom, far at top) with mask.
    pub fn infer(&self, frame: &CameraFrame) -> Result<DepthMap> {
        let w = self.out_width;
        let h = self.out_height;
        let mask_row = (h as f32 * (1.0 - BODY_MASK_FRACTION)) as u32;

        let data: Vec<f32> = (0..h)
            .flat_map(|row| {
                (0..w).map(move |_col| {
                    if row >= mask_row {
                        0.0 // masked (robot body)
                    } else {
                        // Stub gradient: 0.1 (far/ceiling) at row 0 → 0.9 at mask_row
                        0.1 + 0.8 * (row as f32 / mask_row.max(1) as f32)
                    }
                })
            })
            .collect();

        Ok(DepthMap {
            t_ms: frame.t_ms,
            width: w,
            height: h,
            data,
            mask_start_row: mask_row,
        })
    }
}
