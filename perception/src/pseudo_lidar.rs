//! Pseudo-lidar extraction from a MiDaS depth map.
//!
//! Converts the 2-D depth image into N evenly-spaced lidar-like rays spanning
//! the camera's horizontal FOV. Each ray samples a vertical strip of the depth
//! map (excluding masked rows) and takes the minimum depth value (nearest
//! obstacle) to build a conservative obstacle profile.
//!
//! Parameters (requirements_v3.md §5.2):
//!   rays     : 48
//!   max_range: 3 m
//!   H-FOV    : 110° (camera spec)

use core_types::{DepthMap, LidarRay, PseudoLidarScan};

/// Horizontal field of view of the camera in radians.
const HFOV_RAD: f32 = 110.0_f32 * std::f32::consts::PI / 180.0;

/// MiDaS outputs inverse depth (1 = closest). This scale converts the
/// normalised value to metres. Calibrate against real measurements in Phase 6.
const DEPTH_SCALE_M: f32 = 3.0; // 1.0 normalised → 3 m max range

pub struct PseudoLidarExtractor {
    num_rays: usize,
    max_range_m: f32,
}

impl PseudoLidarExtractor {
    pub fn new(num_rays: usize, max_range_m: f32) -> Self {
        Self { num_rays, max_range_m }
    }

    pub fn extract(&self, depth: &DepthMap) -> PseudoLidarScan {
        let w = depth.width as usize;
        let h = depth.height as usize;
        let mask_row = depth.mask_start_row as usize;
        let usable_rows = mask_row.min(h);

        let rays: Vec<LidarRay> = (0..self.num_rays)
            .map(|i| {
                // Map ray index to horizontal angle (0 = forward, +left / -right).
                let frac = i as f32 / (self.num_rays - 1).max(1) as f32;
                let angle_rad = HFOV_RAD * (frac - 0.5); // centred, left positive

                // Column range for this ray.
                let col_start = ((frac * w as f32) as usize).min(w.saturating_sub(1));
                let col_end = ((frac + 1.0 / self.num_rays as f32) * w as f32)
                    .min(w as f32) as usize;
                let col_end = col_end.max(col_start + 1).min(w);

                // Sample the strip: take max depth value (nearest obstacle).
                // Masked rows (robot body) are already 0 and are skipped.
                let mut max_depth = 0.0_f32;
                let mut valid_samples = 0u32;
                for row in 0..usable_rows {
                    for col in col_start..col_end {
                        let v = depth.data[row * w + col];
                        if v > 0.0 {
                            max_depth = max_depth.max(v);
                            valid_samples += 1;
                        }
                    }
                }

                let confidence = if valid_samples > 0 {
                    (valid_samples as f32 / (usable_rows * (col_end - col_start)) as f32)
                        .min(1.0)
                } else {
                    0.0
                };

                // Convert inverse-depth [0,1] → range [0, max_range_m].
                // Inverse: high value = close; range = scale / depth_value.
                let range_m = if max_depth > 0.0 {
                    (DEPTH_SCALE_M / max_depth).min(self.max_range_m)
                } else {
                    self.max_range_m // no obstacle detected → report max range
                };

                LidarRay { angle_rad, range_m, confidence }
            })
            .collect();

        PseudoLidarScan { t_ms: depth.t_ms, rays }
    }
}
