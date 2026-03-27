//! Pseudo-lidar extraction from a MiDaS depth map.
//!
//! Converts the 2-D depth image into N evenly-spaced lidar-like rays spanning
//! the camera's horizontal FOV.
//!
//! ## MiDaS depth is relative, not metric
//!
//! MiDaS normalises inverse depth *within each frame*: the closest point
//! always gets ≈1.0 and the farthest ≈0.0, regardless of actual distances.
//! A naive minimum-ray approach therefore reports a false "close obstacle"
//! whenever the background is the closest thing in the scene.
//!
//! ## Contrast-based obstacle detection
//!
//! Instead of treating every high-depth pixel as close, we compute the
//! scene mean depth across the usable rows, then flag a strip as containing
//! an obstacle only when its peak depth exceeds the scene mean by more than
//! `OBSTACLE_CONTRAST`.  This detects locally close objects (boxes, walls
//! approached head-on) while ignoring the "background is closest" baseline.
//!
//! A strip that does NOT exceed the contrast threshold is reported at
//! `max_range_m` (free space).
//!
//! Parameters (requirements_v3.md §5.2):
//!   rays     : 48
//!   max_range: 3 m
//!   H-FOV    : 110° (camera spec)

use core_types::{DepthMap, LidarRay, PseudoLidarScan};

/// Horizontal field of view of the camera in radians.
const HFOV_RAD: f32 = 110.0_f32 * std::f32::consts::PI / 180.0;

/// How much a strip's peak depth must exceed the scene mean before it is
/// treated as an obstacle.  Tuned empirically: open-floor variance is ≈0.05;
/// a box or wall fills a strip with depth ≈0.15–0.25 above the mean.
const OBSTACLE_CONTRAST: f32 = 0.10;

/// Any computed range below this value is treated as free space rather than
/// a close obstacle.  Values near 0 m arise from floor-pixel artifacts (floor
/// tiles at ~0.43 m anchor MiDaS inverse-depth to 1.0 → formula gives 0 m).
/// Real obstacles at < 0.10 m are handled by the ultrasonic safety interlock.
const MIN_OBS_RANGE_M: f32 = 0.10;

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
        // Use the top 3/4 of usable rows.  Excluding the bottom quarter avoids
        // floor pixels dominating the scene mean; the contrast threshold handles
        // remaining floor discrimination.  Using 3/4 (vs the previous 1/2) lets
        // low obstacles (short boxes, chair legs) appear in the used region when
        // the camera is tilted down 25°.
        let upper_rows = ((usable_rows * 3) / 4).max(1);

        // ── Scene mean depth (first pass) ────────────────────────────────────
        // Computed over the same upper region used for ray sampling so that
        // the contrast threshold is calibrated against the background the
        // robot actually sees.
        let mut depth_sum = 0.0_f32;
        let mut depth_count = 0u32;
        for row in 0..upper_rows {
            for col in 0..w {
                let v = depth.data[row * w + col];
                if v > 0.0 {
                    depth_sum += v;
                    depth_count += 1;
                }
            }
        }
        let scene_mean = if depth_count > 0 { depth_sum / depth_count as f32 } else { 0.5 };
        let obstacle_threshold = scene_mean + OBSTACLE_CONTRAST;

        // ── Ray extraction (second pass) ──────────────────────────────────────
        let rays: Vec<LidarRay> = (0..self.num_rays)
            .map(|i| {
                // Map ray index to horizontal angle (0 = forward, +left / -right).
                // col 0 = left of image = left of robot frame = +angle.
                let frac = i as f32 / (self.num_rays - 1).max(1) as f32;
                let angle_rad = HFOV_RAD * (0.5 - frac);

                // Column range for this ray.
                let col_start = ((frac * w as f32) as usize).min(w.saturating_sub(1));
                let col_end = ((frac + 1.0 / self.num_rays as f32) * w as f32)
                    .min(w as f32) as usize;
                let col_end = col_end.max(col_start + 1).min(w);

                let mut max_depth = 0.0_f32;
                let mut valid_samples = 0u32;
                for row in 0..upper_rows {
                    for col in col_start..col_end {
                        let v = depth.data[row * w + col];
                        if v > 0.0 {
                            max_depth = max_depth.max(v);
                            valid_samples += 1;
                        }
                    }
                }

                let confidence = if valid_samples > 0 {
                    (valid_samples as f32 / (upper_rows * (col_end - col_start)) as f32)
                        .min(1.0)
                } else {
                    0.0
                };

                // Only report an obstacle when this strip's peak depth is
                // notably above the scene average.  Otherwise report free space.
                // Also treat any implausibly-close reading (< MIN_OBS_RANGE_M)
                // as free space — such values are floor/artifact pixels, not
                // real obstacles (ultrasonic handles the true near field).
                let range_m = if max_depth >= obstacle_threshold {
                    let r = self.max_range_m * (1.0 - max_depth);
                    if r < MIN_OBS_RANGE_M { self.max_range_m } else { r }
                } else {
                    self.max_range_m
                };

                LidarRay { angle_rad, range_m, confidence }
            })
            .collect();

        let min_r = rays.iter().map(|r| r.range_m).fold(f32::MAX, f32::min);
        let max_r = rays.iter().map(|r| r.range_m).fold(0.0_f32, f32::max);
        let avg_conf = rays.iter().map(|r| r.confidence).sum::<f32>() / rays.len() as f32;
        tracing::info!(
            min_range_m = min_r, max_range_m = max_r, avg_conf,
            scene_mean_depth = scene_mean, obstacle_threshold,
            "PseudoLidar extracted"
        );
        PseudoLidarScan { t_ms: depth.t_ms, rays }
    }
}
