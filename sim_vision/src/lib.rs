//! SimCamera — Wolfenstein-style first-person raycaster for sim mode.
//!
//! Produces 640×480 RGB frames that give MiDaS usable texture-compression
//! depth cues:
//!   • Walls: procedural brick pattern keyed to world-space hit coordinates so
//!     near walls show large bricks and far walls show fine bricks.
//!   • Floor: checkerboard tiled in world space — a classic depth-from-texture
//!     cue that MiDaS was trained on.
//!   • Ceiling: uniform dark grey (no depth info needed above horizon).
//!
//! Pan offset is baked into the ray fan angle, exactly as
//! FastSim::cast_lidar_pan does for pseudo-lidar.

use std::sync::Arc;
use anyhow::Result;
use async_trait::async_trait;
use core_types::CameraFrame;
use hal::Camera;
use tokio::sync::watch;
use sim_fast::{FastSim, SimStep};
use sim_fast::hal_impls::SimState;
use config::CameraConfig;

/// Total horizontal field of view (radians) — matches FastSim pseudo-lidar.
const HFOV_RAD: f32 = 110.0 * std::f32::consts::PI / 180.0;

/// World-space brick width/height (metres).  Smaller → finer texture at a
/// given distance → more MiDaS frames-per-metre.
const BRICK_M: f32 = 0.40;

/// World-space floor tile size (metres) for checkerboard.
const TILE_M: f32 = 0.50;

/// Camera height above the floor (metres).  Drives the floor projection so
/// the bottom of the frame shows tiles at ~0.4 m instead of ~1 m.  This
/// gives MiDaS the same "close-floor reference" that a real 20 cm mounted
/// camera provides — without it the first wall the robot faces becomes the
/// closest object in the scene and gets depth = 1.0 regardless of its true
/// distance, collapsing range_m to 0.
const CAMERA_HEIGHT_M: f32 = 0.20;

/// Half vertical field of view (radians).  Used together with CAMERA_HEIGHT_M
/// to compute the world distance to each floor row.
const VFOV_HALF_RAD: f32 = 25.0 * std::f32::consts::PI / 180.0;

// ---------------------------------------------------------------------------

/// Pre-computed data for one screen column, shared by wall and floor passes.
struct ColData {
    wall_top: usize,
    wall_bot: usize,
    /// Ray hit distance (metres).
    dist: f32,
    /// World position of the wall hit — used for brick texture U coordinate.
    hit_x: f32,
    hit_y: f32,
    /// Pre-computed cosine/sine of the world ray angle — used for floor tile
    /// world-position calculation.
    ray_cos: f32,
    ray_sin: f32,
}

// ---------------------------------------------------------------------------

/// Camera that renders first-person views of the FastSim wall grid.
pub struct SimCamera {
    sim:     Arc<tokio::sync::Mutex<FastSim>>,
    step_rx: watch::Receiver<Arc<SimStep>>,
    /// Shared with SimGimbal so the pan angle stays in sync.
    pan:     Arc<std::sync::Mutex<f32>>,
    width:   u32,
    height:  u32,
}

impl SimCamera {
    pub fn new(sim_state: &SimState, cfg: &CameraConfig) -> Self {
        Self {
            sim:     Arc::clone(&sim_state.sim),
            step_rx: sim_state.step_rx.clone(),
            pan:     Arc::clone(&sim_state.latest_pan),
            width:   cfg.width,
            height:  cfg.height,
        }
    }
}

#[async_trait]
impl Camera for SimCamera {
    async fn read_frame(&mut self) -> Result<CameraFrame> {
        // Wait for the sim tick task to publish a new step.
        self.step_rx
            .changed()
            .await
            .map_err(|_| anyhow::anyhow!("SimCamera: sim tick task exited"))?;

        let t_ms = self.step_rx.borrow().imu.t_ms;

        // Read current pan angle (degrees → radians, sign matches cast_lidar_pan).
        let pan_rad = -*self.pan.lock().unwrap() * std::f32::consts::PI / 180.0;

        let w = self.width as usize;
        let h = self.height as usize;
        let half_h = h as f32 * 0.5;
        let mut data = vec![0u8; w * h * 3];

        // Lock FastSim for the duration of rendering (fast CPU math, no await).
        let sim = self.sim.lock().await;
        let pose        = sim.pose();
        let robot_theta = pose.theta_rad;
        let robot_x     = pose.x_m;
        let robot_y     = pose.y_m;

        // ── First pass: one ray per screen column ──────────────────────────
        let mut cols = Vec::with_capacity(w);
        for x in 0..w {
            let frac        = x as f32 / (w.saturating_sub(1).max(1)) as f32;
            let local_angle = pan_rad + (-HFOV_RAD * 0.5 + frac * HFOV_RAD);
            let world_angle = robot_theta + local_angle;
            let ray_cos     = world_angle.cos();
            let ray_sin     = world_angle.sin();

            let dist    = sim.cast_single_ray(world_angle).max(0.01);
            let wall_h  = (h as f32 / dist).min(h as f32);
            let wall_top = ((half_h - wall_h * 0.5).max(0.0)) as usize;
            let wall_bot = ((half_h + wall_h * 0.5).min(h as f32)) as usize;

            cols.push(ColData {
                wall_top,
                wall_bot,
                dist,
                hit_x: robot_x + dist * ray_cos,
                hit_y: robot_y + dist * ray_sin,
                ray_cos,
                ray_sin,
            });
        }

        // ── Second pass: pixel fill ────────────────────────────────────────
        for y in 0..h {
            for x in 0..w {
                let col = &cols[x];
                let idx = (y * w + x) * 3;

                if y < col.wall_top {
                    // ── Ceiling ─────────────────────────────────────────────
                    // Uniform dark blue-grey: no depth info needed above horizon.
                    data[idx]     = 50;
                    data[idx + 1] = 50;
                    data[idx + 2] = 60;

                } else if y < col.wall_bot {
                    // ── Wall — brick texture ────────────────────────────────
                    // Base intensity: brighter when closer (useful secondary cue).
                    let base = (255.0 * (1.0 - col.dist / 3.0)).clamp(80.0, 220.0);

                    // Texture coordinates in world space:
                    //   U — lateral offset along the wall face.  We use
                    //       (hit_x + hit_y) / BRICK_M which works for both
                    //       axis-aligned wall orientations because one of
                    //       (hit_x, hit_y) varies across the face while the
                    //       other is approximately constant.
                    //   V — vertical position within the wall strip (0..1).
                    let wall_h_px = (col.wall_bot - col.wall_top).max(1) as f32;
                    let tex_v = (y - col.wall_top) as f32 / wall_h_px;
                    let tex_u = ((col.hit_x + col.hit_y) / BRICK_M).fract().abs();

                    // Brick rows: 6 per full wall-height unit, staggered
                    // every other row by 0.5 to form a running-bond pattern.
                    let brick_row  = (tex_v * 6.0) as usize;
                    let row_frac   = (tex_v * 6.0).fract();
                    let stagger    = if brick_row % 2 == 0 { 0.0f32 } else { 0.5f32 };
                    let brick_u    = (tex_u + stagger).fract();

                    // Mortar lines: 12% of height between rows, 8% at sides.
                    let is_mortar  = row_frac > 0.88 || brick_u < 0.04 || brick_u > 0.96;
                    let factor     = if is_mortar { 0.50 } else { 1.0 };

                    let intensity  = (base * factor) as u8;
                    data[idx]     = intensity;
                    data[idx + 1] = ((intensity as f32) * 0.9 + 10.0).min(255.0) as u8;
                    data[idx + 2] = ((intensity as f32) * 0.8 + 20.0).min(255.0) as u8;

                } else {
                    // ── Floor — world-space checkerboard ───────────────────
                    // Use camera-height projection: the vertical pixel offset
                    // below the horizon maps to a floor distance via the angle
                    // formula  dist = camera_height / tan(row_angle).
                    // This makes the bottom of the frame correspond to ~0.4 m
                    // (same as a real 20 cm mounted camera), giving MiDaS the
                    // close-floor depth anchor it uses for distance calibration.
                    let row_offset     = (y as f32 - half_h).abs().max(0.5);
                    let row_angle      = row_offset / half_h * VFOV_HALF_RAD;
                    let tan_row        = row_angle.tan().max(1e-4);
                    let floor_dist     = CAMERA_HEIGHT_M / tan_row;
                    let fx = robot_x + floor_dist * col.ray_cos;
                    let fy = robot_y + floor_dist * col.ray_sin;

                    // Checkerboard cell index: XOR of integer tile coords.
                    let checker = ((fx / TILE_M).floor() as i32
                                 + (fy / TILE_M).floor() as i32)
                                 .unsigned_abs() % 2;

                    // Gradient brightens toward the bottom of the screen
                    // so very close floor is a little lighter than the horizon.
                    let t = if h > col.wall_bot {
                        (y - col.wall_bot) as f32 / (h - col.wall_bot).max(1) as f32
                    } else {
                        0.0
                    };
                    let base_b   = 60.0 + 40.0 * t;
                    let factor   = if checker == 0 { 1.0 } else { 0.50 };
                    let b        = (base_b * factor) as u8;

                    data[idx]     = b;
                    data[idx + 1] = b;
                    data[idx + 2] = ((b as f32) * 0.9) as u8;
                }
            }
        }

        Ok(CameraFrame { t_ms, width: self.width, height: self.height, data })
    }

    fn resolution(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

// Keep the old stub around so nothing breaks if something imports it.
pub struct VisualSim;
