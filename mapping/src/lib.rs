//! Probabilistic occupancy grid mapping.
//!
//! `Mapper` ingests `PseudoLidarScan` + `Pose2D` and maintains a sparse
//! log-odds occupancy grid. Each `update()` call returns:
//!   - `GridDelta`      — cells changed this scan (absolute log-odds values)
//!   - `Vec<Frontier>`  — current frontier list
//!   - `ExploredStats`  — exploration statistics
//!
//! # Grid parameters
//! - Resolution: 5 cm/cell
//! - Log-odds: hit +0.7, miss −0.35, clamped to [−3, +3]
//! - Free threshold: < −0.5  (log-odds)
//! - Occupied threshold: > +0.5  (log-odds)
//!
//! # Frontier detection
//! A frontier cell is a free cell with at least one unknown 4-neighbor.
//! Frontier cells are BFS-clustered into contiguous regions; each region
//! becomes one `Frontier` with a centroid and cell count.

use std::collections::{HashMap, HashSet, VecDeque};

use core_types::{ExploredStats, Frontier, GridDelta, Pose2D, PseudoLidarScan};
use tracing::debug;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Grid cell side length (metres).
pub const RESOLUTION_M: f32 = 0.05;

const LOG_ODDS_HIT:  f32 =  0.7;
const LOG_ODDS_MISS: f32 = -0.35;
const LOG_ODDS_MIN:  f32 = -3.0;
const LOG_ODDS_MAX:  f32 =  3.0;

/// Cells with log-odds below this are considered free.
pub const FREE_THRESH: f32 = -0.5;
/// Cells with log-odds above this are considered occupied.
pub const OCC_THRESH:  f32 =  0.5;

/// Minimum ray confidence required to update the grid.
const MIN_CONFIDENCE: f32 = 0.1;

/// Rays at or beyond this range are "no obstacle detected" sentinels emitted
/// by the pseudo-lidar when depth contrast is below threshold.  Walking their
/// full 60-cell Bresenham trace applies MISS updates through confirmed obstacle
/// cells, eroding them.  Skip these rays entirely — the immediate-vicinity
/// Bresenham intermediates of shorter obstacle rays still mark corridors free.
const MAX_RANGE_M: f32 = 3.0;

// ── OccupancyGrid ─────────────────────────────────────────────────────────────

/// Sparse log-odds occupancy grid. Cells absent from the map are unknown (0.0).
pub struct OccupancyGrid {
    cells: HashMap<(i32, i32), f32>,
}

impl OccupancyGrid {
    fn new() -> Self {
        Self { cells: HashMap::new() }
    }

    /// Log-odds for cell `(x, y)`. Returns 0.0 (unknown) if not yet observed.
    pub fn get(&self, x: i32, y: i32) -> f32 {
        *self.cells.get(&(x, y)).unwrap_or(&0.0)
    }

    /// Apply a log-odds delta, clamping to bounds.
    fn apply(&mut self, x: i32, y: i32, delta: f32) {
        let v = self.cells.entry((x, y)).or_insert(0.0);
        *v = (*v + delta).clamp(LOG_ODDS_MIN, LOG_ODDS_MAX);
    }

    /// True if the cell is confidently free.
    pub fn is_free(&self, x: i32, y: i32) -> bool {
        self.get(x, y) < FREE_THRESH
    }

    /// True if the cell has never been observed.
    pub fn is_unknown(&self, x: i32, y: i32) -> bool {
        !self.cells.contains_key(&(x, y))
    }

    /// True if the cell is safe to traverse (not confidently occupied).
    pub fn is_passable(&self, x: i32, y: i32) -> bool {
        self.get(x, y) < OCC_THRESH
    }

    /// Total number of observed cells.
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }
}

// ── Mapper ────────────────────────────────────────────────────────────────────

/// Stateful occupancy grid mapper.
pub struct Mapper {
    pub grid: OccupancyGrid,
    resolution: f32,
    /// Frontier list from the most recent `update()` call.
    pub last_frontiers: Vec<Frontier>,
}

impl Mapper {
    pub fn new() -> Self {
        Self {
            grid: OccupancyGrid::new(),
            resolution: RESOLUTION_M,
            last_frontiers: Vec::new(),
        }
    }

    /// Ingest one pseudo-lidar scan at the given pose.
    ///
    /// Returns `(GridDelta, frontiers, stats)`. The `GridDelta` cells field
    /// contains `(grid_x, grid_y, new_log_odds)` for every cell touched.
    pub fn update(
        &mut self,
        scan: &PseudoLidarScan,
        pose: &Pose2D,
    ) -> (GridDelta, Vec<Frontier>, ExploredStats) {
        let res = self.resolution;
        let rx = world_to_cell(pose.x_m, res);
        let ry = world_to_cell(pose.y_m, res);

        // Track touched cells so each appears once in the delta.
        let mut touched: HashMap<(i32, i32), ()> = HashMap::new();

        // First pass: collect HIT endpoint cells for all valid obstacle rays.
        // A HIT cell must not receive a MISS from an adjacent ray's Bresenham
        // walk in the same scan — that would net the cell below OCC_THRESH and
        // render it as uncertain instead of occupied.
        let mut hit_endpoints: HashSet<(i32, i32)> = HashSet::new();
        for ray in &scan.rays {
            if ray.confidence < MIN_CONFIDENCE
                || ray.range_m < 0.05
                || ray.range_m >= MAX_RANGE_M
            {
                continue;
            }
            let world_angle = pose.theta_rad + ray.angle_rad;
            let ex = pose.x_m + ray.range_m * world_angle.cos();
            let ey = pose.y_m + ray.range_m * world_angle.sin();
            hit_endpoints.insert((world_to_cell(ex, res), world_to_cell(ey, res)));
        }

        // Second pass: apply Bresenham updates.
        // Max-range rays are skipped — they carry no reliable obstacle depth.
        for ray in &scan.rays {
            if ray.confidence < MIN_CONFIDENCE
                || ray.range_m < 0.05
                || ray.range_m >= MAX_RANGE_M
            {
                continue;
            }
            let world_angle = pose.theta_rad + ray.angle_rad;
            let end_x = pose.x_m + ray.range_m * world_angle.cos();
            let end_y = pose.y_m + ray.range_m * world_angle.sin();
            let hx = world_to_cell(end_x, res);
            let hy = world_to_cell(end_y, res);

            for (cx, cy) in bresenham(rx, ry, hx, hy) {
                if cx == hx && cy == hy {
                    self.grid.apply(cx, cy, LOG_ODDS_HIT);
                    touched.insert((cx, cy), ());
                } else if !hit_endpoints.contains(&(cx, cy)) {
                    // Only apply MISS to cells that are not a HIT endpoint for
                    // another ray — prevents adjacent Bresenham traces from
                    // immediately overwriting obstacle cells with MISS.
                    self.grid.apply(cx, cy, LOG_ODDS_MISS);
                    touched.insert((cx, cy), ());
                }
            }
        }

        // Build delta with the final log-odds value of each touched cell.
        let cells: Vec<(i32, i32, f32)> = touched
            .keys()
            .map(|&(cx, cy)| (cx, cy, self.grid.get(cx, cy)))
            .collect();
        let grid_delta = GridDelta { t_ms: scan.t_ms, cells };

        let (frontiers, stats) = self.detect_frontiers(scan.t_ms);
        self.last_frontiers = frontiers.clone();
        debug!(
            observed_cells = self.grid.len(),
            frontiers = frontiers.len(),
            "Map updated"
        );
        (grid_delta, frontiers, stats)
    }

    /// Convert world coordinates to grid cell indices.
    pub fn world_to_cell(&self, x: f32, y: f32) -> (i32, i32) {
        (world_to_cell(x, self.resolution), world_to_cell(y, self.resolution))
    }

    /// Convert grid cell indices to the world-frame centroid of that cell.
    pub fn cell_to_world(&self, cx: i32, cy: i32) -> (f32, f32) {
        let res = self.resolution;
        (cx as f32 * res + res * 0.5, cy as f32 * res + res * 0.5)
    }

    // ── Frontier detection ────────────────────────────────────────────────

    fn detect_frontiers(&self, t_ms: u64) -> (Vec<Frontier>, ExploredStats) {
        let mut frontier_cells: HashSet<(i32, i32)> = HashSet::new();
        let mut free_count: u32 = 0;

        for (&(cx, cy), &lo) in &self.grid.cells {
            if lo < FREE_THRESH {
                free_count += 1;
                // Frontier if any 4-neighbor is unknown.
                for (nx, ny) in neighbors4(cx, cy) {
                    if self.grid.is_unknown(nx, ny) {
                        frontier_cells.insert((cx, cy));
                        break;
                    }
                }
            }
        }

        let frontiers = cluster_frontiers(&frontier_cells, self.resolution);
        let stats = ExploredStats {
            t_ms,
            explored_cells: free_count,
            frontier_count: frontiers.len() as u32,
        };
        (frontiers, stats)
    }
}

impl Default for Mapper {
    fn default() -> Self {
        Self::new()
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

pub fn world_to_cell(coord: f32, res: f32) -> i32 {
    (coord / res).floor() as i32
}

fn neighbors4(x: i32, y: i32) -> [(i32, i32); 4] {
    [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
}

/// Bresenham's line algorithm — all grid cells from `(x0,y0)` to `(x1,y1)`.
pub fn bresenham(mut x0: i32, mut y0: i32, x1: i32, y1: i32) -> Vec<(i32, i32)> {
    let mut cells = Vec::new();
    let dx = (x1 - x0).abs();
    let dy = (y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx - dy;
    loop {
        cells.push((x0, y0));
        if x0 == x1 && y0 == y1 { break; }
        let e2 = 2 * err;
        if e2 > -dy { err -= dy; x0 += sx; }
        if e2 <  dx { err += dx; y0 += sy; }
    }
    cells
}

/// BFS-cluster frontier cells into contiguous `Frontier` regions.
fn cluster_frontiers(cells: &HashSet<(i32, i32)>, res: f32) -> Vec<Frontier> {
    let mut visited: HashSet<(i32, i32)> = HashSet::new();
    let mut frontiers = Vec::new();

    for &start in cells {
        if visited.contains(&start) {
            continue;
        }
        let mut cluster = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(start);
        visited.insert(start);

        while let Some(cell) = queue.pop_front() {
            cluster.push(cell);
            for nb in neighbors4(cell.0, cell.1) {
                if cells.contains(&nb) && !visited.contains(&nb) {
                    visited.insert(nb);
                    queue.push_back(nb);
                }
            }
        }

        let n = cluster.len() as f32;
        let cx = cluster.iter().map(|&(x, _)| x as f32).sum::<f32>() / n;
        let cy = cluster.iter().map(|&(_, y)| y as f32).sum::<f32>() / n;
        frontiers.push(Frontier {
            centroid_x_m: cx * res + res * 0.5,
            centroid_y_m: cy * res + res * 0.5,
            size_cells: cluster.len() as u32,
        });
    }
    frontiers
}
