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

const LOG_ODDS_HIT:  f32 =  1.4;
const LOG_ODDS_MISS: f32 = -0.70;
const LOG_ODDS_MIN:  f32 = -3.0;
const LOG_ODDS_MAX:  f32 =  3.0;

/// Cells with log-odds below this are considered free.
pub const FREE_THRESH: f32 = -0.5;
/// Cells with log-odds above this are considered occupied.
pub const OCC_THRESH:  f32 =  0.5;

/// Minimum ray confidence required to update the grid.
const MIN_CONFIDENCE: f32 = 0.02;

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
    /// Cells pre-seeded with ground-truth walls (sim-only).
    /// Re-stamped to LOG_ODDS_MAX after every update so MISS sweeps can never
    /// erode them — prevents frontiers forming outside the arena walls.
    seeded_walls: HashSet<(i32, i32)>,
    /// Arena size in grid cells, set by `seed_walls()`.
    /// When present, `detect_frontiers()` discards any frontier cell outside
    /// `[0, arena_cells)` × `[0, arena_cells)` — the direct guard against
    /// stray Bresenham endpoints outside the seeded margin creating fake clusters.
    arena_cells: Option<i32>,
}

impl Mapper {
    pub fn new() -> Self {
        Self {
            grid: OccupancyGrid::new(),
            resolution: RESOLUTION_M,
            last_frontiers: Vec::new(),
            seeded_walls: HashSet::new(),
            arena_cells: None,
        }
    }

    /// Pre-seed the grid with ground-truth walls (sim-only, called once at startup).
    /// Sets each wall cell to LOG_ODDS_MAX so A* treats them as obstacles immediately
    /// and frontiers never form at room boundaries.
    pub fn seed_walls(&mut self, wall_grid: &[bool], grid_cells: usize) {
        // Remove any previously seeded cells so a second call (e.g. after a
        // sim reset that regenerates the maze) fully replaces the old layout
        // rather than accumulating two different mazes.
        for cell in self.seeded_walls.drain() {
            self.grid.cells.remove(&cell);
        }
        self.arena_cells = Some(grid_cells as i32);
        // Seed actual wall cells.
        for y in 0..grid_cells {
            for x in 0..grid_cells {
                if wall_grid[y * grid_cells + x] {
                    let cell = (x as i32, y as i32);
                    self.grid.cells.insert(cell, LOG_ODDS_MAX);
                    self.seeded_walls.insert(cell);
                }
            }
        }
        // Extend seeding 3 cells beyond each grid edge.
        //
        // The endpoint formula `pose + range * sin(angle)` uses f32 arithmetic.
        // When a ray hits the arena boundary, rounding errors can place the
        // computed endpoint 1–3 cells outside the grid (e.g. cell y = -1).
        // Without seeding those cells, they accumulate uncontrolled HIT/MISS
        // updates and appear as false obstacle walls and frontier clusters
        // outside the physical arena boundary.
        const MARGIN: i32 = 3;
        let g = grid_cells as i32;
        for t in 1..=MARGIN {
            for i in -MARGIN..=(g - 1 + MARGIN) {
                for &cell in &[
                    (i, -t),           // below bottom wall
                    (i, g - 1 + t),    // above top wall
                    (-t, i),           // left of left wall
                    (g - 1 + t, i),    // right of right wall
                ] {
                    self.grid.cells.insert(cell, LOG_ODDS_MAX);
                    self.seeded_walls.insert(cell);
                }
            }
        }
    }

    /// Return a GridDelta containing every seeded wall cell at LOG_ODDS_MAX.
    ///
    /// Call this once after `seed_walls()` and publish the result to the UI bus
    /// so obstacle positions appear immediately without waiting for lidar scans
    /// to touch each cell via a MISS-then-restore cycle.
    ///
    /// Only the interior arena cells (not the 3-cell boundary margin) are
    /// included, since the margin falls outside the 200×200 UI canvas and the
    /// arena_cells bound-check in detect_frontiers already discards them.
    pub fn initial_delta(&self) -> GridDelta {
        let cells: Vec<(i32, i32, f32)> = self.seeded_walls
            .iter()
            .filter(|&&(cx, cy)| {
                // Exclude out-of-bounds margin cells — the UI canvas is 0..arena_cells.
                let g = self.arena_cells.unwrap_or(i32::MAX);
                cx >= 0 && cy >= 0 && cx < g && cy < g
            })
            .map(|&(cx, cy)| (cx, cy, LOG_ODDS_MAX))
            .collect();
        GridDelta { t_ms: 0, cells }
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
                    // In sim mode (seeded_walls non-empty) only stamp HIT on known
                    // obstacle cells.  Float arithmetic places computed endpoints
                    // 1-4 cells from the true wall, creating phantom obstacle
                    // markers in free space that block A* navigation.  All real
                    // obstacles are pre-seeded; anything else is free space.
                    let apply_hit = self.seeded_walls.is_empty()
                        || self.seeded_walls.contains(&(cx, cy));
                    if apply_hit {
                        self.grid.apply(cx, cy, LOG_ODDS_HIT);
                        touched.insert((cx, cy), ());
                    }
                } else if !hit_endpoints.contains(&(cx, cy)) {
                    // Only apply MISS to cells that are not a HIT endpoint for
                    // another ray — prevents adjacent Bresenham traces from
                    // immediately overwriting obstacle cells with MISS.
                    self.grid.apply(cx, cy, LOG_ODDS_MISS);
                    touched.insert((cx, cy), ());
                }
            }
        }

        // Restore seeded wall cells that were eroded by MISS sweeps this tick.
        // Only cells actually changed (eroded below LOG_ODDS_MAX) are added to
        // the delta so the UI copy stays consistent.
        for &cell in &self.seeded_walls {
            let v = self.grid.cells.entry(cell).or_insert(LOG_ODDS_MAX);
            if *v < LOG_ODDS_MAX {
                *v = LOG_ODDS_MAX;
                touched.insert(cell, ());
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
            // Discard cells outside the known arena bounds (sim-only guard).
            // Stray Bresenham endpoints from oblique rays can land past the
            // seeded margin, creating free cells outside the physical arena.
            if let Some(g) = self.arena_cells {
                if cx < 0 || cy < 0 || cx >= g || cy >= g {
                    continue;
                }
            }
            if lo < FREE_THRESH {
                free_count += 1;
                // Frontier if any 4-neighbor is unknown AND the cell has at least one
                // free 4-neighbor.  The second condition rejects isolated free islands
                // caused by stray lidar rays clipping past walls (floating-point endpoints
                // map to the cell just outside the arena).  Real frontier cells are always
                // adjacent to the explored free area behind them.
                let has_unknown_nb = neighbors4(cx, cy).iter()
                    .any(|&(nx, ny)| self.grid.is_unknown(nx, ny));
                if !has_unknown_nb { continue; }
                let has_free_nb = neighbors4(cx, cy).iter()
                    .any(|&(nx, ny)| self.grid.is_free(nx, ny));
                if has_free_nb {
                    frontier_cells.insert((cx, cy));
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
