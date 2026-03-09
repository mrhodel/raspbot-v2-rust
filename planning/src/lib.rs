//! A* path planner over the occupancy grid.
//!
//! `AStarPlanner::plan()` searches from the robot's current cell to a
//! goal position, respecting a configurable clearance margin around obstacles.
//!
//! # Clearance
//! Any cell within `CLEARANCE_CELLS` of a confirmed obstacle is treated as
//! impassable. This keeps the robot body away from walls.
//!
//! # Path output
//! The raw A* path is downsampled — every `WAYPOINT_STEP`-th cell is kept,
//! plus the goal — to produce a compact `Path` of world-frame waypoints.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use core_types::{Path, Pose2D};
use mapping::{Mapper, OCC_THRESH};
use tracing::{debug, warn};

/// Obstacle clearance radius in grid cells.
const CLEARANCE_CELLS: i32 = 2;

/// Keep one waypoint per this many cells along the raw path.
const WAYPOINT_STEP: usize = 10;

/// Maximum A* nodes to expand before giving up.
const MAX_NODES: usize = 50_000;

// ── AStarPlanner ─────────────────────────────────────────────────────────────

pub struct AStarPlanner;

impl AStarPlanner {
    pub fn new() -> Self {
        Self
    }

    /// Find a collision-free path from `start` pose to `goal_m` (world coords).
    ///
    /// Returns `None` if the goal is unreachable within `MAX_NODES` expansions,
    /// or if the start / goal cell is itself inside an inflated obstacle.
    pub fn plan(&self, mapper: &Mapper, start: &Pose2D, goal_m: [f32; 2]) -> Option<Path> {
        let (sx, sy) = mapper.world_to_cell(start.x_m, start.y_m);
        let (gx, gy) = mapper.world_to_cell(goal_m[0], goal_m[1]);

        if !self.passable(mapper, sx, sy) {
            warn!("A*: start cell ({sx},{sy}) is inside an obstacle — cannot plan");
            return None;
        }
        if !self.passable(mapper, gx, gy) {
            warn!("A*: goal cell ({gx},{gy}) is inside an obstacle — cannot plan");
            return None;
        }

        let mut open: BinaryHeap<Node> = BinaryHeap::new();
        let mut g_cost: HashMap<(i32, i32), f32> = HashMap::new();
        let mut came_from: HashMap<(i32, i32), (i32, i32)> = HashMap::new();

        g_cost.insert((sx, sy), 0.0);
        open.push(Node::new((sx, sy), 0.0, heuristic(sx, sy, gx, gy)));

        let mut expanded = 0usize;
        while let Some(node) = open.pop() {
            let (cx, cy) = node.pos;

            if cx == gx && cy == gy {
                debug!(expanded, "A*: path found");
                return Some(self.reconstruct(mapper, &came_from, (sx, sy), (gx, gy), start.t_ms));
            }

            expanded += 1;
            if expanded > MAX_NODES {
                warn!("A*: node limit reached ({MAX_NODES}) — no path found");
                return None;
            }

            let g = g_cost[&(cx, cy)];

            for (nx, ny) in neighbors8(cx, cy) {
                if !self.passable(mapper, nx, ny) {
                    continue;
                }
                // Diagonal moves cost √2, cardinal moves cost 1.
                let step_cost = if (nx - cx).abs() + (ny - cy).abs() == 2 {
                    std::f32::consts::SQRT_2
                } else {
                    1.0
                };
                let ng = g + step_cost;
                if ng < *g_cost.get(&(nx, ny)).unwrap_or(&f32::INFINITY) {
                    g_cost.insert((nx, ny), ng);
                    came_from.insert((nx, ny), (cx, cy));
                    open.push(Node::new((nx, ny), ng, heuristic(nx, ny, gx, gy)));
                }
            }
        }

        warn!("A*: open set exhausted — no path found");
        None
    }

    /// True if `(cx,cy)` and all cells within `CLEARANCE_CELLS` are passable.
    fn passable(&self, mapper: &Mapper, cx: i32, cy: i32) -> bool {
        for dx in -CLEARANCE_CELLS..=CLEARANCE_CELLS {
            for dy in -CLEARANCE_CELLS..=CLEARANCE_CELLS {
                if mapper.grid.get(cx + dx, cy + dy) >= OCC_THRESH {
                    return false;
                }
            }
        }
        true
    }

    fn reconstruct(
        &self,
        mapper: &Mapper,
        came_from: &HashMap<(i32, i32), (i32, i32)>,
        start: (i32, i32),
        goal: (i32, i32),
        t_ms: u64,
    ) -> Path {
        let mut cells = vec![goal];
        let mut cur = goal;
        while cur != start {
            cur = came_from[&cur];
            cells.push(cur);
        }
        cells.reverse();

        let waypoints: Vec<[f32; 2]> = cells
            .iter()
            .enumerate()
            .filter(|(i, _)| i % WAYPOINT_STEP == 0 || *i == cells.len() - 1)
            .map(|(_, &(cx, cy))| {
                let (wx, wy) = mapper.cell_to_world(cx, cy);
                [wx, wy]
            })
            .collect();

        Path { t_ms, waypoints }
    }
}

impl Default for AStarPlanner {
    fn default() -> Self {
        Self::new()
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Octile distance heuristic (consistent, admissible for 8-connected grids).
fn heuristic(x0: i32, y0: i32, x1: i32, y1: i32) -> f32 {
    let dx = (x1 - x0).abs() as f32;
    let dy = (y1 - y0).abs() as f32;
    dx + dy - (2.0 - std::f32::consts::SQRT_2) * dx.min(dy)
}

fn neighbors8(x: i32, y: i32) -> [(i32, i32); 8] {
    [
        (x-1,y), (x+1,y), (x,y-1), (x,y+1),
        (x-1,y-1), (x-1,y+1), (x+1,y-1), (x+1,y+1),
    ]
}

// ── Priority queue node ───────────────────────────────────────────────────────

struct Node {
    f: f32,     // g + h
    pos: (i32, i32),
}

impl Node {
    fn new(pos: (i32, i32), g: f32, h: f32) -> Self {
        Self { f: g + h, pos }
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.f == other.f
    }
}
impl Eq for Node {}

impl Ord for Node {
    // Min-heap: lower f wins.
    fn cmp(&self, other: &Self) -> Ordering {
        other.f.partial_cmp(&self.f).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
