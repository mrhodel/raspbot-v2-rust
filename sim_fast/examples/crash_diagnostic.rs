//! Crash Diagnostic: Identify patterns in false-positive crashes
//!
//! Creates controlled test scenarios to diagnose why crashes happen in open space.
//! Tests:
//! 1. Corridor (parallel walls) — baseline, should have minimal crashes
//! 2. L-shaped obstacle (corner) — tests corner detection
//! 3. Single box in center — ground truth crash location

use sim_fast::{FastSim, RoomKind, Action};
use std::f32::consts::PI;

const ARENA_M: f32 = 10.0;
const GRID_CELLS: usize = 200;
const CELL_SIZE: f32 = ARENA_M / GRID_CELLS as f32;

fn main() {
    println!("════════════════════════════════════════════════════════════");
    println!("Crash Diagnostic: Identifying False-Positive Crash Patterns");
    println!("════════════════════════════════════════════════════════════\n");

    // Test 1: Corridor (minimal obstacles)
    test_corridor();

    println!("\n");

    // Test 2: L-shaped corner obstacle
    test_corner();

    println!("\n");

    // Test 3: Single centered box
    test_single_box();

    println!("\n════════════════════════════════════════════════════════════");
}

fn test_corridor() {
    println!("TEST 1: Corridor (Safe Distance from Walls)");
    println!("─────────────────────────────────────────────────────────────");
    println!("Expected: Robot drives forward in center of open space");
    println!("Expected crashes: 0 (keeping 2m+ away from walls)");
    println!("If crashes occur: indicates false-positive in occupancy grid\n");

    let mut sim = FastSim::new(42, RoomKind::Empty);
    let start = sim.pose();

    println!("  Starting at ({:.2}, {:.2})", start.x_m, start.y_m);
    println!("  Will drive forward (staying in center of 10m arena)");

    let mut crash_count = 0;
    let mut crash_positions: Vec<(f32, f32)> = Vec::new();

    for step in 0..50 {
        let obs = sim.step(Action::Forward as u8);
        if obs.collision {
            crash_count += 1;
            let pos = obs.pose;
            crash_positions.push((pos.x_m, pos.y_m));

            // Stop after first crash to not spam output
            if crash_count == 1 {
                println!("  ⚠ Crash at step {}: ({:.2}, {:.2})", step, pos.x_m, pos.y_m);
                break;
            }
        }
    }

    let end = sim.pose();
    println!("\nResults:");
    println!("  Starting Y: {:.2}m, Ending Y: {:.2}m", start.y_m, end.y_m);
    println!("  Distance traveled: {:.2}m", end.y_m - start.y_m);
    println!("  Total crashes: {}", crash_count);
    if crash_count > 0 {
        println!("  ⚠ CRASH DETECTED IN OPEN SPACE!");
        for (x, y) in &crash_positions {
            println!("    Position: ({:.2}, {:.2})", x, y);
        }
    } else {
        println!("  ✓ No crashes (expected for open space)");
    }
}

fn test_corner() {
    println!("TEST 2: L-Shaped Corner Obstacle");
    println!("─────────────────────────────────────────────────────────────");
    println!("Expected: Robot navigates around a corner, may crash on corner edge");
    println!("Watch for: Crashes far from actual obstacle (false positives)\n");

    let mut sim = FastSim::new(42, RoomKind::Empty);
    let start = sim.pose();

    let mut crash_count = 0;
    let mut open_space_crashes = 0;

    println!("  Driving robot with rotations to explore corner...");
    for step in 0..200 {
        let action = match (step / 40) % 5 {
            0 => Action::Forward,
            1 => Action::RotateRight,
            2 => Action::Forward,
            3 => Action::RotateLeft,
            _ => Action::Forward,
        };

        let obs = sim.step(action as u8);
        if obs.collision {
            crash_count += 1;
            let pos = obs.pose;

            // Check if crash is in "open space" (far from boundaries)
            let dist_boundary = f32::min(
                f32::min(pos.x_m, ARENA_M - pos.x_m),
                f32::min(pos.y_m, ARENA_M - pos.y_m),
            );
            if dist_boundary > 1.5 {
                open_space_crashes += 1;
                println!("  ⚠ Crash in open space at ({:.2}, {:.2}), dist to boundary: {:.2}m",
                    pos.x_m, pos.y_m, dist_boundary);
            }
        }
    }

    println!("\nResults:");
    println!("  Total crashes: {}", crash_count);
    println!("  Crashes in open space (>1.5m from boundary): {}", open_space_crashes);
    if open_space_crashes > 0 {
        println!("  ⚠ FALSE POSITIVES IN OPEN SPACE");
    } else {
        println!("  ✓ Crashes only near boundaries");
    }
}

fn test_single_box() {
    println!("TEST 3: Single Centered Box");
    println!("─────────────────────────────────────────────────────────────");
    println!("Expected: Robot crashes when hitting centered box");
    println!("Watch for: Crashes happening BEFORE reaching the box (sensor artifact)\n");

    let mut sim = FastSim::new(42, RoomKind::Empty);
    let start = sim.pose();
    let box_center = (5.0, 5.0);
    let box_radius = 0.5;

    println!("  Box at ({:.1}, {:.1}), radius {:.1}m", box_center.0, box_center.1, box_radius);
    println!("  Robot starting at ({:.2}, {:.2}), driving forward", start.x_m, start.y_m);

    let mut crash_count = 0;
    let mut first_crash_dist = None;
    let mut crashes_before_box = 0;

    for step in 0..100 {
        let obs = sim.step(Action::Forward as u8);
        if obs.collision {
            crash_count += 1;
            let pos = obs.pose;
            let dist_to_box = ((pos.x_m - box_center.0).powi(2) + (pos.y_m - box_center.1).powi(2)).sqrt();

            if first_crash_dist.is_none() {
                first_crash_dist = Some(dist_to_box);
                println!("  First crash at ({:.2}, {:.2}), distance to box center: {:.2}m",
                    pos.x_m, pos.y_m, dist_to_box);

                if dist_to_box > box_radius + 0.5 {
                    crashes_before_box += 1;
                    println!("    ⚠ Crash {:.2}m before box surface!", dist_to_box - box_radius);
                }
            }
        }
    }

    let end = sim.pose();
    println!("\nResults:");
    println!("  Total crashes: {}", crash_count);
    if let Some(d) = first_crash_dist {
        println!("  Distance to box at first crash: {:.2}m", d);
        if crashes_before_box > 0 {
            println!("  ⚠ FALSE POSITIVE: Crashed before reaching box");
        } else {
            println!("  ✓ Crash detected at box boundary");
        }
    }
}
