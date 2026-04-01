//! Phase 14.2.5: Validate calibration by comparing sim (old vs new kinematics) and robot performance
//!
//! Test scenario: Empty room, 2m corridor width
//! Motion: Drive forward 2m → Rotate 360° → measure final pose error
//!
//! Expected results:
//! - Old kinematics (0.30 m/s, 1.0 rad/s): ~61% forward error, 1230% rotation error
//! - New kinematics (1.61 m/s, 14.3 rad/s): much closer to actual robot motion

use sim_fast::{FastSim, RoomKind, Action};

const OLD_SPEED_M_S: f32 = 0.30;
const OLD_OMEGA_RAD_S: f32 = 1.0;

const NEW_SPEED_M_S: f32 = 0.858;
const NEW_OMEGA_RAD_S: f32 = 7.30;

const FORWARD_DISTANCE_M: f32 = 2.0;
const ROTATION_DEGREES: f32 = 360.0;

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("Phase 14.2.5: Calibration Validation (Sim Old vs New)");
    println!("═══════════════════════════════════════════════════════════\n");

    println!("Test scenario: Empty room, 2m wide");
    println!("  1. Drive forward {}m", FORWARD_DISTANCE_M);
    println!("  2. Rotate {}°", ROTATION_DEGREES);
    println!("  3. Measure final pose error\n");

    // Test with OLD constants
    println!("─────────────────────────────────────────────────────────── ");
    println!("OLD Constants: {} m/s fwd, {} rad/s rotation", OLD_SPEED_M_S, OLD_OMEGA_RAD_S);
    println!("─────────────────────────────────────────────────────────── ");
    let (old_fwd_err, old_rot_err) = run_test(OLD_SPEED_M_S, OLD_OMEGA_RAD_S);

    println!();

    // Test with NEW constants
    println!("─────────────────────────────────────────────────────────── ");
    println!("NEW Constants: {} m/s fwd, {} rad/s rotation (calibrated)", NEW_SPEED_M_S, NEW_OMEGA_RAD_S);
    println!("─────────────────────────────────────────────────────────── ");
    let (new_fwd_err, new_rot_err) = run_test(NEW_SPEED_M_S, NEW_OMEGA_RAD_S);

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("COMPARISON");
    println!("═══════════════════════════════════════════════════════════");
    println!("Forward motion:");
    println!("  Old: {:.2}% error (ideal 0%)", old_fwd_err * 100.0);
    println!("  New: {:.2}% error (ideal 0%)", new_fwd_err * 100.0);
    println!("  Improvement: {:.1}%", (old_fwd_err - new_fwd_err) * 100.0 / old_fwd_err);

    println!("\nRotation:");
    println!("  Old: {:.2}% error (ideal 0%)", old_rot_err * 100.0);
    println!("  New: {:.2}% error (ideal 0%)", new_rot_err * 100.0);
    println!("  Improvement: {:.1}%", (old_rot_err - new_rot_err) * 100.0 / old_rot_err);

    println!("\n  → New constants should match ROBOT behavior better");
    println!("═══════════════════════════════════════════════════════════\n");
}

fn run_test(forward_speed_m_s: f32, omega_rad_s: f32) -> (f32, f32) {
    let mut sim = FastSim::new(42, RoomKind::Empty);
    let start = sim.pose();
    println!("Start: ({:.2}, {:.2}, {:.2}°)", start.x_m, start.y_m, start.theta_rad.to_degrees());

    // Step 1: Drive forward 2m
    let steps_forward = (FORWARD_DISTANCE_M / (forward_speed_m_s * 0.1)).ceil() as u32; // 0.1s per step
    println!("\nPhase 1: Drive forward {}m ({} steps @ 10Hz)", FORWARD_DISTANCE_M, steps_forward);
    for step in 0..steps_forward {
        let obs = sim.step(Action::Forward as u8);
        if obs.collision {
            println!("  ⚠ COLLISION at step {}", step);
            break;
        }
    }

    let mid = sim.pose();
    let forward_distance = ((mid.x_m - start.x_m).powi(2) + (mid.y_m - start.y_m).powi(2)).sqrt();
    println!("  Result: ({:.2}, {:.2}, θ={:.2}°)", mid.x_m, mid.y_m, mid.theta_rad.to_degrees());
    println!("  Distance traveled: {:.2}m (expected {}m)", forward_distance, FORWARD_DISTANCE_M);
    let fwd_error = (forward_distance - FORWARD_DISTANCE_M).abs() / FORWARD_DISTANCE_M;
    println!("  Error: {:.2}%", fwd_error * 100.0);

    // Step 2: Rotate 360°
    let steps_rotation = (ROTATION_DEGREES.to_radians() / (omega_rad_s * 0.1)).ceil() as u32;
    println!("\nPhase 2: Rotate {}° ({} steps @ 10Hz)", ROTATION_DEGREES, steps_rotation);
    for step in 0..steps_rotation {
        let obs = sim.step(Action::RotateLeft as u8);
        if obs.collision {
            println!("  ⚠ COLLISION at step {}", step);
            break;
        }
    }

    let end = sim.pose();
    println!("  Result: ({:.2}, {:.2}, θ={:.2}°)", end.x_m, end.y_m, end.theta_rad.to_degrees());

    // Compute errors
    let final_x_error = (end.x_m - start.x_m).abs();
    let final_y_error = (end.y_m - start.y_m).abs();
    let final_heading = normalize_angle(end.theta_rad - start.theta_rad);
    let rot_error = final_heading.abs() / ROTATION_DEGREES.to_radians();

    println!("\nFinal pose:");
    println!("  Position error: ({:.3}m, {:.3}m)", final_x_error, final_y_error);
    println!("  Heading error: {:.2}° ({:.2}% of 360°)", final_heading.to_degrees(), rot_error * 100.0);

    (fwd_error, rot_error)
}

fn normalize_angle(mut angle: f32) -> f32 {
    let pi = std::f32::consts::PI;
    while angle > pi {
        angle -= 2.0 * pi;
    }
    while angle < -pi {
        angle += 2.0 * pi;
    }
    angle
}
