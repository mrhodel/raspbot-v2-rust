//! Pure-pursuit control task with reactive obstacle avoidance.

use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, info, warn};

use bus::Bus;
use core_types::{CmdVel, Path, SafetyState};
use control::PurePursuitController;

/// Spawn the pure-pursuit control task.
///
/// - `obstacle_slow_m`: depth at which forward speed begins to ramp down.
///   Real=1.20 m (large slow zone compensates for 4-5 frame camera lag);
///   Sim=0.70 m (tighter — sim depth is less noisy after floor-contamination fix).
/// - `obstacle_stop_m`: depth at which forward motion stops and the path is
///   abandoned.  Real=0.60 m (raised from 0.40 to account for camera lag and
///   MiDaS noise); Sim=0.05 m (floor noise reaches ~0.10 m so 0.05 avoids livelock).
/// - `episode_rx`: watch channel that fires on episode reset (sim only).
///   For real mode pass a watch receiver whose sender is immediately dropped —
///   `Ok(()) = changed()` in the select! arm never matches on a closed watch,
///   so the arm is effectively disabled.
///
/// Collision recovery: safety task handles US emergency stops (reverse only,
/// no rotation).  On a sim physics collision the control task backs up 500ms,
/// then holds obstacle_stopped until MiDaS hysteresis confirms clearance (nearest
/// > 2×stop_m for 0.8 s).  No spin — the planner blacklists the crash goal so
/// the new path avoids the crash area.
pub fn spawn_control_task(
    bus: Arc<Bus>,
    ctrl_rx: tokio::sync::mpsc::Receiver<Path>,
    obstacle_slow_m: f32,
    obstacle_stop_m: f32,
    mut episode_rx: tokio::sync::watch::Receiver<u32>,
) -> tokio::task::JoinHandle<()> {
    const CONTROL_HZ: f64 = 10.0;
    const GOAL_TOLERANCE_M: f32 = 0.25;

    let bus_ctrl           = Arc::clone(&bus);
    let mut ctrl_rx        = ctrl_rx;
    let mut rx_pose_c      = bus.slam_pose2d.subscribe();
    let rx_nearest_c       = bus.nearest_obstacle_m.subscribe();
    let rx_nearest_angle_c = bus.nearest_obstacle_angle_rad.subscribe();
    let mut rx_safety_ctrl = bus.safety_state.subscribe();
    let mut rx_us_ctrl     = bus.ultrasonic.subscribe();
    let mut rx_collision_c = bus.collision_count.subscribe();
    tokio::spawn(async move {
        // US-based speed scaling — independent of MiDaS, always forward-facing.
        // Slow zone: 70 cm → 30 cm linearly; below 30 cm: vx=0.
        // us_scale also applied to omega to prevent sweeping into obstacles while rotating.
        // The US emergency stop still fires at 15 cm (safety task), this is the soft pre-stop.
        const US_SLOW_M: f32 = 0.70;
        const US_STOP_M: f32 = 0.30;
        let mut latest_us_m: f32 = f32::MAX;
        info!("Control task started (pure-pursuit {CONTROL_HZ} Hz, lookahead 0.3 m)");
        let controller = PurePursuitController::with_lookahead(0.3);
        let mut current_path: Option<Path> = None;
        let mut tick = tokio::time::interval(Duration::from_secs_f64(1.0 / CONTROL_HZ));
        let mut obstacle_stopped = false;
        let mut obstacle_stopped_by_us = false; // true when US (not camera) triggered the stop
        let mut is_safety_estop = false;  // mirrors safety task state to avoid backup conflict
        let mut was_safety_estop = false; // tracks previous safety state for edge-detection
        let mut backup_until: Option<std::time::Instant> = None;
        // Rotation and speed during backup: set when backup starts.
        // Obstacle on left (angle > 0) → rotate right (omega < 0) to escape.
        // In proximity zone (!safe_to_spin, nearest < stop_m + 0.15 m) use half speed
        // to limit blind reverse distance — crashes 10/11 were caused by backing 0.24 m
        // into a rear wall that the forward-only US couldn't detect.
        let mut backup_escape_omega: f32 = 0.0;
        let mut backup_escape_vx: f32 = -0.30;
        // Camera-obstacle deadlock detection: timer that fires after the robot has
        // been in obstacle_stopped state for 3 s continuously without recovering.
        // When hysteresis never clears (nearest < 2×stop_m because robot can't move),
        // this triggers a 1200 ms backup to create clearance and break the latch.
        let mut cam_obstacle_stopped_since: Option<std::time::Instant> = None;
        // Hysteresis: only clear obstacle_stopped after the nearest obstacle has
        // been above the clear threshold (2× stop distance) for a minimum wall-
        // clock duration.  Using time rather than tick count avoids counting the
        // same stale borrow() value multiple times when perception updates slower
        // than the control tick rate (~3 Hz MiDaS vs 10 Hz control).
        let clear_hysteresis_m = obstacle_stop_m * 2.0;
        const CLEAR_HOLD_S: f32 = 0.8; // must stay clear for 0.8 s before resuming (~2-3 MiDaS frames)
        let mut clear_since: Option<std::time::Instant> = None;
        let mut last_collision_count: u32 = 0;
        // Dropout filter: MiDaS occasionally misses a close obstacle for one frame, causing
        // nearest_m to jump from e.g. 0.23 m → 0.70 m in a single tick, which allows the
        // speed to spike to near-full before the next frame re-detects the wall.
        // Fix: cap upward increases in nearest to MAX_NEAREST_DELTA_PER_TICK (≈ 5× robot speed).
        // Obstacles can appear instantly (safe direction); "clear sky" must be confirmed over
        // ~3 ticks (300 ms) before the robot fully accelerates past the previous closest reading.
        let mut smoothed_nearest: f32 = f32::MAX;
        const MAX_NEAREST_DELTA_PER_TICK: f32 = 0.15; // m per 100 ms tick
        // Near-zone latch: once nearest drops below clear_hysteresis_m, stay latched until
        // nearest has been ABOVE clear_hysteresis_m continuously for CLEAR_HOLD_S (same
        // threshold used by obstacle_stopped).  Distance-only exit was insufficient: at a
        // corner the side wall exits the FOV and smoothed_nearest rises quickly from 0.22 m
        // to 0.67 m over 3 ticks (each at the 0.15 m/tick rate limit), base_scale climbs
        // from 0.40 to 0.95, and the robot accelerates to 27 cm/s while the corner's front
        // wall stays invisible to MiDaS — crash.  Time-based exit (0.8 s above threshold)
        // keeps the latch active for those 3 ticks, capping speed at ~8 cm/s.
        let mut in_near_zone: bool = false;
        let mut near_zone_clear_since: Option<std::time::Instant> = None;
        loop {
            tokio::select! {
                biased;
                // Episode reset (sim) — clear path and latch flag.
                // For real mode the sender is dropped so changed() always returns
                // Err; the Ok(()) pattern never matches and the arm is skipped.
                Ok(()) = episode_rx.changed() => {
                    episode_rx.borrow_and_update();
                    current_path = None;
                    obstacle_stopped = false;
                    obstacle_stopped_by_us = false;
                    clear_since = None;
                    backup_until = None;
                    cam_obstacle_stopped_since = None;
                    smoothed_nearest = f32::MAX;
                    in_near_zone = false;
                    near_zone_clear_since = None;
                }
                // Sim physics collision — clear path and back up if the safety task
                // is not already handling it (safety escape fires only for US < 25 cm;
                // side/rear collisions leave US reading ~3 m so escape never fires).
                Ok(()) = rx_collision_c.changed() => {
                    let current_collision_count = *rx_collision_c.borrow_and_update();
                    if current_collision_count > last_collision_count {
                        last_collision_count = current_collision_count;
                        // Only start a new recovery if not already recovering.
                        // This prevents cascade backups when physics generates multiple
                        // collision events during a single impact (1-2s apart).
                        if !obstacle_stopped {
                            let pose = *rx_pose_c.borrow_and_update();
                            warn!(
                                collision_n = current_collision_count,
                                pose_x = pose.x_m,
                                pose_y = pose.y_m,
                                estop = is_safety_estop,
                                "Control: collision — clearing path"
                            );
                            current_path = None;
                            obstacle_stopped = true;
                            clear_since = None;
                            // Only start our own backup when the safety task is NOT in
                            // emergency-stop (i.e. not already reversing for us).
                            if !is_safety_estop {
                                let now = std::time::Instant::now();
                                // 500ms backup at -0.30 m/s ≈ 15 cm clearance.
                                // No spin — planner blacklists the crash goal so new path avoids the area.
                                backup_until = Some(now + Duration::from_millis(500));
                            }
                        }
                    }
                }
                // US range update — track latest for speed scaling in tick arm.
                Ok(reading) = rx_us_ctrl.recv() => {
                    latest_us_m = reading.range_cm / 100.0;
                }
                // Emergency stop from US safety task — clear path on rising edge.
                // NOTE: the safety task calls safety_state.send() on every US reading
                // (~15 Hz), and tokio watch channels trigger changed() on every send,
                // even when the value is unchanged.  To avoid clearing obstacle_stopped
                // (and the deadlock timer) 15×/second on continuous Safe readings, we
                // edge-detect: only act on Ok→Estop or Estop→Ok transitions.
                Ok(()) = rx_safety_ctrl.changed() => {
                    let is_estop = matches!(
                        *rx_safety_ctrl.borrow_and_update(),
                        SafetyState::EmergencyStop { .. }
                    );
                    let prev_estop = was_safety_estop;
                    was_safety_estop = is_estop;
                    is_safety_estop = is_estop;
                    if is_estop && !prev_estop {
                        // Ok → Estop transition: stop immediately.
                        if !obstacle_stopped {
                            warn!("Control: EmergencyStop — clearing path for replan after latch");
                            current_path = None;
                            obstacle_stopped = true;
                            clear_since = None;
                            backup_until = None;  // safety task has control, cancel our backup
                        }
                    } else if !is_estop && prev_estop {
                        // Estop → Ok transition: release so the planner can assign a
                        // new path.  Keep cam_obstacle_stopped_since alive — the
                        // deadlock timer must keep accumulating across brief estop
                        // releases so the backup fires if the robot is truly cornered.
                        obstacle_stopped = false;
                        obstacle_stopped_by_us = false;
                        clear_since = None;
                    }
                    // Repeated Ok→Ok or Estop→Estop: do nothing — avoids resetting
                    // obstacle_stopped and deadlock timer on every US sensor poll.
                }
                // New path from planner — replace immediately (unless in emergency stop).
                result = ctrl_rx.recv() => {
                    match result {
                        Some(p) => {
                            if obstacle_stopped {
                                debug!(waypoints = p.waypoints.len(), "Control: dropping path — EmergencyStop active");
                            } else {
                                info!(waypoints = p.waypoints.len(), "Control: new path, tracking");
                                current_path = Some(p);
                            }
                        }
                        None => break,
                    }
                }
                // Control tick — re-compute CmdVel from current pose.
                _ = tick.tick() => {
                    // Auto-clear depth stop with time hysteresis — nearest must stay above
                    // clear_hysteresis_m (2× stop distance) continuously for CLEAR_HOLD_S
                    // before resuming.  Any dip below the threshold resets the timer, preventing
                    // noisy MiDaS readings from causing a false clear.  Using wall-clock time
                    // avoids counting the same stale borrow() value across multiple ticks.
                    // Short backup after side/rear collision (safety escape not firing).
                    // Keep obstacle_stopped=true after backup — hysteresis clears it once
                    // the robot has physically separated from the wall (nearest > 2×stop_m
                    // for 0.8 s).  This lets MiDaS map the obstacle before re-planning,
                    // preventing the planner from routing straight back into the same wall.
                    // Backup 500ms at -0.30 m/s (≈15 cm clearance) after collision.
                    // obstacle_stopped remains true after backup; cleared by hysteresis once
                    // nearest > 2×stop_m for 0.8 s.  Planner blacklists crash goal.
                    if let Some(until) = backup_until {
                        if std::time::Instant::now() < until {
                            let _ = bus_ctrl.controller_cmd_vel.try_send(
                                CmdVel { t_ms: 0, vx: backup_escape_vx, vy: 0.0, omega: backup_escape_omega }
                            );
                            continue;
                        } else {
                            backup_until = None;
                            clear_since = None;
                            cam_obstacle_stopped_since = None;
                            // After a forced backup the robot physically moved — clear
                            // obstacle_stopped immediately rather than waiting for
                            // nearest > clear_hysteresis_m.  Near any wall the lidar
                            // nearest always reads < 0.60 m, so hysteresis never fires.
                            obstacle_stopped = false;
                            info!("Control: backup complete — resuming (nearest checked by planner)");
                        }
                    }

                    let raw_nearest = *rx_nearest_c.borrow();
                    // Rate-limit upward jumps (obstacle disappearing) to filter single-frame
                    // MiDaS dropouts.  Downward jumps (obstacle appearing) are always immediate.
                    let nearest = if smoothed_nearest == f32::MAX || raw_nearest <= smoothed_nearest {
                        raw_nearest
                    } else {
                        raw_nearest.min(smoothed_nearest + MAX_NEAREST_DELTA_PER_TICK)
                    };
                    smoothed_nearest = nearest;
                    let nearest_angle = *rx_nearest_angle_c.borrow();
                    if obstacle_stopped {
                        // Deadlock detection: fires regardless of US state (including when
                        // us_stopped keeps reversing but can't clear — e.g. cornered with
                        // forward wall and the straight reverse doesn't help).  Must be
                        // checked before the US-reverse early-return so it isn't skipped.
                        if backup_until.is_none() && !is_safety_estop {
                            let stuck_since = cam_obstacle_stopped_since
                                .get_or_insert_with(std::time::Instant::now);
                            if stuck_since.elapsed() >= Duration::from_secs(3) {
                                // Rotate away from the nearest obstacle during backup.
                                // Obstacle on left (angle > 0) → spin right (omega < 0).
                                // Only spin when obstacle is clearly to one side (> 20°).
                                // Near-center obstacles (±20°): back up straight — spinning
                                // sweeps the robot body into the same wall (crashes 6, 7).
                                //
                                // Proximity guard: only spin when there is enough clearance
                                // for the body to rotate without clipping the obstacle.
                                // At nearest < stop_m + 0.15 m (e.g. < 0.30 m in sim) the
                                // shoulder has < 15 cm to spare — spinning sweeps the robot
                                // body into the wall even when rotating away from it (crash 9).
                                const SPIN_THRESHOLD_RAD: f32 = 0.35; // ~20°
                                const SAFE_SPIN_CLEARANCE_M: f32 = 0.15;
                                let safe_to_spin = nearest >= obstacle_stop_m + SAFE_SPIN_CLEARANCE_M;
                                backup_escape_omega = if !safe_to_spin {
                                    0.0 // too close — straight backup only
                                } else if nearest_angle > SPIN_THRESHOLD_RAD {
                                    -1.0 // obstacle left → spin right
                                } else if nearest_angle < -SPIN_THRESHOLD_RAD {
                                    1.0  // obstacle right → spin left
                                } else {
                                    0.0  // obstacle ≤ 20° off-center: straight backup
                                };
                                // Proximity zone: use half speed to limit blind reverse
                                // distance.  !safe_to_spin means nearest < stop_m+0.15 m —
                                // the robot body has < 15 cm shoulder clearance and the
                                // forward-only US can't see rear obstacles.  At -0.30 m/s
                                // the robot travels 0.36 m in 1.2 s; at -0.15 m/s it
                                // travels only 0.18 m, staying clear of corner rear walls.
                                backup_escape_vx = if safe_to_spin { -0.30 } else { -0.15 };
                                warn!(
                                    nearest_m = nearest,
                                    nearest_angle_deg = nearest_angle.to_degrees(),
                                    backup_escape_omega,
                                    us_m = latest_us_m,
                                    "Control: obstacle stuck for 3 s — backing up with rotation"
                                );
                                backup_until = Some(std::time::Instant::now()
                                    + Duration::from_millis(1200));
                                cam_obstacle_stopped_since = None;
                                clear_since = None;
                            }
                        }

                        // If US is still in the stop zone and not yet running a timed backup,
                        // actively reverse to create distance.
                        if latest_us_m < US_STOP_M && backup_until.is_none() {
                            let _ = bus_ctrl.controller_cmd_vel.try_send(CmdVel { t_ms: 0, vx: -0.15, vy: 0.0, omega: 0.0 });
                            clear_since = None;
                            continue;
                        }
                        // If this was a US-triggered stop and the US has now recovered past
                        // US_SLOW_M (comfortably clear), release immediately.  Do NOT wait
                        // for the camera/lidar nearest hysteresis — the lidar sees ALL walls
                        // in all directions, so near any wall nearest < 0.60 m permanently.
                        if obstacle_stopped_by_us && latest_us_m >= US_SLOW_M {
                            info!(us_m = latest_us_m, "Control: US-stop cleared (US recovered)");
                            obstacle_stopped = false;
                            obstacle_stopped_by_us = false;
                            clear_since = None;
                            // Keep cam_obstacle_stopped_since alive — if the robot immediately
                            // drives back into the same wall the deadlock timer fires quickly.
                        }

                        if nearest > clear_hysteresis_m {
                            let since = clear_since.get_or_insert_with(std::time::Instant::now);
                            if since.elapsed().as_secs_f32() >= CLEAR_HOLD_S {
                                info!(clear_hysteresis_m, CLEAR_HOLD_S, "Control: obstacle cleared (hysteresis satisfied)");
                                obstacle_stopped = false;
                                obstacle_stopped_by_us = false;
                                clear_since = None;
                                // Only reset the deadlock timer if the obstacle is genuinely clear
                                // (nearest > obstacle_slow_m). If nearest is in the slow zone
                                // [clear_hysteresis_m, obstacle_slow_m], a new path with `toward`
                                // will immediately re-trigger obstacle_stopped. Keeping the timer
                                // alive across these brief clear/re-stop cycles lets the 3 s backup
                                // fire even when no single stopped episode lasts 3 s continuously.
                                if nearest > obstacle_slow_m {
                                    cam_obstacle_stopped_since = None;
                                }
                            }
                        } else {
                            clear_since = None; // reset timer on any dip below threshold
                        }
                    }

                    let Some(ref path) = current_path else { continue };
                    let pose = *rx_pose_c.borrow_and_update();

                    // Goal check: within tolerance of the final waypoint → done.
                    if let Some(&[gx, gy]) = path.waypoints.last() {
                        let dx = gx - pose.x_m;
                        let dy = gy - pose.y_m;
                        if dx * dx + dy * dy < GOAL_TOLERANCE_M * GOAL_TOLERANCE_M {
                            // Continue-if-clear: if forward space is open, synthesise a
                            // short coast extension in the current heading instead of
                            // stopping cold.  This fills the A* replan gap (100-500 ms)
                            // with useful forward motion.  The planner's next real path
                            // replaces the coast waypoint immediately when it arrives.
                            // Thresholds are intentionally conservative — if there is any
                            // doubt about forward clearance, stop normally.
                            const COAST_CLEAR_M:  f32 = 1.2; // min obstacle dist to coast
                            const COAST_DIST_M:   f32 = 1.5; // how far to extend
                            let fwd_clear = nearest > COAST_CLEAR_M
                                && latest_us_m > COAST_CLEAR_M * 0.5;
                            if fwd_clear {
                                let ex = pose.x_m + COAST_DIST_M * pose.theta_rad.cos();
                                let ey = pose.y_m + COAST_DIST_M * pose.theta_rad.sin();
                                current_path = Some(Path {
                                    t_ms: pose.t_ms,
                                    waypoints: vec![[ex, ey]],
                                });
                                debug!(coast_x = ex, coast_y = ey,
                                    nearest_m = nearest, us_m = latest_us_m,
                                    "Control: goal reached — coasting forward");
                            } else {
                                info!(nearest_m = nearest, us_m = latest_us_m,
                                    "Control: goal reached — stopping (obstacle ahead)");
                                current_path = None;
                                obstacle_stopped = false;
                                obstacle_stopped_by_us = false;
                            }
                            continue;
                        }
                    }

                    // ── Reactive obstacle avoidance ──────────────────────────
                    // Scale vx [0, 1] based on nearest obstacle in camera FOV.
                    // omega is unaffected — the robot can still rotate away.
                    //
                    // Compute controller output first so we can detect "turning
                    // toward obstacle" on the raw (unscaled) omega before we
                    // decide the speed_scale.
                    let mut cmd_vel = controller.compute(&pose, path);
                    // Final-approach speed cap: when close to the goal (few waypoints left)
                    // halve forward speed. Obstacle corners outside the ±55° sensor FOV
                    // cannot be detected until the robot is very close; slowing down during
                    // the last ~5 waypoints gives the sensor time to detect and stop.
                    if path.waypoints.len() <= 5 && cmd_vel.vx > 0.0 {
                        cmd_vel.vx *= 0.5;
                    }
                    let original_vx = cmd_vel.vx;  // Save for reverse restoration
                    // Is navigation intending to rotate into the detected obstacle?
                    // Use unscaled omega (navigation intent) so US-induced omega
                    // reduction does not mask the dangerous approach condition.
                    // Gate by angle: a wall at ±55° (FOV edge) with tiny omega should
                    // not stop the robot — that's a corridor wall, not a head-on threat.
                    // Only treat as "toward" when the obstacle is within ±40° of forward.
                    const TOWARD_ANGLE_LIMIT: f32 = 40.0 * std::f32::consts::PI / 180.0;
                    let toward = nearest_angle.abs() <= TOWARD_ANGLE_LIMIT
                              && ((cmd_vel.omega > 0.05 && nearest_angle > 0.05)
                               || (cmd_vel.omega < -0.05 && nearest_angle < -0.05));
                    let base_scale = if nearest >= obstacle_slow_m {
                        1.0_f32
                    } else if nearest <= obstacle_stop_m {
                        0.0_f32
                    } else {
                        (nearest - obstacle_stop_m) / (obstacle_slow_m - obstacle_stop_m)
                    };
                    let angle_factor = nearest_angle.cos().max(0.0_f32);
                    // Near-zone latch (time-based exit): enter when nearest < clear_hysteresis_m;
                    // only exit once nearest has been ABOVE clear_hysteresis_m for CLEAR_HOLD_S
                    // (0.8 s) continuously — the same hysteresis used for obstacle_stopped.
                    //
                    // Corner-crash scenario: robot at 0.22 m, side wall exits FOV as robot turns,
                    // smoothed nearest rises 0.22→0.37→0.52→0.67 m at 0.15 m/tick rate-limit
                    // over ~300 ms.  base_scale climbs from 0.40 to 0.95 and vx spikes to 27 cm/s
                    // while the front corner wall is invisible to MiDaS at oblique angles — crash.
                    // With CLEAR_HOLD_S=0.8 s, the latch stays active for those 300 ms, keeping
                    // speed capped at NEAR_ZONE_SPEED_CAP until the area is confirmed clear.
                    if nearest < clear_hysteresis_m {
                        in_near_zone = true;
                        near_zone_clear_since = None;
                    } else if in_near_zone {
                        let since = near_zone_clear_since
                            .get_or_insert_with(std::time::Instant::now);
                        if since.elapsed().as_secs_f32() >= CLEAR_HOLD_S {
                            in_near_zone = false;
                            near_zone_clear_since = None;
                        }
                    }
                    let effective_factor = if toward || in_near_zone {
                        1.0_f32
                    } else {
                        angle_factor
                    };
                    // Hard stop in the stop zone; angle weighting only in slowdown zone.
                    // While in near-zone, cap speed_scale at NEAR_ZONE_SPEED_CAP (base_scale
                    // at the near-zone entry threshold = (0.30-0.15)/(0.70-0.15) ≈ 0.27).
                    // This prevents base_scale from climbing to 0.95 while the latch is held
                    // (corner wall invisible to MiDaS — without cap vx reaches 27 cm/s).
                    let near_zone_speed_cap =
                        (clear_hysteresis_m - obstacle_stop_m) / (obstacle_slow_m - obstacle_stop_m);
                    let speed_scale = if base_scale <= 0.0 {
                        0.0
                    } else {
                        let s = 1.0 - (1.0 - base_scale) * effective_factor;
                        if in_near_zone { s.min(near_zone_speed_cap) } else { s }
                    };

                    // US-based speed scaling (forward sensor, independent of gimbal).
                    // Takes minimum with MiDaS scale — whichever is more restrictive wins.
                    let us_scale = if latest_us_m >= US_SLOW_M {
                        1.0_f32
                    } else if latest_us_m <= US_STOP_M {
                        0.0_f32
                    } else {
                        (latest_us_m - US_STOP_M) / (US_SLOW_M - US_STOP_M)
                    };
                    let combined_scale = speed_scale.min(us_scale);
                    if us_scale < 1.0 {
                        info!(us_m = latest_us_m, us_scale, combined_scale, "Control: US slowing");
                    }

                    cmd_vel.vx *= combined_scale;
                    // Avoid scaling reverse (vx < 0): US is forward-facing only, so blocking
                    // reverse when US is too close is overly conservative. Allow full reverse
                    // to escape forward obstacles. If planning commanded reverse, restore it.
                    if original_vx < 0.0 && cmd_vel.vx < original_vx {
                        cmd_vel.vx = original_vx;
                    }
                    // Also scale omega by us_scale — prevents sweeping the US sensor into an
                    // obstacle while rotating (e.g. omega=0.4 sweeps forward-facing US across
                    // a box, causing sudden 49→15 cm reading with no time to brake).
                    cmd_vel.omega *= us_scale;

                    if combined_scale < 1.0 {
                        let obs_deg     = nearest_angle.to_degrees();
                        let heading_deg = pose.theta_rad.to_degrees();
                        let turn_dir    = if cmd_vel.omega.abs() < 0.05 { "straight" }
                                          else if cmd_vel.omega > 0.0   { "turning-left" }
                                          else                           { "turning-right" };
                        let obs_side = if obs_deg > 5.0 { "left" } else if obs_deg < -5.0 { "right" } else { "center" };
                        // `toward` is already computed above from unscaled omega.
                        // Deep near-zone hard stop: when in_near_zone and already within
                        // 8 cm of the stop threshold (nearest ≤ stop_m + 0.08 m ≈ 0.23 m),
                        // stop immediately rather than creeping forward at 5–17% speed.
                        // Fixes corner-clip crashes where the robot crept at ~7% into a 55°
                        // side wall — combined_scale was > 0 so the block below was skipped.
                        const DEEP_STOP_MARGIN_M: f32 = 0.08;
                        let deep_near_stop = in_near_zone && nearest < obstacle_stop_m + DEEP_STOP_MARGIN_M;

                        if combined_scale <= 0.0 || deep_near_stop {
                            // Hard stop: fire whenever the robot is at or near the stop
                            // threshold, regardless of obstacle angle.
                            //
                            // Previously the "send omega only" path was taken for side
                            // obstacles (obs_deg ±55°) at stop distance, leaving the robot
                            // frozen with obstacle_stopped=false and the deadlock timer
                            // unarmed — causing timeouts.  Always STOP now.
                            let us_stopped     = us_scale <= 0.0;
                            let in_forward_arc = nearest_angle.abs() <= 30f32.to_radians();
                            if !obstacle_stopped {
                                warn!(
                                    nearest_m   = nearest,
                                    us_m        = latest_us_m,
                                    us_stopped,
                                    deep_near_stop,
                                    obs_deg,
                                    obs_side,
                                    heading_deg,
                                    omega       = cmd_vel.omega,
                                    turn_dir,
                                    turning_toward_obstacle = toward,
                                    "Control: obstacle STOP — replan"
                                );
                                obstacle_stopped = true;
                                obstacle_stopped_by_us = !deep_near_stop && us_stopped && !in_forward_arc && !toward;
                                clear_since = None;
                                current_path = None;
                                let _ = bus_ctrl.controller_cmd_vel.try_send(CmdVel { t_ms: 0, vx: 0.0, vy: 0.0, omega: 0.0 });
                            }
                            continue;
                        }
                        info!(
                            nearest_m   = nearest,
                            speed_scale,
                            us_scale,
                            combined_scale,
                            obs_deg,
                            obs_side,
                            heading_deg,
                            vx          = cmd_vel.vx,
                            omega       = cmd_vel.omega,
                            turn_dir,
                            turning_toward_obstacle = toward,
                            "Control: slowing"
                        );
                    }

                    debug!(nearest_m = nearest, vx = cmd_vel.vx, omega = cmd_vel.omega, "Control: cmd_vel");
                    let _ = bus_ctrl.controller_cmd_vel.try_send(cmd_vel);
                }
            }
        }
    })
}
