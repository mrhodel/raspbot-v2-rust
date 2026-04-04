//! Safety task — emergency stop monitor and motor override.

use std::sync::Arc;
use tracing::{error, info, warn};

use bus::Bus;
use core_types::MotorCommand;
use safety::SafetyMonitor;

pub fn spawn_safety_task(
    bus: Arc<Bus>,
    emergency_stop_cm: f32,
    enable_watchdog: bool,
    emstop_latch_s: f64,
    escape_delay_ms: u64,
    escape_duration_ms: u64,
    escape_rotation_ms: u64,
    escape_reverse_spd: i8,
) -> tokio::task::JoinHandle<()> {
    let bus_safety = Arc::clone(&bus);
    let mut rx_us  = bus.ultrasonic.subscribe();
    tokio::spawn(async move {
        let mut monitor  = SafetyMonitor::new();
        monitor.stop_threshold_cm = emergency_stop_cm;
        info!(
            threshold_cm = monitor.stop_threshold_cm,
            watchdog_ms  = monitor.watchdog_timeout_ms,
            "Safety task started"
        );
        let watchdog_dur = tokio::time::Duration::from_millis(monitor.watchdog_timeout_ms);
        let latch_dur    = tokio::time::Duration::from_secs_f64(emstop_latch_s);
        let delay_dur    = tokio::time::Duration::from_millis(escape_delay_ms);
        let reverse_dur  = tokio::time::Duration::from_millis(escape_duration_ms);
        let rotate_dur   = tokio::time::Duration::from_millis(escape_rotation_ms);
        let mut emstop_until:  Option<tokio::time::Instant> = None;
        let mut escape_until:  Option<tokio::time::Instant> = None;
        loop {
            tokio::select! {
                result = rx_us.recv() => {
                    match result {
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                            warn!("Safety task lagged {n} readings");
                        }
                        Ok(reading) => {
                            // Check whether a previous EmergencyStop latch is still active.
                            let now = tokio::time::Instant::now();
                            if emstop_until.map_or(false, |u| now < u) {
                                // Still latched: keep motors stopped but do NOT re-publish
                                // EmergencyStop on the state bus.  The watch channel already
                                // holds EmergencyStop from the initial send; re-sending on every
                                // US poll tick causes changed() to fire on all receivers every
                                // ~150ms, creating a spam loop in the control task that repeatedly
                                // resets obstacle_stopped.  The motor task uses borrow() so it
                                // still sees EmergencyStop without a repeated notification.
                                if escape_until.map_or(true, |u| now >= u) {
                                    let _ = bus_safety.motor_command.try_send(MotorCommand::stop(reading.t_ms));
                                }
                            } else {
                                let (state, maybe_stop) = monitor.evaluate(&reading);
                                let _ = bus_safety.safety_state.send(state);
                                if let Some(cmd) = maybe_stop {
                                    warn!(range_cm = reading.range_cm, "Safety: EMERGENCY STOP — latching for {}s", latch_dur.as_secs());
                                    emstop_until = Some(now + latch_dur);
                                    escape_until = Some(now + delay_dur + reverse_dur + rotate_dur);
                                    // Count each fresh EmergencyStop as a near-miss.
                                    let n = *bus_safety.estop_count.borrow() + 1;
                                    let _ = bus_safety.estop_count.send(n);
                                    if bus_safety.motor_command.try_send(cmd).is_err() {
                                        error!("Safety: motor_command channel full — stop dropped!");
                                    }
                                    // Spawn escape maneuver: wait for stop to take effect,
                                    // reverse to clear the obstacle, then stop again.
                                    let bus_esc = Arc::clone(&bus_safety);
                                    let spd = escape_reverse_spd;
                                    tokio::spawn(async move {
                                        tokio::time::sleep(delay_dur).await;
                                        // Reverse: send repeatedly so sim sustains the command each tick.
                                        info!("Safety: escape — reversing for {}ms", reverse_dur.as_millis());
                                        let reverse = MotorCommand { t_ms: 0, fl: -spd, fr: -spd, rl: -spd, rr: -spd };
                                        let deadline = tokio::time::Instant::now() + reverse_dur;
                                        while tokio::time::Instant::now() < deadline {
                                            let _ = bus_esc.motor_command.try_send(reverse);
                                            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                                        }
                                        // Rotate to break heading deadlock: turn away from the
                                        // nearest-obstacle angle. Positive angle = obstacle to
                                        // left → rotate right (CW); negative → rotate left (CCW).
                                        let obs_angle = *bus_esc.nearest_obstacle_angle_rad.borrow();
                                        let (fl, fr, rl, rr): (i8, i8, i8, i8) = if obs_angle >= 0.0 {
                                            (spd, -spd, spd, -spd)   // CW: obstacle left → turn right
                                        } else {
                                            (-spd, spd, -spd, spd)   // CCW: obstacle right → turn left
                                        };
                                        info!(obs_angle_rad = obs_angle, cw = (obs_angle >= 0.0),
                                              "Safety: escape — rotating for {}ms", rotate_dur.as_millis());
                                        let rotate = MotorCommand { t_ms: 0, fl, fr, rl, rr };
                                        let deadline = tokio::time::Instant::now() + rotate_dur;
                                        while tokio::time::Instant::now() < deadline {
                                            let _ = bus_esc.motor_command.try_send(rotate);
                                            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                                        }
                                        let _ = bus_esc.motor_command.try_send(MotorCommand::stop(0));
                                        info!("Safety: escape complete");
                                    });
                                }
                            }
                        }
                    }
                }
                _ = tokio::time::sleep(watchdog_dur), if enable_watchdog => {
                    warn!("Safety: watchdog timeout — no US reading");
                    let now = tokio::time::Instant::now();
                    // Only arm a fresh latch; don't extend an existing one so the
                    // latch can expire naturally (e.g. during episode reset).
                    if emstop_until.map_or(true, |u| now >= u) {
                        emstop_until = Some(now + latch_dur);
                    }
                    let (state, cmd) = monitor.evaluate_timeout(0);
                    let _ = bus_safety.safety_state.send(state);
                    let _ = bus_safety.motor_command.try_send(cmd);
                }
            }
        }
    })
}
