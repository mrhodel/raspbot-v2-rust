# IMPLEMENTATION_BRIEF.md
## Phase 17 — Sim-to-Real Tuning Handoff

**Last Updated:** 2026-03-12
**Source of Truth:** `REQUIREMENTS.md` (V5.2). If this brief conflicts with it, follow `REQUIREMENTS.md`.

---

## Current State of the Robot

All 16 implementation phases are complete. The robot runs a full autonomous exploration loop:

```
Camera → MiDaS depth → pseudo-lidar → occupancy map → frontier detection
→ RL/classical frontier selector → A* planner → pure-pursuit → motors
```

Safety layers are wired. The system is ready for Phase 17: first real runs, observation, and tuning.

---

## Safety Protocol — Read Before Running

The robot starts **disarmed** (motors blocked) on every launch.

**To arm:**
```bash
kill -USR1 $(pgrep robot)
```

**To stop:**
```bash
kill $(pgrep robot)          # clean shutdown — motors zeroed via Drop
pkill -9 robot               # last resort — SIGKILL does NOT zero motors
```

**Safety interlocks (always active, regardless of arm state):**
- Ultrasonic < 15 cm → emergency stop, transitions to `SafetyStopped`
- 500 ms motor watchdog → zeros motors if command pipeline stalls
- `Drop` on motor controller → zeros motors on shutdown or panic unwind

**After an emergency stop:** robot stays in `SafetyStopped`. Re-arm with a second `kill -USR1`.

**On crash:** motors zero within ~40 ms (Drop impl). SIGKILL is the only case where motors don't stop automatically — keep a hand on the power switch for initial runs.

---

## What Is Built and Working

| Subsystem | Status | Notes |
|---|---|---|
| HAL (motor, US, gimbal, camera, IMU) | Done | Yahboom I2C protocol reverse-engineered and confirmed |
| Executive state machine | Done | `Idle → Exploring` via SIGUSR1; `Exploring → SafetyStopped` on estop |
| Safety layer | Done | US interlock + motor watchdog + Drop crash-stop |
| MiDaS ONNX depth inference | Done | `midas_small.onnx`, 147 ms on Pi 5, 4 threads |
| Event-driven vision gate | Done | Skips inference when scene is unchanged |
| Pseudo-lidar extraction | Done | 48 rays, 3 m max range |
| Occupancy grid mapping | Done | 5 cm/cell, log-odds, frontier BFS clustering |
| A* planner + pure pursuit | Done | 2-cell safety margin; 0.2 m lookahead |
| RL frontier selector | Done | AWR training pipeline; classical fallback if no model |
| sim_fast (2D simulator) | Done | 200×200 grid, 48-ray lidar, mecanum kinematics |
| UI bridge | Done | WebSocket, live map/pose/frontiers at `ws://raspbot:9090` |
| Telemetry | Done | NDJSON log writer; replay tooling not yet built |
| Micro-SLAM | Partial | IMU dead-reckoning only (gyro integration, confidence=0.3) — **drifts** |

---

## Key Gaps for Phase 17

### 1. Operational mode dispatch (blocking)
The binary always runs the full exploration loop. There is no `calibrate` or `slam-debug` mode yet. Implement `std::env::args()` dispatch in `runtime/src/main.rs` so that:
- `robot calibrate` runs IMU bias collection without the exploration stack
- `robot slam-debug` runs camera + IMU + telemetry only (no planning/motors)
- `robot robot-run` (default) runs the full stack as today

### 2. Micro-SLAM visual pipeline (blocking for extended runs)
`micro_slam/src/lib.rs` contains only `ImuDeadReckon` — gyro-only integration. This drifts significantly after a few metres. The architecture requires a visual-inertial pipeline (§8, REQUIREMENTS.md):
- FAST/ORB feature detection on grayscale frames
- Lucas–Kanade optical flow tracking
- IMU-fused pose update
- Keyframe-based drift reduction

Until this is in, map quality degrades quickly and the robot cannot explore beyond a single room reliably.

### 3. HAL I2C priority (watch, not blocking)
Currently each HAL driver makes independent I2C calls with no bus arbiter. Safety is enforced by tokio biased-select in the motor task. This is acceptable for now but should be revisited if safety commands are observed to lag during heavy gimbal/US polling.

---

## Locked Architectural Decisions

Do not change these without asking first.

1. **RL selects frontiers only — not motor commands.** The RL agent outputs a frontier choice; classical A* + pure-pursuit handles motion.
2. **US sensor is a safety interlock, not an RL input.** The policy sees depth maps only.
3. **Gimbal is reactive, not an RL action.** The gimbal controller steers toward free space independently.
4. **No ROS2.** Standalone Rust binary.
5. **Message-bus architecture.** Subsystems communicate via typed async channels on the bus, not direct coupling.
6. **Telemetry is mandatory.** Every run should produce a replayable NDJSON log.
7. **Classical frontier heuristics must work before RL is evaluated.** RL adds value on top of a working classical baseline.

---

## Recommended Phase 17 Sequence

1. **Build and deploy** to Pi — confirm clean compile and startup
2. **Wheels-up test** — arm with SIGUSR1, verify exploration runs (map builds, robot stays still, no segfault)
3. **Add operational mode dispatch** — unblocks calibration and slam-debug
4. **IMU bias calibration** — flat surface, stationary, collect bias offsets, write to `robot_config.yaml`
5. **slam-debug run** — camera + IMU logging only, drive manually, inspect pose drift in telemetry
6. **First real floor run** — wheels down, arm, let it explore one room, observe and log
7. **Tune and iterate** — reward shaping, speed, obstacle inflation based on observed behaviour
8. **Visual-inertial micro-SLAM** — implement feature tracking (see §8 in REQUIREMENTS.md), replace dead-reckoning

---

## Robot Hardware Quick Reference

| Item | Value |
|---|---|
| SSH | `ssh pi@raspbot` (10.0.0.183) |
| ORT dylib | `export ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so` |
| MiDaS model | `models/midas_small.onnx` |
| Build (Pi) | `cargo build --release -p runtime` |
| Build (dev) | `cargo check --workspace --no-default-features` |
| UI bridge | `http://raspbot:9090` (WebSocket) |
| Camera stream | `http://raspbot:8080/` (MJPEG) |
| Emergency stop pin | Physical power switch on robot body |
