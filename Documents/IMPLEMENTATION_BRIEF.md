# IMPLEMENTATION_BRIEF.md

## Goal

Implement the Yahboom Raspbot V2 autonomous indoor exploration stack in Rust on a Raspberry Pi 5.

This is a research-grade indoor robot project focused on reliable autonomous exploration, obstacle avoidance, local mapping, and debuggable system integration.

---

## Source of Truth

Primary architecture/specification document:

- `requirements_v3.md`

This brief is a practical handoff summary for implementation. If there is a conflict, follow `requirements_v3.md`.

---

## Project Summary

The robot should:

- explore unknown indoor spaces
- avoid obstacles
- build and maintain a 2D occupancy map
- estimate pose without wheel encoders
- choose exploration targets autonomously
- navigate to those targets with classical planning/control
- produce replayable telemetry for debugging

The overall design is intentionally hybrid:

- **Learning** is used only for exploration target selection
- **Classical robotics** is used for motion, mapping, control, and safety

---

## Locked Architectural Decisions

These are intentional design choices and should not be changed without asking first.

### 1. RL is only for exploration strategy
The reinforcement learning agent does **not** control motors directly.

RL is used only to choose the next exploration target / frontier selection policy.

### 2. Motion is classical
Robot motion uses:

- occupancy mapping
- frontier detection
- A* planning
- pure-pursuit control
- recovery behaviors

### 3. No wheel encoders
Wheel encoders are not available and are not part of the initial architecture.

### 4. IMU is required
An MPU-class IMU will be added and used for heading stabilization and visual-inertial localization.

### 5. Localization is micro-SLAM, not full SLAM
Initial localization uses a lightweight camera + IMU pipeline:

- feature tracking
- IMU propagation
- local keyframes
- fused `Pose2D`

Initial release does **not** require loop closure or full global SLAM.

### 6. Depth is converted to pseudo-lidar
MiDaS depth is not used directly for mapping. It is converted into lidar-like rays and raycast into the occupancy grid.

### 7. Vision is event-driven
Depth inference should not run continuously if unnecessary. The system should reuse the last depth estimate when the scene has not changed enough to justify another inference.

### 8. Message-bus architecture is required
Subsystems should communicate through typed asynchronous messages rather than direct coupling.

### 9. Telemetry and replay are mandatory
A replayable, timestamped telemetry system is a required part of the architecture, not an optional add-on.

### 10. No ROS2
The project is intentionally standalone Rust without ROS2.

---

## System Overview

High-level pipeline:

```text
Camera
   ↓
Event-Driven Vision Gate
   ↓
MiDaS Depth
   ↓
Pseudo-Lidar Extraction
   ↓
Micro-SLAM Pose Estimation (camera + IMU)
   ↓
Occupancy Grid Mapping
   ↓
Frontier Detection
   ↓
RL Frontier Selection
   ↓
A* Planner
   ↓
Pure Pursuit Controller
   ↓
Motors
```

---

## Primary Implementation Goal

Create a compilable Rust workspace that matches the architecture in `requirements_v3.md`.

Target crates/modules:

```text
robot/
 ├── config/
 ├── core_types/
 ├── bus/
 ├── hal/
 ├── perception/
 ├── micro_slam/
 ├── mapping/
 ├── exploration_rl/
 ├── planning/
 ├── control/
 ├── safety/
 ├── telemetry/
 └── runtime/
```

The initial goal is not “full autonomy immediately.” The initial goal is a sound, testable, debuggable software foundation.

---

## First Milestone

Deliver a non-autonomous debug-capable system that can:

- read camera frames
- read IMU data
- publish typed messages on the bus
- log synchronized telemetry
- run event-driven depth inference
- generate pseudo-lidar
- produce a basic fused pose estimate
- write replayable logs

This milestone is more important than RL or full exploration.

---

## Recommended Implementation Order

Implement in this sequence unless a strong reason appears not to.

1. workspace skeleton
2. `core_types`
3. `bus`
4. `telemetry`
5. `hal` stubs
6. IMU integration
7. camera pipeline
8. event-driven depth
9. pseudo-lidar extraction
10. micro-SLAM prototype
11. occupancy mapping
12. planner + controller
13. safety layer
14. fast simulator
15. RL frontier selector
16. real robot integration
17. simulator/perception refinement

---

## Technical Priorities

### Highest priority
- code clarity
- typed interfaces
- replayability
- deterministic subsystem boundaries
- simple, debuggable implementations first

### Lower priority for early phases
- peak performance optimization
- advanced RL
- full SLAM
- semantic perception
- sophisticated simulator realism

---

## Guidance for Key Subsystems

### Bus
Prefer a simple `tokio`-based bus using appropriate channel types:

- `broadcast` for shared sensor streams
- `watch` for latest-state topics
- `mpsc` for command paths

Messages should be timestamped and typed.

### Telemetry
Telemetry must record enough information to replay runs offline.

At minimum include:

- timestamps
- topic names
- message source
- payload
- event markers

### Perception
Use the simplest pipeline that works first:

- grayscale frame preprocessing
- event-driven inference gate
- MiDaS inference
- pseudo-lidar extraction

Do not overcomplicate depth handling early.

### Micro-SLAM
Keep the first version minimal:

- FAST or ORB features
- Lucas–Kanade optical flow or equivalent short-horizon tracking
- IMU-assisted pose fusion
- 2D pose output
- confidence metric

Avoid loop closure and graph optimization initially.

### Mapping
Standard occupancy grid is acceptable for the first implementation.
Do not switch to TSDF unless specifically asked later.

### RL
RL should remain isolated from motion control.
It should choose frontier targets or frontier-selection heuristics only.

---

## Constraints

- target platform: Raspberry Pi 5
- no wheel encoders
- no ROS2
- keep dependencies modest where reasonable
- prefer simple implementations over sophisticated but fragile ones
- optimize for observability and incremental testing

---

## Known Open Questions

These are still open and can be handled pragmatically during implementation.

- OpenCV bindings vs pure Rust for feature tracking
- exact IMU hardware details and calibration workflow
- exact telemetry storage format
- exact replay tooling format
- ONNX Runtime deployment details on the Pi
- how much of `sim_vision` is needed in the earliest phase

These are implementation choices, not blockers.

---

## What Success Looks Like Early

A successful early implementation will let us:

- boot the system cleanly
- inspect live or logged camera/IMU streams
- confirm messages flow through the bus
- inspect pseudo-lidar output
- inspect pose estimates over short indoor runs
- replay a run and compare subsystem outputs offline

If those work, the project is on a strong path.

---

## What Not To Do Early

Please avoid these until the foundation is working:

- direct RL motor control
- full loop-closure SLAM
- complex graph optimization
- over-engineered planner stacks
- premature simulator complexity
- heavy coupling between crates/modules
- “helpful” architectural rewrites that conflict with the current spec

---

## Delivery Style Requested

Please implement incrementally and keep the project runnable at each stage.

Preferred style:

- small coherent commits / changesets
- clear TODO markers
- compile-first scaffolding
- explicit interfaces
- stubs where needed rather than speculative complexity
- practical notes about tradeoffs when making implementation choices

---

## Final Note

This project is intended to be:

- practical
- debuggable
- modular
- realistic for a Pi-class robot

Please preserve those qualities over elegance or novelty.
