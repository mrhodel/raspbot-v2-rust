# IMPLEMENTATION_BRIEF.md

## Goal

Implement the Yahboom Raspbot V2 autonomous indoor exploration stack in Rust on a Raspberry Pi 5.

This is a research-grade indoor robot project focused on reliable autonomous exploration, obstacle avoidance, local mapping, and debuggable system integration.

---

## Source of Truth

Primary architecture/specification document:

- `requirements_v5.md`

This brief is a practical handoff summary for implementation. If there is a conflict, follow `requirements_v5.md`.

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
- **Classical robotics** is used for motion, mapping, control, safety, and behavior management

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

### 3. The robot must work without RL first
Before any learning is added, the system must support autonomous exploration using classical frontier heuristics.

Required baseline heuristics:

- nearest frontier
- largest frontier

### 4. No wheel encoders
Wheel encoders are not available and are not part of the initial architecture.

### 5. IMU is required
An MPU-class IMU will be added and used for heading stabilization and visual-inertial localization.

### 6. Localization is micro-SLAM, not full SLAM
Initial localization uses a lightweight camera + IMU pipeline:

- feature tracking
- IMU propagation
- local keyframes
- fused `Pose2D`

Initial release does **not** require loop closure or full global SLAM.

### 7. Depth is converted to pseudo-lidar
MiDaS depth is not used directly for mapping. It is converted into lidar-like rays and raycast into the occupancy grid.

### 8. Vision is event-driven
Depth inference should not run continuously if unnecessary. The system should reuse the last depth estimate when the scene has not changed enough to justify another inference.

### 9. Message-bus architecture is required
Subsystems should communicate through typed asynchronous messages rather than direct coupling.

### 10. Telemetry and replay are mandatory
A replayable, timestamped telemetry system is a required part of the architecture, not an optional add-on.

### 11. Calibration is mandatory
Camera, IMU, and extrinsics calibration are first-class deliverables and need an explicit runtime mode.

### 12. A central executive is mandatory
A simple explicit state machine should own high-level state transitions such as exploration, recovery, safety stop, and fault handling.

### 13. No ROS2
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
Classical Frontier Heuristics / RL Frontier Selection
   ↓
A* Planner
   ↓
Pure Pursuit Controller
   ↓
Motors

Executive, Safety, and Telemetry span the full stack
```

---

## Primary Implementation Goal

Create a compilable Rust workspace that matches the architecture in `requirements_v5.md`.

Target crates/modules:

```text
robot/
 ├── config/
 ├── core_types/
 ├── bus/
 ├── calibration/
 ├── executive/
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

The initial goal is not full autonomy immediately. The initial goal is a sound, testable, debuggable software foundation.

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
- report executive state
- write replayable logs

This milestone is more important than RL or full exploration.

---

## Second Milestone

Deliver end-to-end autonomous exploration **without RL** using:

- executive state machine
- occupancy mapping
- frontier detection
- nearest / largest frontier heuristics
- planner + controller
- safety layer
- replay logs

Only after this works should RL be added.

---

## Recommended Implementation Order

Implement in this sequence unless a strong reason appears not to.

1. workspace skeleton
2. `core_types`
3. `bus`
4. `telemetry`
5. `calibration`
6. `executive`
7. `hal` stubs
8. Yahboom I2C protocol capture/documentation
9. IMU integration
10. camera pipeline
11. event-driven depth
12. pseudo-lidar extraction
13. micro-SLAM prototype
14. occupancy mapping
15. classical frontier detection + baseline heuristics
16. planner + controller
17. safety layer
18. fast simulator
19. RL frontier selector
20. real robot integration
21. simulator/perception refinement

---

## Technical Priorities

### Highest priority
- code clarity
- typed interfaces
- replayability
- deterministic subsystem boundaries
- simple, debuggable implementations first
- working baseline without RL
- measurable runtime health and timing

### Lower priority for early phases
- peak performance optimization
- advanced RL tuning
- full SLAM
- semantic perception
- sophisticated simulator realism

---

## Guidance for Key Subsystems

### HAL
Assume Yahboom hardware is controlled through a shared I2C board protocol.
Use the vendor Python implementation as the reference if needed.
Document the protocol explicitly during the HAL phase.

Implement or plan an internal HAL priority policy so urgent commands are not delayed by routine reads.

### Bus
Prefer a simple `tokio`-based bus using appropriate channel types:

- `broadcast` for shared sensor streams
- `watch` for latest-state topics
- `mpsc` for command paths

Messages should be timestamped and typed.

### Executive
Implement a simple explicit state machine first.

Minimum useful states:

- `Idle`
- `Calibrating`
- `Exploring`
- `Recovering`
- `SafetyStopped`
- `Fault`

### Telemetry
Telemetry must record enough information to replay runs offline.

At minimum include:

- timestamps
- topic names
- message source
- payload
- event markers
- executive state changes
- runtime health metrics

### Calibration
Treat calibration as a real subsystem, not just a startup script.

Need support for:

- camera intrinsics
- camera-to-base extrinsics
- IMU bias / orientation
- validated tilt setting

### Perception
Use the simplest pipeline that works first:

- grayscale frame preprocessing
- event-driven inference gate
- initial MiDaS small-class ONNX model
- pseudo-lidar extraction
- optional floor-plane anchoring using known camera geometry

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
Standard occupancy grid is the first implementation target.
Frontier detection and baseline heuristics must work before RL.

### RL
RL should remain isolated from motion control.
It should choose frontier targets or frontier-selection heuristics only.
Its value should be measured against classical baselines.

### Simulation
Implement `sim_fast` first as a 2D kinematic, noisy training/evaluation environment.
Use `sim_vision` later for perception realism and validation, not for the bulk of training.

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
- whether a smaller RL state than a full local occupancy window is preferable

These are implementation choices, not blockers.

---

## What Success Looks Like Early

A successful early implementation will let us:

- boot the system cleanly
- inspect live or logged camera/IMU streams
- confirm messages flow through the bus
- inspect pseudo-lidar output
- inspect pose estimates over short indoor runs
- inspect executive transitions
- replay a run and compare subsystem outputs offline
- observe runtime health metrics and timing

A successful medium-stage implementation will let us:

- autonomously explore a room using classical frontier heuristics
- build a usable occupancy map
- recover from minor tracking or obstacle issues
- evaluate whether RL improves frontier choice

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
- helpful architectural rewrites that conflict with the current spec

---

## Final Note

This project is intended to be:

- practical
- debuggable
- modular
- realistic for a Pi-class robot

It is expected to achieve stronger topological exploration and room-scale autonomy than precise long-range metric reconstruction in the first release. Please preserve those qualities over elegance or novelty.
