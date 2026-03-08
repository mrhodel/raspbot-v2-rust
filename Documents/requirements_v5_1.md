# Yahboom Raspbot V2 — Autonomous Indoor Exploration (Revised Architecture)
## Requirements Specification — Version 1.5 (V5.1)

**Status:** Draft – Implementation-Oriented Architecture  
**Last Updated:** 2026-03-08  
**Authors:** Mike, AI Assistant  

---

# 1. Project Overview

Develop an autonomous indoor exploration robot based on the Yahboom Raspbot V2 and Raspberry Pi 5.

The robot will:

- explore unknown indoor environments
- avoid obstacles
- build and maintain a 2D occupancy map
- estimate pose indoors without wheel encoders
- select exploration targets autonomously
- navigate to targets using classical planning
- support replayable telemetry for debugging and offline evaluation

A hybrid architecture is used:

- **Learning component:** reinforcement learning selects exploration targets
- **Deterministic robotics:** perception, localization, mapping, planning, control, calibration, executive behavior management, safety, and human-observable visualization

This architecture maximizes reliability while maintaining adaptive behavior.

---

# 2. Key Design Principles

1. **Classical control for motion**
2. **Learning only for exploration strategy**
3. **Robot must work with classical frontier heuristics before RL**
4. **Vision converted to pseudo-lidar for mapping**
5. **Micro-SLAM for pose estimation without encoders**
6. **Event-driven perception to reduce compute**
7. **Time-synchronized telemetry for debugging**
8. **Simulation-first development**
9. **Hardware-safe failover layers**
10. **Message-passing architecture to reduce coupling**
11. **Confidence-aware operation under degraded sensing**
12. **Calibration is a first-class subsystem, not an afterthought**
13. **A central executive owns behavior transitions**
14. **Topological exploration quality matters more than precise metric accuracy in v1**
15. **Live human-observable visualization should be available during debug and bring-up**

---

# 3. Hardware Specification

## 3.1 Platform

| Parameter | Value |
|---|---|
| Robot | Yahboom Raspbot V2 |
| Compute | Raspberry Pi 5 (16 GB) |
| Drive | 4-wheel mecanum |
| Camera | USB 1 MP |
| Camera FOV | ~110° |
| Camera height | ~10 cm |
| IMU | MPU-6050 or similar |
| Ultrasonic | HC-SR04 forward |
| Wheel encoders | Not available |

## 3.2 Camera Configuration

Default tilt:

- **−15° downward**

Allowed range:

- −25° to −5°

Benefits:

- better floor visibility
- earlier obstacle detection
- improved mapping stability

## 3.3 Yahboom Expansion Board / HAL Assumption

The Yahboom platform uses a shared controller board accessed via I2C rather than separate direct drivers for each function.
The Rust HAL shall treat this as a **board-level command protocol** problem.

Implications:

- motor control
- servo / gimbal control
- ultrasonic reads

may all be implemented through a shared Yahboom I2C protocol.

**Implementation note:** the official vendor Python library is expected to be the reference for reverse-engineering register writes / command format. Capturing and documenting this protocol is an explicit deliverable of the HAL phase.

---

# 4. System Architecture

```text
                ┌─────────────────────────┐
                │        CAMERA           │
                └──────────┬──────────────┘
                           │
                           ▼
                 Event-Driven Vision Gate
                           │
                    (frame change?)
                           │
                   yes ─────────── no
                    │               │
                    ▼               │
                  MiDaS             │
                    │               │
                    └──── reuse last depth
                           │
                           ▼
                   Pseudo-Lidar Extraction
                           │
                           ▼
            ┌─────────────────────────────────┐
            │  Micro-SLAM / Pose Estimation   │
            │  (camera + IMU, no encoders)    │
            └──────────────┬──────────────────┘
                           │
                           ▼
                   Occupancy Grid Mapping
                           │
                           ▼
                    Frontier Detection
                           │
                           ├──────────────► Classical Frontier Heuristics
                           │
                           ▼
                   RL Frontier Selection
                           │
                           ▼
                       A* Planner
                           │
                           ▼
                  Pure Pursuit Controller
                           │
                           ▼
                         Motors

   Safety / Executive / Telemetry observe and arbitrate across the full stack
```

---

# 5. Perception Pipeline

## 5.1 Event-Driven Depth

Depth inference occurs only when:

- frame difference exceeds threshold
- robot rotates
- gimbal moves
- parallax scan executes
- exploration state changes

Expected reduction in neural-network inference:

**70–85%**

## 5.2 Initial Depth Model and Runtime

Initial depth stack for implementation:

- **MiDaS v3.1 small-class model**
- preferred first candidates:
  - `dpt_levit_224`
  - or another comparable small ONNX-exportable MiDaS variant
- ONNX export required
- quantization is allowed later if needed

Initial runtime assumption:

- ONNX Runtime (`ort` crate) is the preferred first choice
- alternative runtimes may be used later if deployment friction or performance requires it

Performance target:

- raw depth inference on Pi 5 CPU is expected to be modest
- effective system behavior should rely on event-driven inference so that depth is available when needed at roughly **5–10 Hz effective decision support**, not necessarily 5–10 Hz raw model execution

## 5.3 Pseudo-Lidar Extraction

Depth image → lidar-like rays.

Typical configuration:

| Parameter | Value |
|---|---|
| Depth resolution | 192×192 to 256×256 |
| Lidar rays | 48 |
| Max range | 3 m |
| Update rate | 5–10 Hz effective |
| Projection style | nearest-obstacle ray extraction |

Advantages:

- robust mapping
- reduced noise
- fast occupancy updates
- compatibility with classical robotics algorithms

## 5.4 Floor-Plane Anchoring

Where practical, the perception stack should use known geometry to improve metric consistency:

- camera height above floor
- camera tilt angle
- estimated floor plane

This geometric information may be used to anchor or scale pseudo-depth estimates, especially near the floor region.
This does not eliminate monocular ambiguity, but it provides a useful metric constraint.

## 5.5 Parallax Scans

The robot occasionally performs a small lateral scan to improve depth confidence:

```text
left 10 cm
right 20 cm
left 10 cm
```

Recommended frequency:

- every 5–10 seconds
- on entering a new room
- when depth confidence is low

---

# 6. Calibration Subsystem

## 6.1 Purpose

Calibration shall be treated as a required subsystem and operational mode.

The system requires calibration for:

- camera intrinsics
- camera-to-robot extrinsics
- IMU bias
- gravity reference
- default camera tilt validation

## 6.2 Required Calibration Outputs

The calibration process shall produce configuration data for:

- camera intrinsic matrix / distortion parameters
- camera pose relative to robot base frame
- IMU bias offsets
- IMU mounting orientation relative to robot base frame
- validated downward camera tilt value

## 6.3 Operational Mode

An explicit runtime mode shall exist:

- `calibrate`

Sub-modes may include:

- `calibrate-camera`
- `calibrate-imu`
- `calibrate-extrinsics`

## 6.4 Deliverable Expectation

A lightweight `calibration/` crate or equivalent module is recommended.

---

# 7. Executive / Behavior Management

## 7.1 Purpose

A central executive shall own high-level behavior transitions and system state changes.

This prevents responsibility for mode changes from being spread implicitly across planning, safety, mapping, and recovery code.

## 7.2 Responsibilities

The executive shall:

- own the current high-level system state
- react to safety events
- react to degraded localization confidence
- trigger recovery behaviors
- control transitions between calibration, exploration, recovery, paused, and fault states
- expose machine-readable state transitions to telemetry

## 7.3 Minimum State Set

Initial explicit states should include at least:

- `Idle`
- `Calibrating`
- `Exploring`
- `Recovering`
- `SafetyStopped`
- `Fault`

## 7.4 Transition Examples

Examples of transitions the executive should manage:

- `Exploring` → `Recovering` on stuck detection
- `Exploring` → `SafetyStopped` on emergency stop
- `Exploring` → `Recovering` on severe localization confidence drop
- `Recovering` → `Exploring` on successful recovery
- any state → `Fault` on unrecoverable subsystem failure

A simple explicit state machine is preferred for v1 over a more elaborate behavior-tree framework.

---

# 8. Micro-SLAM / Localization

## 8.1 Purpose

Because wheel encoders are unavailable, the robot shall estimate pose using a lightweight
visual-inertial localization pipeline ("micro-SLAM") built from:

- monocular camera
- IMU (gyro + accel)
- short-horizon feature tracking
- sparse local map / keyframes

## 8.2 Scope

Micro-SLAM is intended to provide:

- short- to medium-term pose stability
- reliable heading estimation
- local drift reduction good enough for room-scale exploration
- keyframe-based local consistency

It is **not required** to provide full global loop-closure SLAM in version 1.4.

## 8.3 Functional Design

The initial micro-SLAM implementation should include:

1. **Feature detection**
   - FAST or ORB features on grayscale frames

2. **Feature tracking**
   - pyramidal Lucas-Kanade optical flow or descriptor matching

3. **IMU propagation**
   - gyro integration for heading
   - accel used for short-term stabilization / gravity estimate

4. **Visual-inertial pose update**
   - estimate frame-to-frame motion from tracked features
   - fuse with IMU to stabilize yaw and reject jitter

5. **Keyframe buffer**
   - retain a short rolling set of keyframes
   - use local re-alignment against recent keyframes to reduce drift

6. **Pose output**
   - 2D pose for mapping and planning: `(x, y, theta)`
   - optional internal 3D estimate if useful

## 8.4 Performance Target

Micro-SLAM should run in real time on the Pi 5 at approximately:

- feature tracking: 10–15 Hz
- IMU propagation: 50–100 Hz
- pose output to mapper/controller: 10 Hz

## 8.5 Capability Expectation

Micro-SLAM is expected to be:

- much better than dead reckoning from motor commands
- good enough for room and hallway exploration
- still subject to drift over long paths or low-texture scenes

Expected limitations:

- blank walls may degrade tracking
- rapid lighting changes may reduce feature stability
- no guaranteed global consistency without later loop closure
- mecanum slip and perception latency will limit metric fidelity

## 8.6 Interfaces and Data Structures

### Required outputs

Micro-SLAM shall publish:

- current fused pose
- visual delta pose
- tracking quality metrics
- keyframe events
- confidence estimate

### Reference data structures

```rust
pub struct ImuSample {
    pub t_ms: u64,
    pub gyro_z: f32,
    pub accel_x: f32,
    pub accel_y: f32,
    pub accel_z: f32,
}

pub struct FeaturePoint {
    pub id: u32,
    pub x: f32,
    pub y: f32,
    pub score: f32,
}

pub struct FeatureFrame {
    pub t_ms: u64,
    pub points: Vec<FeaturePoint>,
}

pub struct TrackSet {
    pub t_ms: u64,
    pub matches: Vec<(u32, [f32; 2], [f32; 2])>,
    pub mean_flow: [f32; 2],
    pub inlier_count: usize,
}

pub struct VisualDelta {
    pub t_ms: u64,
    pub dx_m: f32,
    pub dy_m: f32,
    pub dtheta_rad: f32,
    pub confidence: f32,
}

pub struct Pose2D {
    pub t_ms: u64,
    pub x_m: f32,
    pub y_m: f32,
    pub theta_rad: f32,
    pub confidence: f32,
}
```

## 8.7 Runtime Tasks

### IMU task
Runs at 50–100 Hz.

Responsibilities:

- read gyro + accel
- calibrate bias
- estimate gravity direction
- integrate short-term yaw

### Feature task
Runs at camera rate.

Responsibilities:

- grayscale preprocessing
- detect strongest features
- publish feature frame

### Tracking task
Runs at camera rate.

Responsibilities:

- track features frame-to-frame
- reject outliers
- estimate optical flow consistency
- trigger keyframe creation

### Motion estimation task
Runs at camera rate.

Responsibilities:

- compute local visual motion
- estimate short translation and yaw delta
- publish visual delta pose

### Fusion task
Runs at 10–50 Hz.

Responsibilities:

- fuse IMU and visual motion
- generate 2D pose
- publish confidence estimate
- report degraded tracking state

---

# 9. Mapping

Occupancy grid:

| Parameter | Value |
|---|---|
| Cell size | 5 cm |
| Grid size | dynamic |
| Sensor model | raycasting from pseudo-lidar |
| Decay | slow toward unknown |

Mapping shall consume:

- pseudo-lidar rays from depth
- pose estimate from micro-SLAM

Mapping shall publish:

- occupancy grid snapshots or deltas
- frontier candidates
- explored area statistics

## 9.1 Frontier Detection

The system shall detect frontiers as boundaries between known-free and unknown space.

Initial implementation guidance:

- threshold occupancy into free / occupied / unknown
- identify frontier cells as free cells adjacent to unknown cells
- cluster neighboring frontier cells
- compute simple cluster properties such as:
  - centroid
  - size
  - distance from robot
  - heading offset
  - reachable / blocked status if known

This classical frontier detector is required before RL is added.

---

# 10. Exploration Strategy

Exploration uses **frontier-based mapping**.

The robot must support **classical frontier heuristics before reinforcement learning**.

Required baseline strategies:

- nearest frontier
- largest frontier

Optional baseline strategies:

- leftmost frontier
- rightmost frontier
- random valid frontier

RL is evaluated only after baseline exploration is functioning end-to-end.

---

# 11. Reinforcement Learning Design

## 11.1 Role of RL

RL chooses a frontier or frontier-selection heuristic only.
RL does **not** directly control motors.

## 11.2 State

RL state should be based on geometry and exploration context, not raw motor control.

Example contents:

- local 64×64 occupancy window
- robot heading
- candidate frontier descriptors
- local free-space statistics

A smaller frontier-descriptor-based state may be substituted later if it proves simpler and more robust.

## 11.3 Actions

Select a frontier or frontier-selection heuristic.

Example:

| Action | Meaning |
|---|---|
| 0 | nearest frontier |
| 1 | largest frontier |
| 2 | leftmost frontier |
| 3 | rightmost frontier |
| 4 | random valid frontier |

## 11.4 Rewards

| Reward | Intent |
|---|---|
| New map area discovered | positive |
| Frontier reached | positive |
| Collision | large negative |
| Time penalty | small negative |

## 11.5 Training Requirement

The training pipeline must compare learned frontier selection against classical heuristics so the value of RL can be measured rather than assumed.

---

# 12. Motion Control

Classical deterministic navigation.

## 12.1 Planner

- A*

## 12.2 Controller

- pure pursuit

## 12.3 Recovery

If stuck:

1. rotate
2. reverse slightly if safe
3. select a new frontier
4. trigger a parallax scan if needed

---

# 13. Simulation Architecture

Two simulators shall be supported.

## 13.1 sim_fast

Purpose:

- RL training
- reward tuning
- large-scale experiments
- mapping / frontier / planning validation

Scope:

- 2D grid world
- kinematic robot motion
- configurable noise
- no expensive rendering required

Recommended initial design:

- occupancy-style world or procedural 2D floorplan
- robot represented by simple footprint
- commanded motion integrated into pose
- pseudo-lidar or equivalent range observations produced directly from map geometry

Suggested configurable noise / realism:

- translational noise
- rotational noise
- IMU noise
- pseudo-lidar noise
- occasional dropped observations
- randomized sensor quality for robustness testing

Target speed:

- approximately tens of thousands of steps per second on desktop-class hardware

## 13.2 sim_vision

Purpose:

- perception testing
- pseudo-lidar validation
- micro-SLAM validation
- sim-to-real checks

Scope:

- 3D or visually richer simulator
- lower speed acceptable
- used for perception realism, not the bulk of RL training

## 13.3 Episode Definition

Initial RL / exploration episodes should define:

- random initial pose
- generated room or floorplan
- maximum step budget
- success condition based on explored area / frontier exhaustion / target completion
- failure condition based on collision, timeout, or unrecoverable state

---

# 14. Network / GUI Bridge

## 14.1 Purpose

A lightweight read-only network / GUI bridge shall expose live robot state for human visualization during bring-up, debugging, and evaluation.

This bridge is not part of the autonomy core and shall not be required for basic robot operation.

## 14.2 Minimum Scope

The initial bridge should support:

- live occupancy-grid visualization
- robot pose visualization
- frontier candidate visualization
- current executive state display
- selected frontier strategy display
- basic health telemetry display

## 14.3 Transport Guidance

A simple implementation is preferred:

- local or LAN WebSocket server
- browser-based client or simple HTML/JS viewer
- read-only streaming in v1

## 14.4 Data Sources

The bridge should subscribe to existing bus topics rather than introducing parallel data paths.

Likely inputs include:

- `map/grid_delta`
- `map/frontiers`
- `slam/pose2d`
- `executive/state`
- `decision/frontier_choice`
- `health/runtime`

## 14.5 Non-Goals for v1

The bridge does not need to provide:

- full operator teleoperation
- rich mission planning UI
- mandatory cloud connectivity
- safety-critical command control

A small future extension may allow simple operator controls such as start/stop/debug mode switching, but the initial bridge should remain visualization-first.

---

# 15. Message Bus / Runtime Communication

## 14.1 Purpose

Subsystems shall communicate via typed messages over asynchronous channels rather than
tight direct coupling between modules.

Benefits:

- easier debugging
- cleaner ownership boundaries
- isolated subsystem testing
- simpler simulation/real-hardware swapping

## 14.2 Topic Set

Initial logical topics:

```text
camera/frame_raw
camera/frame_gray
imu/raw
imu/orientation
vision/depth
vision/pseudo_lidar
vision/features
vision/tracks
slam/visual_delta
slam/pose2d
slam/keyframe_event
map/grid_delta
map/frontiers
map/explored_stats
decision/frontier_choice
planner/path
controller/cmd_vel
motor/command
executive/state
safety/event
telemetry/event_marker
health/runtime
ui/bridge_status
```

## 14.3 Message Bus Requirements

The runtime bus shall support:

- multiple subscribers per topic
- bounded queues
- timestamped messages
- non-blocking publish where practical
- graceful handling of slow subscribers
- drop detection / counters for debug builds

## 14.4 Implementation Guidance

A `tokio`-based implementation using `broadcast`, `watch`, or `mpsc` is acceptable.

Recommended pattern:

- high-rate sensor streams: bounded `broadcast`
- latest-state topics: `watch`
- command paths: bounded `mpsc`

---

# 16. Telemetry & Replay System

## 15.1 Goal

Every control cycle shall log enough synchronized data to reconstruct and replay robot behavior offline.

## 15.2 Logged Data

Telemetry should include:

- frame reference or compressed camera frame
- depth map and pseudo-lidar
- IMU samples or summaries
- micro-SLAM pose estimate
- occupancy map state or deltas
- frontier candidates
- chosen frontier strategy or RL decision
- planner path
- controller output
- motor commands
- ultrasonic readings
- executive state transitions
- event markers
- confidence / health indicators

## 15.3 Health Telemetry

Health telemetry should include, where available:

- Pi temperature
- CPU utilization or load indicator
- depth inference timing
- micro-SLAM timing
- dropped-frame counters
- I2C error counters
- battery voltage or supply voltage if measurable
- throttle / undervoltage indicators if available

## 15.4 Schema Requirements

Each record shall include:

- monotonic timestamp
- topic name
- source subsystem
- payload
- optional sequence number

### Reference record structure

```rust
pub struct TelemetryRecord<T> {
    pub t_ms: u64,
    pub topic: String,
    pub source: String,
    pub seq: u64,
    pub payload: T,
}
```

A compact binary format is acceptable for runtime logging; a readable export format is recommended for debugging and review.

## 15.5 Event Markers

The system shall record structured event markers, including at least:

- collision
- emergency stop
- frontier selected
- frontier reached
- parallax scan started / completed
- tracking confidence drop
- keyframe inserted
- planner failed
- recovery behavior triggered
- calibration completed
- executive state changed

## 15.6 Replay Requirements

Replay tooling shall support:

- time-synchronized sensor playback
- step-by-step timeline advance
- camera/depth/map side-by-side visualization
- comparison of alternate algorithms against recorded runs

---

# 17. HAL Scheduling and Priority Policy

Because the Yahboom board likely multiplexes control and sensing over I2C, the HAL shall treat bus access as a schedulable shared resource.

## 16.1 Priority Order

Suggested priority order:

1. emergency stop / safety halt
2. motor commands
3. gimbal commands
4. critical status reads needed for control
5. ultrasonic reads
6. low-priority diagnostics / telemetry-only reads

## 16.2 Design Guidance

An internal HAL queue or scheduler is recommended so that urgent commands are not delayed by routine sensor polling.

---

# 18. Safety Layer

Hardware safety is independent of AI.

Triggers:

- ultrasonic < 15 cm
- watchdog timeout
- explicit emergency stop
- invalid pose / control sanity failure

Response:

- immediate motor halt

The safety subsystem shall also publish a machine-readable event to `safety/event`.

---

# 19. Operational Modes

| Mode | Purpose |
|---|---|
| hw-test | test hardware components |
| calibrate | run calibration routines |
| sim-train | RL training |
| sim-run | policy evaluation |
| robot-run | autonomous operation |
| robot-debug | autonomous run with heavy telemetry |
| slam-debug | camera/IMU localization validation |

---

# 20. Software Workspace Layout

```text
robot/
 ├── config/
 ├── core_types/
 ├── bus/
 ├── calibration/
 ├── executive/
 ├── ui_bridge/
 ├── hal/
 ├── perception/
 ├── micro_slam/
 ├── mapping/
 ├── exploration_rl/
 ├── planning/
 ├── control/
 ├── safety/
 ├── sim_fast/
 ├── sim_vision/
 ├── telemetry/
 └── runtime/
```

---

# 21. Development Phases

1. hardware bring-up
2. Yahboom HAL protocol capture / documentation
3. IMU integration and calibration
4. bus + typed message definitions
5. executive state machine
6. network / GUI bridge skeleton
7. camera + event-driven depth pipeline
8. pseudo-lidar extraction
9. telemetry / replay system
10. micro-SLAM prototype
11. occupancy mapping from pseudo-lidar + pose
12. classical frontier detection + baseline heuristics
13. planner + controller
14. fast simulator
15. RL frontier selector
16. real robot integration
17. sim-to-real tuning

---

# 22. Expected Capabilities

The robot should be able to:

- explore rooms autonomously
- avoid furniture and walls
- map small indoor environments
- navigate between discovered spaces
- maintain useful local pose estimates without wheel encoders
- operate and be debugged without RL initially
- expose live map/state/health visualization during debug runs

Target expectations for version 1.4:

- reliable exploration in textured, well-lit rooms
- useful room-scale local maps
- graceful degradation under poor tracking conditions
- research-grade reliability, not product-grade autonomy
- stronger topological understanding than precise metric reconstruction

Expected realistic limitations:

- long-run global map drift may still occur
- low-texture scenes may degrade localization
- monocular depth noise may create occasional false or missed obstacles
- mecanum slip will reduce metric accuracy
- integration effort for Yahboom I2C control may be non-trivial

---

# 23. Out of Scope

- semantic object recognition
- multi-robot coordination
- outdoor navigation
- full loop-closure SLAM in initial release
