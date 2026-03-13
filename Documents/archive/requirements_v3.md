# Yahboom Raspbot V2 — Autonomous Indoor Exploration (Revised Architecture)
## Requirements Specification — Version 1.2 (V3)

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
- **Deterministic robotics:** perception, localization, mapping, planning, and control

This architecture maximizes reliability while maintaining adaptive behavior.

---

# 2. Key Design Principles

1. **Classical control for motion**
2. **Learning only for exploration strategy**
3. **Vision converted to pseudo-lidar for mapping**
4. **Micro-SLAM for pose estimation without encoders**
5. **Event-driven perception to reduce compute**
6. **Time-synchronized telemetry for debugging**
7. **Simulation-first development**
8. **Hardware-safe failover layers**
9. **Message-passing architecture to reduce coupling**
10. **Confidence-aware operation under degraded sensing**

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

## 5.2 Pseudo-Lidar Extraction

Depth image → lidar-like rays.

Typical configuration:

| Parameter | Value |
|---|---|
| Depth resolution | 192×192 to 256×256 |
| Lidar rays | 48 |
| Max range | 3 m |
| Update rate | 5–10 Hz |

Advantages:

- robust mapping
- reduced noise
- fast occupancy updates
- compatibility with classical robotics algorithms

## 5.3 Parallax Scans

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

# 6. Micro-SLAM / Localization

## 6.1 Purpose

Because wheel encoders are unavailable, the robot shall estimate pose using a lightweight
visual-inertial localization pipeline ("micro-SLAM") built from:

- monocular camera
- IMU (gyro + accel)
- short-horizon feature tracking
- sparse local map / keyframes

## 6.2 Scope

Micro-SLAM is intended to provide:

- short- to medium-term pose stability
- reliable heading estimation
- local drift reduction good enough for room-scale exploration
- keyframe-based local consistency

It is **not required** to provide full global loop-closure SLAM in version 1.2.

## 6.3 Functional Design

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

## 6.4 Performance Target

Micro-SLAM should run in real time on the Pi 5 at approximately:

- feature tracking: 10–15 Hz
- IMU propagation: 50–100 Hz
- pose output to mapper/controller: 10 Hz

## 6.5 Capability Expectation

Micro-SLAM is expected to be:

- much better than dead reckoning from motor commands
- good enough for room and hallway exploration
- still subject to drift over long paths or low-texture scenes

Expected limitations:

- blank walls may degrade tracking
- rapid lighting changes may reduce feature stability
- no guaranteed global consistency without later loop closure

## 6.6 Interfaces and Data Structures

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

## 6.7 Runtime Tasks

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

# 7. Mapping

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

---

# 8. Exploration Strategy

Exploration uses **frontier-based mapping**.

Frontiers are:

- boundaries between known and unknown space

The RL policy chooses **which frontier to explore next**.

This avoids learning low-level motion control and reduces sim-to-real fragility.

---

# 9. Reinforcement Learning Design

## 9.1 State

RL state should be based on geometry and exploration context, not raw motor control.

Example contents:

- local 64×64 occupancy window
- robot heading
- candidate frontier descriptors
- local free-space statistics

## 9.2 Actions

Select a frontier or frontier-selection heuristic.

Example:

| Action | Meaning |
|---|---|
| 0 | nearest frontier |
| 1 | largest frontier |
| 2 | leftmost frontier |
| 3 | rightmost frontier |
| 4 | random valid frontier |

## 9.3 Rewards

| Reward | Intent |
|---|---|
| New map area discovered | positive |
| Frontier reached | positive |
| Collision | large negative |
| Time penalty | small negative |

---

# 10. Motion Control

Classical deterministic navigation.

## 10.1 Planner

- A*

## 10.2 Controller

- pure pursuit

## 10.3 Recovery

If stuck:

1. rotate
2. reverse slightly if safe
3. select a new frontier
4. trigger a parallax scan if needed

---

# 11. Message Bus / Runtime Communication

## 11.1 Purpose

Subsystems shall communicate via typed messages over asynchronous channels rather than
tight direct coupling between modules.

Benefits:

- easier debugging
- cleaner ownership boundaries
- isolated subsystem testing
- simpler simulation/real-hardware swapping

## 11.2 Topic Set

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
safety/event
telemetry/event_marker
```

## 11.3 Message Bus Requirements

The runtime bus shall support:

- multiple subscribers per topic
- bounded queues
- timestamped messages
- non-blocking publish where practical
- graceful handling of slow subscribers
- drop detection / counters for debug builds

## 11.4 Implementation Guidance

A `tokio`-based implementation using `broadcast`, `watch`, or `mpsc` is acceptable.

Recommended pattern:

- high-rate sensor streams: bounded `broadcast`
- latest-state topics: `watch`
- command paths: bounded `mpsc`

---

# 12. Telemetry & Replay System

## 12.1 Goal

Every control cycle shall log enough synchronized data to reconstruct and replay robot behavior offline.

## 12.2 Logged Data

Telemetry should include:

- frame reference or compressed camera frame
- depth map and pseudo-lidar
- IMU samples or summaries
- micro-SLAM pose estimate
- occupancy map state or deltas
- frontier candidates
- RL decision
- planner path
- controller output
- motor commands
- ultrasonic readings
- event markers
- confidence / health indicators

## 12.3 Schema Requirements

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

## 12.4 Event Markers

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

## 12.5 Replay Requirements

Replay tooling shall support:

- time-synchronized sensor playback
- step-by-step timeline advance
- camera/depth/map side-by-side visualization
- comparison of alternate algorithms against recorded runs

---

# 13. Safety Layer

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

# 14. Operational Modes

| Mode | Purpose |
|---|---|
| hw-test | test hardware components |
| sim-train | RL training |
| sim-run | policy evaluation |
| robot-run | autonomous operation |
| robot-debug | autonomous run with heavy telemetry |
| slam-debug | camera/IMU localization validation |

---

# 15. Software Workspace Layout

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
 ├── sim_fast/
 ├── sim_vision/
 ├── telemetry/
 └── runtime/
```

---

# 16. Development Phases

1. hardware bring-up
2. IMU integration and calibration
3. bus + typed message definitions
4. camera + event-driven depth pipeline
5. pseudo-lidar extraction
6. telemetry / replay system
7. micro-SLAM prototype
8. occupancy mapping from pseudo-lidar + pose
9. planner + controller
10. fast simulator
11. RL frontier selector
12. real robot integration
13. sim-to-real tuning

---

# 17. Expected Capabilities

The robot should be able to:

- explore rooms autonomously
- avoid furniture and walls
- map small indoor environments
- navigate between discovered spaces
- maintain useful local pose estimates without wheel encoders

Expected realistic limitations in version 1.2:

- long-run global map drift may still occur
- low-texture scenes may degrade localization
- behavior should be considered research-grade, not product-grade

---

# 18. Out of Scope

- semantic object recognition
- multi-robot coordination
- outdoor navigation
- full loop-closure SLAM in initial release
