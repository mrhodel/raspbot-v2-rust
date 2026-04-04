# Yahboom Raspbot V2 — Autonomous Indoor Exploration (Revised Architecture)
## Requirements Specification — Version 1.6 (V5.4)

**Status:** Active — Phase 17 (sim-to-real tuning) in progress; motor swap 1:48 → 1:90 complete
**Last Updated:** 2026-04-03
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
| OS | Debian GNU/Linux 12 (bookworm), aarch64 |
| Drive | 4-wheel mecanum |
| Camera | USB 1 MP, V4L2 `/dev/video0` |
| Camera resolution | 640×480 @ 15 fps YUYV (320×240 not supported by driver) |
| Camera FOV | 110° horizontal (wide-angle, confirmed) |
| Camera height | ~10 cm (measured on wheels-up stand) |
| IMU | MPU-6050 — I2C-6, bit-banged 400 kHz, GPIO22/23 (pins 15/16), addr 0x68 |
| IMU config | ±8g accel (ACCEL_CONFIG=0x10), ±500°/s gyro (GYRO_CONFIG=0x08) |
| Battery monitor | INA226 — I2C-6, addr 0x40 — **not yet wired** |
| Ultrasonic | HC-SR04 forward, I2C-1 via Yahboom board (addr 0x2B) |
| ToF depth sensor | VL53L8CX (Pololu #3419) — I2C-2 (GPIO 4=SDA, GPIO 5=SCL), addr 0x29 |
| Wheel encoders | Not available — dead reckoning only |
| Motor gear ratio | 1:90 (swapped 2026-03-31; was 1:48) |
| Motor stall floor | 1:48 measured: ~10% (FR hardwood), ~15% (RL concrete); 1:90 not yet measured |

## 3.2 Camera Configuration

Default tilt:

- **+30° upward** (`tilt_home_deg: 30` in config; positive = up)

Allowed range:

- −30° to +30° (servo-limited; tilt_neutral raw servo value = 30 points level)

**Note on servo convention:** The tilt servo is mounted inverted — higher raw servo value = camera up. `tilt_home_deg` is expressed in physical degrees where positive = upward tilt.

Rationale for +30° up:

- The ToF sensor is mounted above the camera on the gimbal; tilting up keeps the ToF field of view above the floor, preventing false floor detections
- Horizon placed in the upper ~40% of the camera frame, giving the perception pipeline a useful mix of forward scene and near-floor view
- Obstacle detection relies primarily on the ToF and pseudo-lidar pipeline, not floor-plane geometry

## 3.3 Yahboom Expansion Board / HAL Assumption

The Yahboom platform uses a shared controller board accessed via I2C rather than separate direct drivers for each function.
The Rust HAL shall treat this as a **board-level command protocol** problem.

Implications:

- motor control
- servo / gimbal control
- ultrasonic reads

may all be implemented through a shared Yahboom I2C protocol.

**Motor I2C protocol (captured and documented):**

| Field | Value |
|---|---|
| I2C address | 0x2B |
| Register | 0x01 |
| Write format | `[motor_id, direction, speed]` (3 bytes) |
| motor_id | 0=FL, 1=RL, 2=FR, 3=RR |
| direction | 0=forward, 1=backward |
| speed | 0–255 (duty% × 255 / 100) |
| Inter-write delay | ~10 ms between successive motor writes |

Duty-cycle conversion: if duty ≥ 0 → dir=0, speed=duty×255/100; if duty < 0 → dir=1, speed=|duty|×255/100.

Example: FL forward 30% → `[0, 0, 77]`; FL backward 30% → `[0, 1, 77]`

**Mecanum forward kinematics:** all four motors same sign = translate; FL+RL positive, FR+RR negative = rotate CW.

## 3.4 ToF Sensor Specification

The VL53L8CX is a 64-zone (8×8) time-of-flight ranging sensor used as the primary near-field obstacle detector.

| Parameter | Value |
|---|---|
| Model | VL53L8CX (Pololu carrier board #3419) |
| I2C bus | 2 (`/dev/i2c-2`, GPIO 4=SDA, GPIO 5=SCL) |
| I2C address | 0x29 (7-bit) |
| Ranging mode | 1 (short range, ≤135 cm) |
| Integration time | 5 ms |
| Max scan rate | Up to 60 Hz |
| Output format | 8 rays spanning ±19.69° horizontal (5.625° column spacing) |
| Row filtering | Rows 0–7 (all rows) used at 30° tilt; configurable via `row_min`/`row_max` to exclude floor-facing zones |
| Safety cone | Inner 6 of 8 columns (±14.06°) used for nearest-obstacle detection |
| Per-ray value | Column minimum across used rows |
| Runtime output | `nearest_obstacle_m` (scalar) and 8-ray pseudo-lidar scan to mapper |

**Why ToF instead of relying solely on camera depth:**
The VL53L8CX provides metric ranging independent of lighting and texture — essential for safety-critical near-field detection where MiDaS monocular depth is unreliable at close range.

**Row filtering rationale:** At +30° camera/gimbal tilt, lower rows of the ToF FOV point toward the floor. `row_min`/`row_max` clip floor-facing rows to prevent the floor from registering as a close obstacle.

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

Expected reduction in neural-network inference: **70–85%**

**EventGate parameters:**

| Parameter | Value |
|---|---|
| Frame difference threshold | 8.0 (mean absolute pixel difference) |
| Max suppressed frames | 10 (force re-inference after 10 consecutive skips) |

When the gate returns false, the previous depth map is reused unchanged.

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

**Model and runtime details:**

| Parameter | Value |
|---|---|
| Model file | `models/midas_small.onnx` |
| Input spatial size | **256×256 pixels** (bilinear resize from camera frame) |
| Input normalisation | ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]` |
| Output size | Configurable (`depth_out_width` × `depth_out_height`); typical 32×32 for RL |
| Runtime | ONNX Runtime (`ort` crate) |
| Export script | `scripts/export_midas.py` (MiDaS v3.1 small-class) |

Performance target:

- raw depth inference on Pi 5 CPU is expected to be modest
- effective system behavior should rely on event-driven inference so that depth is available when needed at roughly **5–10 Hz effective decision support**, not necessarily 5–10 Hz raw model execution

## 5.3 Pseudo-Lidar Extraction

Depth image → 48 lidar-like rays for occupancy mapping.

| Parameter | Value |
|---|---|
| Ray count | 48 (hardcoded) |
| Horizontal FOV | 110° (camera spec) → ~2.29° per ray |
| Max range | 3.0 m |
| Min range | 0.20 m (clips floor artefacts and very close dark objects) |
| Update rate | 5–10 Hz effective (EventGate-limited) |
| Detection method | Contrast-based: ray obstacle only if depth exceeds scene mean by ≥ 0.10 (contrast threshold) |
| Row usage | Top 75% of depth frame; bottom 25% excluded to suppress floor returns |
| Per-ray extraction | Column minimum across used rows |

**Detection rationale:** A simple nearest-depth rule incorrectly triggers on the floor in front of the robot. The contrast threshold (`depth > scene_mean + 0.10`) rejects near-uniform-depth scenes (open floor, blank walls) and only flags genuine foreground obstacles.

Advantages:

- robust mapping independent of texture
- reduced false positives from floor and ceiling
- fast occupancy updates
- compatible with classical robotics planning algorithms

## 5.4 Floor-Plane Anchoring

**Status: Not implemented** (post-Phase-17 gap).

Where practical, the perception stack could use known geometry to improve metric consistency:

- camera height above floor (~10 cm)
- camera tilt angle (+30° up)
- estimated floor plane

**Note:** With the camera tilted +30° upward and the bottom 25% of the depth frame excluded (§5.3), floor-plane anchoring is lower priority than originally planned. The ToF sensor (§3.4) provides reliable metric near-field ranging independently of depth model geometry.

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

## 6.5 Kinematics Calibration Results

Measured on 2026-03-18 with 1:48 gear ratio motors (wheels on garage floor).
Theoretical 1:90 values scaled by 48/90 ≈ 0.533 — **re-measure after physical swap**.

| Metric | 1:48 Measured | 1:90 Theoretical | Config key |
|---|---|---|---|
| Forward speed @ 30% duty | 0.484 m/s | — | — |
| Forward speed @ 100% duty | 1.61 m/s | **0.858 m/s** | `kinematics.forward_speed_m_s` |
| Rotation rate @ 30% duty (wheels-on) | 4.103 rad/s | — | — |
| Rotation rate @ 100% duty (wheels-on) | 13.7 rad/s | **7.30 rad/s** | `kinematics.rotation_rate_rad_s` |
| Escape rotation duration (110° @ 35%) | 400 ms | **750 ms** | `safety.escape_rotation_ms` |
| Navigation forward cap (`max_vx`) | 0.3 m/s | 0.3 m/s (unchanged) | `kinematics.max_vx` |
| Navigation omega cap (`max_omega_rad_s`) | 4.0 rad/s | 4.0 rad/s (unchanged) | `kinematics.max_omega_rad_s` |

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

## 7.5 Arm / Disarm *(implemented)*

The robot starts in `Idle` on every launch. Motors are disarmed — CmdVel commands are silently dropped until explicitly armed.

**Arm mechanism:** SIGUSR1 sent to the robot process (`kill -USR1 <pid>`).

Arm transitions:
- `Idle` → `Exploring` (normal arm)
- `SafetyStopped` → `Idle` → `Exploring` (re-arm after estop)

After an emergency stop the robot stays in `SafetyStopped` and must be re-armed with a second SIGUSR1. It does **not** automatically resume when an obstacle is cleared.

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
| LOG_ODDS HIT | +1.4 (tuned 2026-03-18; was +0.7) |
| LOG_ODDS MISS | −0.70 (tuned 2026-03-18; was −0.35) |
| MIN_CONFIDENCE | 0.02 — minimum ray confidence required to update a cell (tuned 2026-03-18; was 0.1) |
| Decay rate | 0.01 per step (robot_config.yaml; was 0.001 — 10× faster fade of false obstacles) |
| Free threshold | −0.5 log-odds (cell considered navigable) |
| Occupied threshold | +0.5 log-odds (cell considered obstacle) |
| Log-odds clamp | [−3.0, +3.0] (prevents saturation) |
| Sim boundary guard | Max-range MISS rays skip a 1 m (20-cell) buffer at arena edge to prevent false frontiers at seeded walls |

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

A* on the occupancy grid with configurable clearance inflation.

| Parameter | Value |
|---|---|
| Default clearance | 7 cells (35 cm) — safe navigation in open space |
| Tight-corridor fallback | 5 cells (25 cm) — attempted when clearance=7 finds no path |
| Max node budget | 50,000 nodes per A* call |
| Waypoint downsampling | Every 8 cells (40 cm) along path |
| Goal BFS pre-screen | Reachability checked before A* to avoid wasted planning |

If clearance=7 fails and clearance=5 succeeds, the path is flagged as `tight`. If the robot makes no progress on a tight-corridor path, the goal is hard-blacklisted for 10× the normal duration.

## 12.2 Controller

Pure pursuit with heading-error speed scaling.

| Parameter | Value | Config key |
|---|---|---|
| Lookahead distance | 0.20 m | `kinematics.lookahead_dist_m` |
| Max forward speed | 0.3 m/s | `kinematics.max_vx` |
| Max angular velocity | 2.2 rad/s (1:90 motors) | `kinematics.max_omega_rad_s` |
| Heading proportional gain | 2.0 rad/s per radian | hardcoded |
| US slow zone | 0.70 m — reduce speed | `kinematics.us_slow_m` |
| US stop zone | 0.30 m — stop forward motion | `kinematics.us_stop_m` |
| Speed scaling range | Linear over ±120° heading error | hardcoded |

## 12.3 Recovery

**Collision recovery** is a two-phase state machine triggered on any collision event:

1. **BackingUp** — reverse at −0.50 m/s until the forward ultrasonic reads > 0.3 m (exit the safety zone). US interlock does not block reverse motion.
2. **Spinning** — spin at ±1.5 rad/s for 750 ms, direction chosen away from `nearest_obstacle_angle` (obstacle on left → spin CW; obstacle on right → spin CCW). Breaks heading deadlock.
3. **Recovery complete** — clear path flag set, replanning requested.

**Collision cascade suppression:** duplicate collision events within a 500 ms window are ignored to prevent the robot re-triggering recovery while still in contact with the wall.

**Tight-corridor hard-blacklisting:** if the planner used its clearance=5 fallback (tight corridor) and the robot still made no progress toward the goal, the goal is hard-blacklisted for 10× the normal blacklist duration. Prevents repeated navigation into bottlenecks the robot cannot traverse at safe clearance.

**Safety-task escape sequence** (triggered by US emergency stop):

| Phase | Duration | Config key |
|---|---|---|
| Stop (latch) | 5.0 s total latch | `safety.emstop_latch_s` |
| Delay before reverse | 200 ms | `safety.escape_delay_ms` |
| Reverse | 300 ms at 35% duty | `safety.escape_duration_ms` / `escape_reverse_spd` |
| Rotate away from obstacle | 870 ms at 35% duty | `safety.escape_rotation_ms` |

Rotation direction is chosen away from `nearest_obstacle_angle_rad`: obstacle on left → rotate CW; obstacle on right → rotate CCW. Motor commands sent every 50 ms during escape phases.

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
vision/features          (defined; unused until visual SLAM implemented)
vision/tracks            (defined; unused until visual SLAM implemented)
slam/visual_delta        (defined; unused until visual SLAM implemented)
slam/pose2d
slam/keyframe_event      (defined; unused until visual SLAM implemented)
map/grid_delta
map/frontiers
map/explored_stats
decision/frontier_choice
planner/path
controller/cmd_vel
motor/command
executive/state
safety/event
safety/nearest_obstacle_m        (ToF scalar nearest range)
safety/nearest_obstacle_angle    (ToF bearing to nearest obstacle, rad)
safety/estop_count               (cumulative emergency stop counter)
safety/collision_count           (cumulative IMU crash spike counter)
gimbal/pan_deg
gimbal/tilt_deg
ui/manual_cmd_vel                (joystick velocity from UI bridge)
ui/manual_gimbal_cmd             (pan/tilt command from UI bridge)
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

## 16.1 Goal

Every control cycle shall log enough synchronized data to reconstruct and replay robot behavior offline.

## 16.2 Log Formats

Two distinct logging layers exist:

**Operational log (stdout / log file)**
- Framework: structured `tracing` (Rust) or `rclpy` logging (ROS2)
- Levels: INFO, WARN, ERROR with key=value structured fields
- Destination: stdout + `~/robot_ws/robot.log` on Pi; `/tmp/robot_sim.log` in sim
- Purpose: real-time diagnostics, crash analysis

**Telemetry log (data recording)**
- Current format: **NDJSON** (newline-delimited JSON), one record per line
- ROS2 equivalent: **rosbag2** (replaces the custom NDJSON writer entirely)
- Purpose: offline replay, algorithm comparison, training data collection

For a ROS2 implementation, `ros2 bag record` and `ros2 bag play` replace the custom telemetry writer and replay tooling described in §16.5.

## 16.3 Logged Data

Telemetry should include:

- frame reference or compressed camera frame
- depth map and pseudo-lidar scan
- IMU samples or summaries
- micro-SLAM pose estimate
- occupancy map state or deltas
- frontier candidates
- chosen frontier strategy or RL decision
- planner path
- controller output (CmdVel)
- motor commands
- ultrasonic and ToF readings
- executive state transitions
- event markers (§16.4)
- confidence / health indicators

## 16.4 Event Markers

The system shall record structured event markers, including at least:

- collision (IMU spike)
- emergency stop
- frontier selected
- frontier reached
- tracking confidence drop
- planner failed
- recovery behavior triggered
- executive state changed

## 16.5 Health Telemetry

Health telemetry should include, where available:

- Pi temperature
- CPU utilization or load indicator
- depth inference timing
- micro-SLAM timing
- dropped-frame counters
- I2C error counters
- battery voltage (INA226, not yet wired)
- throttle / undervoltage indicators if available

**Status: Not implemented** — `HealthMetrics` struct defined but no publishing task.

## 16.6 Telemetry Record Schema

Each record shall include:

- monotonic timestamp (ms)
- topic name
- source subsystem
- payload
- optional sequence number

```python
# Python / ROS2 equivalent
@dataclass
class TelemetryRecord:
    t_ms: int
    topic: str
    source: str
    seq: int
    payload: dict
```

## 16.7 Replay

**Rust implementation:** NDJSON log writer done; step-by-step replay and offline comparison not implemented.

**ROS2 implementation:** use `ros2 bag record` to capture all topics during a run and `ros2 bag play` for replay. No custom tooling required.

---

# 17. HAL Scheduling and Priority Policy

Because the Yahboom board likely multiplexes control and sensing over I2C, the HAL shall treat bus access as a schedulable shared resource.

## 17.1 Priority Order

Suggested priority order:

1. emergency stop / safety halt
2. motor commands
3. gimbal commands
4. critical status reads needed for control
5. ultrasonic reads
6. low-priority diagnostics / telemetry-only reads

## 17.2 Design Guidance

An internal HAL queue or scheduler is recommended so that urgent commands are not delayed by routine sensor polling.

---

# 18. Safety Layer

Hardware safety is independent of AI.

**Safety thresholds:**

| Threshold | Value | Config key |
|---|---|---|
| US emergency stop | 20 cm | `safety.emergency_stop_cm` |
| US slow zone | 0.70 m — controller reduces speed | `kinematics.us_slow_m` |
| US stop zone | 0.30 m — controller halts forward motion | `kinematics.us_stop_m` |
| EmStop latch duration | 5.0 s | `safety.emstop_latch_s` |
| Motor watchdog timeout | 500 ms | `motor.watchdog_ms` |
| Crash accel threshold | 15.0 m/s² (IMU horizontal) | `agent.crash.accel_threshold_m_s2` |
| Crash debounce | 2000 ms | `agent.crash.debounce_ms` |

**Note on emergency stop distance:** 20 cm accounts for gimbal protrusion (~5 cm), giving ~15 cm of physical clearance at the robot body.

Triggers:

- ultrasonic < 20 cm
- watchdog timeout (500 ms without a US reading)
- IMU crash spike > 15 m/s²
- explicit emergency stop (executive command)

Response:

- immediate motor halt via `MotorCommand::stop()` on priority channel
- escape sequence executes (§12.3)
- executive transitions to `SafetyStopped` (blocks all further CmdVel)

The safety subsystem shall also publish a machine-readable event to `safety/event`.

**US interlock is forward-only:** the ultrasonic proximity stop does not block reverse motion, allowing collision recovery to back away from an obstacle unimpeded.

**Collision cascade suppression:** a 500 ms cooldown after any collision event suppresses duplicate triggers caused by the robot body still contacting the wall during the backup maneuver.

## 18.1 Motor Crash-Stop *(implemented)*

`YahboomMotorController` implements `Drop`. On drop (normal shutdown, SIGTERM, panic unwind) all four motors are synchronously zeroed via blocking I2C writes (~40 ms). SIGKILL cannot be caught.

## 18.2 Motor Watchdog *(implemented)*

The motor execution task sends a zero command if no motor command (safety or CmdVel) is received within **500 ms**. Covers: control loop stall, task hang, or any bug that stops the command pipeline without crashing the process.

---

# 19. Operational Modes

| Mode | Purpose | Status |
|---|---|---|
| hw-test | test individual hardware components | **Not implemented** |
| calibrate | IMU bias collection | **Partial** (IMU only; no camera/extrinsics) |
| sim-train | RL training (Python-based, separate pipeline) | **Not in runtime** |
| sim | Fast 2D simulator with exploration | **Implemented** |
| robot-run | autonomous operation | **Implemented** |
| robot-debug | autonomous run with heavy telemetry | **Same as robot-run** |
| slam-debug | camera/IMU localization validation, no autonomy | **Implemented** |

Mode is selected via first CLI argument: `./robot <mode>`. Unknown modes fall through to `robot-run`.

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

| # | Phase | Status |
|---|---|---|
| 1 | hardware bring-up | **Done** |
| 2 | Yahboom HAL protocol capture / documentation | **Done** |
| 3 | IMU integration and calibration | **Done** (MPU-6050, I2C-6 bit-banged) |
| 4 | bus + typed message definitions | **Done** |
| 5 | executive state machine | **Done** (arm/disarm via SIGUSR1) |
| 6 | network / GUI bridge skeleton | **Done** |
| 7 | camera + event-driven depth pipeline | **Done** (640×480, MiDaS ONNX, EventGate) |
| 8 | pseudo-lidar extraction | **Done** (48-ray) |
| 9 | telemetry / replay system | **Done** (NDJSON writer; replay tooling TODO) |
| 10 | micro-SLAM prototype | **Partial** (IMU dead-reckoning; visual-inertial pipeline TODO) |
| 11 | occupancy mapping from pseudo-lidar + pose | **Done** |
| 12 | classical frontier detection + baseline heuristics | **Done** |
| 13 | planner + controller | **Done** (A* + pure pursuit) |
| 14 | fast simulator | **Done** (sim_fast 2D, 200×200 grid) |
| 15 | RL frontier selector | **Done** (AWR training pipeline + classical fallback) |
| 16 | real robot integration | **Done** (runtime HAL wired, motor safety gates) |
| 17 | sim-to-real tuning | **In progress** |

### Phase 17 sub-phases

| # | Sub-phase | Status |
|---|---|---|
| 14.2.1 | Obstacle maze validation (18-obstacle sim) | **Done** |
| 14.2.2 | HW safety tests (5 systems) | **Done** |
| 14.2.3 | Kinematics calibration (forward speed, rotation rate) | **Done** — 1:48 motors measured 2026-03-18 |
| 14.2.4 | Wheels-on rotation re-measurement | **Done** — 13.7 rad/s confirmed |
| 14.2.5 | Occupancy grid & collision recovery fixes | **Done** — LOG_ODDS, decay, cascade suppression |
| 14.2.6 | Runtime modularization (main.rs 2613 → 2123 lines) | **Done** — control/safety/mapping tasks split out |
| 14.2.7 | Distance-based two-phase collision recovery | **Done** — BackingUp + Spinning state machine |
| 14.2.8 | Motor swap 1:48 → 1:90; theoretical constants updated | **Done** — re-measure pending physical swap |

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

---

# 24. Implementation Gap Analysis *(as of 2026-04-03)*

| § | Feature | Status | Notes |
|---|---|---|---|
| 3.4 | ToF sensor (VL53L8CX) | **Done** | 8-ray, ±19.69°, I2C-2/0x29, safety + mapping |
| 5.1 | Event-driven depth (EventGate) | **Done** | Threshold=8.0, max suppressed=10 |
| 5.3 | Pseudo-lidar extraction (48-ray) | **Done** | 110° FOV, contrast-based, 0.20m min range |
| 5.4 | Floor-plane anchoring | **Deprioritised** | Camera tilted up +30°; ToF handles near-field metric ranging |
| 5.5 | Parallax scans | **Missing** | No strafe-scan logic in perception or runtime |
| 6 | Calibration subsystem | **Partial** | IMU bias calibration implemented; no camera intrinsics or extrinsics routines |
| 6.3 | `calibrate` operational mode | **Partial** | Mode dispatch implemented; IMU bias only |
| 7 | Executive state machine | **Done** | arm/disarm via SIGUSR1 (§7.5) |
| 8 | Micro-SLAM (visual-inertial) | **Partial** | `ImuDeadReckon` (gyro+cmdvel feedforward, confidence fixed 0.3); feature detection / optical flow / keyframes not implemented |
| 9 | Mapping + frontier detection | **Done** | |
| 10 | A* planner + pure pursuit | **Done** | Clearance=7 normal, clearance=5 tight fallback |
| 11 | RL frontier selector | **Done** | AWR training + classical fallback |
| 13.1 | sim_fast (2D fast simulator) | **Done** | RoomKind: Empty / Random(N) / SingleBox |
| 13.2 | sim_vision (3D visual simulator) | **Missing** | Crate exists as pure stub |
| 14 | Network / GUI bridge | **Done** | WebSocket, live map/pose/frontier/manual-drive |
| 15.3 | Health telemetry (Pi temp, CPU, battery) | **Partial** | `HealthMetrics` struct defined; no publishing task; INA226 not wired |
| 15.6 | Replay tooling | **Partial** | NDJSON log writer done; step-by-step replay not implemented |
| 17 | HAL I2C priority queue | **Missing** | Safety priority enforced by tokio biased-select only; no explicit bus arbiter |
| 19 | hw-test mode | **Missing** | Not implemented |

### Gaps blocking Phase 17 (sim-to-real tuning)

1. **Micro-SLAM visual pipeline** — IMU+cmdvel dead reckoning drifts too fast for useful mapping beyond one room; visual-inertial tracking is the next major subsystem
2. **HAL I2C priority** — currently acceptable via biased-select; revisit if safety commands lag during heavy I2C use

### Lower-priority gaps (post Phase 17)

- Parallax scans (§5.5) — depth confidence in ambiguous scenes
- Health telemetry publication (§15.3) — diagnostics during long runs
- Replay tooling (§15.6) — offline debugging and algorithm comparison
- sim_vision stub (§13.2) — perception/SLAM validation
- Motor stall floor re-measurement for 1:90 motors
- hw-test operational mode (§19)
- Camera intrinsics and extrinsics calibration routines (§6)
