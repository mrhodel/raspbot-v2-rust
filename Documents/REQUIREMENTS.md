# Yahboom Raspbot V2 — Autonomous Navigation (Rust)
### Requirements Specification — Living Document

| Field | Value |
|---|---|
| Status | **Draft — In Review** |
| Version | 0.2 |
| Last Updated | 2026-03-08 |
| Authors | Mike, Claude |

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Hardware Specification](#2-hardware-specification)
3. [System Architecture](#3-system-architecture)
4. [Simulation Design](#4-simulation-design)
5. [Perception Pipeline](#5-perception-pipeline)
6. [RL Design — Action Space, State & Reward](#6-rl-design)
7. [Navigation](#7-navigation)
8. [Operational Modes](#8-operational-modes)
9. [Implementation Phases](#9-implementation-phases)
10. [Decisions Log](#10-decisions-log)
11. [Open Questions](#11-open-questions)
12. [Change Log](#12-change-log)

---

## 1. Project Overview

Build a fully autonomous indoor navigation system for the Yahboom Raspbot V2 robot car, implemented in Rust. The robot must navigate around a home or garage environment using its gimbal-mounted camera as the primary sensor. The ultrasonic sensor is used solely as a hardware safety interlock.

**Primary goal:** Explore and navigate unknown indoor spaces without collisions, using vision-derived depth as the main obstacle sensor.

**Non-goals (current scope):**
- Object identification or semantic labeling
- Multi-robot coordination
- Outdoor navigation
- SLAM with loop closure

---

## 2. Hardware Specification

### 2.1 Platform
| Parameter | Value |
|---|---|
| Robot model | Yahboom Raspbot V2 |
| Compute | Raspberry Pi 5, 16 GB RAM |
| Drive type | Mecanum (4-wheel, holonomic) |
| Motor duty-cycle range | 20–80 (below 20 = stall) |
| Max duty-cycle | 80 |
| Robot body length | ~25 cm |
| Camera mount height | ~10 cm above ground (measured with robot on test stand) |

### 2.2 I2C Bus Layout
All hardware peripherals share a single Yahboom expansion board:

| Peripheral | I2C Bus | Address |
|---|---|---|
| Motor driver | 1 | 0x2B |
| Ultrasonic (HC-SR04) | 1 | 0x2B |
| Gimbal servos | 1 | 0x2B |

### 2.3 Ultrasonic Sensor
- Fixed, forward-facing only (does not move with gimbal)
- Range: 3–400 cm
- Role: **Hardware safety interlock only** — not an RL input (see §6.1)
- Median filter: 3 samples per reading

### 2.4 Gimbal
- 2-DOF: pan (horizontal) and tilt (vertical)
- Pan range: ±90° from centre
- Tilt range: −45° (down) to +30° (up); neutral (level) = 30° raw servo angle
- Max angular velocity: 120 °/s (software-limited)
- Driven by Yahboom expansion board via I2C (not GPIO/PCA9685)

### 2.5 Camera
- 1 MP USB camera, mounted on gimbal
- Interface: V4L2 (`/dev/v4l/by-id/usb-DHZJ-...`)
- Capture resolution: 320×240 @ 15 fps
- **Horizontal FOV: 110° (wide-angle)**
- Approximate vertical FOV: ~83° (estimated from 4:3 aspect ratio and 110° H-FOV)
- Stream: MJPEG on port 8080 (optional, for remote monitoring)

### 2.6 Camera Geometry & Robot Body Masking

The camera sits on the gimbal at the front of the robot (~10 cm high, 110° H-FOV).
At neutral tilt (level), the bottom edge of the frame is ~41° below horizontal.
At that angle with a 10 cm mount height, the bottom of the frame sees the floor
approximately **11.5 cm ahead of the camera**. Because the camera is at the front of the
25 cm body, the robot's own chassis is mostly behind the camera mount and unlikely to
appear in the forward-looking frame at neutral tilt. However, the gimbal arm and front
bumper may appear at the very bottom rows at downward tilt angles.

**Geometric estimate:** bottom ~15–20% of rows (rows 192–240 of a 240-row frame)
may show the gimbal mount or chassis at steep downward tilt. A conservative initial
mask of **bottom 20% (48 rows)** will be used until hardware calibration.

- **Implication for depth map:** masked rows are zeroed before MiDaS input and
  excluded from RL state and proximity reward calculations
- **Sim equivalent:** the virtual camera model must reproduce this occlusion region
- **Test method:** robot placed on wheels-up stand (available) for safe static camera tests

> **TODO (Phase 5 — Camera HW test):** Photograph the robot (on wheels-up stand) at
> neutral tilt and at maximum downward tilt (-45°). Measure the pixel row where the
> chassis/gimbal arm begins and update the mask row constant in `robot_config.yaml`.

---

## 3. System Architecture

### 3.1 Crate Workspace Layout
```
robot/                         ← Cargo workspace root
├── Cargo.toml
├── robot_config.yaml          ← Runtime configuration (no constants in code)
├── crates/
│   ├── config/                ← YAML config loader → typed structs (shared)
│   ├── hal/                   ← Hardware abstraction layer
│   │   ├── motor.rs           ← Mecanum drive
│   │   ├── ultrasonic.rs      ← HC-SR04 via I2C
│   │   ├── gimbal.rs          ← Pan/tilt servo control
│   │   └── camera.rs          ← V4L2 capture + MJPEG stream
│   ├── perception/            ← Vision pipeline
│   │   ├── midas.rs           ← MiDaS depth inference (ONNX)
│   │   └── features.rs        ← MobileNetV2 feature extraction (future)
│   ├── mapping/               ← Probabilistic occupancy grid
│   ├── navigation/            ← A* planner + pure-pursuit controller
│   ├── rl/                    ← PPO agent (train + infer)
│   └── sim/                   ← Bevy-based 2D/3D simulator
├── src/
│   └── main.rs                ← Agent orchestrator binary
└── models/
    ├── midas_small.onnx
    └── checkpoints/
```

### 3.2 Key Technology Choices
| Concern | Choice | Rationale |
|---|---|---|
| Language | Rust | Performance, safety, single binary deployment |
| ML inference | `ort` (ONNX Runtime) | Best model compatibility; models pre-exported from PyTorch |
| I2C / GPIO | `rppal` | Pi-native, well-maintained |
| Camera | `v4l` crate (V4L2) | Direct kernel interface, no OpenCV dependency |
| Simulator | Bevy (game engine) | All-Rust, renders 3D camera view, z-buffer depth available |
| Async runtime | `tokio` | Concurrent tasks: camera capture, I2C polling, control loop |
| ROS2 | **Not used** | Adds build complexity; standalone Rust binary preferred |
| Visualization | `rerun` (optional) | Lightweight telemetry without ROS |

### 3.3 Runtime Concurrency Model
```
tokio runtime
├── Task: camera_capture      (15 Hz)  → shared frame buffer
├── Task: sensor_poll         (20 Hz)  → US reading + gimbal angle
├── Task: gimbal_controller   (10 Hz)  → reactive, reads depth map
├── Task: control_loop        (10 Hz)  → perception → RL → motor commands
└── Task: mjpeg_stream        (async)  → optional HTTP stream
```

---

## 4. Simulation Design

### 4.1 Simulator: Bevy-Based 3D Environment

The simulator is implemented in Rust using the Bevy game engine. It provides:
- A 3D rendered environment with a virtual camera matching the real robot's gimbal camera
- Ground-truth z-buffer depth (available directly from the renderer)
- MiDaS depth inference run on the rendered RGB frame (for sim-to-real transfer)
- Simple rigid-body kinematics (no physics engine needed for flat-floor navigation)

### 4.2 Environment Generation

Environments are procedurally generated at the start of each training episode:
- Rectangular room with configurable size range
- Random interior walls (maze-like partitions)
- Random obstacle objects: boxes and cylinders representing furniture (chairs, tables, etc.)
- Robot spawns at a random free cell with a random heading
- Minimum clearance around spawn point guaranteed

**Environment parameters (configurable):**
| Parameter | Range |
|---|---|
| Room width | 4–10 m |
| Room depth | 4–10 m |
| Wall segments | 2–6 per episode |
| Obstacle objects | 3–10 per episode |
| Obstacle types | Box (table), cylinder (chair leg/bin) |

### 4.3 Virtual Camera & Robot Geometry

The virtual camera must match the real robot's physical configuration:
- **Position:** mounted at the front of the robot, `camera_forward_offset_m` ahead of the robot centre
- **Height:** `camera_height_m` above the ground plane
- **Field of view:** match real camera horizontal FOV (to be measured on hardware)
- **Gimbal articulation:** pan and tilt angles applied to camera transform
- **Robot body mask:** same mask region as real camera (§2.6) applied to rendered frames

> Robot body occlusion in the virtual camera view must match the real robot.
> This requires accurate modelling of the robot's physical length and camera mount height.
> Values to be calibrated during Phase 5 (Camera HW test) and updated here.

### 4.4 Depth Strategy: MiDaS in Simulation

MiDaS small is run on the rendered RGB frame rather than using raw z-buffer depth.
This improves sim-to-real transfer since the policy learns to interpret MiDaS output,
not perfect ground-truth depth.

**Timing estimates (MiDaS small, 256×256 input):**

| Hardware | MiDaS inference | Vision every N=3 steps | Per episode (500 steps) | 1000 episodes |
|---|---|---|---|---|
| Dev CPU (modern) | ~60 ms | ~167 calls | ~10 s overhead + sim | ~3–4 hours |
| RTX 3050 (training machine) | ~5 ms | ~167 calls | ~1 s overhead + sim | ~20–30 min |
| Pi 5 (inference only, deploy) | ~15–20 ms | — | — | — |

**Training machine: Nvidia RTX 3050** — confirmed available. ~1000 episodes estimated in
20–30 minutes; 10,000 episodes (sufficient for a navigation policy) in ~3–5 hours.

**Implications:**

- Train on dev machine (RTX 3050), deploy checkpoint to Pi 5 for fine-tuning
- `vision_every_n_steps = 3` is the default; benchmark in Phase 6 to confirm optimal value
- A "fast mode" option (z-buffer depth only, no MiDaS) should be available for rapid
  environment/reward iteration before committing to full MiDaS training runs

### 4.5 Sim-to-Real Transfer Strategy

| Sim element | Real robot equivalent |
|---|---|
| Bevy renderer RGB → MiDaS depth | Camera RGB → MiDaS depth |
| Procedural wall/obstacle geometry | Actual walls, furniture |
| Dead-reckoning odometry (command integration) | Same (no encoders) |
| Gimbal angle from servo command | Servo command feedback |
| Collision = any body overlap | US < 15 cm OR depth map saturation |

---

## 5. Perception Pipeline

### 5.1 MiDaS Depth Estimation
- Model: MiDaS small (exported to ONNX)
- Input: 256×256 RGB (resized from 320×240 capture)
- Output: inverse-depth map, normalised to [0, 1] (1 = closest)
- Robot body mask applied **before** passing to RL state
- Downsampled to **32×32** for RL state input (1024 values)
- Run every N steps (configurable, default 3)

### 5.2 MobileNetV2 Feature Extraction
- **Status: Deferred.** The initial design excludes MobileNetV2; MiDaS depth map is
  the sole visual RL input. MobileNetV2 may be reintroduced if the policy needs richer
  visual features beyond depth.

### 5.3 Object Detection (YOLOv5)
- **Status: Removed from scope.** Object distance and size are derived from the MiDaS
  depth map. Semantic classification is not required.

### 5.4 Sensor Fusion
- US sensor is **not fused into RL state**. It operates as an independent safety layer.
- Depth map is the sole obstacle representation fed to the policy.

---

## 6. RL Design

### 6.1 Design Philosophy

The Python predecessor used the ultrasonic sensor as the primary obstacle sensor, with
the camera as a secondary feature extractor. This caused the policy to over-rely on the
single forward US beam and under-utilise the wide-angle camera view.

**The Rust redesign inverts this hierarchy:**
- The **camera depth map is the primary obstacle sensor** (wide angle, 2D spatial info)
- The **US sensor is a hardware safety interlock only** — it triggers an unconditional
  emergency stop below `emergency_stop_cm` (15 cm) and is invisible to the RL policy
- **Gimbal control is separated from the RL action space** — a reactive gimbal controller
  steers the camera toward the direction of maximum free space, ensuring the policy
  always has a useful view without spending actions on gimbal management

### 6.2 Action Space (6 discrete actions)
| ID | Action | Motor behaviour |
|---|---|---|
| 0 | Forward | All wheels forward |
| 1 | Rotate left | Left wheels back, right wheels forward |
| 2 | Rotate right | Left wheels forward, right wheels back |
| 3 | Strafe left | Mecanum lateral left |
| 4 | Strafe right | Mecanum lateral right |
| 5 | Stop | All wheels stop |

Removed from Python version: `backward`, `gimbal_pan_*`, `gimbal_tilt_*`.
- Backward removed: rarely useful for exploration, prone to wall-backing exploit
- Gimbal actions removed: handled by reactive gimbal controller (see §6.5)

Speed constants (duty-cycle):
| Action | Speed |
|---|---|
| Forward | 55 |
| Rotate | 45 |
| Strafe | 35 |

### 6.3 State Space (1031 values)
| Component | Size | Source | Notes |
|---|---|---|---|
| Depth map (flattened, masked) | 1024 | MiDaS 32×32 | Robot body rows zeroed |
| Gimbal pan angle (sin, cos) | 2 | Servo command | Normalised |
| Robot velocity estimate (vx, vy, ω) | 3 | Dead reckoning | From motor commands + timing |
| Heading (sin θ, cos θ) | 2 | Integrated odometry | |
| **Total** | **1031** | | |

Removed from Python version: US rays (not an RL input), local occupancy map window
(simplification for initial training; may reintroduce).

### 6.4 Reward Structure
| Signal | Value | Condition |
|---|---|---|
| Exploration bonus | +0.02 per cell | Newly revealed occupancy cell |
| Visual clearance bonus | +0.1 | Mean depth in central 1/3 of frame > 0.6 (open space ahead) |
| Proximity penalty | −2.0 × (fraction of depth map pixels > 0.8) | Camera-derived, not US |
| Collision penalty | −50.0 | Any collision (sim body overlap; robot US < 15 cm on hardware) |
| Time step cost | −0.01 | Every step |
| Spin penalty | −0.3 per step | After 3+ consecutive rotate actions |

**Key change from Python version:** Proximity penalty is derived entirely from the depth map
(fraction of visible pixels indicating close obstacles), not from the US reading. This forces
the policy to learn obstacle avoidance from the camera.

### 6.5 Reactive Gimbal Controller

The gimbal is controlled by a separate low-level reactive controller, not the RL policy:

```
Each control tick:
  1. Divide the current 32×32 depth map into left / centre / right columns
  2. Compute mean free space (low depth value) in each column
  3. Steer pan angle toward the column with the most free space
     (proportional control with max ±10°/tick, clamped to pan_range)
  4. Hold tilt at a fixed look-ahead angle (tuned for camera height and
     typical obstacle height — to be set during HW calibration)
```

This ensures the camera is always looking toward navigable space, providing the RL
policy with useful depth information without consuming action budget on gimbal management.

### 6.6 PPO Hyperparameters
| Parameter | Value |
|---|---|
| Learning rate | 3.0e-4 |
| Gamma | 0.99 |
| GAE lambda | 0.95 |
| Clip epsilon | 0.2 |
| Value loss coeff | 0.5 |
| Entropy coeff | 0.01 |
| Epochs per update | 4 |
| Batch size | 64 |
| Rollout steps (sim) | 512 |
| Rollout steps (hardware) | 64 |
| Hidden dim | 256 |
| Max steps per episode | 500 |
| Checkpoint interval | 10 episodes |

---

## 7. Navigation

### 7.1 Occupancy Grid Mapping
- Resolution: 5 cm/cell
- Initial size: 200×200 cells (10 m × 10 m); expands dynamically
- Probabilistic update (log-odds): p_occ_hit=0.75, p_occ_miss=0.40
- Obstacle inflation: Gaussian blur, σ=0.8 cells, every 10 steps
- Decay toward prior (0.5) at rate 0.001/step

### 7.2 Path Planner
- Algorithm: A* with Euclidean heuristic
- Safety margin: 2 cells from obstacles
- Path smoothing: 3 iterations
- Replan on: path blocked, goal reached, timeout

### 7.3 Pure-Pursuit Controller
| Parameter | Value |
|---|---|
| Lookahead distance | 0.20 m |
| Goal tolerance | 0.10 m |
| Angle Kp | 1.2 |
| Linear Kp | 0.8 |
| Max angular velocity | 60 (duty-cycle units) |
| Max linear velocity | 70 (duty-cycle units) |

### 7.4 Recovery Behaviour
On stuck detection (no progress for N steps):
1. Rotate 30°
2. Brief reverse if US permits (≥ 30 cm rear estimate — note: no rear sensor,
   so reverse is only attempted after a timed spin to face rearward)
3. Up to 5 attempts before declaring navigation failure

### 7.5 Frontier-Based Exploration
- Frontier: boundary between known-free and unknown cells
- Minimum frontier size: 3 cells
- Goal selection: nearest frontier centroid
- Gimbal sweep (full pan) every 5 seconds during EXPLORE phase
- Gimbal sweep suppressed during EXECUTE (path following) phase

---

## 8. Operational Modes

| Mode | Description |
|---|---|
| `hw-test` | Individual hardware component tests (wheels, gimbal, US, camera) |
| `sim-train` | Run PPO training in Bevy simulator |
| `sim-run` | Load trained policy, run in simulator (evaluation) |
| `robot-run` | Load trained policy, run on real robot |
| `robot-finetune` | Run on real robot with policy updates enabled |

All modes load `robot_config.yaml` at startup. The `driver` fields (`yahboom` vs `simulation`)
switch between real HAL and simulated HAL transparently.

---

## 9. Implementation Phases

| Phase | Crate(s) | Deliverable | Status |
|---|---|---|---|
| 1 | `config` | Parse `robot_config.yaml` → typed Rust structs | Pending |
| 2 | `hal/motor` | Drive each wheel independently; HW test mode | Pending |
| 3 | `hal/ultrasonic` | Read distance via I2C; calibrate min/max | Pending |
| 4 | `hal/gimbal` | Pan/tilt servo control; range limits; reactive controller stub | Pending |
| 5 | `hal/camera` | V4L2 frame capture; MJPEG stream; robot body mask calibration | Pending |
| 6 | `perception` | MiDaS ONNX inference; depth map pipeline; timing benchmark | Pending |
| 7 | `mapping` | Probabilistic occupancy grid; visualisation via rerun | Pending |
| 8 | `navigation` | A* planner; pure-pursuit controller; frontier exploration | Pending |
| 9 | `sim` | Bevy 3D simulator; procedural environment generation; virtual camera | Pending |
| 10 | `rl` | PPO implementation; train in sim; evaluate episode timing | Pending |
| 11 | `main` | Full orchestrator; all modes wired together | Pending |
| 12 | — | On-robot fine-tuning; sim-to-real evaluation | Pending |

---

## 10. Decisions Log

| # | Date | Decision | Rationale |
|---|---|---|---|
| D-01 | 2026-03-07 | Language: Rust | Performance, single binary, safety |
| D-02 | 2026-03-07 | ML inference: ONNX Runtime (`ort` crate) | Best model compat; models pre-exported from PyTorch |
| D-03 | 2026-03-07 | Simulator: Bevy (custom, all-Rust) | Tight RL integration; no external process; z-buffer + MiDaS |
| D-04 | 2026-03-07 | Sim environments: procedural maze-like, new per episode | Prevents overfitting to fixed layout |
| D-05 | 2026-03-07 | Depth in sim: MiDaS on rendered RGB (not raw z-buffer) | Better sim-to-real transfer |
| D-06 | 2026-03-07 | US sensor role: hardware safety interlock only | Force policy to learn from camera; eliminate US dependency |
| D-07 | 2026-03-07 | Gimbal: reactive controller, not RL action | Frees action budget; camera always points toward free space |
| D-08 | 2026-03-07 | Action space: 6 discrete (no backward, no gimbal) | Simpler, removes exploits; backward added back if needed |
| D-09 | 2026-03-07 | Training: sim-only → on-robot fine-tuning | Safe; faster iteration |
| D-10 | 2026-03-07 | ROS2: not used | Build complexity; standalone Rust binary preferred |
| D-11 | 2026-03-07 | YOLOv5 / object detection: removed | Depth map sufficient; no semantic classification needed |
| D-12 | 2026-03-07 | MobileNetV2: deferred | MiDaS depth is primary visual input; revisit if needed |
| D-13 | 2026-03-08 | Odometry: dead reckoning from motor commands + timing | No wheel encoders available |
| D-14 | 2026-03-08 | Training machine: Nvidia RTX 3050 | ~10k episodes in 3–5 hours; train on dev, deploy to Pi 5 |
| D-15 | 2026-03-08 | Camera FOV: 110° wide-angle, mount height 10 cm | Measured/confirmed from Yahboom spec and physical measurement |
| D-16 | 2026-03-08 | Body mask initial estimate: bottom 20% of frame (48 rows) | Geometric calculation; calibrate in Phase 5 on wheels-up stand |
| D-17 | 2026-03-08 | Wheels-up test stand available for static HW testing | Safe way to run motors and camera tests without the robot driving away |

---

## 11. Open Questions

| # | Question | Owner | Priority |
|---|---|---|---|
| ~~Q-01~~ | ~~Robot physical length and camera mount height~~ | **Resolved** — length ~25 cm, height ~10 cm (measured on wheels-up stand) | — |
| ~~Q-02~~ | ~~Real camera horizontal FOV~~ | **Resolved** — 110° wide-angle (Yahboom spec) | — |
| ~~Q-03~~ | ~~GPU on training machine~~ | **Resolved** — Nvidia RTX 3050 | — |
| Q-04 | Should `robot-finetune` mode update the full network or only the policy head (value/actor output layers)? | Design decision | Medium — needed for Phase 12 |
| Q-05 | Should the occupancy map be included in the RL state in a later iteration? | Design decision | Low — deferred |
| Q-06 | What `vision_every_n_steps` value should be used for sim training? Benchmark in Phase 6. | Phase 6 output | Medium |

---

## 12. Change Log

| Version | Date | Summary |
|---|---|---|
| 0.1 | 2026-03-08 | Initial draft — requirements and architecture defined |
