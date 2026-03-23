# Phase 14.2.2: Manual Hardware Safety Test

## Goal
Validate that all hardware components (motors, gimbal, camera, IMU, ultrasonic) work correctly
via direct I2C/V4L2 commands **before** running autonomous exploration on the real robot.

**Time**: ~15 minutes
**Risk**: Minimal — direct component tests, wheels elevated (off ground)
**Prerequisites**: Robot on stand, wheels off ground, power cable plugged in

---

## Pre-Test Checklist

- [ ] Robot on elevated stand (wheels off ground)
- [ ] SSH access to pi@raspbot (10.0.0.183)
- [ ] Soft surface beneath robot (carpet/blanket for safety)
- [ ] Human operator ready to power off immediately if issues arise
- [ ] Telemetry window open to monitor (optional): `ssh raspbot "tail -f logs/robot.ndjson"`

---

## Test 1: Motors (5 min)

**Goal**: Verify all 4 wheels spin at 30% duty without grinding or jamming.

```bash
# SSH to robot
ssh raspbot

# Build motor test if needed
cd ~/Robot && cargo build -p hal --example motor_test --release 2>&1 | tail -5

# Run interactive motor test
ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so \
  ./target/release/examples/motor_test
```

**Expected behavior**:
- Prompts: "Motor: FL [Y/n]?" (front-left, front-right, rear-left, rear-right)
- When you press Y, wheel spins smoothly at ~30% duty
- No grinding, no sudden jerks, no smoke

**Acceptance criteria**:
- ✅ All 4 wheels spin when commanded
- ✅ Speed consistent (all reach ~0.3 m/s within ±10%)
- ✅ No thermal warnings or I2C errors

**Failure response**:
- ❌ Wheel doesn't spin → motor driver I2C issue, don't proceed
- ❌ Grinding/grinding noise → mechanical jam, do NOT continue
- ❌ I2C error 121 (Device Busy) → controller hung, power cycle robot

---

## Test 2: Gimbal Pan/Tilt (3 min)

**Goal**: Verify gimbal servo actuators respond without grinding or jamming.

```bash
# Still on robot
ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so \
  ./target/release/examples/gimbal_test
```

**Expected behavior**:
- Gimbal homes to neutral (0°, 0°) on startup
- Prompts for pan and tilt angles
- Servos move smoothly, respond to commands

**Acceptance criteria**:
- ✅ Pan moves left-right (±90°) without grinding
- ✅ Tilt moves down-up (-30° to +30°) without grinding
- ✅ Both return to neutral without overshoot

**Failure response**:
- ❌ No response to servo commands → gimbal I2C not responding
- ❌ Grinding/squealing → servo jam or mechanical friction
- ❌ Overshooting target angles → firmware calibration issue (rare)

---

## Test 3: Camera / V4L2 Stream (2 min)

**Goal**: Verify USB camera is detected and can stream frames.

```bash
# On dev machine (where you run `make sim`)
# Point browser to robot's MJPEG server
open "http://raspbot:8080/" &

# OR check from robot:
ssh raspbot "curl -s http://localhost:8080/ | head -20"
```

**Expected behavior**:
- MJPEG stream loads in browser
- Video shows robot's forward view (gimbal neutral = straight ahead)
- Frame rate ~15 fps (may vary ±2 fps)
- No frozen frames or corruption artifacts

**Acceptance criteria**:
- ✅ Stream accessible at `http://raspbot:8080/`
- ✅ Video updates smoothly (not frozen)
- ✅ Camera gimbal controlled by gimbal test → pan/tilt changes view

**Failure response**:
- ❌ Connection refused (8080) → MJPEG server not running
- ❌ Frozen frame → V4L2 driver issue, may need `sudo modprobe -r uvcvideo && sudo modprobe uvcvideo`
- ❌ Black frame → camera not mounted or disconnected

---

## Test 4: Ultrasonic Sensor (2 min)

**Goal**: Verify HC-SR04 can detect obstacles at known distances.

```bash
# On robot
ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so \
  ./target/release/examples/ultrasonic_test
```

**Expected behavior**:
- Sensor reads distance every ~100 ms
- You hold hand/object at ~30 cm → reads ~300 mm
- Move hand away to ~60 cm → reads ~600 mm
- Clean values, no spikes or zeros (except very close)

**Acceptance criteria**:
- ✅ Detects obstacles (hand) at 20–100 cm with ±3 cm error
- ✅ Readings stable (not wildly fluctuating)
- ✅ No dist=0 readings in mid-range (only at <5 cm)

**Failure response**:
- ❌ Only zeros → sensor not wired or MCU not responding (check 0x2B I2C)
- ❌ Random spikes to 400+ cm → US blind spot or noise, acceptable (safety interlock works)
- ❌ Constant max-range reads → sensor enabled but no echo, check wiring

---

## Test 5: IMU (Gyroscope + Accelerometer) (3 min)

**Goal**: Verify 6-axis IMU (MPU-6050) responds to motion.

```bash
# On robot
ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so \
  ./target/release/examples/imu_test
```

**Expected behavior**:
- Prints gyro (roll/pitch/yaw) and accel (x/y/z) continuously
- When stationary: gyro ≈ [0, 0, 0] rad/s, accel ≈ [0, 0, 9.81] m/s²
- When tilted: gyro values change
- When moved: accel changes

**Acceptance criteria**:
- ✅ Gyro readings < 0.1 rad/s when stationary (bias acceptable)
- ✅ Accel Z ≈ 9.81 m/s² when level (gravity reference)
- ✅ Gyro/accel change when you move the robot

**Failure response**:
- ❌ All zeros → IMU not wired (check I2C-6, address 0x68)
- ❌ Frozen readings → IMU communication stalled
- ❌ No response to motion → sensor likely not mounted or dead

---

## Post-Test Actions

**If all tests PASS** ✅
1. Power down robot: `ssh raspbot "sudo shutdown -h now"` or unplug
2. Move to Phase 14.2.3: Controlled Measurement Runs
   - Will run 6 short (~5-10 sec) autonomous tests: forward speed, rotation rate, etc.
   - Same elevated stand, same soft surface beneath
   - Operator at ready with `make stop-pi` command

**If any test FAILS** ❌
1. Note which test failed and error message
2. Do NOT proceed to Phase 14.2.3
3. Troubleshoot:
   - Check I2C devices: `i2cdetect -y 1` (motors/US/gimbal) or `i2cdetect -y 6` (IMU)
   - Check USB camera: `lsusb | grep "HD USB"` or similar
   - Power cycle robot: unplug, wait 5s, reconnect
4. Re-run failing test only
5. Proceed only after all tests pass

---

## Build Notes

If you need to rebuild examples on the Pi:

```bash
ssh raspbot "cd ~/Robot && \
  cargo build -p hal --example motor_test --release && \
  cargo build -p hal --example gimbal_test --release && \
  cargo build -p hal --example ultrasonic_test --release && \
  cargo build -p hal --example imu_test --release"
```

(May take ~30 sec on Pi 5; dev machine is faster with cross-compilation.)

---

## Expected Duration

| Test | Time | Status |
|------|------|--------|
| Motors | 5 min | Direct I2C @ 0x2B |
| Gimbal | 3 min | Servo commands @ 0x2B |
| Camera | 2 min | V4L2 stream @ :8080 |
| Ultrasonic | 2 min | HC-SR04 @ 0x2B |
| IMU | 3 min | MPU-6050 @ I2C-6 0x68 |
| **Total** | **~15 min** | |

---

## Safety Reminders

- **Keep robot elevated** throughout all tests — wheels must not touch ground
- **Operator ready** — hand near power switch or `make stop-pi` command ready
- **Soft surface** beneath — if robot escapes, padded landing reduces camera damage risk
- **No autonomous motion** — all tests are direct commands, no RL/planning

**Proceed cautiously to Phase 14.2.3 only after all components validated.**
