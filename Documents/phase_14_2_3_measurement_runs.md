# Phase 14.2.3: Controlled Measurement Runs

## Goal
Collect real-world calibration measurements (6 brief autonomous tests) to update sim constants.
Each test runs **5-10 seconds max** with human operator ready to execute `make stop-pi` immediately.

**Time**: ~45 minutes (6 tests × 5-10 sec each, plus setup/data collection between runs)
**Risk**: Low — short duration, elevated stand, soft surface beneath, emergency stop ready
**Prerequisites**: Phase 14.2.2 hardware tests all passing

---

## Pre-Test Checklist

- [ ] Robot on elevated stand (wheels off ground, verified)
- [ ] Soft surface beneath (carpet/blanket for safety)
- [ ] Human operator sitting at laptop ready to type `make stop-pi`
- [ ] `make logs/measurements.yaml` created and ready for logging
- [ ] SSH access verified: `ssh raspbot "ls ~/Robot/logs/" | grep measurements`
- [ ] Terminal A: Monitoring telemetry on robot
  ```bash
  ssh raspbot "tail -f logs/robot.ndjson | jq .pose 2>/dev/null || tail -f logs/robot.ndjson"
  ```
- [ ] Terminal B: Ready to execute stop command
  ```bash
  cd ~/Robot && make stop-pi
  ```

---

## Test 1: Forward Speed (5-7 sec)

**Measurement**: Distance traveled / time = m/s at 30% motor duty

**Setup**:
1. Place a reference marker on the floor directly in front of the robot (tape/chalk)
2. Arm robot: `ssh raspbot 'kill -USR1 $(pgrep robot)'` (or via UI ARM button)
3. Send forward command via exploration system (natural exploration, just happens to go forward)
4. Let it run for ~6 seconds (enough to travel ~1.8m)
5. Execute `make stop-pi` — motors immediately zero

**Data Collection**:
- Read IMU integration or odometry from telemetry
- Look for distance in logs: search for "x_m" / "y_m" deltas
- Measure physical distance robot traveled (tape measure from marker)
- **Formula**: `speed_m_s = distance_m / time_sec`
- Log result: `calibration_run_1: {test: "forward_speed_30pct", distance_m: X, time_sec: 6, computed_m_s: X/6, measured_m_s: Y, notes: "..."}`

**Expected**:
- Sim constant: 0.30 m/s
- Real robot: likely 0.25–0.35 m/s (±15%)
- Surface: wheels off ground (less friction bias)

---

## Test 2: Rotation Rate (5-7 sec)

**Measurement**: Integrated gyro Z over time = rad/s at 30% duty rotation

**Setup**:
1. Arm robot again
2. Force it into rotation mode (rotate_left or rotate_right command)
   - Can do this by sending command via exploration override, or let exploration naturally rotate
3. Run for ~6 seconds (full rotation should take ~6-7 sec at 1.0 rad/s)
4. Execute `make stop-pi`

**Data Collection**:
- Extract gyro_z from IMU telemetry logs over the 6 second window
- Integrate: `cumulative_rotation_rad = sum(gyro_z * dt)` over the period
- **Formula**: `omega_rad_s = cumulative_rotation_rad / time_sec`
- Log result: `calibration_run_2: {test: "rotation_rate_30pct", cumulative_rotation_rad: X, time_sec: 6, computed_rad_s: X/6, sim_constant: 1.0, notes: "..."}`

**Expected**:
- Sim constant: 1.0 rad/s
- Real robot: likely 0.8–1.2 rad/s (±20%)

---

## Test 3: Strafe Speed (5-7 sec)

**Measurement**: Lateral distance traveled / time = m/s at 30% duty strafe

**Setup**:
1. Arm robot
2. Command strafe_left (or let exploration trigger it naturally)
3. Run for ~6 seconds (expect ~1.8m lateral travel)
4. Execute `make stop-pi`

**Data Collection**:
- Measure lateral displacement from IMU/odometry OR physical tape measure
- **Formula**: `strafe_m_s = lateral_distance_m / time_sec`
- Compare vs forward speed (expect ~0.7× forward on mecanum wheels)
- Log result: `calibration_run_3: {test: "strafe_speed_30pct", distance_m: X, time_sec: 6, computed_m_s: X/6, vs_forward: "ratio", notes: "..."}`

**Expected**:
- Ratio to forward speed: ~0.7–0.9 (mecanum wheels less efficient laterally)
- Actual value: ~0.21–0.32 m/s if forward is 0.30 m/s

---

## Test 4: Gimbal Response Timing (5-10 sec)

**Measurement**: Pan/tilt speed when gimbal commanded to pan ±90°

**Setup**:
1. Start gimbal test directly:
   ```bash
   ssh raspbot "ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so ./target/release/examples/gimbal_test"
   ```
2. When prompted, command:
   - Pan: 90° (full right)
   - Then: -90° (full left)
   - Then: 0° (center)
3. Time each movement with stopwatch or logs

**Data Collection**:
- Measure time for servo to complete pan 90° → -90° (180° total, expect ~0.8 sec)
- Measure time for tilt -30° → +30° (60° total, expect ~0.4 sec)
- **Formula**: `pan_speed_deg_s = 180 / time_sec`, `tilt_speed_deg_s = 60 / time_sec`
- Log result: `calibration_run_4: {test: "gimbal_response", pan_180deg_time_sec: X, pan_speed_deg_s: Y, tilt_60deg_time_sec: A, tilt_speed_deg_s: B, notes: "..."}`

**Expected**:
- Pan speed: ~225 °/sec (180° in 0.8s)
- Tilt speed: ~150 °/sec (60° in 0.4s)
- Actual may vary ±20% (servo firmware dependent)

---

## Test 5: MiDaS Depth Noise Profile (3 tests, ~10 sec each)

**Measurement**: MiDaS depth readings at known distances to quantify noise

**Setup**:
1. Position obstacles at controlled distances:
   - 0.5 m away (can use cardboard box or measured wall)
   - 1.0 m away
   - 2.0 m away
2. For each distance:
   - Arm robot
   - Record 100 depth frames → save to logs
   - Extract MiDaS values (focus on center column, obstacle distance should be stable)
   - Compute mean and std dev

**Data Collection**:
- Run for ~10 seconds per distance, collect MiDaS readings
- Extract latest 100 depth samples from telemetry
- **Formula**:
  - `mean_depth = avg(MiDaS readings)`
  - `std_depth = stdev(MiDaS readings)`
  - `cv = std_depth / mean_depth` (coefficient of variation = noise as % of signal)
- Log result:
  ```yaml
  calibration_run_5a:
    test: "midas_depth_noise"
    distance_m: 0.5
    num_samples: 100
    mean_range_m: X
    std_range_m: Y
    cv_percent: Z
    notes: "..."

  calibration_run_5b:
    test: "midas_depth_noise"
    distance_m: 1.0
    num_samples: 100
    mean_range_m: X
    std_range_m: Y
    cv_percent: Z
    notes: "..."

  calibration_run_5c:
    test: "midas_depth_noise"
    distance_m: 2.0
    num_samples: 100
    mean_range_m: X
    std_range_m: Y
    cv_percent: Z
    notes: "..."
  ```

**Expected**:
- Mean should match actual distance (validate calibration)
- Std dev should be ~5-10% at 0.5m, increasing to ~10-20% at 2.0m
- Noise increases with distance (expected for monocular depth)

---

## Test 6: Emergency Stop Latency (5-7 sec)

**Measurement**: How far does robot travel after `make stop-pi` issued?

**Setup**:
1. Arm robot and command forward motion
2. Let it accelerate for ~3 seconds (reach cruising speed)
3. Issue `make stop-pi` and immediately start timer
4. Measure how far it coasts before stopping
5. Safe margin: should stop within ~0.1-0.2m

**Data Collection**:
- Measure stopping distance with tape measure
- **Formula**: `stop_latency_m = distance_traveled_after_stop_command`
- Log result: `calibration_run_6: {test: "emergency_stop_latency", stop_distance_m: X, coast_time_sec: "~Y", notes: "..."}`

**Expected**:
- Motor command executes at ~10ms latency (network)
- Motor inertia coasts ~0.05-0.15m before all wheels zero
- If > 0.2m, may indicate motor response lag issue

---

## Data Logging Format

Create `logs/measurements.yaml` with entries like:

```yaml
phase: 14.2.3
date: "2026-03-18"
robot: "RaspbotV2"
surface: "wheels_elevated"

calibration_runs:
  - run: 1
    test: "forward_speed_30pct"
    duration_sec: 6
    distance_m_measured: 1.82
    computed_speed_m_s: 0.303
    sim_expected_m_s: 0.30
    error_percent: 1.0
    notes: "Clean run, stable speed"

  - run: 2
    test: "rotation_rate_30pct"
    duration_sec: 6
    cumulative_rotation_rad: 6.03
    computed_omega_rad_s: 1.005
    sim_expected_rad_s: 1.0
    error_percent: 0.5
    notes: "361° rotation, drift minimal"

  # ... (continue for tests 3-6)

summary:
  all_tests_passed: true
  ready_for_phase_14_2_4: true
  confidence: "HIGH - all measurements within ±15% of sim"
```

---

## Safety Protocol

**Operator Checklist** (run before EACH test):
- [ ] Robot still on elevated stand
- [ ] No objects under robot that could interfere
- [ ] Soft surface beneath intact
- [ ] Terminal B ready with `make stop-pi` command
- [ ] Eyes on robot, ready to execute stop if needed

**Emergency Procedures**:
- ❌ Motor doesn't stop on `make stop-pi` → **pull power cable immediately**
- ❌ Robot falls off stand → **do NOT catch with hands** (rotation hazard), let it land on soft surface
- ❌ Unusual noise/grinding → **execute `make stop-pi` immediately**, do NOT continue test

**Post-Test**:
- Verify telemetry logged: `tail logs/robot.ndjson`
- Verify no crashes in logs: `grep -i crash logs/robot.ndjson || echo "No crashes"`
- Power down between tests: `ssh raspbot 'sudo poweroff'` (optional, reduces thermal stress)

---

## Expected Duration

| Test | Duration | Notes |
|------|----------|-------|
| Test 1 (forward) | 7 sec + 2 min logging | Straightforward |
| Test 2 (rotation) | 7 sec + 2 min logging | Check gyro integration |
| Test 3 (strafe) | 7 sec + 2 min logging | Mecanum ratio estimate |
| Test 4 (gimbal) | 30 sec + 1 min logging | Manual servo timing |
| Test 5a (depth 0.5m) | 10 sec + 2 min analysis | Obstacle setup |
| Test 5b (depth 1.0m) | 10 sec + 2 min analysis | Reposition obstacle |
| Test 5c (depth 2.0m) | 10 sec + 2 min analysis | Reposition obstacle |
| Test 6 (stop latency) | 7 sec + 1 min logging | Final safety check |
| **Total** | ~**70 min** (with breaks) | |

---

## Next Phase (14.2.4)

After all measurements complete:
1. Parse `logs/measurements.yaml` for calibration values
2. Update `robot_config.yaml` with measured constants:
   - `sim.speed_m_s` ← from Test 1
   - `sim.omega_rad_s` ← from Test 2
   - `sim.strafe_scale` ← from Test 3 (ratio to forward)
   - `hal.gimbal.pan_speed_deg_s` ← from Test 4
   - Depth noise profile ← from Test 5 (inform perception tuning)
   - Emergency stop latency ← from Test 6 (safety verification)
3. Rebuild and validate with Phase 14.2.1 scenarios
4. Proceed to Phase 14.2.5: Confidence Checkpoint

---

## Success Criteria

✅ All 6 tests complete without robot escaping or crashing
✅ Measurements self-consistent (repeat run ±10% of first)
✅ Sim constants match real measurements within ±15%
✅ Emergency stop latency < 0.2m
✅ No I2C/hardware errors during tests

**If any criterion fails**, re-run affected test or troubleshoot before Phase 14.2.4.

---

## Quick Start Command

```bash
# Robot side: arm the robot and start logging
ssh raspbot "cd ~/Robot && \
  RUST_LOG=info cargo run --release --no-default-features -- robot-run 2>&1 | \
  tee logs/measurements_$(date +%Y%m%d_%H%M%S).log"

# Dev side: monitor in real-time
ssh raspbot "tail -f logs/robot.ndjson | jq -r '[.t_ms, .pose.x_m, .pose.y_m, .pose.theta_rad] | @tsv'"

# Emergency stop (always ready):
make stop-pi
```

Ready to start Phase 14.2.3 measurement runs?
