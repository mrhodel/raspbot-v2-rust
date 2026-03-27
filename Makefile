TARGET    := aarch64-unknown-linux-gnu
PI        := raspbot
PI_DIR    := ~/robot_ws
BINARY    := target/$(TARGET)/release/robot
PI_CONFIG := robot_config.yaml

.PHONY: cross deploy check run-sim run-sim-box stop-sim logs-sim stop-pi run-pi logs-pi arm arm-local \
         motor-test-cross motor-test-deploy motor-test-run motor-test-sweep motor-test-plot \
         tof-test

## Fast local check (no default features — skips V4L2/MPU6050).
check:
	cargo check --workspace --no-default-features

## Cross-compile runtime for the Pi.
cross:
	cargo build --release --target $(TARGET) -p runtime

ORT_LIB := $(CURDIR)/.ort/libonnxruntime.so
SIM_LOG  := /tmp/robot_sim.log

## Run sim (maze, ToF) in background and auto-arm.  Pass ARGS="seed=42" etc.
run-sim: stop-sim
	ORT_DYLIB_PATH=$(ORT_LIB) cargo run --no-default-features --features onnx -- sim explore tof $(ARGS) 2>&1 | tee $(SIM_LOG) &
	@echo "Sim starting — waiting 5 s for init…"
	sleep 5
	pkill -USR1 -x robot
	@echo "Sim armed. Logs: make logs-sim"

## Run sim with a single centered box (crash isolation), auto-armed.
run-sim-box: stop-sim
	ORT_DYLIB_PATH=$(ORT_LIB) cargo run --no-default-features --features onnx -- sim explore single-box tof $(ARGS) 2>&1 | tee $(SIM_LOG) &
	@echo "Sim (single-box) starting — waiting 5 s for init…"
	sleep 5
	pkill -USR1 -x robot
	@echo "Sim armed. Logs: make logs-sim"

## Stop the local sim.
stop-sim:
	-pkill -x robot 2>/dev/null; sleep 0.2

## Stream sim logs.
logs-sim:
	tail -f $(SIM_LOG)

## Stop robot on Pi.
stop-pi:
	ssh $(PI) "pkill -x robot 2>/dev/null || true"

## Cross-compile + rsync binary AND config to Pi.
deploy: cross
	rsync -az $(BINARY) $(PI_CONFIG) $(PI):$(PI_DIR)/runtime/
	@echo "Deployed to $(PI):$(PI_DIR)/runtime/"

PI_ORT    := /usr/local/lib/libonnxruntime.so

## Deploy then start robot on Pi in background.
run-pi: deploy stop-pi
	ssh $(PI) "cp $(PI_DIR)/runtime/robot_config.yaml $(PI_DIR)/robot_config.yaml && cd $(PI_DIR) && nohup env ORT_DYLIB_PATH=$(PI_ORT) ./runtime/robot robot-run > robot.log 2>&1 &"
	@echo "Robot started — tail with: make logs-pi"

## Stream robot logs from Pi.
logs-pi:
	ssh $(PI) "tail -f $(PI_DIR)/robot.log"

## Arm the robot on Pi (sends SIGUSR1 to running process).
arm:
	ssh $(PI) "pkill -USR1 -x robot"

## Arm the robot running locally.
arm-local:
	pkill -USR1 -x robot

# ─────────────────────────────────────────────────────────────────────────

## Cross-compile motor_test example for Pi.
MOTOR_TEST := target/$(TARGET)/release/examples/motor_test
motor-test-cross:
	cargo build --release --target $(TARGET) -p hal --example motor_test

## Deploy motor_test binary to Pi.
motor-test-deploy: motor-test-cross
	scp $(MOTOR_TEST) $(PI):$(PI_DIR)/motor_test
	@echo "Deployed to $(PI):$(PI_DIR)/motor_test"

## Run motor_test interactively on Pi (prompts for surface & pass/fail feedback).
motor-test-run: motor-test-deploy
	ssh -t $(PI) "cd $(PI_DIR) && ./motor_test"

## Run motor_test in --sweep mode on Pi (stall speed characterization).
##   Pass SPEED=NN to override default 40%.  SURFACES=hardwood,carpet,tile to batch test.
motor-test-sweep: motor-test-deploy
	ssh -t $(PI) "cd $(PI_DIR) && ./motor_test --sweep --speed $(or $(SPEED),40) --surfaces $(or $(SURFACES),hardwood)"

## Plot motor test CSVs from runs/ locally. Requires matplotlib + numpy.
motor-test-plot:
	python3 scripts/plot_motor_test.py

# ─────────────────────────────────────────────────────────────────────────

TOF_TEST := target/$(TARGET)/release/examples/tof_test

## Build, deploy, and run the VL53L8CX sanity test on the Pi.
tof-test:
	cargo build --release --target $(TARGET) -p hal --example tof_test
	scp $(TOF_TEST) $(PI):$(PI_DIR)/tof_test
	ssh -t $(PI) "cd $(PI_DIR) && ./tof_test"
