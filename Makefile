TARGET    := aarch64-unknown-linux-gnu
PI        := raspbot
PI_DIR    := ~/robot_ws
BINARY    := target/$(TARGET)/release/robot
PI_CONFIG := robot_config.yaml

.PHONY: cross deploy check stop sim sim-armed sim-box sim-box-armed sim-tof sim-tof-armed sim-box-tof sim-box-tof-armed stop-pi run-pi logs-pi arm arm-local \
         motor-test-cross motor-test-deploy motor-test-run motor-test-sweep motor-test-plot

## Fast local check (no default features — skips V4L2/MPU6050).
check:
	cargo check --workspace --no-default-features

## Cross-compile runtime for the Pi.
cross:
	cargo build --release --target $(TARGET) -p runtime

## Stop any running robot process locally.
stop:
	-pkill -x robot 2>/dev/null; sleep 0.2

## Run sim mode locally (stops first).  Pass ARGS="--seed 42" etc.
## Skips rppal/V4L2 but enables ONNX (MiDaS) via local ORT shared library.
## Runs in explore mode by default: no episode resets, continuous run.
ORT_LIB := $(CURDIR)/.ort/libonnxruntime.so
SIM_LOG := /tmp/robot_sim.log
sim: stop
	ORT_DYLIB_PATH=$(ORT_LIB) cargo run --no-default-features --features onnx -- sim explore $(ARGS) 2>&1 | tee $(SIM_LOG)

## Run sim and auto-arm after startup (background; logs → SIM_LOG).
sim-armed: stop
	ORT_DYLIB_PATH=$(ORT_LIB) cargo run --no-default-features --features onnx -- sim explore $(ARGS) 2>&1 | tee $(SIM_LOG) &
	@echo "Sim starting — waiting 5 s for init…"
	sleep 5
	pkill -USR1 -x robot
	@echo "Sim armed. Logs: tail -f $(SIM_LOG)"

## Run sim with a single centered 1m×1m obstacle (clean crash isolation).
sim-box: stop
	ORT_DYLIB_PATH=$(ORT_LIB) cargo run --no-default-features --features onnx -- sim explore single-box $(ARGS) 2>&1 | tee $(SIM_LOG)

## Run sim-box and auto-arm after startup (background; logs → SIM_LOG).
sim-box-armed: stop
	ORT_DYLIB_PATH=$(ORT_LIB) cargo run --no-default-features --features onnx -- sim explore single-box $(ARGS) 2>&1 | tee $(SIM_LOG) &
	@echo "Sim (single-box) starting — waiting 5 s for init…"
	sleep 5
	pkill -USR1 -x robot
	@echo "Sim armed. Logs: tail -f $(SIM_LOG)"

## Run sim with VL53L5CX ToF sensor model (8 rays, ±19.69°, no dropout).
sim-tof: stop
	ORT_DYLIB_PATH=$(ORT_LIB) cargo run --no-default-features --features onnx -- sim explore tof $(ARGS) 2>&1 | tee $(SIM_LOG)

## Run sim-tof and auto-arm after startup (background; logs → SIM_LOG).
sim-tof-armed: stop
	ORT_DYLIB_PATH=$(ORT_LIB) cargo run --no-default-features --features onnx -- sim explore tof $(ARGS) 2>&1 | tee $(SIM_LOG) &
	@echo "Sim (ToF) starting — waiting 5 s for init…"
	sleep 5
	pkill -USR1 -x robot
	@echo "Sim (ToF) armed. Logs: tail -f $(SIM_LOG)"

## Run sim with single-box + ToF sensor model.
sim-box-tof: stop
	ORT_DYLIB_PATH=$(ORT_LIB) cargo run --no-default-features --features onnx -- sim explore single-box tof $(ARGS) 2>&1 | tee $(SIM_LOG)

## Run sim-box-tof and auto-arm after startup (background; logs → SIM_LOG).
sim-box-tof-armed: stop
	ORT_DYLIB_PATH=$(ORT_LIB) cargo run --no-default-features --features onnx -- sim explore single-box tof $(ARGS) 2>&1 | tee $(SIM_LOG) &
	@echo "Sim (single-box ToF) starting — waiting 5 s for init…"
	sleep 5
	pkill -USR1 -x robot
	@echo "Sim (single-box ToF) armed. Logs: tail -f $(SIM_LOG)"

## Stop robot on Pi.
stop-pi:
	ssh $(PI) "pkill -x robot 2>/dev/null || true"

## Cross-compile + rsync binary AND config to Pi.
deploy: cross
	rsync -az $(BINARY) $(PI_CONFIG) $(PI):$(PI_DIR)/runtime/
	@echo "Deployed to $(PI):$(PI_DIR)/runtime/"

## Deploy then start robot on Pi in background.
run-pi: deploy stop-pi
	ssh $(PI) "cd $(PI_DIR)/runtime && nohup ./robot robot-run > ../robot.log 2>&1 &"
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
