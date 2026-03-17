TARGET    := aarch64-unknown-linux-gnu
PI        := raspbot
PI_DIR    := ~/robot_ws
BINARY    := target/$(TARGET)/release/robot
PI_CONFIG := robot_config.yaml

.PHONY: cross deploy check stop sim stop-pi run-pi logs-pi arm arm-local

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
ORT_LIB := $(CURDIR)/.ort/libonnxruntime.so
sim: stop
	ORT_DYLIB_PATH=$(ORT_LIB) cargo run --no-default-features --features onnx -- sim $(ARGS)

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
