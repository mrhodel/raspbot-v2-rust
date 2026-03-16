TARGET   := aarch64-unknown-linux-gnu
PI       := raspbot
PI_DIR   := ~/robot_ws
BINARY   := target/$(TARGET)/release/robot

.PHONY: cross deploy check

## Cross-compile runtime for the Pi then rsync the binary.
deploy: cross
	rsync -az $(BINARY) $(PI):$(PI_DIR)/runtime
	@echo "Deployed to $(PI):$(PI_DIR)/runtime"

## Cross-compile only.
cross:
	cargo build --release --target $(TARGET) -p runtime

## Fast local check (no default features — skips V4L2/MPU6050).
check:
	cargo check --workspace --no-default-features
