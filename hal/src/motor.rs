//! Motor controller HAL trait and stub.
//!
//! TODO (Phase 2): implement Yahboom I2C motor commands via rppal.

use anyhow::Result;
use async_trait::async_trait;
use core_types::MotorCommand;
use tracing::debug;

#[async_trait]
pub trait MotorController: Send + Sync {
    async fn send_command(&mut self, cmd: MotorCommand) -> Result<()>;
    async fn emergency_stop(&mut self) -> Result<()>;
}

// ── Stub ──────────────────────────────────────────────────────────────────────

pub struct StubMotorController;

#[async_trait]
impl MotorController for StubMotorController {
    async fn send_command(&mut self, cmd: MotorCommand) -> Result<()> {
        debug!(?cmd, "StubMotor: command");
        Ok(())
    }
    async fn emergency_stop(&mut self) -> Result<()> {
        debug!("StubMotor: emergency stop");
        Ok(())
    }
}
