//! SimMotorController — stores the latest MotorCommand for the sim tick task.

use std::sync::Arc;
use anyhow::Result;
use async_trait::async_trait;
use core_types::MotorCommand;
use hal::MotorController;

pub struct SimMotorController {
    latest: Arc<std::sync::Mutex<Option<MotorCommand>>>,
}

impl SimMotorController {
    pub fn new(latest: Arc<std::sync::Mutex<Option<MotorCommand>>>) -> Self {
        Self { latest }
    }
}

#[async_trait]
impl MotorController for SimMotorController {
    async fn send_command(&mut self, cmd: MotorCommand) -> Result<()> {
        *self.latest.lock().unwrap() = Some(cmd);
        Ok(())
    }

    async fn emergency_stop(&mut self) -> Result<()> {
        *self.latest.lock().unwrap() = Some(MotorCommand { t_ms: 0, fl: 0, fr: 0, rl: 0, rr: 0 });
        Ok(())
    }
}
