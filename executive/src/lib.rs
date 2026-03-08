//! Central executive state machine.
//!
//! The executive owns the high-level robot state and is the single authority
//! that decides when to transition between `Idle`, `Calibrating`, `Exploring`,
//! `Recovering`, `SafetyStopped`, and `Fault`.
//!
//! Every transition is:
//!   1. Logged via `tracing`.
//!   2. Published on `bus.executive_state` (watch channel).
//!   3. Emitted as an `EventMarker::ExecutiveTransition` on `bus.telemetry_event`.
//!
//! Subsystems read the watch channel to react to state changes; they never
//! call `transition()` directly — they request it through a future command
//! channel (planned for Milestone 2).

use std::sync::Arc;

use core_types::{EventMarker, ExecutiveState};
use tracing::info;

/// Executive state machine.
pub struct Executive {
    state: ExecutiveState,
    bus: Arc<bus::Bus>,
}

impl Executive {
    /// Create a new executive starting in `Idle`.
    pub fn new(bus: Arc<bus::Bus>) -> Self {
        let _ = bus.executive_state.send(ExecutiveState::Idle);
        Self { state: ExecutiveState::Idle, bus }
    }

    /// Current state (read-only).
    pub fn state(&self) -> &ExecutiveState {
        &self.state
    }

    /// Attempt a state transition. Returns `Err` if the transition is not
    /// allowed from the current state.
    pub fn transition(&mut self, next: ExecutiveState) -> Result<(), String> {
        if !self.is_allowed(&next) {
            return Err(format!(
                "transition {:?} → {:?} not permitted",
                self.state, next
            ));
        }
        let from = format!("{:?}", self.state);
        let to   = format!("{:?}", next);
        info!("Executive: {from} → {to}");
        self.state = next;
        let _ = self.bus.executive_state.send(self.state.clone());
        let _ = self.bus.telemetry_event.try_send(
            EventMarker::ExecutiveTransition { from, to },
        );
        Ok(())
    }

    /// State transition guard.
    ///
    /// Allowed transitions:
    /// ```text
    ///   Idle            → Calibrating | Exploring | Fault
    ///   Calibrating     → Idle | Exploring | Fault
    ///   Exploring       → Recovering | SafetyStopped | Idle | Fault
    ///   Recovering      → Exploring | SafetyStopped | Idle | Fault
    ///   SafetyStopped   → Idle | Fault
    ///   Fault           → Idle  (manual reset only)
    /// ```
    fn is_allowed(&self, next: &ExecutiveState) -> bool {
        use ExecutiveState::*;
        matches!(
            (&self.state, next),
            (Idle,          Calibrating   )
            | (Idle,        Exploring     )
            | (Idle,        Fault { .. }  )
            | (Calibrating, Idle          )
            | (Calibrating, Exploring     )
            | (Calibrating, Fault { .. }  )
            | (Exploring,   Recovering    )
            | (Exploring,   SafetyStopped )
            | (Exploring,   Idle          )
            | (Exploring,   Fault { .. }  )
            | (Recovering,  Exploring     )
            | (Recovering,  SafetyStopped )
            | (Recovering,  Idle          )
            | (Recovering,  Fault { .. }  )
            | (SafetyStopped, Idle        )
            | (SafetyStopped, Fault { .. })
            | (Fault { .. }, Idle         )
        )
    }
}
