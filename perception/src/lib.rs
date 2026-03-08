//! Perception pipeline.
//!
//! Subsystems:
//!   - `EventGate`   — decides when to run depth inference
//!   - `depth`       — MiDaS ONNX inference (stub; real impl in Phase 6)
//!   - `pseudo_lidar` — converts depth map → lidar rays
//!
//! The event gate reuses the previous depth estimate when the scene has not
//! changed enough to justify another inference, targeting a 70–85% reduction
//! in neural-network calls (requirements_v3.md §5.1).

pub mod depth;
pub mod event_gate;
pub mod pseudo_lidar;

pub use depth::DepthInference;
pub use event_gate::EventGate;
pub use pseudo_lidar::PseudoLidarExtractor;
