//! Perception pipeline.
//!
//! Subsystems:
//!   - `EventGate`        — decides when to run depth inference
//!   - `depth`            — MiDaS ONNX inference (real or stub fallback)
//!   - `pseudo_lidar`     — converts depth map → lidar rays
//!
//! The event gate reuses the previous depth estimate when the scene has not
//! changed enough to justify another inference, targeting a 70–85% reduction
//! in neural-network calls.

pub mod depth;
pub mod event_gate;
pub mod pseudo_lidar;

pub use depth::DepthInference;
pub use event_gate::EventGate;
pub use pseudo_lidar::PseudoLidarExtractor;
