//! Hardware Abstraction Layer.
//!
//! Defines traits for each hardware subsystem and provides stub
//! implementations. Real implementations (Yahboom I2C, V4L2, rppal) will be
//! added in Phase 2 hardware bring-up.
//!
//! All drivers implement their trait via `async-trait` so they can be used as
//! `Box<dyn Camera>`, `Box<dyn Imu>`, etc.

pub mod camera;
pub mod gimbal;
pub mod imu;
pub mod motor;
pub mod ultrasonic;

// Re-export trait and stub types at crate root for convenience.
pub use camera::{Camera, StubCamera};
#[cfg(feature = "usb-camera")]
pub use camera::V4L2Camera;
pub use gimbal::{Gimbal, StubGimbal, YahboomGimbal};
pub use imu::{Imu, StubImu};
#[cfg(feature = "mpu6050")]
pub use imu::Mpu6050Imu;
pub use motor::{MotorController, StubMotorController, YahboomMotorController};
pub use ultrasonic::{Ultrasonic, StubUltrasonic, YahboomUltrasonic};
