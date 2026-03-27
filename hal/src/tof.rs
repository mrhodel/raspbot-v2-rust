//! VL53L8CX Time-of-Flight sensor HAL — trait and implementations.
//!
//! # Hardware (Pololu carrier #3419)
//!
//!   VL53L8CX on hardware I2C2 (`/dev/i2c-2`), GPIO 4/5 (SDA/SCL).
//!   Default 7-bit I2C address: 0x29.
//!
//!   The Pololu carrier has an on-board level shifter and regulators — power
//!   from 3.3V (Pi pin 1 or 17) and it handles the rest.
//!
//!   **No XSHUT pin** is exposed on the Pololu #3419 carrier.  The SPI/I2C
//!   mode-select pin on the carrier must be tied to GND to enable I2C; the
//!   easiest way is to bridge SPI/I2C to the carrier's GND pin.
//!
//! # Wiring
//!
//!   | Pololu pin | Pi header pin | GPIO  |
//!   |------------|---------------|-------|
//!   | VIN        | Pin 1  (3.3V) | —     |
//!   | GND        | Pin 34 (GND)  | —     |
//!   | SDA        | Pin 7         | GPIO4 |
//!   | SCL        | Pin 29        | GPIO5 |
//!   | SPI/I2C    | Pin 34 (GND)  | —     | ← must be LOW to select I2C
//!
//!   `/boot/firmware/config.txt` overlay required:
//!     `dtoverlay=i2c2-pi5`  (hardware I2C2, GPIO 4/5, Pi 5 only)
//!
//! # Output
//!
//!   `read_scan()` returns a `PseudoLidarScan` with 8 rays spanning ±19.69°
//!   (column centres of the 8×8 zone grid, 5.625° spacing).  Each ray is the
//!   minimum range across the 8 rows of that column — conservative for safety.
//!
//! # Implementation
//!
//!   When the `vl53l8cx` feature is enabled the real driver calls the ST ULD C
//!   library (compiled via the `cc` crate in `build.rs`) through a thin FFI
//!   wrapper (`vl53l8cx_uld/wrapper.c`).  On dev machines without the hardware
//!   the `StubTof` implementation returns open-space scans at ~10 Hz.
//!
//! # Feature flags
//!
//!   `vl53l8cx` — compile and link the C ULD + FFI wrapper (Pi target only)

use anyhow::Result;
use async_trait::async_trait;
use core_types::{LidarRay, PseudoLidarScan};
use std::time::Instant;

// ── Trait ─────────────────────────────────────────────────────────────────────

#[async_trait]
pub trait Tof: Send {
    /// Read one ToF scan: 8 rays spanning ±19.69° in robot frame.
    async fn read_scan(&mut self) -> Result<PseudoLidarScan>;
}

// ── Stub ──────────────────────────────────────────────────────────────────────

/// Returns all rays at 3.0 m (open space) at ~10 Hz. Used on dev machines
/// and in sim-mode where the sim provides its own ToF geometry.
pub struct StubTof {
    t0: Instant,
}

impl StubTof {
    pub fn new() -> Self {
        Self { t0: Instant::now() }
    }
}

impl Default for StubTof {
    fn default() -> Self { Self::new() }
}

#[async_trait]
impl Tof for StubTof {
    async fn read_scan(&mut self) -> Result<PseudoLidarScan> {
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        let t_ms = self.t0.elapsed().as_millis() as u64;
        Ok(open_space_scan(t_ms))
    }
}

// ── VL53L8CX real implementation via C ULD FFI ────────────────────────────────

#[cfg(feature = "vl53l8cx")]
mod vl53l8cx_ffi {
    use anyhow::{bail, Result};
    use async_trait::async_trait;
    use core_types::{LidarRay, PseudoLidarScan};
    use std::ffi::CStr;
    use std::os::raw::{c_char, c_float, c_int, c_void};
    use tracing::info;
    use super::Tof;

    // ── FFI declarations (matches wrapper.h) ─────────────────────────────────

    extern "C" {
        fn tof_open(
            bus: u8,
            addr_7bit: u8,
            ranging_mode: u8,
            integration_ms: u32,
            errbuf: *mut c_char,
            errbuf_len: c_int,
        ) -> *mut c_void;

        fn tof_read(
            handle: *mut c_void,
            out_ranges_m: *mut c_float,
            out_t_ms: *mut u64,
        ) -> c_int;

        fn tof_close(handle: *mut c_void);
    }

    // ── Column geometry (matches FastSim::cast_tof) ───────────────────────────

    const COL_STEP_RAD: f32 = 5.625 * std::f32::consts::PI / 180.0;

    fn col_angle_rad(col: usize) -> f32 {
        (col as f32 - 3.5) * COL_STEP_RAD
    }

    // ── Driver ────────────────────────────────────────────────────────────────

    pub struct Vl53l8cxTof {
        handle: *mut c_void,
    }

    // The C wrapper owns the only reference to the handle.
    // We never share it across threads, and tokio::task::block_in_place
    // ensures synchronous access.
    unsafe impl Send for Vl53l8cxTof {}

    impl Vl53l8cxTof {
        pub fn new(
            i2c_bus: u8,
            i2c_addr_7bit: u8,
            ranging_mode: u8,
            integration_time_ms: u32,
        ) -> Result<Self> {
            let mut errbuf = [0u8; 256];
            let handle = unsafe {
                tof_open(
                    i2c_bus,
                    i2c_addr_7bit,
                    ranging_mode,
                    integration_time_ms,
                    errbuf.as_mut_ptr() as *mut c_char,
                    errbuf.len() as c_int,
                )
            };
            if handle.is_null() {
                let msg = CStr::from_bytes_until_nul(&errbuf)
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|_| "unknown error".to_string());
                bail!("VL53L8CX init failed: {msg}");
            }
            info!(
                "VL53L8CX: /dev/i2c-{i2c_bus} @ 0x{i2c_addr_7bit:02X}, \
                 8×8 zones, mode={ranging_mode}, integration={integration_time_ms} ms"
            );
            Ok(Self { handle })
        }

        fn read_scan_sync(&mut self) -> Result<PseudoLidarScan> {
            let mut ranges = [0.0f32; 8];
            let mut t_ms: u64 = 0;
            let rc = unsafe {
                tof_read(self.handle, ranges.as_mut_ptr(), &mut t_ms)
            };
            if rc != 0 {
                anyhow::bail!("tof_read error: rc={rc}");
            }
            let rays = ranges
                .iter()
                .enumerate()
                .map(|(col, &range_m)| LidarRay {
                    angle_rad:  col_angle_rad(col),
                    range_m,
                    confidence: 1.0,
                })
                .collect();
            Ok(PseudoLidarScan { t_ms, rays })
        }
    }

    impl Drop for Vl53l8cxTof {
        fn drop(&mut self) {
            unsafe { tof_close(self.handle) };
        }
    }

    #[async_trait]
    impl Tof for Vl53l8cxTof {
        async fn read_scan(&mut self) -> Result<PseudoLidarScan> {
            tokio::task::block_in_place(|| self.read_scan_sync())
        }
    }
}

#[cfg(feature = "vl53l8cx")]
pub use vl53l8cx_ffi::Vl53l8cxTof;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// 8-ray open-space scan (all ranges 3.0 m) matching `cast_tof` geometry.
pub fn open_space_scan(t_ms: u64) -> PseudoLidarScan {
    const COL_STEP_RAD: f32 = 5.625 * std::f32::consts::PI / 180.0;
    let rays = (0..8)
        .map(|col| LidarRay {
            angle_rad:  (col as f32 - 3.5) * COL_STEP_RAD,
            range_m:    3.0,
            confidence: 1.0,
        })
        .collect();
    PseudoLidarScan { t_ms, rays }
}
