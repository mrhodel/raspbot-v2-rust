//! Quick VL53L8CX sanity test.
//!
//! Initialises the sensor, reads 20 scans, and prints the 8 column-minimum
//! distances.  Run with:
//!
//!   make build-tof-test
//!   make deploy-tof-test
//!   ssh raspbot ./Robot/tof_test

#[cfg(feature = "vl53l8cx")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    use hal::{Tof, Vl53l8cxTof};

    println!("VL53L8CX test — initialising on /dev/i2c-2 (0x29)...");
    let mut tof = Vl53l8cxTof::new(
        2,      // i2c_bus
        0x29,   // i2c_address (7-bit)
        1,      // ranging_mode: 1=short ≤135 cm
        5,      // integration_time_ms
    )?;
    println!("Sensor ready. Reading 20 scans...\n");

    for i in 0..20 {
        let scan = tof.read_scan().await?;
        let distances: Vec<String> = scan.rays.iter()
            .map(|r| format!("{:5.2}m", r.range_m))
            .collect();
        println!("scan {:02}: [{}]", i + 1, distances.join("  "));
    }

    println!("\nDone.");
    Ok(())
}

#[cfg(not(feature = "vl53l8cx"))]
fn main() {
    eprintln!("Build with --features vl53l8cx");
}
