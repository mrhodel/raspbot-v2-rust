//! MiDaS depth inference latency benchmark.
//!
//! Build with ONNX (default):
//!   cargo build --release -p perception --example depth_bench
//!
//! Usage:
//!   ./target/release/examples/depth_bench [model_path] [num_threads] [runs]
//!
//! Defaults:
//!   model_path  = models/midas_small.onnx
//!   num_threads = 2
//!   runs        = 20

use std::{env, time::Instant};

use anyhow::Result;
use core_types::CameraFrame;
use perception::DepthInference;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("perception=info".parse().unwrap()),
        )
        .init();

    let args: Vec<String> = env::args().collect();
    let model_path  = args.get(1).map(|s| s.as_str()).unwrap_or("models/midas_small.onnx");
    let num_threads = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4usize);
    let runs        = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(20u32);

    println!("Depth bench  model={model_path}  threads={num_threads}  runs={runs}");

    let mut engine = DepthInference::new(
        model_path,
        32, 32,   // out_width, out_height
        0,        // mask_rows (calibrated later)
        num_threads,
    );

    println!("Engine: {}", if engine.is_real() { "ONNX (real)" } else { "stub" });

    // Synthetic 640x480 RGB frame (matches real camera).
    let frame = CameraFrame {
        t_ms:   0,
        width:  640,
        height: 480,
        data:   (0..640 * 480 * 3).map(|i| (i % 251) as u8).collect(),
    };

    // Warmup.
    for _ in 0..3 {
        let _ = engine.infer(&frame)?;
    }

    // Timed runs.
    let mut times_ms = Vec::with_capacity(runs as usize);
    for _ in 0..runs {
        let t = Instant::now();
        let depth = engine.infer(&frame)?;
        times_ms.push(t.elapsed().as_secs_f64() * 1000.0);
        let _ = depth; // prevent optimisation
    }

    times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
    let p50  = times_ms[times_ms.len() / 2];
    let p95  = times_ms[(times_ms.len() as f32 * 0.95) as usize];
    let min  = times_ms[0];
    let max  = times_ms[times_ms.len() - 1];

    println!(
        "\n{runs} runs (after 3 warmup)\n\
         mean={mean:.1}ms  p50={p50:.1}ms  p95={p95:.1}ms  min={min:.1}ms  max={max:.1}ms"
    );

    // Budget check: target < 200ms on Pi 5 (10 Hz loop, vision every 3 steps).
    let budget_ms = 200.0;
    if mean < budget_ms {
        println!("Budget OK: {mean:.1}ms < {budget_ms}ms");
    } else {
        println!("OVER BUDGET: {mean:.1}ms >= {budget_ms}ms — reduce threads or input size");
    }

    Ok(())
}
