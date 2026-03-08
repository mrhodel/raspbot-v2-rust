//! Telemetry writer — newline-delimited JSON (NDJSON).
//!
//! Every record includes a monotonic timestamp, topic name, source subsystem,
//! sequence number, and a JSON payload. Records are appended to a file and can
//! be replayed offline by reading line-by-line.
//!
//! # Replay format
//! Each line is a self-contained JSON object:
//! ```json
//! {"t_ms":1234,"topic":"imu/raw","source":"hal","seq":5,"payload":{...}}
//! ```

use anyhow::Context;
use core_types::Ms;
use serde::Serialize;
use std::path::Path;
use tokio::io::{AsyncWriteExt, BufWriter};

// ── Record ────────────────────────────────────────────────────────────────────

/// Envelope written for every telemetry record.
#[derive(Serialize)]
struct Envelope<'a, T: Serialize> {
    t_ms: Ms,
    topic: &'a str,
    source: &'a str,
    seq: u64,
    payload: &'a T,
}

// ── Writer ────────────────────────────────────────────────────────────────────

/// Async telemetry writer. Flushes after each record for crash safety.
///
/// Create one instance per run; pass an `Arc<Mutex<TelemetryWriter>>` if
/// multiple tasks need to write, or fan-in through the `bus.telemetry_event`
/// mpsc channel and have a single dedicated writer task.
pub struct TelemetryWriter {
    file: BufWriter<tokio::fs::File>,
    seq: u64,
    t0: std::time::Instant,
}

impl TelemetryWriter {
    /// Open (or create) a log file. The file is appended to if it already exists.
    pub async fn open(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        // Ensure parent directory exists.
        if let Some(parent) = path.as_ref().parent() {
            tokio::fs::create_dir_all(parent).await.ok();
        }
        let file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .await
            .with_context(|| format!("opening telemetry log {:?}", path.as_ref()))?;
        Ok(Self {
            file: BufWriter::new(file),
            seq: 0,
            t0: std::time::Instant::now(),
        })
    }

    /// Write one record. The payload must implement `serde::Serialize`.
    pub async fn write<T: Serialize>(
        &mut self,
        topic: &str,
        source: &str,
        payload: &T,
    ) -> anyhow::Result<()> {
        let t_ms = self.t0.elapsed().as_millis() as Ms;
        let seq = self.seq;
        self.seq += 1;

        let envelope = Envelope { t_ms, topic, source, seq, payload };
        let mut line = serde_json::to_string(&envelope)
            .context("serialising telemetry record")?;
        line.push('\n');

        self.file.write_all(line.as_bytes()).await.context("writing telemetry record")?;
        self.file.flush().await.context("flushing telemetry")?;
        Ok(())
    }

    /// Write a plain string event marker record.
    pub async fn event(&mut self, source: &str, marker: &str) -> anyhow::Result<()> {
        self.write("telemetry/event_marker", source, &marker).await
    }
}
