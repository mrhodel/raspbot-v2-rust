//! Read-only WebSocket visualization bridge.
//!
//! Subscribes to key bus topics and fans out JSON-serialized snapshots to
//! all connected browser clients. The bridge is entirely read-only — clients
//! cannot send commands.
//!
//! # Protocol
//! Each WebSocket message is a single JSON object:
//! ```json
//! { "topic": "slam/pose2d", "t_ms": 1234, "data": { ... } }
//! ```
//!
//! # Topics published to clients
//! - `slam/pose2d`         — latest robot pose
//! - `map/grid_delta`      — incremental occupancy grid updates
//! - `map/frontiers`       — current frontier list
//! - `executive/state`     — executive state machine state
//! - `health/runtime`      — CPU / memory / timing snapshot
//!
//! # Default port
//! 9000 (configurable via `UiBridgeConfig`).

use std::sync::Arc;

use anyhow::Result;
use core_types::BridgeStatus;
use futures_util::SinkExt;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::broadcast;
use tokio_tungstenite::tungstenite::Message;
use tracing::{info, warn};

/// Configuration for the WebSocket bridge.
#[derive(Debug, Clone)]
pub struct UiBridgeConfig {
    pub port: u16,
}

impl Default for UiBridgeConfig {
    fn default() -> Self {
        Self { port: 9000 }
    }
}

/// Internal fan-out channel capacity.
const FANOUT_CAP: usize = 64;

/// Start the UI bridge. This function spawns Tokio tasks internally and
/// returns immediately. Call from the runtime main after building the bus.
///
/// Publishes `BridgeStatus` updates on `bus.ui_bridge_status`.
pub async fn start(bus: Arc<bus::Bus>, cfg: UiBridgeConfig) -> Result<()> {
    // Internal broadcast channel: bridge aggregator → per-client tasks.
    let (fanout_tx, _) = broadcast::channel::<Arc<String>>(FANOUT_CAP);
    let fanout_tx = Arc::new(fanout_tx);

    // ── Aggregator task: subscribe to bus, serialize, fan out ────────────
    let fanout_agg = Arc::clone(&fanout_tx);
    let bus_agg = Arc::clone(&bus);
    tokio::spawn(async move {
        let mut rx_pose     = bus_agg.slam_pose2d.subscribe();
        let mut rx_grid     = bus_agg.map_grid_delta.subscribe();
        let mut rx_frontier = bus_agg.map_frontiers.subscribe();
        let mut rx_exec     = bus_agg.executive_state.subscribe();
        let mut rx_health   = bus_agg.health_runtime.subscribe();

        loop {
            tokio::select! {
                Ok(()) = rx_exec.changed() => {
                    let state = rx_exec.borrow_and_update().clone();
                    if let Ok(msg) = serde_json::to_string(&serde_json::json!({
                        "topic": "executive/state",
                        "data":  state,
                    })) {
                        let _ = fanout_agg.send(Arc::new(msg));
                    }
                }
                Ok(()) = rx_pose.changed() => {
                    let pose = *rx_pose.borrow_and_update();
                    if let Ok(msg) = serde_json::to_string(&serde_json::json!({
                        "topic": "slam/pose2d",
                        "t_ms":  pose.t_ms,
                        "data":  pose,
                    })) {
                        let _ = fanout_agg.send(Arc::new(msg));
                    }
                }
                Ok(delta) = rx_grid.recv() => {
                    if let Ok(msg) = serde_json::to_string(&serde_json::json!({
                        "topic": "map/grid_delta",
                        "t_ms":  delta.t_ms,
                        "data":  delta,
                    })) {
                        let _ = fanout_agg.send(Arc::new(msg));
                    }
                }
                Ok(frontiers) = rx_frontier.recv() => {
                    if let Ok(msg) = serde_json::to_string(&serde_json::json!({
                        "topic": "map/frontiers",
                        "data":  frontiers,
                    })) {
                        let _ = fanout_agg.send(Arc::new(msg));
                    }
                }
                Ok(health) = rx_health.recv() => {
                    if let Ok(msg) = serde_json::to_string(&serde_json::json!({
                        "topic": "health/runtime",
                        "t_ms":  health.t_ms,
                        "data":  health,
                    })) {
                        let _ = fanout_agg.send(Arc::new(msg));
                    }
                }
                else => break,
            }
        }
    });

    // ── Listener task: accept connections, spawn per-client handlers ──────
    let addr = format!("0.0.0.0:{}", cfg.port);
    let listener = TcpListener::bind(&addr).await?;
    info!("UI bridge WebSocket listening on ws://{addr}");

    let bus_status = Arc::clone(&bus);
    let fanout_listener = Arc::clone(&fanout_tx);
    let mut connected: u32 = 0;
    let bytes_sent: u64 = 0;

    tokio::spawn(async move {
        loop {
            match listener.accept().await {
                Ok((stream, peer)) => {
                    connected += 1;
                    info!("UI bridge: client connected from {peer}");
                    let rx = fanout_listener.subscribe();
                    tokio::spawn(handle_client(stream, rx));

                    let status = BridgeStatus {
                        t_ms: 0, // runtime fills real timestamps later
                        connected_clients: connected,
                        bytes_sent_total: bytes_sent,
                        last_send_ok: true,
                    };
                    let _ = bus_status.ui_bridge_status.send(status);
                }
                Err(e) => warn!("UI bridge accept error: {e}"),
            }
        }
    });

    Ok(())
}

/// Per-client task: receive fan-out messages and forward to WebSocket.
async fn handle_client(
    stream: TcpStream,
    mut rx: broadcast::Receiver<Arc<String>>,
) {
    let ws = match tokio_tungstenite::accept_async(stream).await {
        Ok(ws) => ws,
        Err(e) => {
            warn!("UI bridge WS handshake failed: {e}");
            return;
        }
    };
    let (mut sink, _source) = futures_util::StreamExt::split(ws);

    loop {
        match rx.recv().await {
            Ok(msg) => {
                if sink.send(Message::Text(msg.as_str().to_owned().into())).await.is_err() {
                    break; // client disconnected
                }
            }
            Err(broadcast::error::RecvError::Lagged(n)) => {
                warn!("UI bridge client lagged by {n} messages");
            }
            Err(broadcast::error::RecvError::Closed) => break,
        }
    }
}
