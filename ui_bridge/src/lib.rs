//! Read-only WebSocket visualization bridge with HTTP overlay page.
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
//! - `sensor/ultrasonic`   — latest ultrasonic distance reading
//!
//! # HTTP overlay page
//! Plain HTTP GET requests to port 9000 receive an HTML page that embeds
//! the MJPEG stream (port 8080) and overlays telemetry from the WebSocket.
//!
//! # Default port
//! 9000 (configurable via `UiBridgeConfig`).

use std::sync::Arc;

use anyhow::Result;
use core_types::BridgeStatus;
use futures_util::SinkExt;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;
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

/// HTML overlay page served to plain HTTP clients.
const OVERLAY_HTML: &str = r#"<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Robot View</title>
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; background: #111; display: flex; flex-direction: column; align-items: center; justify-content: flex-start; min-height: 100vh; font-family: monospace; }
    #container { position: relative; display: inline-block; margin-top: 8px; }
    #stream { display: block; max-width: 100%; border: 1px solid #333; }
    #overlay { position: absolute; top: 8px; left: 8px; background: rgba(0,0,0,0.65); color: #0f0; padding: 8px 12px; font-size: 14px; line-height: 1.8; border-radius: 4px; min-width: 200px; }
    .label { color: #8a8; }
    .val { color: #0f0; }
    .warn { color: #fa0; }
    .err  { color: #f44; }
    h3 { margin: 0 0 4px 0; color: #fff; font-size: 12px; letter-spacing: 2px; border-bottom: 1px solid #444; padding-bottom: 4px; }
  </style>
</head>
<body>
  <div id="container">
    <img id="stream" src="" alt="MJPEG stream loading...">
    <div id="overlay">
      <h3>ROBOT TELEMETRY</h3>
      <div><span class="label">MODE </span><span id="mode" class="val">--</span></div>
      <div><span class="label">US   </span><span id="us" class="val">--</span> cm</div>
      <div><span class="label">X    </span><span id="x" class="val">--</span> m</div>
      <div><span class="label">Y    </span><span id="y" class="val">--</span> m</div>
      <div><span class="label">HDG  </span><span id="hdg" class="val">--</span>&deg;</div>
      <div><span class="label">WS   </span><span id="ws_st" class="warn">connecting</span></div>
    </div>
  </div>
  <script>
    const host = window.location.hostname;
    document.getElementById('stream').src = 'http://' + host + ':8080/';

    function fmt(v, d) { return (v != null && v.toFixed) ? v.toFixed(d) : '--'; }

    function stateLabel(data) {
      if (typeof data === 'string') return data;
      if (typeof data === 'object' && data !== null) return Object.keys(data)[0];
      return '--';
    }

    function connect() {
      const ws = new WebSocket('ws://' + host + ':9000/');
      const wsSt = document.getElementById('ws_st');
      wsSt.className = 'warn'; wsSt.textContent = 'connecting';

      ws.onopen  = () => { wsSt.className = 'val'; wsSt.textContent = 'ok'; };
      ws.onclose = () => {
        wsSt.className = 'err'; wsSt.textContent = 'reconnecting';
        setTimeout(connect, 2000);
      };
      ws.onerror = () => ws.close();

      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          switch (msg.topic) {
            case 'executive/state':
              document.getElementById('mode').textContent = stateLabel(msg.data);
              break;
            case 'sensor/ultrasonic':
              document.getElementById('us').textContent = fmt(msg.data && msg.data.range_cm, 1);
              break;
            case 'slam/pose2d':
              document.getElementById('x').textContent   = fmt(msg.data && msg.data.x, 2);
              document.getElementById('y').textContent   = fmt(msg.data && msg.data.y, 2);
              const rad = msg.data && msg.data.theta_rad;
              document.getElementById('hdg').textContent = rad != null ? (rad * 180 / Math.PI).toFixed(1) : '--';
              break;
          }
        } catch(e) {}
      };
    }
    connect();
  </script>
</body>
</html>
"#;

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
        let mut rx_us       = bus_agg.ultrasonic.subscribe();

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
                Ok(reading) = rx_us.recv() => {
                    if let Ok(msg) = serde_json::to_string(&serde_json::json!({
                        "topic": "sensor/ultrasonic",
                        "t_ms":  reading.t_ms,
                        "data":  reading,
                    })) {
                        let _ = fanout_agg.send(Arc::new(msg));
                    }
                }
                else => break,
            }
        }
    });

    // ── Listener task: accept connections, route to WS or HTTP handler ────
    let addr = format!("0.0.0.0:{}", cfg.port);
    // SO_REUSEADDR lets us rebind immediately after a crash or fast restart.
    let socket = tokio::net::TcpSocket::new_v4()?;
    socket.set_reuseaddr(true)?;
    socket.bind(addr.parse()?)?;
    let listener = socket.listen(128)?;
    info!("UI bridge listening on ws://{addr}  (overlay: http://{addr}/)");

    let bus_status = Arc::clone(&bus);
    let fanout_listener = Arc::clone(&fanout_tx);
    let mut connected: u32 = 0;
    let bytes_sent: u64 = 0;

    tokio::spawn(async move {
        loop {
            match listener.accept().await {
                Ok((stream, peer)) => {
                    // Peek at the request to distinguish WebSocket upgrade
                    // from a plain HTTP GET (browser opening the overlay page).
                    let mut peek = [0u8; 512];
                    let is_ws = match stream.peek(&mut peek).await {
                        Ok(n) => {
                            let h = std::str::from_utf8(&peek[..n]).unwrap_or("");
                            h.contains("Upgrade: websocket") || h.contains("upgrade: websocket")
                        }
                        Err(_) => false,
                    };

                    if is_ws {
                        connected += 1;
                        info!("UI bridge: WS client connected from {peer}");
                        let rx = fanout_listener.subscribe();
                        tokio::spawn(handle_client(stream, rx));

                        let status = BridgeStatus {
                            t_ms: 0,
                            connected_clients: connected,
                            bytes_sent_total: bytes_sent,
                            last_send_ok: true,
                        };
                        let _ = bus_status.ui_bridge_status.send(status);
                    } else {
                        tokio::spawn(serve_overlay(stream));
                    }
                }
                Err(e) => warn!("UI bridge accept error: {e}"),
            }
        }
    });

    Ok(())
}

/// Serve the HTML overlay page to a plain HTTP client.
async fn serve_overlay(mut stream: TcpStream) {
    let body = OVERLAY_HTML.as_bytes();
    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(header.as_bytes()).await;
    let _ = stream.write_all(body).await;
    // Flush and half-close the write side so the OS sends FIN instead of RST.
    // Without this the kernel may reset the connection before all bytes leave
    // the send buffer (especially when there is unread request data pending).
    let _ = stream.flush().await;
    let _ = stream.shutdown().await;
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
