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
//! - `sensor/nearest_m`    — nearest obstacle from pseudo-lidar (camera-based, all directions)
//!
//! - `vision/pseudo_lidar`  — current lidar scan (rays with angle + range)
//!
//! # HTTP pages
//! - `GET /`    — video overlay: MJPEG stream + telemetry HUD
//! - `GET /map` — 2-D map canvas: occupancy grid, robot, frontiers, lidar rays
//!
//! # Default port
//! 9000 (configurable via `UiBridgeConfig`).

use std::sync::Arc;

use anyhow::Result;
use bus::BridgeCommand;
use core_types::{BridgeStatus, CmdVel};
use futures_util::{SinkExt, StreamExt};
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;
use tokio::sync::{broadcast, watch};
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

/// 2-D map canvas page served at `/map`.
const MAP_HTML: &str = r#"<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Robot Map</title>
  <style>
    * { box-sizing: border-box; margin: 0; }
    body { background: #0d0d0d; display: flex; height: 100vh; overflow: hidden; font-family: monospace; }
    #map-wrap { flex: 1; overflow: auto; position: relative; padding: 8px; }
    #map-bg, #map-fg { position: absolute; top: 8px; left: 8px; image-rendering: pixelated; }
    #cam-panel { width: 320px; background: #111; padding: 8px; display: flex; flex-direction: column; gap: 4px; flex-shrink: 0; border-left: 1px solid #333; overflow-y: auto; }
    #cam-stream { width: 304px; height: 228px; display: block; object-fit: contain; background: #000; border: 1px solid #444; }
    #cam-placeholder { width: 304px; height: 228px; background: #1a1a1a; border: 1px solid #333; display: flex; align-items: center; justify-content: center; color: #555; font-size: 12px; }
    .panel-label { color: #666; font-size: 10px; letter-spacing: 1px; margin-top: 6px; }
    #depth-stream { width: 256px; height: 256px; display: block; image-rendering: pixelated; background: #000; border: 1px solid #444; }
    #depth-placeholder { width: 256px; height: 256px; background: #1a1a1a; border: 1px solid #333; display: flex; align-items: center; justify-content: center; color: #555; font-size: 12px; }
    #lidar-canvas { display: block; background: #0d0d0d; border: 1px solid #444; }
    #sidebar { width: 170px; background: #111; padding: 8px 12px; font-size: 13px; line-height: 2; flex-shrink: 0; border-left: 1px solid #333; overflow-y: auto; }
    .label { color: #8a8; }
    .val  { color: #0f0; }
    .warn { color: #fa0; }
    .err  { color: #f44; }
    h3 { margin: 0 0 4px; color: #fff; font-size: 11px; letter-spacing: 2px; border-bottom: 1px solid #444; padding-bottom: 4px; }
    #legend { margin-top: 12px; font-size: 11px; color: #aaa; }
    .ls { display:inline-block; width:12px; height:12px; margin-right:4px; vertical-align:middle; border: 1px solid #555; }
  </style>
</head>
<body>
  <div id="map-wrap">
    <canvas id="map-bg"></canvas>
    <canvas id="map-fg"></canvas>
  </div>
  <div id="cam-panel">
    <div class="panel-label">CAMERA</div>
    <img id="cam-stream" src="" alt="camera">
    <div id="cam-placeholder">NO FEED</div>
    <div class="panel-label">DEPTH MAP &nbsp;<span style="color:#888;font-size:9px">(white=near &nbsp;red=mask)</span></div>
    <img id="depth-stream" src="" alt="depth">
    <div id="depth-placeholder">NO DEPTH</div>
    <div class="panel-label">LIDAR POLAR &nbsp;<span style="color:#888;font-size:9px">(fwd=up &nbsp;orange=nearest)</span></div>
    <canvas id="lidar-canvas" width="290" height="260"></canvas>
  </div>
  <div id="sidebar">
    <h3>TELEMETRY</h3>
    <div><span class="label">MODE  </span><span id="mode" class="val">--</span></div>
    <div><span class="label">US    </span><span id="us" class="val">--</span> cm</div>
    <div><span class="label">VIS   </span><span id="vis" class="val">--</span> cm</div>
    <div><span class="label">X     </span><span id="x" class="val">--</span> m</div>
    <div><span class="label">Y     </span><span id="y" class="val">--</span> m</div>
    <div><span class="label">HDG   </span><span id="hdg" class="val">--</span>&deg;</div>
    <div><span class="label">CONF  </span><span id="conf" class="val">--</span></div>
    <div><span class="label">PAN   </span><span id="pan" class="val">--</span>&deg;</div>
    <div><span class="label">TILT  </span><span id="tilt" class="val">--</span>&deg;</div>
    <div><span class="label">CELLS  </span><span id="cells" class="val">0</span></div>
    <div><span class="label">RUN TIME </span><span id="runtime" class="val">--:--</span></div>
    <div><span class="label">CRASHES  </span><span id="crashes" class="val">0</span></div>
    <div><span class="label">NEAR MISS </span><span id="estops" class="val">0</span></div>
    <div><span class="label">EPISODE  </span><span id="episode" class="val">0</span></div>
    <div><span class="label">EP TIMEOUT </span><span id="ep_timeout" class="val">0</span></div>
    <div><span class="label">WS     </span><span id="ws_st" class="warn">connecting</span></div>
    <div style="margin-top:12px">
      <button onclick="ws&&ws.send('arm')" style="background:#1a1;color:#fff;padding:6px 14px;border:none;cursor:pointer;font-family:monospace;width:100%;margin-bottom:4px">ARM</button>
      <button onclick="ws&&ws.send('stop')" style="background:#a11;color:#fff;padding:6px 14px;border:none;cursor:pointer;font-family:monospace;width:100%;margin-bottom:4px">STOP</button>
      <button onclick="sendManual()" style="background:#225;color:#fff;padding:6px 14px;border:none;cursor:pointer;font-family:monospace;width:100%;margin-bottom:4px">MANUAL</button>
      <button onclick="sendAuto()" style="background:#522;color:#fff;padding:6px 14px;border:none;cursor:pointer;font-family:monospace;width:100%">AUTO</button>
    </div>
    <div id="manual-hint" style="margin-top:6px;font-size:11px;color:#555">WASD=drive QE=strafe</div>
    <div id="legend">
      <h3 style="margin-top:8px">LEGEND</h3>
      <div><span class="ls" style="background:#1a4a1e"></span>free</div>
      <div><span class="ls" style="background:#555"></span>unknown</div>
      <div><span class="ls" style="background:#c0c0c0"></span>obstacle</div>
      <div><span class="ls" style="background:#00cccc"></span>frontier</div>
      <div><span class="ls" style="background:#00ff44"></span>robot</div>
    </div>
  </div>
  <script>
    const PX   = 4;      // canvas pixels per grid cell
    const RES  = 0.05;   // metres per cell
    const host = window.location.hostname;

    // ── Map state ──────────────────────────────────────────────────────────
    // Cells store absolute log-odds values from GridDelta.
    const grid = new Map();   // "cx,cy" → log_odds (absolute)
    let crashCount = 0;
    let crashMarkers = [];
    let lastRobotPx = 0, lastRobotPy = 0;
    let lastExecState = null;
    let simStartMs = null;   // server epoch ms when sim process started
    let minCX = null, maxCX = null, minCY = null, maxCY = null;

    // ── Ground-truth sim walls ──────────────────────────────────────────────
    let truWalls = null;   // Uint8Array, 200×200, truWalls[y*200+x]=1 means wall
    let truW = 0, truH = 0, truRes = 0.05;

    function drawTrueWalls() {
      if (!truWalls || minCX === null) return;
      bgX.fillStyle = '#606060';   // visible even before the robot scans these cells
      for (let y = 0; y < truH; y++) {
        for (let x = 0; x < truW; x++) {
          if (truWalls[y * truW + x]) {
            const [px, py] = cellPx(x, y);
            bgX.fillRect(px, py, PX, PX);
          }
        }
      }
    }

    // ── Overlay state ──────────────────────────────────────────────────────
    let pose = null;
    let frontiers = [];
    let lidarRays = null;
    let lastPan = 0;       // gimbal pan in degrees; 0 = forward
    let flashFrames = 0;
    let flashColor = '#ff4444';
    let flashLabel = '';

    // ── Canvas elements ────────────────────────────────────────────────────
    const bgC = document.getElementById('map-bg');
    const fgC = document.getElementById('map-fg');
    const bgX = bgC.getContext('2d');
    const fgX = fgC.getContext('2d');

    // Initialise with tiny placeholder so canvas exists.
    bgC.width = bgC.height = fgC.width = fgC.height = 4;

    // ── Coordinate helpers ─────────────────────────────────────────────────
    // World y increases up; canvas y increases down — flip Y so the map
    // renders with physical orientation (north = up).
    function cellPx(cx, cy) {
      return [(cx - minCX) * PX, (maxCY - cy) * PX];
    }
    function worldPx(xm, ym) {
      return cellPx(xm / RES, ym / RES);
    }

    // ── Cell colour ────────────────────────────────────────────────────────
    function cellColor(lo) {
      if (lo >  1.5) return '#c0c0c0';  // definitely occupied
      if (lo >  0.5) return '#707070';  // probably occupied
      if (lo < -0.5) return '#0f2010';  // free
      return '#2a2a2a';                 // uncertain
    }

    // ── Draw a single cell onto the bg canvas ─────────────────────────────
    function drawCell(cx, cy, lo) {
      const [px, py] = cellPx(cx, cy);
      const isWall = truWalls && cx >= 0 && cx < truW && cy >= 0 && cy < truH
                     && truWalls[cy * truW + cx];
      // Wall cells: show discovery when scanned, but never go darker than base.
      bgX.fillStyle = (isWall && lo < 0.5) ? '#606060' : cellColor(lo);
      bgX.fillRect(px, py, PX, PX);
    }

    // ── Full bg redraw (after canvas resize) ──────────────────────────────
    function redrawBg() {
      bgX.fillStyle = '#1a1a1a';
      bgX.fillRect(0, 0, bgC.width, bgC.height);
      drawTrueWalls();                          // base pass: unscanned walls
      for (const [key, lo] of grid) {
        const c = key.split(',');
        drawCell(+c[0], +c[1], lo);            // occupancy; walls clamped above
      }
    }

    // ── Apply a GridDelta message ──────────────────────────────────────────
    function applyDelta(cells) {
      if (!cells || cells.length === 0) return;

      let boundsChanged = false;
      for (const [cx, cy, lo] of cells) {
        grid.set(cx + ',' + cy, lo);
        if (minCX === null) {
          minCX = maxCX = cx; minCY = maxCY = cy; boundsChanged = true;
        } else {
          if (cx < minCX) { minCX = cx; boundsChanged = true; }
          if (cx > maxCX) { maxCX = cx; boundsChanged = true; }
          if (cy < minCY) { minCY = cy; boundsChanged = true; }
          if (cy > maxCY) { maxCY = cy; boundsChanged = true; }
        }
      }
      // If ground truth is loaded, never let the canvas shrink below the
      // full sim grid — this prevents a resize from blacking out the walls.
      if (truWalls) {
        if (minCX === null) {
          minCX = 0; maxCX = truW - 1; minCY = 0; maxCY = truH - 1; boundsChanged = true;
        } else {
          if (minCX > 0)         { minCX = 0;         boundsChanged = true; }
          if (maxCX < truW - 1)  { maxCX = truW - 1;  boundsChanged = true; }
          if (minCY > 0)         { minCY = 0;         boundsChanged = true; }
          if (maxCY < truH - 1)  { maxCY = truH - 1;  boundsChanged = true; }
        }
      }

      const w = (maxCX - minCX + 2) * PX;
      const h = (maxCY - minCY + 2) * PX;

      if (boundsChanged || bgC.width !== w || bgC.height !== h) {
        bgC.width = bgC.height = fgC.width = fgC.height = 1; // force clear
        bgC.width  = fgC.width  = w;
        bgC.height = fgC.height = h;
        redrawBg();
      } else {
        for (const [cx, cy, lo] of cells) drawCell(cx, cy, lo);
      }

      document.getElementById('cells').textContent = grid.size;
    }

    // ── Overlay (robot + frontiers + lidar rays) ───────────────────────────
    function drawOverlay() {
      requestAnimationFrame(drawOverlay);
      // Update run-time clock every frame.
      if (simStartMs !== null) {
        const secs = Math.floor((Date.now() - simStartMs) / 1000);
        const m = Math.floor(secs / 60).toString().padStart(2, '0');
        const s = (secs % 60).toString().padStart(2, '0');
        document.getElementById('runtime').textContent = m + ':' + s;
      }
      fgX.clearRect(0, 0, fgC.width, fgC.height);
      if (!pose || minCX === null) return;

      const [rx, ry] = worldPx(pose.x_m, pose.y_m);
      const theta = pose.theta_rad;

      // Lidar rays (faint green).
      if (lidarRays) {
        fgX.strokeStyle = 'rgba(0,255,0,0.12)';
        fgX.lineWidth = 1;
        for (const ray of lidarRays) {
          const wa = theta + ray.angle_rad;
          const rp = ray.range_m / RES * PX;
          fgX.beginPath();
          fgX.moveTo(rx, ry);
          fgX.lineTo(rx + Math.cos(wa) * rp, ry - Math.sin(wa) * rp);
          fgX.stroke();
        }
      }

      // Frontiers (cyan circles, sized by frontier area).
      // True-wall overlay on fg — drawn after rays so wall cells aren't
      // visually dominated by the green ray strokes terminating there.
      if (truWalls) {
        fgX.fillStyle = 'rgba(120,120,120,0.55)';
        for (let ty = 0; ty < truH; ty++) {
          for (let tx = 0; tx < truW; tx++) {
            if (truWalls[ty * truW + tx]) {
              const [px, py] = cellPx(tx, ty);
              fgX.fillRect(px, py, PX, PX);
            }
          }
        }
      }

      // Frontiers (cyan circles, sized by frontier area).
      for (const f of frontiers) {
        const [fx, fy] = worldPx(f.centroid_x_m, f.centroid_y_m);
        const r = Math.max(3, Math.sqrt(f.size_cells) * PX * 0.3);
        fgX.strokeStyle = '#00cccc';
        fgX.lineWidth = 2;
        fgX.beginPath();
        fgX.arc(fx, fy, r, 0, Math.PI * 2);
        fgX.stroke();
      }

      // Crash markers — permanent red dots at world-coordinate collision positions
      fgX.fillStyle = '#ff2200';
      for (const cm of crashMarkers) {
        const [cpx, cpy] = worldPx(cm.x_m, cm.y_m);
        fgX.beginPath();
        fgX.arc(cpx, cpy, 6, 0, Math.PI * 2);
        fgX.fill();
      }

      // Robot triangle.
      const sz = 8;
      lastRobotPx = rx; lastRobotPy = ry;
      fgX.save();
      fgX.translate(rx, ry);
      fgX.rotate(-theta);   // negate: Y is flipped
      fgX.fillStyle = flashFrames > 0 ? flashColor : '#00ff44';
      if (flashFrames > 0) flashFrames--;
      fgX.beginPath();
      fgX.moveTo( sz,  0);
      fgX.lineTo(-sz, -sz * 0.6);
      fgX.lineTo(-sz,  sz * 0.6);
      fgX.closePath();
      fgX.fill();
      fgX.restore();

      // Camera / gimbal pan direction — yellow line from robot centre.
      // Shows where the camera is looking relative to the heading triangle.
      const panRad = lastPan * Math.PI / 180;
      const lookAngle = theta - panRad;  // pan_deg>0=right → subtract (CCW-positive frame)
      const lookLen = sz * 3.5;
      fgX.strokeStyle = '#ffee00';
      fgX.lineWidth = 2;
      fgX.beginPath();
      fgX.moveTo(rx, ry);
      fgX.lineTo(rx + Math.cos(lookAngle) * lookLen, ry - Math.sin(lookAngle) * lookLen);
      fgX.stroke();

      // Episode-end label (CRASH / TIMEOUT).
      if (flashFrames > 0 && flashLabel) {
        fgX.save();
        fgX.font = 'bold 28px monospace';
        fgX.textAlign = 'center';
        fgX.fillStyle = flashColor;
        fgX.globalAlpha = flashFrames / 60;
        fgX.fillText(flashLabel, fgC.width / 2, fgC.height / 2);
        fgX.restore();
      }
      // Draw lidar polar plot on every overlay frame.
      drawLidarPolar();
    }

    // ── Telemetry helpers ──────────────────────────────────────────────────
    function fmt(v, d) { return (v != null && v.toFixed) ? v.toFixed(d) : '--'; }
    function stateLabel(d) {
      if (typeof d === 'string') return d;
      if (typeof d === 'object' && d !== null) return Object.keys(d)[0];
      return '--';
    }
    function distClass(cm) {
      if (cm == null || isNaN(cm)) return 'val';
      return cm < 30 ? 'err' : cm < 60 ? 'warn' : 'val';
    }

    // ── Manual drive ───────────────────────────────────────────────────────
    let manualMode = false;
    const keysDown = new Set();

    function manualVelFromKeys() {
      let vx = 0, vy = 0, omega = 0;
      if (keysDown.has('w')) vx    =  0.8;
      if (keysDown.has('s')) vx    = -0.8;
      if (keysDown.has('a')) omega =  2.0;
      if (keysDown.has('d')) omega = -2.0;
      if (keysDown.has('q')) vy    =  0.8;
      if (keysDown.has('e')) vy    = -0.8;
      return 'vel:' + vx + ',' + vy + ',' + omega;
    }

    document.addEventListener('keydown', (ev) => {
      if (!manualMode || !ws) return;
      const k = ev.key.toLowerCase();
      if ('wasdqe'.includes(k) && k.length === 1 && !keysDown.has(k)) {
        keysDown.add(k);
        ws.send(manualVelFromKeys());
        ev.preventDefault();
      }
    });
    document.addEventListener('keyup', (ev) => {
      if (!manualMode || !ws) return;
      const k = ev.key.toLowerCase();
      if ('wasdqe'.includes(k) && k.length === 1) {
        keysDown.delete(k);
        ws.send(manualVelFromKeys());
        ev.preventDefault();
      }
    });

    // Key-hold repeat: re-send held velocity at sim-tick rate (100 ms) so the
    // motor task always has a fresh command and the robot doesn't stutter.
    // All keys use the same 100 ms rate — fast H-bridge braking (BRAKE_TAU_S)
    // keeps rotation fine-grained without needing a separate skip interval.
    setInterval(() => {
      if (manualMode && ws && keysDown.size > 0) ws.send(manualVelFromKeys());
    }, 100);

    function sendManual() { manualMode = true;  ws && ws.send('manual'); }
    function sendAuto()   { manualMode = false; keysDown.clear(); ws && ws.send('auto'); }

    // ── WebSocket ──────────────────────────────────────────────────────────
    var ws;
    function connect() {
      ws = new WebSocket('ws://' + host + ':9000/');
      const wsSt = document.getElementById('ws_st');
      wsSt.className = 'warn'; wsSt.textContent = 'connecting';
      ws.onopen  = () => { wsSt.className = 'val'; wsSt.textContent = 'ok'; startCamStream(); };
      ws.onclose = () => {
        wsSt.className = 'err'; wsSt.textContent = 'reconnecting';
        if (simStartMs !== null) {
          const secs = Math.floor((Date.now() - simStartMs) / 1000);
          const m = Math.floor(secs / 60).toString().padStart(2,'0');
          const s = (secs % 60).toString().padStart(2,'0');
          document.getElementById('runtime').textContent = m + ':' + s + ' (stopped)';
          simStartMs = null;
        }
        setTimeout(connect, 2000);
      };
      ws.onerror = () => ws.close();

      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          switch (msg.topic) {
            case 'map/grid_delta':
              if (msg.data && msg.data.cells) applyDelta(msg.data.cells);
              break;
            case 'map/frontiers':
              frontiers = msg.data || [];
              break;
            case 'vision/pseudo_lidar':
              lidarRays = msg.data && msg.data.rays;
              break;
            case 'slam/pose2d':
              pose = msg.data;
              document.getElementById('x').textContent    = fmt(pose.x_m, 2);
              document.getElementById('y').textContent    = fmt(pose.y_m, 2);
              document.getElementById('conf').textContent = fmt(pose.confidence, 2);
              document.getElementById('hdg').textContent  = (pose.theta_rad * 180 / Math.PI).toFixed(1);
              break;
            case 'executive/state': {
              const lbl = stateLabel(msg.data);
              document.getElementById('mode').textContent = lbl;
              manualMode = (lbl === 'ManualDrive');
              if (!manualMode) keysDown.clear();
              document.getElementById('manual-hint').style.color = manualMode ? '#0af' : '#555';
              if (lbl === 'SafetyStopped' && lastExecState !== 'SafetyStopped') {
                flashColor = '#ffaa00'; flashLabel = 'NEAR MISS'; flashFrames = 60;
              } else if (lbl === 'Fault' && lastExecState !== 'Fault') {
                flashColor = '#ff2200'; flashLabel = 'CRASH';   flashFrames = 60;
              }
              lastExecState = lbl;
              break;
            }
            case 'sensor/ultrasonic': {
              const cm = msg.data && msg.data.range_cm;
              const el = document.getElementById('us');
              el.textContent = fmt(cm, 1); el.className = distClass(cm);
              break;
            }
            case 'sensor/nearest_m': {
              const cm = msg.data != null ? msg.data * 100 : null;
              const el = document.getElementById('vis');
              el.textContent = fmt(cm, 0); el.className = distClass(cm);
              break;
            }
            case 'gimbal/pan':
              lastPan = msg.data || 0;
              document.getElementById('pan').textContent = fmt(msg.data, 1);
              break;
            case 'gimbal/tilt':
              document.getElementById('tilt').textContent = fmt(msg.data, 1);
              break;
            case 'sim/start_time':
              simStartMs = msg.data || null;
              // Initial session start — clear everything including counts.
              grid.clear();
              crashMarkers = []; crashCount = 0;
              frontiers = []; lidarRays = null; pose = null;
              truWalls = null; truW = 0; truH = 0;
              minCX = maxCX = minCY = maxCY = null;
              bgC.width = bgC.height = fgC.width = fgC.height = 4;
              document.getElementById('crashes').textContent = '0';
              document.getElementById('crashes').className = 'val';
              document.getElementById('episode').textContent = '0';
              document.getElementById('ep_timeout').textContent = '0';
              break;
            case 'robot/episode': {
              // New room started — clear map state but keep cumulative crash/estop counts.
              const epNum = msg.data || 0;
              grid.clear();
              crashMarkers = [];
              frontiers = []; lidarRays = null; pose = null;
              truWalls = null; truW = 0; truH = 0;
              minCX = maxCX = minCY = maxCY = null;
              bgC.width = bgC.height = fgC.width = fgC.height = 4;
              document.getElementById('episode').textContent = epNum;
              flashColor = '#00aaff'; flashLabel = 'ROOM ' + epNum; flashFrames = 90;
              break;
            }
            case 'robot/episode_timeout': {
              const n = msg.data || 0;
              const el = document.getElementById('ep_timeout');
              el.textContent = n;
              el.className = n > 0 ? 'warn' : 'val';
              break;
            }
            case 'robot/collision': {
              const prev = crashCount;
              crashCount = msg.data || 0;
              if (crashCount > prev) {
                flashColor = '#ff2200'; flashLabel = 'CRASH'; flashFrames = 60;
                if (msg.x != null && msg.y != null) {
                  crashMarkers.push({ x_m: msg.x, y_m: msg.y });
                } else if (pose) {
                  crashMarkers.push({ x_m: pose.x_m, y_m: pose.y_m });
                }
              }
              const el = document.getElementById('crashes');
              el.textContent = crashCount;
              el.className = crashCount > 0 ? 'err' : 'val';
              break;
            }
            case 'robot/estop': {
              const estopCount = msg.data || 0;
              const el = document.getElementById('estops');
              el.textContent = estopCount;
              el.className = estopCount > 0 ? 'warn' : 'val';
              break;
            }
            case 'sim/ground_truth': {
              truW = msg.width; truH = msg.height; truRes = msg.res;
              truWalls = new Uint8Array(msg.walls);
              // Size canvas to full arena on first episode only — keep occupancy
              // cells across episodes so the map accumulates coverage over time.
              if (minCX === null) {
                minCX = 0; maxCX = truW - 1; minCY = 0; maxCY = truH - 1;
                const w = (truW + 1) * PX, h = (truH + 1) * PX;
                bgC.width = fgC.width = w;
                bgC.height = fgC.height = h;
              }
              // Repaint: new maze walls behind accumulated occupancy overlay.
              bgX.clearRect(0, 0, bgC.width, bgC.height);
              drawTrueWalls();
              for (const [key, lo] of grid) {
                const c = key.split(',');
                drawCell(+c[0], +c[1], lo);
              }
              break;
            }
          }
        } catch(e) {}
      };
    }

    connect();
    requestAnimationFrame(drawOverlay);

    // Camera stream (MJPEG on port 8080).
    function startCamStream() {
      const img = document.getElementById('cam-stream');
      const ph  = document.getElementById('cam-placeholder');
      img.onload  = () => { img.style.display = 'block'; ph.style.display = 'none'; };
      img.onerror = () => { img.style.display = 'none';  ph.style.display = 'flex';
                            setTimeout(startCamStream, 3000); };
      img.src = 'http://' + host + ':8080/?_=' + Date.now();
    }

    // Depth map stream (MJPEG on port 8081 — 32×32 f32 scaled to 256×256 JPEG).
    function startDepthStream() {
      const img = document.getElementById('depth-stream');
      const ph  = document.getElementById('depth-placeholder');
      img.onload  = () => { img.style.display = 'block'; ph.style.display = 'none'; };
      img.onerror = () => { img.style.display = 'none';  ph.style.display = 'flex';
                            setTimeout(startDepthStream, 3000); };
      img.src = 'http://' + host + ':8081/?_=' + Date.now();
    }

    // ── Lidar polar plot ───────────────────────────────────────────────────
    const lidarC = document.getElementById('lidar-canvas');
    const lidarX = lidarC.getContext('2d');
    const LIDAR_MAX_M = 3.0;
    const LIDAR_FOV   = 55 * Math.PI / 180;  // ±55° display cone
    const STOP_M      = 0.15;                 // obstacle-stop threshold ring

    function drawLidarPolar() {
      const w = lidarC.width, h = lidarC.height;
      const cx = w / 2;
      const cy = h * 0.52;
      const maxPx = Math.min(cx - 2, cy - 14);
      const scale = maxPx / LIDAR_MAX_M;

      lidarX.fillStyle = '#0d0d0d';
      lidarX.fillRect(0, 0, w, h);

      // Distance rings + labels.
      lidarX.lineWidth = 1;
      for (let r = 0.5; r <= LIDAR_MAX_M + 0.01; r += 0.5) {
        const rp = r * scale;
        lidarX.strokeStyle = r % 1.0 < 0.01 ? '#333' : '#222';
        lidarX.beginPath(); lidarX.arc(cx, cy, rp, 0, Math.PI * 2); lidarX.stroke();
        if (r % 1.0 < 0.01) {
          lidarX.fillStyle = '#444'; lidarX.font = '9px monospace';
          lidarX.fillText(r.toFixed(0) + 'm', cx + 2, cy - rp + 9);
        }
      }

      // Cardinal spokes (every 45°).
      lidarX.strokeStyle = '#1e1e1e'; lidarX.lineWidth = 1;
      for (let i = 0; i < 8; i++) {
        const a = i * Math.PI / 4;
        lidarX.beginPath();
        lidarX.moveTo(cx, cy);
        lidarX.lineTo(cx - Math.sin(a) * maxPx, cy - Math.cos(a) * maxPx);
        lidarX.stroke();
      }

      // FOV cone fill (±55° from forward).
      lidarX.fillStyle = 'rgba(0,90,0,0.15)';
      lidarX.beginPath();
      lidarX.moveTo(cx, cy);
      lidarX.arc(cx, cy, maxPx, -Math.PI / 2 - LIDAR_FOV, -Math.PI / 2 + LIDAR_FOV, false);
      lidarX.closePath(); lidarX.fill();

      // FOV cone edges.
      lidarX.strokeStyle = 'rgba(0,180,0,0.3)'; lidarX.lineWidth = 1;
      for (const sign of [-1, 1]) {
        const a = sign * LIDAR_FOV;
        lidarX.beginPath();
        lidarX.moveTo(cx, cy);
        lidarX.lineTo(cx - Math.sin(a) * maxPx, cy - Math.cos(a) * maxPx);
        lidarX.stroke();
      }

      // Rays.
      if (lidarRays && lidarRays.length > 0) {
        let nearR = Infinity, nearIdx = -1;
        for (let i = 0; i < lidarRays.length; i++) {
          if (lidarRays[i].range_m < nearR) { nearR = lidarRays[i].range_m; nearIdx = i; }
        }
        lidarX.lineWidth = 1;
        for (let i = 0; i < lidarRays.length; i++) {
          const ray = lidarRays[i];
          const rp  = Math.min(ray.range_m, LIDAR_MAX_M) * scale;
          const ex  = cx - Math.sin(ray.angle_rad) * rp;
          const ey  = cy - Math.cos(ray.angle_rad) * rp;
          if (i === nearIdx) {
            lidarX.strokeStyle = '#ff6600'; lidarX.lineWidth = 2;
            lidarX.beginPath(); lidarX.moveTo(cx, cy); lidarX.lineTo(ex, ey); lidarX.stroke();
            lidarX.fillStyle = '#ff6600';
            lidarX.beginPath(); lidarX.arc(ex, ey, 3, 0, Math.PI * 2); lidarX.fill();
            lidarX.lineWidth = 1;
          } else {
            lidarX.strokeStyle = 'rgba(0,210,0,0.4)';
            lidarX.beginPath(); lidarX.moveTo(cx, cy); lidarX.lineTo(ex, ey); lidarX.stroke();
          }
        }
      }

      // Obstacle-stop threshold ring (dashed red, 0.15 m).
      lidarX.strokeStyle = 'rgba(255,60,60,0.7)'; lidarX.lineWidth = 1.5;
      lidarX.setLineDash([3, 3]);
      lidarX.beginPath(); lidarX.arc(cx, cy, STOP_M * scale, 0, Math.PI * 2); lidarX.stroke();
      lidarX.setLineDash([]);

      // Robot dot + forward tick.
      lidarX.fillStyle = '#00ff44';
      lidarX.beginPath(); lidarX.arc(cx, cy, 4, 0, Math.PI * 2); lidarX.fill();
      lidarX.strokeStyle = '#00ff44'; lidarX.lineWidth = 2;
      lidarX.beginPath(); lidarX.moveTo(cx, cy); lidarX.lineTo(cx, cy - 10); lidarX.stroke();

      // Labels.
      lidarX.fillStyle = '#555'; lidarX.font = '9px monospace'; lidarX.textAlign = 'center';
      lidarX.fillText('FWD', cx, cy - maxPx - 3);
      lidarX.fillText('L', cx - maxPx - 8, cy + 4);
      lidarX.fillText('R', cx + maxPx + 6, cy + 4);
      lidarX.textAlign = 'left';
    }

    // Integrate polar draw into the existing overlay rAF loop.
    const _origDrawOverlay = drawOverlay;

    startCamStream();
    startDepthStream();
  </script>
</body>
</html>
"#;

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
    #sensor-row { display: flex; gap: 12px; margin-top: 8px; align-items: flex-start; }
    .sensor-block { display: flex; flex-direction: column; gap: 3px; }
    .sensor-label { color: #666; font-size: 10px; letter-spacing: 1px; }
    #depth-stream { width: 256px; height: 256px; display: block; image-rendering: pixelated; background: #000; border: 1px solid #444; }
    #depth-placeholder { width: 256px; height: 256px; background: #1a1a1a; border: 1px solid #333; display: flex; align-items: center; justify-content: center; color: #555; font-size: 12px; }
    #lidar-canvas-ov { display: block; background: #0d0d0d; border: 1px solid #444; }
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
    <img id="stream" src="" alt="Camera stream">
    <div id="cam-placeholder" style="display:none;width:640px;height:480px;background:#222;border:1px solid #333;display:none;align-items:center;justify-content:center;color:#555;font-size:14px">No camera stream (simulation mode)</div>
    <div id="overlay">
      <h3>ROBOT TELEMETRY</h3>
      <div><span class="label">MODE </span><span id="mode" class="val">--</span></div>
      <div><span class="label">US   </span><span id="us" class="val">--</span> cm</div>
      <div><span class="label">VIS  </span><span id="vis" class="val">--</span> cm</div>
      <div><span class="label">X    </span><span id="x" class="val">--</span> m</div>
      <div><span class="label">Y    </span><span id="y" class="val">--</span> m</div>
      <div><span class="label">HDG  </span><span id="hdg" class="val">--</span>&deg;</div>
      <div><span class="label">CONF </span><span id="conf" class="val">--</span></div>
      <div><span class="label">PAN  </span><span id="pan" class="val">--</span>&deg;</div>
      <div><span class="label">TILT </span><span id="tilt" class="val">--</span>&deg;</div>
      <div><span class="label">CELLS </span><span id="cells" class="val">0</span></div>
      <div><span class="label">RUN TIME </span><span id="runtime" class="val">--:--</span></div>
      <div><span class="label">CRASHES </span><span id="crashes" class="val">0</span></div>
      <div><span class="label">NEAR MISS </span><span id="estops" class="val">0</span></div>
      <div><span class="label">EPISODE </span><span id="episode" class="val">0</span></div>
      <div><span class="label">EP TIMEOUT </span><span id="ep_timeout" class="val">0</span></div>
      <div><span class="label">WS   </span><span id="ws_st" class="warn">connecting</span></div>
    </div>
  </div>
  <div style="margin:8px;display:flex;gap:8px">
    <button onclick="ws&&ws.send('arm')" style="background:#1a1;color:#fff;padding:8px 18px;border:none;cursor:pointer;font-family:monospace">ARM</button>
    <button onclick="ws&&ws.send('stop')" style="background:#a11;color:#fff;padding:8px 18px;border:none;cursor:pointer;font-family:monospace">STOP</button>
  </div>
  <div id="sensor-row">
    <div class="sensor-block">
      <div class="sensor-label">DEPTH MAP &nbsp;<span style="font-size:9px">(white=near &nbsp;red=mask)</span></div>
      <img id="depth-stream" src="" alt="depth">
      <div id="depth-placeholder">NO DEPTH</div>
    </div>
    <div class="sensor-block">
      <div class="sensor-label">LIDAR POLAR &nbsp;<span style="font-size:9px">(fwd=up &nbsp;orange=nearest)</span></div>
      <canvas id="lidar-canvas-ov" width="280" height="260"></canvas>
    </div>
  </div>
  <script>
    const host = window.location.hostname;

    function startStream() {
      const img = document.getElementById('stream');
      const ph  = document.getElementById('cam-placeholder');
      img.onload  = () => { img.style.display = 'block'; ph.style.display = 'none'; };
      img.onerror = () => { img.style.display = 'none';  ph.style.display = 'flex'; };
      img.src = 'http://' + host + ':8080/';
    }
    startStream();

    let crashCount = 0;
    let simStartMs = null;

    function fmt(v, d) { return (v != null && v.toFixed) ? v.toFixed(d) : '--'; }

    function stateLabel(data) {
      if (typeof data === 'string') return data;
      if (typeof data === 'object' && data !== null) return Object.keys(data)[0];
      return '--';
    }

    // Color-code a distance display: red <30 cm, yellow <60 cm, green otherwise.
    function distClass(cm) {
      if (cm == null || isNaN(cm)) return 'val';
      if (cm < 30) return 'err';
      if (cm < 60) return 'warn';
      return 'val';
    }

    // Depth map stream (port 8081).
    function startDepthStreamOv() {
      const img = document.getElementById('depth-stream');
      const ph  = document.getElementById('depth-placeholder');
      img.onload  = () => { img.style.display = 'block'; ph.style.display = 'none'; };
      img.onerror = () => { img.style.display = 'none';  ph.style.display = 'flex';
                            setTimeout(startDepthStreamOv, 3000); };
      img.src = 'http://' + host + ':8081/?_=' + Date.now();
    }
    startDepthStreamOv();

    // ── Lidar polar plot ───────────────────────────────────────────────────
    let lidarRaysOv = null;
    const lidarCov = document.getElementById('lidar-canvas-ov');
    const lidarXov = lidarCov.getContext('2d');
    const LIDAR_MAX_M_OV = 3.0;
    const LIDAR_FOV_OV   = 55 * Math.PI / 180;
    const STOP_M_OV      = 0.15;

    function drawLidarPolarOv() {
      const w = lidarCov.width, h = lidarCov.height;
      const cx = w / 2, cy = h * 0.52;
      const maxPx = Math.min(cx - 2, cy - 14);
      const scale = maxPx / LIDAR_MAX_M_OV;

      lidarXov.fillStyle = '#0d0d0d';
      lidarXov.fillRect(0, 0, w, h);

      lidarXov.lineWidth = 1;
      for (let r = 0.5; r <= LIDAR_MAX_M_OV + 0.01; r += 0.5) {
        const rp = r * scale;
        lidarXov.strokeStyle = r % 1.0 < 0.01 ? '#333' : '#222';
        lidarXov.beginPath(); lidarXov.arc(cx, cy, rp, 0, Math.PI * 2); lidarXov.stroke();
        if (r % 1.0 < 0.01) {
          lidarXov.fillStyle = '#444'; lidarXov.font = '9px monospace';
          lidarXov.fillText(r.toFixed(0) + 'm', cx + 2, cy - rp + 9);
        }
      }

      lidarXov.strokeStyle = '#1e1e1e'; lidarXov.lineWidth = 1;
      for (let i = 0; i < 8; i++) {
        const a = i * Math.PI / 4;
        lidarXov.beginPath();
        lidarXov.moveTo(cx, cy);
        lidarXov.lineTo(cx - Math.sin(a) * maxPx, cy - Math.cos(a) * maxPx);
        lidarXov.stroke();
      }

      lidarXov.fillStyle = 'rgba(0,90,0,0.15)';
      lidarXov.beginPath(); lidarXov.moveTo(cx, cy);
      lidarXov.arc(cx, cy, maxPx, -Math.PI / 2 - LIDAR_FOV_OV, -Math.PI / 2 + LIDAR_FOV_OV, false);
      lidarXov.closePath(); lidarXov.fill();

      lidarXov.strokeStyle = 'rgba(0,180,0,0.3)'; lidarXov.lineWidth = 1;
      for (const sign of [-1, 1]) {
        const a = sign * LIDAR_FOV_OV;
        lidarXov.beginPath(); lidarXov.moveTo(cx, cy);
        lidarXov.lineTo(cx - Math.sin(a) * maxPx, cy - Math.cos(a) * maxPx);
        lidarXov.stroke();
      }

      if (lidarRaysOv && lidarRaysOv.length > 0) {
        let nearR = Infinity, nearIdx = -1;
        for (let i = 0; i < lidarRaysOv.length; i++) {
          if (lidarRaysOv[i].range_m < nearR) { nearR = lidarRaysOv[i].range_m; nearIdx = i; }
        }
        lidarXov.lineWidth = 1;
        for (let i = 0; i < lidarRaysOv.length; i++) {
          const ray = lidarRaysOv[i];
          const rp  = Math.min(ray.range_m, LIDAR_MAX_M_OV) * scale;
          const ex  = cx - Math.sin(ray.angle_rad) * rp;
          const ey  = cy - Math.cos(ray.angle_rad) * rp;
          if (i === nearIdx) {
            lidarXov.strokeStyle = '#ff6600'; lidarXov.lineWidth = 2;
            lidarXov.beginPath(); lidarXov.moveTo(cx, cy); lidarXov.lineTo(ex, ey); lidarXov.stroke();
            lidarXov.fillStyle = '#ff6600';
            lidarXov.beginPath(); lidarXov.arc(ex, ey, 3, 0, Math.PI * 2); lidarXov.fill();
            lidarXov.lineWidth = 1;
          } else {
            lidarXov.strokeStyle = 'rgba(0,210,0,0.4)';
            lidarXov.beginPath(); lidarXov.moveTo(cx, cy); lidarXov.lineTo(ex, ey); lidarXov.stroke();
          }
        }
      }

      lidarXov.strokeStyle = 'rgba(255,60,60,0.7)'; lidarXov.lineWidth = 1.5;
      lidarXov.setLineDash([3, 3]);
      lidarXov.beginPath(); lidarXov.arc(cx, cy, STOP_M_OV * scale, 0, Math.PI * 2); lidarXov.stroke();
      lidarXov.setLineDash([]);

      lidarXov.fillStyle = '#00ff44';
      lidarXov.beginPath(); lidarXov.arc(cx, cy, 4, 0, Math.PI * 2); lidarXov.fill();
      lidarXov.strokeStyle = '#00ff44'; lidarXov.lineWidth = 2;
      lidarXov.beginPath(); lidarXov.moveTo(cx, cy); lidarXov.lineTo(cx, cy - 10); lidarXov.stroke();

      lidarXov.fillStyle = '#555'; lidarXov.font = '9px monospace'; lidarXov.textAlign = 'center';
      lidarXov.fillText('FWD', cx, cy - maxPx - 3);
      lidarXov.fillText('L', cx - maxPx - 8, cy + 4);
      lidarXov.fillText('R', cx + maxPx + 6, cy + 4);
      lidarXov.textAlign = 'left';
    }

    setInterval(drawLidarPolarOv, 100);

    function connect() {
      var ws = window.ws = new WebSocket('ws://' + host + ':9000/');
      const wsSt = document.getElementById('ws_st');
      wsSt.className = 'warn'; wsSt.textContent = 'connecting';

      ws.onopen  = () => { wsSt.className = 'val'; wsSt.textContent = 'ok'; startStream(); };
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
            case 'sensor/ultrasonic': {
              const cm = msg.data && msg.data.range_cm;
              const el = document.getElementById('us');
              el.textContent = fmt(cm, 1);
              el.className = distClass(cm);
              break;
            }
            case 'sensor/nearest_m': {
              const cm = msg.data != null ? msg.data * 100 : null;
              const el = document.getElementById('vis');
              el.textContent = fmt(cm, 0);
              el.className = distClass(cm);
              break;
            }
            case 'slam/pose2d':
              document.getElementById('x').textContent    = fmt(msg.data && msg.data.x_m, 2);
              document.getElementById('y').textContent    = fmt(msg.data && msg.data.y_m, 2);
              document.getElementById('conf').textContent = fmt(msg.data && msg.data.confidence, 2);
              const rad = msg.data && msg.data.theta_rad;
              document.getElementById('hdg').textContent  = rad != null ? (rad * 180 / Math.PI).toFixed(1) : '--';
              break;
            case 'gimbal/pan':
              document.getElementById('pan').textContent = fmt(msg.data, 1);
              break;
            case 'gimbal/tilt':
              document.getElementById('tilt').textContent = fmt(msg.data, 1);
              break;
            case 'sim/start_time':
              simStartMs = msg.data || null;
              document.getElementById('episode').textContent = '0';
              document.getElementById('ep_timeout').textContent = '0';
              break;
            case 'robot/episode':
              document.getElementById('episode').textContent = msg.data || 0;
              break;
            case 'robot/episode_timeout': {
              const n = msg.data || 0;
              const el = document.getElementById('ep_timeout');
              el.textContent = n;
              el.className = n > 0 ? 'warn' : 'val';
              break;
            }
            case 'robot/collision': {
              const prev = crashCount;
              crashCount = msg.data || 0;
              const el = document.getElementById('crashes');
              el.textContent = crashCount;
              el.className = crashCount > 0 ? 'err' : 'val';
              break;
            }
            case 'robot/estop': {
              const estopCount = msg.data || 0;
              const el = document.getElementById('estops');
              el.textContent = estopCount;
              el.className = estopCount > 0 ? 'warn' : 'val';
              break;
            }
            case 'vision/pseudo_lidar':
              lidarRaysOv = msg.data && msg.data.rays;
              break;
            case 'map/grid_delta': {
              // Track cell count for CELLS telemetry field.
              if (msg.data && msg.data.cells) {
                if (!window._gridSet) window._gridSet = new Set();
                for (const [cx, cy] of msg.data.cells) window._gridSet.add(cx + ',' + cy);
                document.getElementById('cells').textContent = window._gridSet.size;
              }
              break;
            }
          }
        } catch(e) {}
      };
    }

    // Run-time clock update.
    setInterval(() => {
      if (simStartMs !== null) {
        const secs = Math.floor((Date.now() - simStartMs) / 1000);
        const m = Math.floor(secs / 60).toString().padStart(2, '0');
        const s = (secs % 60).toString().padStart(2, '0');
        document.getElementById('runtime').textContent = m + ':' + s;
      }
    }, 1000);

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

    // Cache the latest sim/ground_truth JSON so new WS clients immediately
    // receive it on connect (broadcast misses messages sent before subscribe).
    let last_ground_truth: Arc<std::sync::Mutex<Option<Arc<String>>>> =
        Arc::new(std::sync::Mutex::new(None));

    // Sim process start time — sent once to every new client so the browser
    // can compute elapsed time without resetting on refresh or episode reset.
    let sim_start_ms: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let sim_start_json: Arc<String> = Arc::new(
        serde_json::to_string(&serde_json::json!({
            "topic": "sim/start_time",
            "data":  sim_start_ms,
        })).expect("static JSON"),
    );

    // ── Pose ticker: sample slam_pose2d watch at 10 Hz and fan out ───────
    // Keeping the watch poll in a dedicated task avoids a known footgun
    // where `watch::changed()` in a multi-arm `select!` fires only once:
    // when another arm wins, the `changed()` future is dropped without
    // calling `borrow_and_update()`, and subsequent polls can silently
    // stall depending on tokio scheduler ordering.
    let fanout_pose = Arc::clone(&fanout_tx);
    let bus_pose    = Arc::clone(&bus);
    tokio::spawn(async move {
        let rx_pose = bus_pose.slam_pose2d.subscribe();
        let rx_exec = bus_pose.executive_state.subscribe();
        let rx_pan  = bus_pose.gimbal_pan_deg.subscribe();
        let rx_tilt = bus_pose.gimbal_tilt_deg.subscribe();
        let rx_near = bus_pose.nearest_obstacle_m.subscribe();
        let mut ticker = tokio::time::interval(std::time::Duration::from_millis(100));
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            ticker.tick().await;
            let pose  = *rx_pose.borrow();
            let state = rx_exec.borrow().clone();
            let pan   = *rx_pan.borrow();
            let tilt  = *rx_tilt.borrow();
            let near  = *rx_near.borrow();
            if let Ok(msg) = serde_json::to_string(&serde_json::json!({
                "topic": "slam/pose2d",
                "t_ms":  pose.t_ms,
                "data":  pose,
            })) {
                let _ = fanout_pose.send(Arc::new(msg));
            }
            if let Ok(msg) = serde_json::to_string(&serde_json::json!({
                "topic": "executive/state",
                "data":  state,
            })) {
                let _ = fanout_pose.send(Arc::new(msg));
            }
            if let Ok(msg) = serde_json::to_string(&serde_json::json!({
                "topic": "gimbal/pan",
                "data":  pan,
            })) {
                let _ = fanout_pose.send(Arc::new(msg));
            }
            if let Ok(msg) = serde_json::to_string(&serde_json::json!({
                "topic": "gimbal/tilt",
                "data":  tilt,
            })) {
                let _ = fanout_pose.send(Arc::new(msg));
            }
            // Only publish when we have a real reading (f32::MAX = no inference yet).
            if near < f32::MAX {
                if let Ok(msg) = serde_json::to_string(&serde_json::json!({
                    "topic": "sensor/nearest_m",
                    "data":  near,
                })) {
                    let _ = fanout_pose.send(Arc::new(msg));
                }
            }
        }
    });

    // ── Aggregator task: subscribe to bus, serialize, fan out ────────────
    let fanout_agg = Arc::clone(&fanout_tx);
    let bus_agg = Arc::clone(&bus);
    let gt_cache_agg = Arc::clone(&last_ground_truth);
    tokio::spawn(async move {
        let mut rx_grid       = bus_agg.map_grid_delta.subscribe();
        let mut rx_frontier   = bus_agg.map_frontiers.subscribe();
        let mut rx_exec       = bus_agg.executive_state.subscribe();
        let mut rx_health     = bus_agg.health_runtime.subscribe();
        let mut rx_us         = bus_agg.ultrasonic.subscribe();
        let mut rx_lidar      = bus_agg.vision_pseudo_lidar.subscribe();
        let mut rx_collisions      = bus_agg.collision_count.subscribe();
        let mut rx_estops          = bus_agg.estop_count.subscribe();
        let mut rx_episodes        = bus_agg.episode_count.subscribe();
        let mut rx_ep_timeouts     = bus_agg.episode_timeout_count.subscribe();
        let mut rx_sim_truth   = bus_agg.sim_ground_truth.subscribe();
        let rx_pose_agg        = bus_agg.slam_pose2d.subscribe();

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
                Ok(scan) = rx_lidar.recv() => {
                    if let Ok(msg) = serde_json::to_string(&serde_json::json!({
                        "topic": "vision/pseudo_lidar",
                        "t_ms":  scan.t_ms,
                        "data":  *scan,
                    })) {
                        let _ = fanout_agg.send(Arc::new(msg));
                    }
                }
                Ok(()) = rx_collisions.changed() => {
                    let count = *rx_collisions.borrow_and_update();
                    let pose  = *rx_pose_agg.borrow();
                    if let Ok(msg) = serde_json::to_string(&serde_json::json!({
                        "topic": "robot/collision",
                        "data":  count,
                        "x": pose.x_m,
                        "y": pose.y_m,
                    })) {
                        let _ = fanout_agg.send(Arc::new(msg));
                    }
                }
                Ok(()) = rx_estops.changed() => {
                    let count = *rx_estops.borrow_and_update();
                    if let Ok(msg) = serde_json::to_string(&serde_json::json!({
                        "topic": "robot/estop",
                        "data":  count,
                    })) {
                        let _ = fanout_agg.send(Arc::new(msg));
                    }
                }
                Ok(()) = rx_episodes.changed() => {
                    let count = *rx_episodes.borrow_and_update();
                    if let Ok(msg) = serde_json::to_string(&serde_json::json!({
                        "topic": "robot/episode",
                        "data":  count,
                    })) {
                        let _ = fanout_agg.send(Arc::new(msg));
                    }
                }
                Ok(()) = rx_ep_timeouts.changed() => {
                    let count = *rx_ep_timeouts.borrow_and_update();
                    if let Ok(msg) = serde_json::to_string(&serde_json::json!({
                        "topic": "robot/episode_timeout",
                        "data":  count,
                    })) {
                        let _ = fanout_agg.send(Arc::new(msg));
                    }
                }
                Ok(walls) = rx_sim_truth.recv() => {
                    if let Ok(msg) = serde_json::to_string(&serde_json::json!({
                        "topic": "sim/ground_truth",
                        "width": 200u32,
                        "height": 200u32,
                        "res":   0.05f32,
                        "walls": *walls,
                    })) {
                        let msg = Arc::new(msg);
                        // Cache so new WS clients get it immediately on connect.
                        if let Ok(mut g) = gt_cache_agg.lock() { *g = Some(Arc::clone(&msg)); }
                        let _ = fanout_agg.send(msg);
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
    let bus_listener = Arc::clone(&bus);
    let fanout_listener = Arc::clone(&fanout_tx);
    let gt_cache_listener = Arc::clone(&last_ground_truth);
    let sim_start_json_listener = Arc::clone(&sim_start_json);
    let mut connected: u32 = 0;
    let bytes_sent: u64 = 0;

    tokio::spawn(async move {
        loop {
            match listener.accept().await {
                Ok((stream, peer)) => {
                    // Peek at the request to distinguish WebSocket upgrade
                    // from a plain HTTP GET (browser opening the overlay page).
                    let mut peek = [0u8; 512];
                    let (is_ws, is_map) = match stream.peek(&mut peek).await {
                        Ok(n) => {
                            let h = std::str::from_utf8(&peek[..n]).unwrap_or("");
                            let ws  = h.contains("Upgrade: websocket") || h.contains("upgrade: websocket");
                            let map = !ws && h.starts_with("GET /map");
                            (ws, map)
                        }
                        Err(_) => (false, false),
                    };

                    if is_ws {
                        connected += 1;
                        info!("UI bridge: WS client connected from {peer}");
                        let rx = fanout_listener.subscribe();
                        // Build initial-state burst: start time, current counts, ground truth.
                        let mut initial: Vec<Arc<String>> = vec![Arc::clone(&sim_start_json_listener)];
                        let crash_count   = *bus_listener.collision_count.borrow();
                        let estop_count   = *bus_listener.estop_count.borrow();
                        let episode_count = *bus_listener.episode_count.borrow();
                        let ep_timeout    = *bus_listener.episode_timeout_count.borrow();
                        if let Ok(m) = serde_json::to_string(&serde_json::json!({"topic":"robot/collision","data":crash_count})) {
                            initial.push(Arc::new(m));
                        }
                        if let Ok(m) = serde_json::to_string(&serde_json::json!({"topic":"robot/estop","data":estop_count})) {
                            initial.push(Arc::new(m));
                        }
                        if let Ok(m) = serde_json::to_string(&serde_json::json!({"topic":"robot/episode","data":episode_count})) {
                            initial.push(Arc::new(m));
                        }
                        if let Ok(m) = serde_json::to_string(&serde_json::json!({"topic":"robot/episode_timeout","data":ep_timeout})) {
                            initial.push(Arc::new(m));
                        }
                        if let Some(gt) = gt_cache_listener.lock().ok().and_then(|g| g.clone()) {
                            initial.push(gt);
                        }
                        tokio::spawn(handle_client(stream, rx, initial, bus_listener.bridge_cmd.clone(), bus_listener.manual_cmd_vel.clone()));

                        let status = BridgeStatus {
                            t_ms: 0,
                            connected_clients: connected,
                            bytes_sent_total: bytes_sent,
                            last_send_ok: true,
                        };
                        let _ = bus_status.ui_bridge_status.send(status);
                    } else if is_map {
                        tokio::spawn(serve_page(stream, MAP_HTML));
                    } else {
                        tokio::spawn(serve_page(stream, OVERLAY_HTML));
                    }
                }
                Err(e) => warn!("UI bridge accept error: {e}"),
            }
        }
    });

    Ok(())
}

/// Serve an HTML page to a plain HTTP client.
///
/// Flushes and half-closes the write side so the OS sends FIN instead of RST
/// (without this the kernel may reset the connection before all bytes leave
/// the send buffer, especially when there is unread request data pending).
async fn serve_page(mut stream: TcpStream, html: &'static str) {
    let body = html.as_bytes();
    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(header.as_bytes()).await;
    let _ = stream.write_all(body).await;
    let _ = stream.flush().await;
    let _ = stream.shutdown().await;
}

/// Per-client task: receive fan-out messages and forward to WebSocket;
/// also read inbound WebSocket messages and forward commands.
async fn handle_client(
    stream: TcpStream,
    mut rx: broadcast::Receiver<Arc<String>>,
    initial: Vec<Arc<String>>,
    cmd_tx: watch::Sender<BridgeCommand>,
    manual_vel_tx: watch::Sender<CmdVel>,
) {
    let ws = match tokio_tungstenite::accept_async(stream).await {
        Ok(ws) => ws,
        Err(e) => {
            warn!("UI bridge WS handshake failed: {e}");
            return;
        }
    };
    let (mut sink, mut source) = futures_util::StreamExt::split(ws);

    // Spawn reader task: handle ARM/STOP/MANUAL/AUTO/vel commands from browser.
    tokio::spawn(async move {
        while let Some(Ok(msg)) = source.next().await {
            if let Message::Text(txt) = msg {
                let txt = txt.trim();
                match txt {
                    "arm"    => { let _ = cmd_tx.send(BridgeCommand::Arm); }
                    "stop"   => { let _ = cmd_tx.send(BridgeCommand::Stop); }
                    "manual" => { let _ = cmd_tx.send(BridgeCommand::Manual); }
                    "auto"   => { let _ = cmd_tx.send(BridgeCommand::Auto); }
                    _ if txt.starts_with("vel:") => {
                        let parts: Vec<f32> = txt[4..].split(',')
                            .filter_map(|s| s.parse().ok())
                            .collect();
                        if parts.len() == 3 {
                            let vel = CmdVel { t_ms: 0, vx: parts[0], vy: parts[1], omega: parts[2] };
                            let _ = manual_vel_tx.send(vel);
                        }
                    }
                    _ => {}
                }
            }
        }
    });

    // Replay initial state: start time, counts, ground truth.
    for msg in initial {
        if sink.send(Message::Text(msg.as_str().to_owned().into())).await.is_err() {
            return;
        }
    }

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
