//! Camera HAL — USB V4L2 capture + optional in-process MJPEG HTTP stream.
//!
//! # V4L2 capture
//!
//!   Format: YUYV 4:2:2 (universally supported by USB webcams).
//!   A dedicated `std::thread` runs the V4L2 mmap stream and sends each
//!   converted RGB frame over a `tokio::sync::watch` channel.
//!   `read_frame()` awaits the next frame from that channel.
//!
//! # MJPEG stream (`stream_enabled: true` in config)
//!
//!   When enabled, each captured frame is also published to a
//!   `tokio::sync::broadcast` channel.  A Tokio task subscribes, JPEG-encodes
//!   each frame, and serves `multipart/x-mixed-replace` HTTP on `stream_port`
//!   (default 8080).  Open `http://<robot-ip>:8080/` in any browser or VLC.
//!
//!   The server is fully independent of the capture loop — a dead client or
//!   server crash does not affect frame capture.
//!
//! # Body mask
//!
//!   `body_mask_bottom_rows` in `CameraConfig` (default 0) is NOT applied here.
//!   Perception computes `mask_start_row = height − body_mask_bottom_rows` and
//!   stores it in every `DepthMap`.  Calibrate via `camera_test --save-frame`.
//!
//! # Feature flag
//!
//!   The real `V4L2Camera` is compiled only when the `usb-camera` feature is
//!   enabled (the default).  On machines without Linux V4L2 headers / libclang,
//!   build with `--no-default-features` to skip it.

use std::time::Instant;

use anyhow::Result;
use async_trait::async_trait;
use core_types::CameraFrame;

// ── Trait ─────────────────────────────────────────────────────────────────────

#[async_trait]
pub trait Camera: Send + Sync {
    /// Block until the next frame is available, then return it.
    async fn read_frame(&mut self) -> Result<CameraFrame>;
    /// Return configured (width, height) in pixels.
    fn resolution(&self) -> (u32, u32);
}

// ── Real V4L2 implementation ──────────────────────────────────────────────────

#[cfg(feature = "usb-camera")]
mod usb {
    use std::sync::Arc;
    use std::time::Instant;

    use anyhow::{Context, Result};
    use async_trait::async_trait;
    use core_types::CameraFrame;
    use tracing::{debug, info, warn};

    use super::Camera;

    /// Captures from a USB V4L2 camera in a background thread.
    ///
    /// Optionally serves an in-process MJPEG HTTP stream so the feed can be
    /// monitored in a browser during training and deployment.
    pub struct V4L2Camera {
        /// Receives the latest frame from the capture thread.
        frame_rx:        tokio::sync::watch::Receiver<Option<Arc<CameraFrame>>>,
        width:           u32,
        height:          u32,
        /// Keeps the broadcast sender alive (dropping it stops the MJPEG server).
        _broadcast_tx:   Option<tokio::sync::broadcast::Sender<Arc<CameraFrame>>>,
        /// Keeps the capture thread alive (runs until the process exits).
        _capture_thread: std::thread::JoinHandle<()>,
    }

    impl V4L2Camera {
        /// Open the V4L2 device and start capturing.
        ///
        /// * `device_path`  — `/dev/video0` or a stable by-id path
        /// * `width/height` — requested resolution (driver may round)
        /// * `fps`          — requested frame rate
        /// * `stream_port`  — `Some(port)` → start MJPEG server; `None` → disabled
        ///
        /// Must be called from within a Tokio runtime (spawns a Tokio task for
        /// the MJPEG server when `stream_port` is `Some`).
        pub fn new(
            device_path: &str,
            width:       u32,
            height:      u32,
            fps:         u32,
            stream_port: Option<u16>,
        ) -> Result<Self> {
            use v4l::buffer::Type;
            use v4l::io::mmap::Stream as MmapStream;
            use v4l::prelude::*;
            use v4l::video::capture::Parameters;
            use v4l::{Device, FourCC};

            // Open device and negotiate format.
            let dev = Device::with_path(device_path)
                .with_context(|| format!("open {device_path}"))?;

            let mut fmt = dev.format().context("read camera format")?;
            fmt.width  = width;
            fmt.height = height;
            fmt.fourcc = FourCC::new(b"YUYV");
            let fmt = dev.set_format(&fmt).context("set camera format")?;

            let actual_w = fmt.width;
            let actual_h = fmt.height;
            info!(
                "V4L2Camera: {device_path} {actual_w}×{actual_h} YUYV \
                 (requested {width}×{height})"
            );

            // Request frame rate (best-effort — some drivers ignore this).
            if let Err(e) = dev.set_parameters(&Parameters::with_fps(fps)) {
                warn!("V4L2Camera: set fps={fps} ignored by driver: {e}");
            }

            // Watch channel: capture thread → read_frame().
            let dummy = Arc::new(CameraFrame {
                t_ms:   0,
                width:  actual_w,
                height: actual_h,
                data:   vec![0u8; (actual_w * actual_h * 3) as usize],
            });
            let (watch_tx, frame_rx) =
                tokio::sync::watch::channel(Some(dummy));

            // Broadcast channel: capture thread → MJPEG server.
            let (broadcast_tx, _) =
                tokio::sync::broadcast::channel::<Arc<CameraFrame>>(4);

            let broadcast_opt =
                stream_port.map(|_| broadcast_tx.clone());

            // Spawn MJPEG server task when requested.
            if let Some(port) = stream_port {
                let tx = broadcast_tx.clone();
                tokio::spawn(async move {
                    run_mjpeg_server(port, tx).await;
                });
                info!("V4L2Camera: MJPEG stream → http://0.0.0.0:{port}/");
            }

            // Move device into the capture thread.
            let t0 = Instant::now();
            let stream_tx_for_thread = broadcast_opt.clone();

            let capture_thread = std::thread::Builder::new()
                .name("v4l2-capture".into())
                .spawn(move || {
                    let mut mmap = match MmapStream::with_buffers(
                        &dev, Type::VideoCapture, 4,
                    ) {
                        Ok(s)  => s,
                        Err(e) => {
                            warn!("V4L2: failed to start mmap stream: {e}");
                            return;
                        }
                    };

                    loop {
                        let (yuyv, _meta) = match mmap.next() {
                            Ok(r)  => r,
                            Err(e) => {
                                warn!("V4L2: capture error: {e}");
                                std::thread::sleep(
                                    std::time::Duration::from_millis(50)
                                );
                                continue;
                            }
                        };

                        // Convert YUYV → RGB (copies data, releasing mmap borrow).
                        let rgb = yuyv_to_rgb(yuyv, actual_w, actual_h);
                        let frame = Arc::new(CameraFrame {
                            t_ms:   t0.elapsed().as_millis() as u64,
                            width:  actual_w,
                            height: actual_h,
                            data:   rgb,
                        });

                        let _ = watch_tx.send(Some(Arc::clone(&frame)));

                        if let Some(ref tx) = stream_tx_for_thread {
                            let _ = tx.send(Arc::clone(&frame));
                        }

                        debug!("V4L2: frame t={}ms", frame.t_ms);
                    }
                })
                .context("spawn V4L2 capture thread")?;

            Ok(Self {
                frame_rx,
                width: actual_w,
                height: actual_h,
                _broadcast_tx: broadcast_opt,
                _capture_thread: capture_thread,
            })
        }
    }

    #[async_trait]
    impl Camera for V4L2Camera {
        async fn read_frame(&mut self) -> Result<CameraFrame> {
            self.frame_rx
                .changed()
                .await
                .context("V4L2 capture thread exited")?;
            let arc = self
                .frame_rx
                .borrow()
                .clone()
                .ok_or_else(|| anyhow::anyhow!("no frame available yet"))?;
            Ok((*arc).clone())
        }

        fn resolution(&self) -> (u32, u32) {
            (self.width, self.height)
        }
    }

    // ── YUYV 4:2:2 → packed RGB ───────────────────────────────────────────────

    /// Convert a YUYV (YUV422) buffer to packed RGB.
    ///
    /// YUYV layout: for every 4 bytes [Y0, U, Y1, V] → 2 RGB pixels.
    fn yuyv_to_rgb(yuyv: &[u8], width: u32, height: u32) -> Vec<u8> {
        let n_pairs = (width * height / 2) as usize;
        let mut rgb = vec![0u8; (width * height * 3) as usize];

        for i in 0..n_pairs {
            let y0 = yuyv[i * 4]     as i32;
            let u  = yuyv[i * 4 + 1] as i32 - 128;
            let y1 = yuyv[i * 4 + 2] as i32;
            let v  = yuyv[i * 4 + 3] as i32 - 128;

            let conv = |y: i32| -> (u8, u8, u8) {
                let r = (y + 1402 * v / 1000).clamp(0, 255) as u8;
                let g = (y -  344 * u / 1000 - 714 * v / 1000).clamp(0, 255) as u8;
                let b = (y + 1772 * u / 1000).clamp(0, 255) as u8;
                (r, g, b)
            };

            let (r0, g0, b0) = conv(y0);
            let (r1, g1, b1) = conv(y1);
            let base = i * 6;
            rgb[base]     = r0;  rgb[base + 1] = g0;  rgb[base + 2] = b0;
            rgb[base + 3] = r1;  rgb[base + 4] = g1;  rgb[base + 5] = b1;
        }
        rgb
    }

    // ── MJPEG HTTP server ─────────────────────────────────────────────────────

    async fn run_mjpeg_server(
        port: u16,
        tx:   tokio::sync::broadcast::Sender<Arc<CameraFrame>>,
    ) {
        use tokio::net::TcpListener;

        let listener = match TcpListener::bind(format!("0.0.0.0:{port}")).await {
            Ok(l)  => l,
            Err(e) => { warn!("MJPEG server: bind port {port} failed: {e}"); return; }
        };

        loop {
            let Ok((socket, addr)) = listener.accept().await else { continue };
            info!("MJPEG: client connected from {addr}");
            tokio::spawn(serve_mjpeg_client(socket, tx.subscribe()));
        }
    }

    async fn serve_mjpeg_client(
        mut socket: tokio::net::TcpStream,
        mut rx:     tokio::sync::broadcast::Receiver<Arc<CameraFrame>>,
    ) {
        use tokio::io::AsyncWriteExt;

        let hdr = b"HTTP/1.1 200 OK\r\n\
                    Content-Type: multipart/x-mixed-replace;boundary=frame\r\n\
                    Cache-Control: no-cache\r\n\
                    Connection: close\r\n\r\n";
        if socket.write_all(hdr).await.is_err() { return; }

        loop {
            let frame = match rx.recv().await {
                Ok(f)  => f,
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(_) => break,
            };

            let jpeg = encode_jpeg(&frame);
            let part = format!(
                "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
                jpeg.len()
            );
            if socket.write_all(part.as_bytes()).await.is_err() { break; }
            if socket.write_all(&jpeg).await.is_err()           { break; }
            if socket.write_all(b"\r\n").await.is_err()         { break; }
        }
    }

    /// JPEG-encode an RGB `CameraFrame` at quality 75.
    fn encode_jpeg(frame: &CameraFrame) -> Vec<u8> {
        use image::codecs::jpeg::JpegEncoder;
        use image::RgbImage;

        let Some(img) = RgbImage::from_raw(
            frame.width, frame.height, frame.data.clone()
        ) else {
            warn!("encode_jpeg: frame data size mismatch");
            return Vec::new();
        };

        let mut buf = Vec::new();
        if let Err(e) = JpegEncoder::new_with_quality(&mut buf, 75).encode_image(&img) {
            warn!("encode_jpeg: {e}");
            return Vec::new();
        }
        buf
    }
}

#[cfg(feature = "usb-camera")]
pub use usb::V4L2Camera;

// ── Stub ──────────────────────────────────────────────────────────────────────

/// Produces synthetic solid-colour frames for tests and simulation.
pub struct StubCamera {
    width:       u32,
    height:      u32,
    t0:          Instant,
    frame_count: u64,
    interval_ms: u64,
}

impl StubCamera {
    pub fn new(cfg: &config::CameraConfig) -> Self {
        Self {
            width:       cfg.width,
            height:      cfg.height,
            t0:          Instant::now(),
            frame_count: 0,
            interval_ms: 1000 / cfg.fps.max(1) as u64,
        }
    }
}

#[async_trait]
impl Camera for StubCamera {
    async fn read_frame(&mut self) -> Result<CameraFrame> {
        tokio::time::sleep(tokio::time::Duration::from_millis(self.interval_ms)).await;
        self.frame_count += 1;
        let grey = ((self.frame_count % 32) * 4 + 64) as u8;
        Ok(CameraFrame {
            t_ms:   self.t0.elapsed().as_millis() as u64,
            width:  self.width,
            height: self.height,
            data:   vec![grey; (self.width * self.height * 3) as usize],
        })
    }

    fn resolution(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}
