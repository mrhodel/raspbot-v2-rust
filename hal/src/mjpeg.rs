//! MJPEG HTTP streaming server.
//!
//! Shared by the real V4L2Camera and the sim camera path.
//! Call [`run_server`] inside a `tokio::spawn()`.

use std::sync::Arc;

use core_types::{CameraFrame, DepthMap};
use tokio::sync::broadcast;
use tracing::{info, warn};

/// Bind a TCP MJPEG server on `port`.  Each connecting client gets its own
/// streaming task driven by a fresh [`broadcast::Receiver`] on `tx`.
pub async fn run_server(
    port: u16,
    tx:   broadcast::Sender<Arc<CameraFrame>>,
) {
    use tokio::net::TcpListener;

    let listener = match TcpListener::bind(format!("0.0.0.0:{port}")).await {
        Ok(l)  => l,
        Err(e) => { warn!("MJPEG server: bind port {port} failed: {e}"); return; }
    };
    info!("MJPEG server: listening on port {port}");

    loop {
        let Ok((socket, addr)) = listener.accept().await else { continue };
        info!("MJPEG: client connected from {addr}");
        tokio::spawn(serve_client(socket, tx.subscribe()));
    }
}

async fn serve_client(
    mut socket: tokio::net::TcpStream,
    mut rx:     broadcast::Receiver<Arc<CameraFrame>>,
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
            Err(broadcast::error::RecvError::Lagged(_)) => continue,
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

// ── Depth map MJPEG server ────────────────────────────────────────────────────

/// Display size for the scaled-up depth image.
/// Source is 32×32 (RL-sized); 8× zoom → 256×256 with crisp pixels.
const DEPTH_DISPLAY_PX: u32 = 256;

/// Bind a TCP MJPEG server on `port` that streams the depth map.
/// Each frame is colorised (white = nearest, red tint = body-mask rows) and JPEG-encoded.
pub async fn run_depth_server(port: u16, tx: broadcast::Sender<Arc<DepthMap>>) {
    use tokio::net::TcpListener;
    let listener = match TcpListener::bind(format!("0.0.0.0:{port}")).await {
        Ok(l)  => l,
        Err(e) => { warn!("Depth MJPEG server: bind port {port} failed: {e}"); return; }
    };
    info!("Depth MJPEG server: listening on port {port}");
    loop {
        let Ok((socket, _)) = listener.accept().await else { continue };
        tokio::spawn(serve_depth_client(socket, tx.subscribe()));
    }
}

async fn serve_depth_client(
    mut socket: tokio::net::TcpStream,
    mut rx:     broadcast::Receiver<Arc<DepthMap>>,
) {
    use tokio::io::AsyncWriteExt;
    let hdr = b"HTTP/1.1 200 OK\r\n\
                Content-Type: multipart/x-mixed-replace;boundary=frame\r\n\
                Cache-Control: no-cache\r\n\
                Connection: close\r\n\r\n";
    if socket.write_all(hdr).await.is_err() { return; }
    loop {
        let depth = match rx.recv().await {
            Ok(d)  => d,
            Err(broadcast::error::RecvError::Lagged(_)) => continue,
            Err(_) => break,
        };
        let jpeg = encode_depth_jpeg(&depth);
        if jpeg.is_empty() { continue; }
        let part = format!(
            "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
            jpeg.len()
        );
        if socket.write_all(part.as_bytes()).await.is_err() { break; }
        if socket.write_all(&jpeg).await.is_err()           { break; }
        if socket.write_all(b"\r\n").await.is_err()         { break; }
    }
}

/// JPEG-encode a [`DepthMap`] as a colourised image scaled up for display.
///
/// Colour convention:
/// - Valid rows: grayscale (white = nearest, black = farthest)
/// - Body-mask rows: reddish tint so the masked region is visually distinct
pub fn encode_depth_jpeg(depth: &DepthMap) -> Vec<u8> {
    use image::codecs::jpeg::JpegEncoder;
    use image::{Rgb, RgbImage};

    let sw = depth.width;
    let sh = depth.height;
    if sw == 0 || sh == 0 || depth.data.len() < (sw * sh) as usize {
        return Vec::new();
    }

    let mut src = RgbImage::new(sw, sh);
    for y in 0..sh {
        for x in 0..sw {
            let v = depth.data[(y * sw + x) as usize].clamp(0.0, 1.0);
            let g = (v * 255.0) as u8;
            let pixel = if y >= depth.mask_start_row {
                Rgb([g.saturating_add(60), g / 2, g / 2])   // reddish: body mask
            } else {
                Rgb([g, g, g])
            };
            src.put_pixel(x, y, pixel);
        }
    }

    let big = image::imageops::resize(
        &src, DEPTH_DISPLAY_PX, DEPTH_DISPLAY_PX, image::imageops::FilterType::Nearest,
    );

    let mut buf = Vec::new();
    if let Err(e) = JpegEncoder::new_with_quality(&mut buf, 85).encode_image(&big) {
        warn!("Depth MJPEG encode: {e}");
        return Vec::new();
    }
    buf
}

// ── Camera MJPEG ──────────────────────────────────────────────────────────────

/// JPEG-encode an RGB [`CameraFrame`] at quality 75.
pub fn encode_jpeg(frame: &CameraFrame) -> Vec<u8> {
    use image::codecs::jpeg::JpegEncoder;
    use image::RgbImage;

    let Some(img) = RgbImage::from_raw(frame.width, frame.height, frame.data.clone()) else {
        warn!("MJPEG encode: frame data size mismatch");
        return Vec::new();
    };

    let mut buf = Vec::new();
    if let Err(e) = JpegEncoder::new_with_quality(&mut buf, 75).encode_image(&img) {
        warn!("MJPEG encode: {e}");
        return Vec::new();
    }
    buf
}
