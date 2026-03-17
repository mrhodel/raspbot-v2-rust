//! MJPEG HTTP streaming server.
//!
//! Shared by the real V4L2Camera and the sim camera path.
//! Call [`run_server`] inside a `tokio::spawn()`.

use std::sync::Arc;

use core_types::CameraFrame;
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
