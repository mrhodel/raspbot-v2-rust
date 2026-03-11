//! MiDaS depth inference.
//!
//! # Inference pipeline
//!
//!   1. Resize `CameraFrame` RGB to `INPUT_HW × INPUT_HW` (256 px, bilinear).
//!   2. Convert to f32, normalise with ImageNet mean/std.
//!   3. Run MiDaS-small ONNX → single-channel inverse-depth [INPUT_HW × INPUT_HW].
//!   4. Normalise depth output to [0, 1].
//!   5. Resize to `out_width × out_height` (32 × 32 for RL).
//!   6. Apply body mask (bottom `mask_rows` rows set to 0).
//!
//! # Model file
//!
//!   `models/midas_small.onnx` — export once with `scripts/export_midas.py`.
//!   If the file is absent the stub gradient is returned (so simulation and
//!   unit tests work without the model).
//!
//! # Feature flag
//!
//!   ONNX inference is compiled only when the `onnx` feature is enabled
//!   (default).  Disable on targets without the `ort` shared library:
//!     cargo check --no-default-features

use anyhow::Result;
use core_types::{CameraFrame, DepthMap};

// ── Constants ──────────────────────────────────────────────────────────────────

/// Spatial size fed to MiDaS (must match the exported ONNX model).
#[allow(dead_code)]
const INPUT_HW: usize = 256;

/// ImageNet normalisation constants.
#[allow(dead_code)]
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
#[allow(dead_code)]
const STD:  [f32; 3] = [0.229, 0.224, 0.225];

// ── Real ONNX implementation ───────────────────────────────────────────────────

#[cfg(feature = "onnx")]
mod onnx_impl {
    use anyhow::Result;
    use core_types::{CameraFrame, DepthMap};
    use ort::session::{Session, builder::GraphOptimizationLevel};
    use ort::value::Tensor;
    use tracing::{debug, info};

    use super::{INPUT_HW, MEAN, STD};

    pub struct OnnxDepth {
        session:    Session,
        out_width:  u32,
        out_height: u32,
        mask_rows:  u32,
    }

    impl OnnxDepth {
        pub fn new(
            model_path:  &str,
            out_width:   u32,
            out_height:  u32,
            mask_rows:   u32,
            num_threads: usize,
        ) -> Result<Self> {
            // Build session — ort::Error doesn't implement std::error::Error,
            // so we convert each Result via map_err.
            let session = Session::builder()
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .with_intra_threads(num_threads)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .commit_from_file(model_path)
                .map_err(|e| anyhow::anyhow!("load MiDaS ONNX from {model_path}: {e}"))?;

            info!(
                "DepthInference: loaded {model_path}  \
                 -> {out_width}x{out_height} depth map  \
                 mask_rows={mask_rows}  threads={num_threads}"
            );
            Ok(Self { session, out_width, out_height, mask_rows })
        }

        pub fn infer(&mut self, frame: &CameraFrame) -> Result<DepthMap> {
            // 1. Resize to INPUT_HW x INPUT_HW (bilinear).
            let resized = resize_rgb(
                &frame.data,
                frame.width as usize,
                frame.height as usize,
                INPUT_HW, INPUT_HW,
            );

            // 2. Normalise -> CHW f32 flat vec [1 * 3 * H * W].
            let data = rgb_to_chw_flat(&resized, INPUT_HW, INPUT_HW);

            // 3. Run ONNX.  Use (shape, Vec<f32>) — no ndarray version conflict.
            let shape = [1usize, 3, INPUT_HW, INPUT_HW];
            let input_tensor = Tensor::<f32>::from_array((shape, data))
                .map_err(|e| anyhow::anyhow!("create input Tensor: {e}"))?;
            let outputs = self.session
                .run(ort::inputs![input_tensor])
                .map_err(|e| anyhow::anyhow!("MiDaS inference: {e}"))?;

            let (_shape, raw_slice) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| anyhow::anyhow!("extract depth tensor: {e}"))?;

            // 4. Normalise to [0, 1].
            let mut min_v = f32::MAX;
            let mut max_v = f32::MIN;
            for &v in raw_slice { min_v = min_v.min(v); max_v = max_v.max(v); }
            let range = (max_v - min_v).max(1e-6);
            let normalised: Vec<f32> =
                raw_slice.iter().map(|&v| (v - min_v) / range).collect();

            // 5. Resize to out_width x out_height.
            let ow = self.out_width as usize;
            let oh = self.out_height as usize;
            let mut data = resize_depth(&normalised, INPUT_HW, INPUT_HW, ow, oh);

            // 6. Apply body mask.
            let mask_start = oh.saturating_sub(self.mask_rows as usize);
            for row in mask_start..oh {
                for col in 0..ow { data[row * ow + col] = 0.0; }
            }

            debug!("depth infer raw=[{:.3},{:.3}] out={}x{}", min_v, max_v, ow, oh);

            Ok(DepthMap {
                t_ms:           frame.t_ms,
                width:          self.out_width,
                height:         self.out_height,
                data,
                mask_start_row: mask_start as u32,
            })
        }
    }

    fn resize_rgb(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
        let mut dst = vec![0u8; dw * dh * 3];
        for dy in 0..dh {
            for dx in 0..dw {
                let sx = dx as f32 * (sw as f32 / dw as f32);
                let sy = dy as f32 * (sh as f32 / dh as f32);
                let x0 = (sx as usize).min(sw - 1);
                let y0 = (sy as usize).min(sh - 1);
                let x1 = (x0 + 1).min(sw - 1);
                let y1 = (y0 + 1).min(sh - 1);
                let fx = sx - x0 as f32;
                let fy = sy - y0 as f32;
                for c in 0..3 {
                    let v = src[(y0*sw+x0)*3+c] as f32 * (1.0-fx)*(1.0-fy)
                          + src[(y0*sw+x1)*3+c] as f32 * fx       *(1.0-fy)
                          + src[(y1*sw+x0)*3+c] as f32 * (1.0-fx)* fy
                          + src[(y1*sw+x1)*3+c] as f32 * fx       * fy;
                    dst[(dy*dw+dx)*3+c] = v as u8;
                }
            }
        }
        dst
    }

    /// Pack HWC u8 → CHW f32 flat vec with ImageNet normalisation.
    /// Output length = 3 * h * w  (channel-first, no batch dim needed for shape).
    fn rgb_to_chw_flat(rgb: &[u8], h: usize, w: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; 3 * h * w];
        for row in 0..h {
            for col in 0..w {
                for c in 0..3 {
                    let v = rgb[(row*w+col)*3+c] as f32 / 255.0;
                    out[c * h * w + row * w + col] = (v - MEAN[c]) / STD[c];
                }
            }
        }
        out
    }

    fn resize_depth(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
        let mut dst = vec![0.0f32; dw * dh];
        for dy in 0..dh {
            for dx in 0..dw {
                let sx = dx as f32 * (sw as f32 / dw as f32);
                let sy = dy as f32 * (sh as f32 / dh as f32);
                let x0 = (sx as usize).min(sw - 1);
                let y0 = (sy as usize).min(sh - 1);
                let x1 = (x0 + 1).min(sw - 1);
                let y1 = (y0 + 1).min(sh - 1);
                let fx = sx - x0 as f32;
                let fy = sy - y0 as f32;
                dst[dy*dw+dx] = src[y0*sw+x0] * (1.0-fx)*(1.0-fy)
                              + src[y0*sw+x1] * fx       *(1.0-fy)
                              + src[y1*sw+x0] * (1.0-fx)* fy
                              + src[y1*sw+x1] * fx       * fy;
            }
        }
        dst
    }

    pub use OnnxDepth as Impl;
}

// ── Stub ───────────────────────────────────────────────────────────────────────

struct StubDepth {
    out_width:  u32,
    out_height: u32,
    mask_rows:  u32,
}

impl StubDepth {
    fn new(out_width: u32, out_height: u32, mask_rows: u32) -> Self {
        Self { out_width, out_height, mask_rows }
    }

    fn infer(&self, frame: &CameraFrame) -> Result<DepthMap> {
        let w = self.out_width as usize;
        let h = self.out_height as usize;
        let mask_start = h.saturating_sub(self.mask_rows as usize);
        let data: Vec<f32> = (0..h).flat_map(|row| {
            (0..w).map(move |_| {
                if row >= mask_start { 0.0 }
                else { 0.1 + 0.8 * (row as f32 / mask_start.max(1) as f32) }
            })
        }).collect();
        Ok(DepthMap {
            t_ms: frame.t_ms,
            width: self.out_width,
            height: self.out_height,
            data,
            mask_start_row: mask_start as u32,
        })
    }
}

// ── Public façade ──────────────────────────────────────────────────────────────

/// Depth inference engine.
///
/// Loads `midas_small.onnx` when the `onnx` feature is enabled and the file
/// exists.  Falls back to a synthetic gradient stub otherwise.
pub struct DepthInference {
    #[cfg(feature = "onnx")]
    onnx:     Option<onnx_impl::Impl>,
    stub:     StubDepth,
    use_onnx: bool,
}

impl DepthInference {
    /// Create an inference engine.
    ///
    /// * `model_path`       — path to `midas_small.onnx`
    /// * `out_width/height` — target depth-map size (32x32 for RL)
    /// * `mask_rows`        — bottom rows to zero (body mask)
    /// * `num_threads`      — CPU intra-op threads for ONNX runtime
    pub fn new(
        model_path:  &str,
        out_width:   u32,
        out_height:  u32,
        mask_rows:   u32,
        num_threads: usize,
    ) -> Self {
        // Suppress unused-variable warnings when onnx feature is off.
        let _ = (model_path, num_threads);
        let stub = StubDepth::new(out_width, out_height, mask_rows);

        #[cfg(feature = "onnx")]
        {
            match onnx_impl::Impl::new(
                model_path, out_width, out_height, mask_rows, num_threads,
            ) {
                Ok(engine) => return Self {
                    onnx: Some(engine), stub, use_onnx: true,
                },
                Err(e) => tracing::warn!(
                    "MiDaS ONNX unavailable ({e}); using stub depth"
                ),
            }
        }

        Self {
            #[cfg(feature = "onnx")]
            onnx: None,
            stub,
            use_onnx: false,
        }
    }

    /// `true` if the real ONNX engine is active.
    pub fn is_real(&self) -> bool { self.use_onnx }

    pub fn infer(&mut self, frame: &CameraFrame) -> Result<DepthMap> {
        #[cfg(feature = "onnx")]
        if self.use_onnx {
            if let Some(ref mut engine) = self.onnx {
                return engine.infer(frame);
            }
        }
        self.stub.infer(frame)
    }
}
