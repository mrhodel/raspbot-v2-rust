#!/usr/bin/env python3
"""Export MiDaS small to ONNX for use with the `ort` crate.

Requirements:
    pip install torch torchvision timm

Output:
    models/midas_small.onnx

The model takes a batch of RGB images in ImageNet-normalised CHW format
and outputs a single-channel inverse-depth map of the same spatial size.
We fix the input to 256x256 (MiDaS-small's smallest practical input; the
robot runs at this size for latency) and export with opset 17.

Inference pipeline (mirrors perception/src/depth.rs):
  1. Resize CameraFrame to 256x256
  2. Convert to f32, divide by 255
  3. Subtract ImageNet mean [0.485, 0.456, 0.406]
  4. Divide by ImageNet std  [0.229, 0.224, 0.225]
  5. Add batch dim  → [1, 3, 256, 256]
  6. Run ONNX       → [1, 256, 256]
  7. Normalise output to [0, 1]
  8. Resize to 32x32 (bilinear, nearest)
  9. Apply body mask
"""

import sys
from pathlib import Path

try:
    import torch
    import timm  # noqa: F401 — needed by midas hub load
except ImportError:
    sys.exit("Install torch and timm:  pip install torch torchvision timm")

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH  = REPO_ROOT / "models" / "midas_small.onnx"
INPUT_HW  = 256   # spatial size fed to MiDaS
OPSET     = 17

def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading MiDaS small from torch.hub …")
    model = torch.hub.load(
        "intel-isl/MiDaS",
        "MiDaS_small",
        pretrained=True,
        trust_repo=True,
    )
    model.eval()

    dummy = torch.randn(1, 3, INPUT_HW, INPUT_HW)

    print(f"Exporting to {OUT_PATH}  (opset {OPSET}) …")
    torch.onnx.export(
        model,
        dummy,
        str(OUT_PATH),
        opset_version=OPSET,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch"},
            "output": {0: "batch"},
        },
        dynamo=False,   # use legacy TorchScript exporter
    )

    # Quick sanity-check with onnxruntime if available.
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(str(OUT_PATH), providers=["CPUExecutionProvider"])
        out  = sess.run(None, {"input": dummy.numpy()})
        print(f"Sanity check:  output shape = {out[0].shape}  "
              f"range [{out[0].min():.3f}, {out[0].max():.3f}]")
    except ImportError:
        print("(onnxruntime not installed – skipping sanity check)")

    size_mb = OUT_PATH.stat().st_size / 1_048_576
    print(f"Saved {OUT_PATH}  ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()
