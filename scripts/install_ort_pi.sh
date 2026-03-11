#!/usr/bin/env bash
# Install ONNX Runtime 1.24.3 shared library on Raspberry Pi (aarch64).
# Required by the perception crate when built with the `onnx` feature.
#
# Usage: bash scripts/install_ort_pi.sh
# Then set ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so at runtime
# (or add it to ~/.bashrc / a systemd unit).

set -euo pipefail

VER=1.24.3
URL="https://github.com/microsoft/onnxruntime/releases/download/v${VER}/onnxruntime-linux-aarch64-${VER}.tgz"
TMP=$(mktemp -d)

echo "Downloading ONNX Runtime ${VER} for aarch64..."
curl -fsSL "$URL" -o "$TMP/ort.tgz"

echo "Extracting..."
tar -xzf "$TMP/ort.tgz" -C "$TMP"

echo "Installing to /usr/local/lib/ ..."
sudo cp "$TMP/onnxruntime-linux-aarch64-${VER}/lib/"libonnxruntime*.so* /usr/local/lib/
sudo ldconfig

rm -rf "$TMP"
echo "Done.  ONNX Runtime ${VER} installed."
echo "Set ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so before running."
