# @embedeer/ort-win32-x64-cuda

> ⚠️ **Not yet available** — CUDA on Windows is not included in `onnxruntime-node` prebuilt binaries.

This package is a placeholder for future Windows CUDA support.

## Use DirectML instead

For GPU acceleration on Windows, use DirectML — it supports NVIDIA, AMD, and Intel GPUs without CUDA:

```bash
npm install @embedeer/ort-win32-x64-dml
```

See [`@embedeer/ort-win32-x64-dml`](../ort-win32-x64-dml/README.md) for full documentation.

## Why CUDA isn't available on Windows

`onnxruntime-node` prebuilt binaries include CUDA support on **Linux x64** only (CUDA 12 + cuDNN 9). Windows CUDA support would require either:
- A future official ONNX Runtime release with Windows CUDA prebuilts
- A custom `onnxruntime-node` build against CUDA on Windows

See [ONNX Runtime build docs](https://onnxruntime.ai/docs/build/inferencing.html) if you need Windows CUDA.
