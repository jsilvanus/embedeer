# @embedeer/ort-linux-x64-cuda

CUDA execution provider for [embedeer](https://github.com/jsilvanus/embedeer) on **Linux x64**.

Install this package alongside `embedeer` to enable GPU-accelerated embeddings using NVIDIA CUDA.

---

## Installation (two-step)

```bash
# Step 1 — install embedeer
npm install embedeer

# Step 2 — install the CUDA provider for Linux x64
npm install @embedeer/ort-linux-x64-cuda
```

> **Requirements**
> - Linux x86_64
> - NVIDIA GPU with CUDA drivers installed (CUDA 12.x recommended)
> - NVIDIA CUDA Toolkit matching the binary version

---

## Usage

Once installed, embedeer automatically detects and uses this provider:

```js
import { Embedder } from 'embedeer';

// Auto-detect GPU (falls back to CPU if no provider is installed)
const embedder = await Embedder.create('Xenova/all-MiniLM-L6-v2', {
  device: 'auto',
});

// Require GPU (throws if not available)
const embedder = await Embedder.create('Xenova/all-MiniLM-L6-v2', {
  device: 'gpu',
});

// Explicitly request CUDA
const embedder = await Embedder.create('Xenova/all-MiniLM-L6-v2', {
  provider: 'cuda',
});
```

CLI:

```bash
# Auto GPU (falls back to CPU)
npx embedeer --model Xenova/all-MiniLM-L6-v2 --device auto --data "Hello GPU"

# Require GPU
npx embedeer --model Xenova/all-MiniLM-L6-v2 --device gpu --data "Hello GPU"

# Explicit CUDA provider
npx embedeer --model Xenova/all-MiniLM-L6-v2 --provider cuda --data "Hello GPU"
```

---

## How it works

This package provides:

1. **`install.js`** — runs on `npm install` to download (or build) a CUDA-enabled
   ONNX Runtime Node.js binding into `vendor/`.

2. **`index.js`** — exports `activate()` and `getDevice()`. The `activate()` function
   verifies that the native binary is present and configures ONNX Runtime to use the
   CUDA execution provider. `getDevice()` returns `'cuda'` so embedeer passes the
   correct device string to `@huggingface/transformers` `pipeline()`.

---

## Current status (stub)

> ⚠️ The binary download in `install.js` is currently **stubbed** — no real CUDA
> binary is downloaded yet. GPU execution is not functional until the TODO in
> `install.js` is implemented.
>
> See `install.js` for the full TODO list and skeleton download code.

### What needs to be done

1. Build a CUDA-enabled `onnxruntime-node` binding:
   ```bash
   # Clone ORT
   git clone --recursive https://github.com/microsoft/onnxruntime
   cd onnxruntime
   # Build with CUDA
   ./build.sh --config Release --build_nodejs --use_cuda \
     --cuda_home /usr/local/cuda \
     --cudnn_home /usr/local/cuda
   ```

2. Upload the resulting `.node` file as a GitHub Release asset.

3. Update `install.js` to download and verify the binary.

4. Update `index.js` to wire the binary into ONNX Runtime's module resolution.

---

## Platform

| Platform | Architecture | Provider | Package |
|----------|-------------|----------|---------|
| Linux    | x64          | CUDA     | `@embedeer/ort-linux-x64-cuda` ← **this package** |
| Windows  | x64          | CUDA     | `@embedeer/ort-win32-x64-cuda` |
| Windows  | x64          | DirectML | `@embedeer/ort-win32-x64-dml`  |
