# embedeer

A Node.js tool for generating text embeddings using models from [Hugging Face](https://huggingface.co/models).  
Supports **batched** input, **parallel** execution, optional **GPU acceleration** (CUDA / DirectML), quantization, and Hugging Face auth.

This repository is a **monorepo** managed with npm workspaces.

---

## Packages

| Package | Description |
|---------|-------------|
| [`@jsilvanus/embedeer`](packages/embedeer) | Main embeddings package (CPU + optional GPU) |
| [`@jsilvanus/ort-linux-x64-cuda`](packages/ort-linux-x64-cuda) | CUDA provider for Linux x64 |
| [`@jsilvanus/ort-win32-x64-dml`](packages/ort-win32-x64-dml)   | DirectML provider for Windows x64 |

---

## Quick Start

### CPU (default, works everywhere)

```bash
npm install @jsilvanus/embedeer
npx embedeer --model Xenova/all-MiniLM-L6-v2 --data "Hello world"
```

```js
import { Embedder } from '@jsilvanus/embedeer';
const embedder = await Embedder.create('Xenova/all-MiniLM-L6-v2');
const vectors = await embedder.embed(['Hello', 'World']);
await embedder.destroy();
```

---

## GPU — Two-Step Install

### Linux x64 + NVIDIA CUDA (GPU MVP)

**System requirements:** NVIDIA GPU + driver ≥ 525, CUDA 12, cuDNN 9

`onnxruntime-node` v1.24.x ships `libonnxruntime_providers_cuda.so` on Linux x64. No custom binary needed — just install CUDA 12 + cuDNN 9 system libraries and the npm package:

```bash
# Install CUDA 12 + cuDNN 9 (Ubuntu/Debian)
sudo apt install cuda-toolkit-12-6 libcudnn9-cuda-12

# Install embedeer and the CUDA provider package
npm install @jsilvanus/embedeer
npm install @jsilvanus/ort-linux-x64-cuda

# Run with GPU
npx embedeer --model Xenova/all-MiniLM-L6-v2 --device gpu --data "Hello GPU"
```

### Docker + NVIDIA CUDA

Use an [NVIDIA CUDA Docker image](https://hub.docker.com/r/nvidia/cuda) as your base — it ships all required CUDA 12 + cuDNN 9 libraries, so no manual `apt install` is needed in your Dockerfile.

**Requirements on the host:**
- NVIDIA GPU driver ≥ 525
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed

**Example `Dockerfile`:**

```dockerfile
# CUDA 12 + cuDNN 9 runtime — all required libs are pre-installed
FROM nvidia/cuda:12.6.3-cudnn9-runtime-ubuntu24.04

WORKDIR /app

# Install Node.js (e.g. via NodeSource)
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Install embedeer + CUDA provider
COPY package.json ./
RUN npm install @jsilvanus/embedeer && \
    npm install @jsilvanus/ort-linux-x64-cuda

COPY . .
```

**Build and run:**

```bash
docker build -t my-embedeer-app .

# --gpus all enables NVIDIA GPU access inside the container
docker run --rm --gpus all my-embedeer-app \
  npx embedeer --model Xenova/all-MiniLM-L6-v2 --device gpu --data "Hello GPU"
```

**docker-compose:**

```yaml
services:
  embedeer:
    build: .
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: >
      npx embedeer --model Xenova/all-MiniLM-L6-v2
                   --device gpu
                   --data "Hello GPU"
```

### Windows x64 + DirectML (any GPU)

**System requirements:** Windows 10 (1903+) or 11, any DirectX 12 GPU, up-to-date drivers

```bash
npm install @jsilvanus/embedeer
npm install @jsilvanus/ort-win32-x64-dml

npx embedeer --model Xenova/all-MiniLM-L6-v2 --device gpu --data "Hello GPU"
```

### GPU API options

```js
import { Embedder } from '@jsilvanus/embedeer';

// Auto-detect GPU, silent CPU fallback if unavailable
const e1 = await Embedder.create(model, { device: 'auto' });

// Require GPU — throws if no GPU provider is available
const e2 = await Embedder.create(model, { device: 'gpu' });

// Explicit provider
const e3 = await Embedder.create(model, { provider: 'cuda' }); // Linux CUDA
const e4 = await Embedder.create(model, { provider: 'dml' });  // Windows DirectML
```

```bash
npx embedeer --device auto    # try GPU, fall back to CPU
npx embedeer --device gpu     # require GPU
npx embedeer --provider cuda  # explicit CUDA (Linux)
npx embedeer --provider dml   # explicit DirectML (Windows)
```

---

## Provider Selection Logic

| Platform | `device='auto'` or `device='gpu'` order |
|----------|-----------------------------------------|
| Linux x64 | CUDA → (CPU fallback) |
| Windows x64 | CUDA → DirectML → (CPU fallback) |
| Other | CPU only |

For `device='auto'`: silently falls back to CPU if no GPU provider is available.  
For `device='gpu'`: throws with a clear error and install instructions.  
For explicit `--provider cuda/dml`: throws if libraries are missing, with install instructions.

---

## Monorepo Development

```bash
npm install       # install all workspace packages
npm test          # run tests (packages/embedeer)
```

---

## Documentation

Full API documentation, CLI reference, and all options: [`packages/embedeer/README.md`](packages/embedeer/README.md)
