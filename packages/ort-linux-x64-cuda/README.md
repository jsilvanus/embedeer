# @embedeer/ort-linux-x64-cuda

CUDA execution provider for [embedeer](https://github.com/jsilvanus/embedeer) on **Linux x64**.

Install this package alongside `embedeer` to enable GPU-accelerated embeddings using NVIDIA CUDA on Linux.

## How it works

`onnxruntime-node` v1.14+ ships `libonnxruntime_providers_cuda.so` on Linux x64 as part of its standard npm package — **no additional binary download is required**.

This package verifies that the required CUDA 12 system libraries are present, then returns `device='cuda'` so that `@huggingface/transformers` pipeline runs on the GPU.

## System Requirements

| Requirement | Version |
|-------------|---------|
| NVIDIA GPU Driver | ≥ 525 (CUDA 12 compatible) |
| CUDA Toolkit | 12.x (`libcudart.so.12`, `libcublas.so.12`, `libcublasLt.so.12`, `libcurand.so.10`, `libcufft.so.11`) |
| cuDNN | 9.x (`libcudnn.so.9`) |
| OS | Linux x64 |

### Installing CUDA 12 + cuDNN 9

**Ubuntu/Debian (recommended):**
```bash
# Add NVIDIA package repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Install CUDA 12 and cuDNN 9
sudo apt install cuda-toolkit-12-6 libcudnn9-cuda-12

# Add to PATH / LD_LIBRARY_PATH
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

**CUDA Toolkit installer:** https://developer.nvidia.com/cuda-downloads  
**cuDNN download:** https://developer.nvidia.com/cudnn-downloads  

Verify installation:
```bash
nvidia-smi         # confirm GPU is detected
nvcc --version     # confirm CUDA toolkit is installed
```

## Installation

```bash
# Step 1 — main package
npm install embedeer

# Step 2 — CUDA provider
npm install @embedeer/ort-linux-x64-cuda
```

## Usage

```js
import { Embedder } from 'embedeer';

// Auto-detect GPU (falls back to CPU if CUDA unavailable)
const embedder = await Embedder.create('Xenova/all-MiniLM-L6-v2', { device: 'auto' });

// Require GPU (throws with clear error if CUDA unavailable)
const embedder = await Embedder.create('Xenova/all-MiniLM-L6-v2', { device: 'gpu' });

// Explicit CUDA
const embedder = await Embedder.create('Xenova/all-MiniLM-L6-v2', { provider: 'cuda' });
```

```bash
# CLI — auto GPU
npx embedeer --model Xenova/all-MiniLM-L6-v2 --device auto --data "Hello GPU"

# CLI — explicit CUDA
npx embedeer --model Xenova/all-MiniLM-L6-v2 --provider cuda --data "Hello CUDA"
```

## Error messages

If CUDA libraries are missing, you'll see:

```
@embedeer/ort-linux-x64-cuda: Missing CUDA system libraries: libcudart.so.12, libcudnn.so.9

onnxruntime-node CUDA requires CUDA 12 + cuDNN 9. Install them:

  # Option A — CUDA 12 + cuDNN 9 via apt (Ubuntu/Debian)
  sudo apt install cuda-toolkit-12-6 libcudnn9-cuda-12
  ...
```
