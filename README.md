# embedeer

A Node.js tool for generating text embeddings using models from [Hugging Face](https://huggingface.co/models).  
Supports **batched** input, **parallel** execution, optional **GPU acceleration** (CUDA / DirectML), quantization, and Hugging Face auth.

This repository is a **monorepo** managed with npm workspaces.

---

## Packages

| Package | Description |
|---------|-------------|
| [`embedeer`](packages/embedeer) | Main embeddings package (CPU + optional GPU) |
| [`@embedeer/ort-linux-x64-cuda`](packages/ort-linux-x64-cuda) | CUDA provider for Linux x64 |
| [`@embedeer/ort-win32-x64-cuda`](packages/ort-win32-x64-cuda) | CUDA provider for Windows x64 |
| [`@embedeer/ort-win32-x64-dml`](packages/ort-win32-x64-dml)   | DirectML provider for Windows x64 |

---

## Quick Start

### CPU (default)

```bash
# Install
npm install embedeer

# CLI
npx embedeer --model Xenova/all-MiniLM-L6-v2 --data "Hello world"

# API
import { Embedder } from 'embedeer';
const embedder = await Embedder.create('Xenova/all-MiniLM-L6-v2');
const vectors = await embedder.embed(['Hello', 'World']);
await embedder.destroy();
```

### GPU (two-step install)

```bash
# Step 1 - install embedeer
npm install embedeer

# Step 2 - install GPU provider for your platform
npm install @embedeer/ort-linux-x64-cuda   # Linux x64 NVIDIA CUDA
npm install @embedeer/ort-win32-x64-cuda   # Windows x64 NVIDIA CUDA
npm install @embedeer/ort-win32-x64-dml    # Windows x64 DirectML (any GPU)

# CLI - auto-detect GPU, fall back to CPU
npx embedeer --model Xenova/all-MiniLM-L6-v2 --device auto --data "Hello GPU"

# CLI - require GPU
npx embedeer --model Xenova/all-MiniLM-L6-v2 --device gpu --data "Hello GPU"

# API
import { Embedder } from 'embedeer';
const embedder = await Embedder.create('Xenova/all-MiniLM-L6-v2', {
  device: 'auto',     // 'auto' | 'cpu' | 'gpu'
  // provider: 'cuda', // explicit override: 'cuda' | 'dml'
});
```

---

## Monorepo Development

```bash
# Install all workspace packages
npm install

# Run tests (packages/embedeer)
npm test

# Run tests in a specific package
npm test --workspace=packages/embedeer
```

---

## GPU Provider Status

> The native binary download in GPU provider packages is currently **stubbed**.
> The JS API structure, dynamic loading hooks, and runtime selection logic are fully implemented.
> Actual CUDA/DirectML binaries will be added in a future release.
> See each provider package's `install.js` for the full TODO list.

---

## Documentation

Full API documentation, CLI reference, and options are in [`packages/embedeer/README.md`](packages/embedeer/README.md).
