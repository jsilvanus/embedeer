# @jsilvanus/embedeer-ort-win32-x64-dml

DirectML execution provider for [embedeer](https://github.com/jsilvanus/embedeer) on **Windows x64**.

Install this package alongside `embedeer` to enable GPU-accelerated embeddings using DirectML on Windows. Supports **NVIDIA, AMD, and Intel GPUs** — no CUDA installation required.

## How it works

`onnxruntime-node` ships DirectML support bundled on Windows x64 — **no additional binary download is required**.

DirectML is a Microsoft API built into Windows 10 (1903+) and Windows 11 that accelerates machine learning inference across all DirectX 12-capable GPUs.

## System Requirements

| Requirement | Version |
|-------------|---------|
| Windows | 10 (1903+) or Windows 11 |
| GPU | Any DirectX 12-capable GPU (NVIDIA, AMD, Intel — most GPUs from 2016+) |
| GPU Driver | Up-to-date drivers from your GPU vendor |

No CUDA installation needed.

## Installation

```bash
# Step 1 — main package
npm install @jsilvanus/embedeer

# Step 2 — DirectML provider
npm install @jsilvanus/embedeer-ort-win32-x64-dml
```

## Usage

```js
import { Embedder } from 'embedeer';

// Auto-detect GPU (DirectML is tried first on Windows)
const embedder = await Embedder.create('Xenova/all-MiniLM-L6-v2', { device: 'auto' });

// Require GPU
const embedder = await Embedder.create('Xenova/all-MiniLM-L6-v2', { device: 'gpu' });

// Explicit DirectML
const embedder = await Embedder.create('Xenova/all-MiniLM-L6-v2', { provider: 'dml' });
```

```bash
# CLI — auto GPU (uses DirectML on Windows)
npx @jsilvanus/embedeer --model Xenova/all-MiniLM-L6-v2 --device auto --data "Hello GPU"

# CLI — explicit DirectML
npx @jsilvanus/embedeer --model Xenova/all-MiniLM-L6-v2 --provider dml --data "Hello DML"
```
