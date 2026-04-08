# @embedeer/ort-win32-x64-dml

DirectML execution provider for [embedeer](https://github.com/jsilvanus/embedeer) on **Windows x64**.

Install this package alongside `embedeer` to enable GPU-accelerated embeddings using DirectML on Windows (supports NVIDIA, AMD, and Intel GPUs — no CUDA required).

## Installation

```bash
npm install embedeer
npm install @embedeer/ort-win32-x64-dml
```

## Usage

```js
// On Windows, device='gpu' prefers CUDA first, then DirectML
const embedder = await Embedder.create('Xenova/all-MiniLM-L6-v2', { device: 'gpu' });

// Explicitly use DirectML
const embedder = await Embedder.create('Xenova/all-MiniLM-L6-v2', { provider: 'dml' });
```

```bash
npx embedeer --model Xenova/all-MiniLM-L6-v2 --provider dml --data "Hello"
```

> ⚠️ **Stub** — binary download not yet implemented. See `install.js` for TODO.

See [packages/ort-linux-x64-cuda/README.md](../ort-linux-x64-cuda/README.md) for full documentation.
