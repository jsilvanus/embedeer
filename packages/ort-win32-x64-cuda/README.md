# @embedeer/ort-win32-x64-cuda

CUDA execution provider for [embedeer](https://github.com/jsilvanus/embedeer) on **Windows x64**.

Install this package alongside `embedeer` to enable GPU-accelerated embeddings using NVIDIA CUDA on Windows.

## Installation

```bash
npm install embedeer
npm install @embedeer/ort-win32-x64-cuda
```

## Usage

```js
const embedder = await Embedder.create('Xenova/all-MiniLM-L6-v2', { device: 'gpu' });
```

```bash
npx embedeer --model Xenova/all-MiniLM-L6-v2 --device gpu --data "Hello"
```

> ⚠️ **Stub** — binary download not yet implemented. See `install.js` for TODO.

See [packages/ort-linux-x64-cuda/README.md](../ort-linux-x64-cuda/README.md) for full documentation.
