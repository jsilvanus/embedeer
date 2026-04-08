# embedeer

A Node.js tool for generating text embeddings using models from [Hugging Face](https://huggingface.co/models).  
Supports **batched** input, **parallel** execution, isolated **child-process** workers (default) or **in-process threads**, quantization, optional GPU acceleration, and Hugging Face auth.

---

## Features

- Downloads any Hugging Face feature-extraction model on first use (cached in `~/.embedeer/models`)
- **Isolated processes** (default) — a worker crash cannot bring down the caller
- **In-process threads** — opt-in via `mode: 'thread'` for lower overhead
- **Sequential** execution when `concurrency: 1`
- Configurable batch size and concurrency
- **GPU acceleration** — optional via separate provider packages (see below)
- Hugging Face API token support (`--token` / `HF_TOKEN` env var)
- Quantization via `dtype` (`fp32` · `fp16` · `q8` · `q4` · `q4f16` · `auto`)
- Rich CLI: pull model, embed from file, dump output as JSON / TXT / SQL

---

## Installation

```bash
# CPU (default, works everywhere)
npm install @jsilvanus/embedeer

# GPU — Linux x64 + NVIDIA CUDA
npm install @jsilvanus/ort-linux-x64-cuda

# GPU — Windows x64 + NVIDIA CUDA
npm install @jsilvanus/ort-win32-x64-cuda

# GPU — Windows x64 + DirectML (any GPU: NVIDIA / AMD / Intel)
npm install @jsilvanus/ort-win32-x64-dml
```

---

## Programmatic API

### Embed texts (CPU — default)

```js
import { Embedder } from '@jsilvanus/embedeer';

const embedder = await Embedder.create('Xenova/all-MiniLM-L6-v2', {
  batchSize:   32,          // texts per worker task   (default: 32)
  concurrency: 2,           // parallel workers        (default: 2)
  mode:       'process',    // 'process' | 'thread'    (default: 'process')
  pooling:    'mean',       // 'mean' | 'cls' | 'none' (default: 'mean')
  normalize:   true,        // L2-normalise vectors    (default: true)
  token:      'hf_...',     // HF API token (optional; also reads HF_TOKEN env)
  dtype:      'q8',         // quantization dtype      (optional)
  cacheDir:   '/my/cache',  // override model cache    (default: ~/.embedeer/models)
});

const vectors = await embedder.embed(['Hello world', 'Foo bar baz']);
// → number[][]  (one 384-dim vector per text for all-MiniLM-L6-v2)

await embedder.destroy(); // shut down worker processes
```

### Embed texts with GPU

```js
import { Embedder } from '@jsilvanus/embedeer';

// Auto-detect GPU (falls back to CPU if no provider is installed)
const embedder = await Embedder.create('Xenova/all-MiniLM-L6-v2', {
  device: 'auto',
});

// Require GPU (throws if no provider is available)
const embedder = await Embedder.create('Xenova/all-MiniLM-L6-v2', {
  device: 'gpu',
});

// Explicitly select an execution provider
const embedder = await Embedder.create('Xenova/all-MiniLM-L6-v2', {
  provider: 'cuda',  // 'cuda' | 'dml'
});
```

### Pull (pre-cache) a model

Like `ollama pull` — downloads the model once so workers start instantly:

```js
import { loadModel } from '@jsilvanus/embedeer';

const { modelName, cacheDir } = await loadModel('Xenova/all-MiniLM-L6-v2', {
  token: 'hf_...',   // optional
  dtype: 'q8',       // optional
});
```

---

## CLI

```
npx embedeer [options]

Model management (pull / cache model):
  npx embedeer --model <name>

Embed texts:
  npx embedeer --model <name> --data "text1" "text2" ...
  npx embedeer --model <name> --data '["text1","text2"]'
  npx embedeer --model <name> --file texts.txt
  echo '["t1","t2"]' | npx embedeer --model <name>

Options:
  -m, --model <name>        Hugging Face model (default: Xenova/all-MiniLM-L6-v2)
  -d, --data <text...>      Text(s) or JSON array to embed
      --file <path>         Input file: JSON array or one text per line
      --dump <path>         Write output to file instead of stdout
      --output json|txt|sql Output format (default: json)
  -b, --batch-size <n>      Texts per worker batch (default: 32)
  -c, --concurrency <n>     Parallel workers (default: 2)
      --mode process|thread Worker mode (default: process)
  -p, --pooling <mode>      mean|cls|none (default: mean)
      --no-normalize        Disable L2 normalisation
      --dtype <type>        Quantization: fp32|fp16|q8|q4|q4f16|auto
      --token <tok>         Hugging Face API token (or set HF_TOKEN env)
      --cache-dir <path>    Model cache directory (default: ~/.embedeer/models)
      --device <mode>       Compute device: auto|cpu|gpu (default: cpu)
      --provider <name>     Execution provider override: cpu|cuda|dml
  -h, --help                Show this help
```

### CLI Examples

```bash
# Pull a model (like ollama pull)
npx embedeer --model Xenova/all-MiniLM-L6-v2

# Embed a few strings, output JSON (CPU)
npx embedeer --model Xenova/all-MiniLM-L6-v2 --data "Hello" "World"

# Auto-detect GPU, fall back to CPU if unavailable
npx embedeer --model Xenova/all-MiniLM-L6-v2 --device auto --data "Hello GPU"

# Require GPU (error if no provider installed)
npx embedeer --model Xenova/all-MiniLM-L6-v2 --device gpu --data "Hello GPU"

# Use CUDA explicitly (requires @jsilvanus/ort-linux-x64-cuda or ort-win32-x64-cuda)
npx embedeer --model Xenova/all-MiniLM-L6-v2 --provider cuda --data "Hello CUDA"

# Use DirectML on Windows (requires @jsilvanus/ort-win32-x64-dml)
npx embedeer --model Xenova/all-MiniLM-L6-v2 --provider dml --data "Hello DML"

# Embed from a file, dump SQL to disk
npx embedeer --model Xenova/all-MiniLM-L6-v2 \
  --file texts.txt --output sql --dump out.sql

# Use quantized model, in-process threads, private model with token
npx embedeer --model my-org/private-model \
  --token hf_xxx --dtype q8 --mode thread \
  --data "embed me"
```

---

## GPU Provider Packages

GPU support requires an additional provider package that ships a CUDA-enabled (or DirectML-enabled) ONNX Runtime binary.

| Platform       | Provider  | Package                          |
|----------------|-----------|----------------------------------|
| Linux x64      | CUDA      | `@jsilvanus/ort-linux-x64-cuda`   |
| Windows x64    | CUDA      | `@jsilvanus/ort-win32-x64-cuda`   |
| Windows x64    | DirectML  | `@jsilvanus/ort-win32-x64-dml`    |

### Provider selection logic

| `device` | `provider` | Behavior |
|----------|-----------|----------|
| `cpu` (default) | — | Always CPU |
| `auto` | — | Try GPU providers for the platform in order; silent CPU fallback |
| `gpu` | — | Try GPU providers; **throw** if none available |
| any | `cuda` | Load CUDA provider; **throw** if not available or not supported |
| any | `dml` | Load DirectML provider; **throw** if not available or not supported |
| any | `cpu` | Always CPU |

On Linux x64: GPU order is `cuda`.  
On Windows x64: GPU order is `cuda → dml`.

---

## How it works

```
embed(texts)
  │
  ├─ split into batches of batchSize
  │
  └─ Promise.all(batches) ──► WorkerPool
                                 │
                                 ├─ [process mode] ChildProcessWorker 0
                                 │   resolveProvider(device, provider)
                                 │   → pipeline('feature-extraction', model, { device: 'cuda' })
                                 │   → embed batch A
                                 │
                                 └─ [process mode] ChildProcessWorker 1
                                     resolveProvider(device, provider)
                                     → pipeline(...) → embed batch B
```

Workers load the model **once** at startup and reuse it for all batches.  
Provider activation happens per-worker before the pipeline is created.

---

## Testing

```bash
cd packages/embedeer && npm test
# or from the monorepo root:
npm test
```

Tests use Node's built-in `node:test` runner. No real model download required.
