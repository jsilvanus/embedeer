# embedeer

A Node.js tool for generating text embeddings using models from [Hugging Face](https://huggingface.co/models).  
Supports **batched** input, **parallel** execution, isolated **child-process** workers (default) or **in-process threads**, quantization, and Hugging Face auth.

---

## Features

- Downloads any Hugging Face feature-extraction model on first use (cached in `~/.embedeer/models`)
- **Isolated processes** (default) — a worker crash cannot bring down the caller
- **In-process threads** — opt-in via `mode: 'thread'` for lower overhead
- **Sequential** execution when `concurrency: 1`
- Configurable batch size and concurrency
- Hugging Face API token support (`--token` / `HF_TOKEN` env var)
- Quantization via `dtype` (`fp32` · `fp16` · `q8` · `q4` · `q4f16` · `auto`)
- Rich CLI: pull model, embed from file, dump output as JSON / TXT / SQL

---

## Installation

```bash
npm install
```

---

## Programmatic API

### Embed texts

```js
import { Embedder } from 'embedeer';

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

### Pull (pre-cache) a model

Like `ollama pull` — downloads the model once so workers start instantly:

```js
import { loadModel } from 'embedeer';

const { modelName, cacheDir } = await loadModel('Xenova/all-MiniLM-L6-v2', {
  token: 'hf_...',   // optional
  dtype: 'q8',       // optional
});
```

### Sequential execution

```js
const embedder = await Embedder.create('Xenova/all-MiniLM-L6-v2', { concurrency: 1 });
```

### In-process threads (same process, lower overhead)

```js
const embedder = await Embedder.create('Xenova/all-MiniLM-L6-v2', { mode: 'thread' });
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
  -h, --help                Show this help
```

### Examples

```bash
# Pull a model (like ollama pull)
npx embedeer --model Xenova/all-MiniLM-L6-v2

# Embed a few strings, output JSON
npx embedeer --model Xenova/all-MiniLM-L6-v2 --data "Hello" "World"

# Embed from a file, dump SQL to disk
npx embedeer --model Xenova/all-MiniLM-L6-v2 \
  --file texts.txt --output sql --dump out.sql

# Use quantized model, in-process threads, private model with token
npx embedeer --model my-org/private-model \
  --token hf_xxx --dtype q8 --mode thread \
  --data "embed me"
```

---

## How it works

```
embed(texts)
  │
  ├─ split into batches of batchSize
  │
  └─ Promise.all(batches) ──► WorkerPool
                                 │
                                 ├─ [process mode] ChildProcessWorker 0 → batch A
                                 ├─ [process mode] ChildProcessWorker 1 → batch B
                                 │   (OS-level isolation; crash → reject only that task)
                                 │
                                 ├─ [thread mode]  ThreadWorker 0 → batch A
                                 └─ [thread mode]  ThreadWorker 1 → batch B
```

Workers load the model **once** at startup and reuse it for all batches, avoiding
repeated download overhead. Models are cached in `~/.embedeer/models` so
subsequent runs start instantly.

---

## Testing

```bash
npm test
```

Tests use Node's built-in `node:test` runner. Worker behaviour is tested with
lightweight fake/echo workers — no real model download required.
