# embedeer

A Node.js tool for generating text embeddings using models from [Hugging Face](https://huggingface.co/models).  
Supports **batched** input and **parallel** processing via isolated worker processes.

---

## Features

- Downloads any Hugging Face feature-extraction model on first use (cached locally)
- Splits large input arrays into configurable batches
- Runs batches in parallel across multiple isolated worker processes
- Simple async API and a CLI

---

## Installation

```bash
npm install
```

---

## Programmatic API

```js
import { Embedder } from 'embedeer';

// Create an embedder backed by 2 parallel workers (default)
const embedder = await Embedder.create('Xenova/all-MiniLM-L6-v2', {
  batchSize: 32,   // texts per worker task  (default: 32)
  concurrency: 2,  // parallel worker processes (default: 2)
  pooling: 'mean', // 'mean' | 'cls' | 'none' (default: 'mean')
  normalize: true, // L2-normalise vectors    (default: true)
});

const texts = ['Hello world', 'Foo bar baz', 'Another sentence'];
const vectors = await embedder.embed(texts);
// vectors → number[][]  (one 384-dim vector per text for all-MiniLM-L6-v2)

await embedder.destroy(); // shut down worker processes
```

You can also construct and initialise separately:

```js
import { Embedder } from 'embedeer';

const embedder = new Embedder('Xenova/all-MiniLM-L6-v2', { concurrency: 4 });
await embedder.initialize(); // download/cache the model
const vectors = await embedder.embed(['text 1', 'text 2']);
await embedder.destroy();
```

---

## CLI

```
npx embedeer [options] "text1" "text2" ...
echo '["text1","text2"]' | npx embedeer [options]

Options:
  -m, --model <name>      Hugging Face model  (default: Xenova/all-MiniLM-L6-v2)
  -b, --batch-size <n>    Texts per batch     (default: 32)
  -c, --concurrency <n>   Parallel worker processes (default: 2)
  -p, --pooling <mode>    mean | cls | none   (default: mean)
      --no-normalize      Disable L2 normalisation
  -h, --help              Show help
```

Example:

```bash
node src/cli.js "Hello world" "Foo bar"
# outputs a JSON array of embedding vectors
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
                                 ├─ Process 0: pipeline('feature-extraction', model) → batch A
                                 ├─ Process 1: pipeline('feature-extraction', model) → batch B
                                 └─ ...
```

Each worker process loads the model **once** and processes many batches, avoiding
repeated model-download overhead.  Running as separate OS processes means a crash
in one worker does not affect the calling program — the pool rejects only that
worker's in-flight task and the rest continue normally.

---

## Testing

```bash
npm test
```

Tests use Node's built-in `node:test` runner with mocked workers (no real model download required).
