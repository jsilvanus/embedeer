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
npm install @jsilvanus/embedeer-ort-linux-x64-cuda

# GPU — Windows x64 + DirectML (any GPU: NVIDIA / AMD / Intel)
npm install @jsilvanus/embedeer-ort-win32-x64-dml
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
npx @jsilvanus/embedeer [options]

Model management (pull / cache model):
  npx @jsilvanus/embedeer --model <name>

Embed texts:
  npx @jsilvanus/embedeer --model <name> --data "text1" "text2" ...
  npx @jsilvanus/embedeer --model <name> --data '["text1","text2"]'
  npx @jsilvanus/embedeer --model <name> --file texts.txt
  echo '["t1","t2"]' | npx @jsilvanus/embedeer --model <name>
  printf 'a\0b\0c' | npx @jsilvanus/embedeer --model <name> --delimiter '\0'

Options:
  -m, --model <name>           Hugging Face model (default: Xenova/all-MiniLM-L6-v2)
  -d, --data <text...>         Text(s) or JSON array to embed
      --file <path>            Input file: JSON array or delimited texts
  -D, --delimiter <str>        Record separator for stdin/file (default: \n)
                               Escape sequences supported: \0 \n \t \r
      --dump <path>            Write output to file instead of stdout
      --output <format>        Output: json|jsonl|csv|txt|sql (default: json)
      --with-text              Include source text alongside each embedding
  -b, --batch-size <n>         Texts per worker batch (default: 32)
  -c, --concurrency <n>        Parallel workers (default: 2)
      --mode process|thread    Worker mode (default: process)
  -p, --pooling <mode>         mean|cls|none (default: mean)
      --no-normalize           Disable L2 normalisation
      --dtype <type>           Quantization: fp32|fp16|q8|q4|q4f16|auto
      --token <tok>            Hugging Face API token (or set HF_TOKEN env)
      --cache-dir <path>       Model cache directory (default: ~/.embedeer/models)
      --device <mode>          Compute device: auto|cpu|gpu (default: cpu)
      --provider <name>        Execution provider override: cpu|cuda|dml
  -h, --help                   Show this help
```

---

## Input Sources

Texts can be provided in any of these ways (checked in order):

| Source | How |
|--------|-----|
| Inline args | `--data "text1" "text2" "text3"` |
| Inline JSON | `--data '["text1","text2"]'` |
| File | `--file texts.txt` (JSON array or one record per line) |
| Stdin | Pipe or redirect — auto-detected; TTY is skipped |

**Stdin auto-detection:** when `stdin` is not a TTY (i.e. data is piped or redirected), embedeer reads it before deciding what to do. JSON arrays are accepted directly; otherwise records are split on the delimiter.

### Configurable delimiter (`-D` / `--delimiter`)

By default records in stdin and files are split on newline (`\n`). Use `--delimiter` to change it:

```bash
# Newline-delimited (default)
printf 'Hello\nWorld\n' | npx @jsilvanus/embedeer --model Xenova/all-MiniLM-L6-v2

# Null-byte delimited — safe with filenames/texts that contain newlines
printf 'Hello\0World\0' | npx @jsilvanus/embedeer --model Xenova/all-MiniLM-L6-v2 --delimiter '\0'

# Tab-delimited
printf 'Hello\tWorld' | npx @jsilvanus/embedeer --model Xenova/all-MiniLM-L6-v2 --delimiter '\t'

# Custom multi-character delimiter
printf 'Hello|||World|||Foo' | npx @jsilvanus/embedeer --model Xenova/all-MiniLM-L6-v2 --delimiter '|||'

# File with null-byte delimiter
npx @jsilvanus/embedeer --model Xenova/all-MiniLM-L6-v2 --file records.bin --delimiter '\0'

# Integrate with find -print0 (handles filenames with spaces / newlines)
find ./docs -name '*.txt' -print0 | \
  xargs -0 cat | \
  npx @jsilvanus/embedeer --model Xenova/all-MiniLM-L6-v2 --delimiter '\0'
```

Supported escape sequences in `--delimiter`:

| Sequence | Character |
|----------|-----------|
| `\0` | Null byte (U+0000) |
| `\n` | Newline (U+000A) |
| `\t` | Tab (U+0009) |
| `\r` | Carriage return (U+000D) |

---

## Output Formats

| Format | Description |
|--------|-------------|
| `json` (default) | JSON array of float arrays: `[[0.1,0.2,...],[...]]` |
| `json --with-text` | JSON array of objects: `[{"text":"...","embedding":[...]}]` |
| `jsonl` | Newline-delimited JSON, one object per line: `{"text":"...","embedding":[...]}` |
| `csv` | CSV with header: `text,dim_0,dim_1,...,dim_N` |
| `txt` | Space-separated floats, one vector per line |
| `txt --with-text` | Tab-separated: `<original text>\t<float float ...>` |
| `sql` | `INSERT INTO embeddings (text, vector) VALUES ...;` |

Use `--dump <path>` to write the output to a file instead of stdout. Progress messages always go to stderr so they never interfere with piped output.

### Piping examples

```bash
MODEL=Xenova/all-MiniLM-L6-v2

# --- json (default) ---
# Embed and pretty-print with jq
echo '["Hello","World"]' | npx @jsilvanus/embedeer --model $MODEL | jq '.[0] | length'

# --- jsonl ---
# One object per line — pipe to jq, grep, awk, etc.
npx @jsilvanus/embedeer --model $MODEL --data "foo" "bar" --output jsonl

# Filter by similarity: extract embedding for downstream processing
npx @jsilvanus/embedeer --model $MODEL --data "query text" --output jsonl \
  | jq -c '.embedding'

# Stream a large file and store as JSONL
npx @jsilvanus/embedeer --model $MODEL --file big.txt --output jsonl --dump out.jsonl

# --- json --with-text ---
# Keep the source text next to each vector (useful for building a search index)
npx @jsilvanus/embedeer --model $MODEL --output json --with-text \
  --data "cat" "dog" "fish" \
  | jq '.[] | {text, dims: (.embedding | length)}'

# --- csv ---
# Embed then open in Python/pandas
npx @jsilvanus/embedeer --model $MODEL --file texts.txt --output csv --dump vectors.csv
python3 -c "import pandas as pd; df = pd.read_csv('vectors.csv'); print(df.shape)"

# --- txt ---
# Raw floats — useful for awk/paste/numpy text loading
npx @jsilvanus/embedeer --model $MODEL --data "Hello" "World" --output txt \
  | awk '{print NF, "dimensions"}'

# txt --with-text: original text + tab + floats, easy to parse
npx @jsilvanus/embedeer --model $MODEL --file texts.txt --output txt --with-text \
  | while IFS=$'\t' read -r text vec; do echo "TEXT: $text"; done

# --- sql ---
# Generate INSERT statements for a vector DB or SQLite
npx @jsilvanus/embedeer --model $MODEL --file texts.txt --output sql --dump inserts.sql
sqlite3 mydb.sqlite < inserts.sql

# --- Chaining with other tools ---
# Embed stdin from another command
cat docs/*.txt | npx @jsilvanus/embedeer --model $MODEL --output jsonl > embeddings.jsonl

# Null-byte input from find (handles any filename or text with newlines)
find ./corpus -name '*.txt' -print0 \
  | xargs -0 cat \
  | npx @jsilvanus/embedeer --model $MODEL --delimiter '\0' --output jsonl
```

---

### CLI Examples

```bash
# Pull a model (like ollama pull)
npx @jsilvanus/embedeer --model Xenova/all-MiniLM-L6-v2

# Embed a few strings, output JSON (CPU)
npx @jsilvanus/embedeer --model Xenova/all-MiniLM-L6-v2 --data "Hello" "World"

# Auto-detect GPU, fall back to CPU if unavailable
# (uses CUDA on Linux, DirectML on Windows, CPU everywhere else)
npx @jsilvanus/embedeer --model Xenova/all-MiniLM-L6-v2 --device auto --data "Hello"

# Require GPU (throws with install instructions if no provider found)
npx @jsilvanus/embedeer --model Xenova/all-MiniLM-L6-v2 --device gpu --data "Hello GPU"

# Explicit CUDA (Linux — requires @jsilvanus/embedeer-ort-linux-x64-cuda)
npx @jsilvanus/embedeer --model Xenova/all-MiniLM-L6-v2 --provider cuda --data "Hello CUDA"

# Explicit DirectML (Windows — requires @jsilvanus/embedeer-ort-win32-x64-dml)
npx @jsilvanus/embedeer --model Xenova/all-MiniLM-L6-v2 --provider dml --data "Hello DML"

# Embed from a file, dump SQL to disk
npx @jsilvanus/embedeer --model Xenova/all-MiniLM-L6-v2 \
  --file texts.txt --output sql --dump out.sql

# Use quantized model, in-process threads, private model with token
npx @jsilvanus/embedeer --model my-org/private-model \
  --token hf_xxx --dtype q8 --mode thread \
  --data "embed me"
```

---

### Using GPU with npx

Install the provider package for your platform, then pass `--device auto` to use the GPU
wherever available, with silent CPU fallback.

**Linux x64 — NVIDIA CUDA:**

```bash
# One-time: install CUDA 12 system libraries (Ubuntu/Debian)
sudo apt install cuda-toolkit-12-6 libcudnn9-cuda-12

# Install both packages
npm install @jsilvanus/embedeer
npm install @jsilvanus/embedeer-ort-linux-x64-cuda

# Auto-detect: uses CUDA here, CPU fallback on any other machine
npx @jsilvanus/embedeer --model Xenova/all-MiniLM-L6-v2 --device auto --data "Hello"

# Hard-require CUDA (error + install hint if unavailable):
npx @jsilvanus/embedeer --model Xenova/all-MiniLM-L6-v2 --device gpu --data "Hello GPU"

# Explicit CUDA provider:
npx @jsilvanus/embedeer --model Xenova/all-MiniLM-L6-v2 --provider cuda --data "Hello CUDA"
```

**Windows x64 — DirectML (any GPU: NVIDIA / AMD / Intel):**

```bash
npm install @jsilvanus/embedeer
npm install @jsilvanus/embedeer-ort-win32-x64-dml

npx @jsilvanus/embedeer --model Xenova/all-MiniLM-L6-v2 --device auto --data "Hello"
npx @jsilvanus/embedeer --model Xenova/all-MiniLM-L6-v2 --device gpu  --data "Hello GPU"
npx @jsilvanus/embedeer --model Xenova/all-MiniLM-L6-v2 --provider dml --data "Hello DML"
```

---

## GPU Provider Packages

GPU support requires an additional provider package that ships a CUDA-enabled (or DirectML-enabled) ONNX Runtime binary.

| Platform       | Provider  | Package                                       |
|----------------|-----------|-----------------------------------------------|
| Linux x64      | CUDA      | `@jsilvanus/embedeer-ort-linux-x64-cuda`      |
| Windows x64    | DirectML  | `@jsilvanus/embedeer-ort-win32-x64-dml`       |

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
