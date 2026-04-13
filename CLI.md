# CLI

```
npx @jsilvanus/embedeer [options]

Model management (pull / cache model):
  npx @jsilvanus/embedeer --model <name>

Embed texts (batch):
  npx @jsilvanus/embedeer --model <name> --data "text1" "text2" ...
  npx @jsilvanus/embedeer --model <name> --data '["text1","text2"]'
  npx @jsilvanus/embedeer --model <name> --file texts.txt
  echo '["t1","t2"]' | npx @jsilvanus/embedeer --model <name>
  printf 'a\0b\0c' | npx @jsilvanus/embedeer --model <name> --delimiter '\0'

Interactive / streaming line-reader:
  npx @jsilvanus/embedeer --model <name> --interactive --dump out.jsonl
  cat big.txt | npx @jsilvanus/embedeer --model <name> -i --output csv --dump out.csv

Options:
  -m, --model <name>           Hugging Face model (default: Xenova/all-MiniLM-L6-v2)
  -d, --data <text...>         Text(s) or JSON array to embed
      --file <path>            Input file: JSON array or delimited texts
  -D, --delimiter <str>        Record separator for stdin/file (default: \n)
                               Escape sequences supported: \0 \n \t \r
  -i, --interactive            Interactive line-reader (see below)
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
| Interactive | `--interactive` / `-i` — line-reader, embeds as you type |

**Stdin auto-detection:** when `stdin` is not a TTY (i.e. data is piped or redirected), embedeer reads it before deciding what to do. JSON arrays are accepted directly; otherwise records are split on the delimiter.

---

### Interactive Line-Reader Mode (`-i` / `--interactive`)

The interactive mode opens a line-by-line reader that starts embedding as records arrive — ideal for pasting large datasets into a terminal or streaming data from another process.

```bash
# Open an interactive session (paste lines, Ctrl+D when done)
npx @jsilvanus/embedeer --model Xenova/all-MiniLM-L6-v2 --interactive --dump embeddings.jsonl

# Stream a large file through interactive mode with CSV output
cat big.txt | npx @jsilvanus/embedeer --model Xenova/all-MiniLM-L6-v2 \
  --interactive --output csv --dump embeddings.csv

# Interactive with GPU, custom batch size, txt output
npx @jsilvanus/embedeer --model Xenova/all-MiniLM-L6-v2 \
  --interactive --device auto --batch-size 16 --output txt --dump vecs.txt
```

**How it works:**

| Event | What happens |
|-------|-------------|
| Type a line, press Enter | Record is buffered |
| Buffer reaches `--batch-size` | Auto-flush: embed + append to output |
| Type an empty line | Manual flush: embed whatever is buffered |
| Ctrl+D (EOF) | Flush remaining records and exit |
| Ctrl+C | Flush remaining records and exit |

**Behaviour notes:**

- Progress messages (`Batch N: M record(s) → file`) always go to **stderr** — they never pollute piped output.
- When stdin is a TTY, a `> ` prompt is shown on stderr.
- Output defaults to **stdout** if `--dump` is omitted; a tip is printed when running in TTY mode.
- `--output json` and `--output sql` are automatically promoted to `jsonl` since they produce complete documents that cannot be appended to incrementally.
- `--output csv` writes the dimension header (`text,dim_0,dim_1,...`) on the first batch only; subsequent batches append data rows.
- Each interactive session **clears** the `--dump` file on start so you always get a fresh output file.

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

# Explicit CUDA (Linux x64 — requires CUDA 12 system libraries)
npx @jsilvanus/embedeer --model Xenova/all-MiniLM-L6-v2 --provider cuda --data "Hello CUDA"

# Explicit DirectML (Windows x64)
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

### Using GPU

No additional packages are needed — `onnxruntime-node` (installed with `@jsilvanus/embedeer`) already
bundles the CUDA provider on Linux x64 and DirectML on Windows x64.

**Linux x64 — NVIDIA CUDA:**

```bash
# One-time: install CUDA 12 system libraries (Ubuntu/Debian)
sudo apt install cuda-toolkit-12-6 libcudnn9-cuda-12

# Auto-detect: uses CUDA here, CPU fallback on any other machine
npx @jsilvanus/embedeer --model Xenova/all-MiniLM-L6-v2 --device auto --data "Hello"

# Hard-require CUDA (throws with diagnostic error if unavailable):
npx @jsilvanus/embedeer --model Xenova/all-MiniLM-L6-v2 --device gpu --data "Hello GPU"

# Explicit CUDA provider:
npx @jsilvanus/embedeer --model Xenova/all-MiniLM-L6-v2 --provider cuda --data "Hello CUDA"
```

**Windows x64 — DirectML (any GPU: NVIDIA / AMD / Intel):**

```bash
npx @jsilvanus/embedeer --model Xenova/all-MiniLM-L6-v2 --device auto --data "Hello"
npx @jsilvanus/embedeer --model Xenova/all-MiniLM-L6-v2 --device gpu  --data "Hello GPU"
npx @jsilvanus/embedeer --model Xenova/all-MiniLM-L6-v2 --provider dml --data "Hello DML"
```
