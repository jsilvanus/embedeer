#!/usr/bin/env node
/**
 * embedeer CLI
 *
 * Model management:
 *   embedeer --model <name>                        Pull / cache a model
 *
 * Embedding:
 *   embedeer --model <name> --data "text1" "text2" ...
 *   embedeer --model <name> --data '["text1","text2"]'
 *   embedeer --model <name> --file texts.txt
 *   echo '["t1","t2"]' | embedeer --model <name>
 *
 * Options:
 *   -m, --model <name>        Hugging Face model (default: Xenova/all-MiniLM-L6-v2)
 *   -d, --data <text...>      Text(s) to embed (JSON array or individual strings)
 *       --file <path>         File of texts (JSON array or one text per line)
 *       --dump <path>         Write output to file instead of stdout
 *       --output json|txt|sql Output format (default: json)
 *   -b, --batch-size <n>      Texts per worker batch (default: 32)
 *   -c, --concurrency <n>     Parallel worker processes/threads (default: 2)
 *       --mode process|thread Worker mode: isolated processes or in-process threads (default: process)
 *   -p, --pooling <mode>      Pooling: mean|cls|none (default: mean)
 *       --no-normalize        Disable L2 normalisation
 *       --dtype <type>        Quantization: fp32|fp16|q8|q4|q4f16|auto
 *       --token <tok>         Hugging Face API token (overrides HF_TOKEN env var)
 *       --cache-dir <path>    Custom model cache directory (default: ~/.embedeer/models)
 *       --device <mode>       Compute device: auto|cpu|gpu (default: cpu)
 *       --provider <name>     Execution provider override: cpu|cuda|dml
 *   -h, --help                Show this help
 */

import { Embedder } from './embedder.js';
import { getCacheDir, DEFAULT_CACHE_DIR } from './model-cache.js';
import { readFileSync, writeFileSync } from 'fs';

// ── Argument parsing ────────────────────────────────────────────────────────

const args = process.argv.slice(2);

function printHelp() {
  console.log(`
embedeer — parallel batched embeddings from Hugging Face

Model management (pull / cache):
  embedeer --model <name>

Embedding:
  embedeer --model <name> [--data "text1" "text2" ...]
  embedeer --model <name> --file texts.txt
  echo '["t1","t2"]' | embedeer --model <name>

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
      --token <tok>         Hugging Face API token
      --cache-dir <path>    Model cache directory (default: ${DEFAULT_CACHE_DIR})
      --device <mode>       Compute device: auto|cpu|gpu (default: cpu)
      --provider <name>     Execution provider override: cpu|cuda|dml
  -h, --help                Show this help
`.trim());
}

// Known flag names. Used to distinguish flags from text values when parsing
// --data so that negative numbers or hyphen-prefixed strings work correctly.
const KNOWN_FLAGS = new Set([
  '--help', '-h', '--model', '-m', '--data', '-d', '--file', '--dump',
  '--output', '--batch-size', '-b', '--concurrency', '-c', '--mode',
  '--pooling', '-p', '--no-normalize', '--dtype', '--token', '--cache-dir',
  '--device', '--provider',
]);
  model: 'Xenova/all-MiniLM-L6-v2',
  data: null,       // --data texts (array)
  file: null,       // --file path
  dump: null,       // --dump path
  output: 'json',   // json | txt | sql
  batchSize: 32,
  concurrency: 2,
  mode: 'process',
  pooling: 'mean',
  normalize: true,
  dtype: undefined,
  token: undefined,
  cacheDir: undefined,
  device: undefined,
  provider: undefined,
};

const positional = [];

for (let i = 0; i < args.length; i++) {
  const arg = args[i];
  if (arg === '--help' || arg === '-h') {
    printHelp();
    process.exit(0);
  } else if (arg === '--model' || arg === '-m') {
    options.model = args[++i];
  } else if (arg === '--data' || arg === '-d') {
    // Consume subsequent args that are not known flags so that text values
    // beginning with '-' (e.g. negative numbers) are handled correctly.
    options.data = [];
    while (i + 1 < args.length && !KNOWN_FLAGS.has(args[i + 1])) {
      options.data.push(args[++i]);
    }
  } else if (arg === '--file') {
    options.file = args[++i];
  } else if (arg === '--dump') {
    options.dump = args[++i];
  } else if (arg === '--output') {
    options.output = args[++i];
  } else if (arg === '--batch-size' || arg === '-b') {
    options.batchSize = parseInt(args[++i], 10);
  } else if (arg === '--concurrency' || arg === '-c') {
    options.concurrency = parseInt(args[++i], 10);
  } else if (arg === '--mode') {
    options.mode = args[++i];
  } else if (arg === '--pooling' || arg === '-p') {
    options.pooling = args[++i];
  } else if (arg === '--no-normalize') {
    options.normalize = false;
  } else if (arg === '--dtype') {
    options.dtype = args[++i];
  } else if (arg === '--token') {
    options.token = args[++i];
  } else if (arg === '--cache-dir') {
    options.cacheDir = args[++i];
  } else if (arg === '--device') {
    options.device = args[++i];
  } else if (arg === '--provider') {
    options.provider = args[++i];
  } else {
    positional.push(arg);
  }
}

// ── Output formatting ───────────────────────────────────────────────────────

function formatOutput(texts, embeddings, format) {
  switch (format) {
    case 'txt':
      return embeddings.map((vec) => vec.join(' ')).join('\n');

    case 'sql': {
      const rows = texts.map((text, i) => {
        const safeText = text.replace(/'/g, "''");
        const vector = JSON.stringify(embeddings[i]);
        return `  ('${safeText}', '${vector}')`;
      });
      return (
        'INSERT INTO embeddings (text, vector) VALUES\n' +
        rows.join(',\n') +
        ';'
      );
    }

    default: // json
      return JSON.stringify(embeddings);
  }
}

function writeOutput(content, dumpPath) {
  if (dumpPath) {
    writeFileSync(dumpPath, content + '\n', 'utf8');
    console.error(`Output written to ${dumpPath}`);
  } else {
    console.log(content);
  }
}

// ── Input reading ───────────────────────────────────────────────────────────

function parseTexts(raw) {
  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) throw new Error('Expected a JSON array');
    return parsed;
  } catch {
    return raw.split('\n').filter(Boolean);
  }
}

async function readStdin() {
  // Return empty string immediately when running interactively (TTY) so the
  // CLI doesn't hang waiting for stdin that will never arrive.
  if (process.stdin.isTTY) return '';
  return new Promise((resolve) => {
    let data = '';
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', (chunk) => { data += chunk; });
    process.stdin.on('end', () => resolve(data.trim()));
  });
}

// ── Main ────────────────────────────────────────────────────────────────────

async function main() {
  const resolvedCacheDir = getCacheDir(options.cacheDir);

  // ── Model-only mode (pull / cache) ──────────────────────────────────────
  const hasDataSource = options.data || options.file || positional.length > 0;
  if (!hasDataSource) {
    const stdinRaw = await readStdin();
    if (!stdinRaw) {
      // No data — download/cache the model and exit.
      console.error(`Pulling model: ${options.model}`);
      console.error(`Cache directory: ${resolvedCacheDir}`);
      await Embedder.loadModel(options.model, {
        token: options.token,
        dtype: options.dtype,
        cacheDir: resolvedCacheDir,
      });
      console.error('Model ready.');
      return;
    }
    // Stdin provided — treat as text input.
    const texts = parseTexts(stdinRaw);
    return runEmbedding(texts, resolvedCacheDir);
  }

  // ── Collect texts ────────────────────────────────────────────────────────
  let texts = [];

  if (options.file) {
    const raw = readFileSync(options.file, 'utf8').trim();
    texts = parseTexts(raw);
  } else if (options.data && options.data.length > 0) {
    // --data may be a JSON array in a single arg or multiple plain strings
    if (options.data.length === 1) {
      texts = parseTexts(options.data[0]);
    } else {
      texts = options.data;
    }
  } else if (positional.length > 0) {
    texts = positional;
  }

  if (texts.length === 0) {
    console.error('Error: no input texts found.');
    process.exit(1);
  }

  return runEmbedding(texts, resolvedCacheDir);
}

async function runEmbedding(texts, cacheDir) {
  const embedder = await Embedder.create(options.model, {
    batchSize: options.batchSize,
    concurrency: options.concurrency,
    mode: options.mode,
    pooling: options.pooling,
    normalize: options.normalize,
    dtype: options.dtype,
    token: options.token,
    cacheDir,
    device: options.device,
    provider: options.provider,
  });

  try {
    const embeddings = await embedder.embed(texts);
    const content = formatOutput(texts, embeddings, options.output);
    writeOutput(content, options.dump);
  } finally {
    await embedder.destroy();
  }
}

main().catch((err) => {
  console.error('Error:', err.message);
  process.exit(1);
});

