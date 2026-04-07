#!/usr/bin/env node
/**
 * embedeer CLI
 *
 * Usage:
 *   embedeer [options] "text1" "text2" ...
 *   echo '["text1","text2"]' | embedeer [options]
 *
 * Options:
 *   --model,       -m  Hugging Face model name  (default: Xenova/all-MiniLM-L6-v2)
 *   --batch-size,  -b  Texts per worker batch   (default: 32)
 *   --concurrency, -c  Parallel worker threads  (default: 2)
 *   --pooling,     -p  Pooling strategy         (default: mean)
 *   --no-normalize     Disable L2 normalisation
 *   --help,        -h  Show this help
 */

import { Embedder } from './embedder.js';
import { readFileSync } from 'fs';

// ── Argument parsing ────────────────────────────────────────────────────────

const args = process.argv.slice(2);

function printHelp() {
  console.log(`
embedeer — parallel batched embeddings from Hugging Face

Usage:
  embedeer [options] "text1" "text2" ...
  echo '["text1","text2"]' | embedeer [options]

Options:
  -m, --model <name>      Hugging Face model (default: Xenova/all-MiniLM-L6-v2)
  -b, --batch-size <n>    Texts per worker batch (default: 32)
  -c, --concurrency <n>   Parallel worker processes (default: 2)
  -p, --pooling <mode>    Pooling strategy: mean|cls|none (default: mean)
      --no-normalize      Disable L2 normalisation
  -h, --help              Show this help
`.trim());
}

const options = {
  model: 'Xenova/all-MiniLM-L6-v2',
  batchSize: 32,
  concurrency: 2,
  pooling: 'mean',
  normalize: true,
};

const positional = [];

for (let i = 0; i < args.length; i++) {
  const arg = args[i];
  if (arg === '--help' || arg === '-h') {
    printHelp();
    process.exit(0);
  } else if (arg === '--model' || arg === '-m') {
    options.model = args[++i];
  } else if (arg === '--batch-size' || arg === '-b') {
    options.batchSize = parseInt(args[++i], 10);
  } else if (arg === '--concurrency' || arg === '-c') {
    options.concurrency = parseInt(args[++i], 10);
  } else if (arg === '--pooling' || arg === '-p') {
    options.pooling = args[++i];
  } else if (arg === '--no-normalize') {
    options.normalize = false;
  } else {
    positional.push(arg);
  }
}

// ── Collect input texts ─────────────────────────────────────────────────────

async function readStdin() {
  return new Promise((resolve) => {
    let data = '';
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', (chunk) => { data += chunk; });
    process.stdin.on('end', () => resolve(data.trim()));
  });
}

async function main() {
  let texts = positional;

  if (texts.length === 0) {
    // Try reading from stdin
    const raw = await readStdin();
    if (!raw) {
      console.error('Error: provide texts as arguments or via stdin as a JSON array.');
      printHelp();
      process.exit(1);
    }
    try {
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) throw new Error('Expected a JSON array');
      texts = parsed;
    } catch {
      // Treat stdin as newline-separated texts
      texts = raw.split('\n').filter(Boolean);
    }
  }

  if (texts.length === 0) {
    console.error('Error: no input texts provided.');
    process.exit(1);
  }

  const embedder = await Embedder.create(options.model, {
    batchSize: options.batchSize,
    concurrency: options.concurrency,
    pooling: options.pooling,
    normalize: options.normalize,
  });

  try {
    const embeddings = await embedder.embed(texts);
    console.log(JSON.stringify(embeddings));
  } finally {
    await embedder.destroy();
  }
}

main().catch((err) => {
  console.error('Error:', err.message);
  process.exit(1);
});
