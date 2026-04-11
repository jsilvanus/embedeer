#!/usr/bin/env node
/**
 * embedeer CLI
 *
 * Model management:
 *   embedeer --model <name>                        Pull / cache a model
 *
 * Embedding (batch):
 *   embedeer --model <name> --data "text1" "text2" ...
 *   embedeer --model <name> --data '["text1","text2"]'
 *   embedeer --model <name> --file texts.txt
 *   echo '["t1","t2"]' | embedeer --model <name>
 *   printf 'a\0b\0c' | embedeer --model <name> --delimiter '\0'
 *
 * Interactive / streaming line-reader:
 *   embedeer --model <name> --interactive --dump out.jsonl
 *   cat big.txt | embedeer --model <name> --interactive --output csv --dump out.csv
 *
 * Options:
 *   -m, --model <name>           Hugging Face model (default: nomic-embed-text)
 *   -d, --data <text...>         Text(s) to embed (JSON array or individual strings)
 *       --file <path>            File of texts (JSON array or one text per line)
 *   -D, --delimiter <str>        Record separator for stdin/file input (default: \n)
 *                                Escape sequences: \0 (null byte), \n, \t, \r
 *   -i, --interactive            Interactive line-reader: embed as lines arrive
 *       --dump <path>            Write output to file instead of stdout
 *       --output <format>        Output format: json|jsonl|csv|txt|sql (default: json)
 *       --with-text              Include source text in json/txt output
 *   -b, --batch-size <n>         Texts per worker batch (default: 32)
 *   -c, --concurrency <n>        Parallel worker processes/threads (default: 2)
 *       --mode process|thread    Worker mode (default: process)
 *   -p, --pooling <mode>         Pooling: mean|cls|none (default: mean)
 *       --no-normalize           Disable L2 normalisation
 *       --dtype <type>           Quantization: fp32|fp16|q8|q4|q4f16|auto
 *       --token <tok>            Hugging Face API token (overrides HF_TOKEN env var)
 *       --cache-dir <path>       Custom model cache directory (default: ~/.embedeer/models)
 *       --device <mode>          Compute device: auto|cpu|gpu (default: cpu)
 *       --provider <name>        Execution provider override: cpu|cuda|dml
 *       --prefix <str>           Text prepended to every input before embedding
 *   -h, --help                   Show this help
 */

import { getCacheDir, DEFAULT_CACHE_DIR } from './model-cache.js';
import { readFileSync, writeFileSync, appendFileSync } from 'fs';
import readline from 'readline';
import { fileURLToPath } from 'url';

// ── Argument parsing ────────────────────────────────────────────────────────

const args = process.argv.slice(2);

function printHelp() {
  console.log(`
embedeer — parallel batched embeddings from Hugging Face

Model management (pull / cache):
  embedeer --model <name>

Embedding (batch):
  embedeer --model <name> [--data "text1" "text2" ...]
  embedeer --model <name> --file texts.txt
  echo '["t1","t2"]' | embedeer --model <name>
  printf 'a\\0b\\0c' | embedeer --model <name> --delimiter '\\0'

Interactive / streaming line-reader:
  embedeer --model <name> --interactive --dump out.jsonl
  cat big.txt | embedeer --model <name> -i --output csv --dump out.csv

Options:
  -m, --model <name>           Hugging Face model (default: nomic-embed-text)
  -d, --data <text...>         Text(s) or JSON array to embed
      --file <path>            Input file: JSON array or delimited texts
  -D, --delimiter <str>        Record separator for stdin/file (default: \\n)
                               Escape sequences supported: \\0 \\n \\t \\r
  -i, --interactive            Interactive line-reader: embed as lines arrive
                               (empty line or full batch triggers immediate flush)
      --dump <path>            Write output to file instead of stdout
      --output <format>        Output: json|jsonl|csv|txt|sql (default: json)
      --with-text              Include source text alongside each embedding
  -b, --batch-size <n>         Texts per worker batch (default: 32)
  -c, --concurrency <n>        Parallel workers (default: 2)
      --mode process|thread    Worker mode (default: process)
  -p, --pooling <mode>         mean|cls|none (default: mean)
      --no-normalize           Disable L2 normalisation
      --dtype <type>           Quantization: fp32|fp16|q8|q4|q4f16|auto
      --token <tok>            Hugging Face API token
      --cache-dir <path>       Model cache directory (default: ${DEFAULT_CACHE_DIR})
      --device <mode>          Compute device: auto|cpu|gpu (default: cpu)
      --provider <name>        Execution provider override: cpu|cuda|dml
      --prefix <str>           Text prepended to every input before embedding
                               (e.g. "search_query: " for nomic-embed-text)
      --timer                  Print elapsed wall-clock time to stderr when done
  -h, --help                   Show this help
`.trim());
}

// Known flag names. Used to distinguish flags from text values when parsing
// --data so that negative numbers or hyphen-prefixed strings work correctly.
const KNOWN_FLAGS = new Set([
  '--help', '-h', '--model', '-m', '--data', '-d', '--file', '--dump',
  '--output', '--with-text', '--batch-size', '-b', '--concurrency', '-c',
  '--mode', '--pooling', '-p', '--no-normalize', '--dtype', '--token',
  '--cache-dir', '--device', '--provider', '--delimiter', '-D',
  '--interactive', '-i', '--prefix', '--timer',
]);
const options = {
  model: 'nomic-embed-text',
  data: null,         // --data texts (array)
  file: null,         // --file path
  delimiter: '\n',    // --delimiter record separator for stdin/file
  interactive: false, // --interactive / -i: line-reader mode
  dump: null,         // --dump path
  output: 'json',     // json | jsonl | csv | txt | sql
  withText: false,    // --with-text: include source text in output
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
  prefix: undefined,
  timer: false,
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
  } else if (arg === '--delimiter' || arg === '-D') {
    options.delimiter = parseDelimiter(args[++i]);
  } else if (arg === '--interactive' || arg === '-i') {
    options.interactive = true;
  } else if (arg === '--dump') {
    options.dump = args[++i];
  } else if (arg === '--output') {
    options.output = args[++i];
  } else if (arg === '--with-text') {
    options.withText = true;
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
  } else if (arg === '--prefix') {
    options.prefix = args[++i];
  } else if (arg === '--timer') {
    options.timer = true;
  } else {
    positional.push(arg);
  }
}

// ── Output formatting ───────────────────────────────────────────────────────

export function formatOutput(texts, embeddings, format, withText) {
  switch (format) {
    case 'jsonl':
      return texts
        .map((text, i) => JSON.stringify({ text, embedding: embeddings[i] }))
        .join('\n');

    case 'csv': {
      if (embeddings.length === 0) return '';
      const dims = embeddings[0].length;
      const header = ['text', ...Array.from({ length: dims }, (_, k) => `dim_${k}`)].join(',');
      const rows = texts.map((text, i) => {
        const safeText = '"' + text.replace(/"/g, '""') + '"';
        return [safeText, ...embeddings[i]].join(',');
      });
      return [header, ...rows].join('\n');
    }

    case 'txt':
      if (withText) {
        return texts.map((text, i) => `${text}\t${embeddings[i].join(' ')}`).join('\n');
      }
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
      if (withText) {
        return JSON.stringify(
          texts.map((text, i) => ({ text, embedding: embeddings[i] }))
        );
      }
      return JSON.stringify(embeddings);
  }
}

export function writeOutput(content, dumpPath) {
  if (dumpPath) {
    writeFileSync(dumpPath, content + '\n', 'utf8');
    console.error(`Output written to ${dumpPath}`);
  } else {
    console.log(content);
  }
}

// ── Input reading ───────────────────────────────────────────────────────────

/**
 * Convert a user-supplied delimiter string, resolving common escape sequences.
 * Supports: \0 (null byte), \n (newline), \t (tab), \r (carriage return).
 */
export function parseDelimiter(str) {
  return str
    .replace(/\\0/g, '\0')
    .replace(/\\n/g, '\n')
    .replace(/\\t/g, '\t')
    .replace(/\\r/g, '\r');
}

/**
 * Parse a block of text into an array of strings.
 * First tries to parse as a JSON array; if that fails, splits on `delimiter`.
 */
export function parseTexts(raw, delimiter = '\n') {
  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) throw new Error('Expected a JSON array');
    return parsed;
  } catch {
    return raw.split(delimiter).filter(Boolean);
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

// ── Interactive / streaming line-reader mode ────────────────────────────────

/**
 * Interactive mode: read one text record per line from stdin, embed in
 * configurable batches, and stream results to a file (or stdout).
 *
 * Flushing triggers:
 *   • Batch reaches --batch-size lines (auto-flush)
 *   • User types an empty line (manual flush)
 *   • EOF / Ctrl+D (flush remaining records and exit)
 *   • Ctrl+C (flush remaining records and exit)
 *
 * Output:
 *   • Formats json and sql are not appendable — they are promoted to jsonl.
 *   • csv writes the dimension header once (on the first batch).
 *   • All other formats append each batch as independent lines.
 *   • Progress/prompt messages always go to stderr.
 */
async function runInteractive(cacheDir) {
  const t0 = options.timer ? performance.now() : 0;
  // json and sql produce complete documents that can't be appended to
  // incrementally; switch to jsonl so each batch emits self-contained lines.
  if (options.output === 'json' || options.output === 'sql') {
    console.error(
      `Warning: --output ${options.output} is not suitable for interactive mode. Switching to jsonl.`
    );
    options.output = 'jsonl';
  }

  const isTTY = process.stdin.isTTY;
  const outputFile = options.dump;

  if (isTTY && !outputFile) {
    console.error(
      'Tip: use --dump <path> to write output to a file so it does not mix with input.'
    );
  }

  // Load the model before opening the reader so we are ready to embed immediately.
  const { Embedder } = await import('./embedder.js');
  console.error(`Loading model: ${options.model}…`);
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

  if (isTTY) {
    console.error('Model ready. Paste records below, one per line.');
    console.error(`Batch size: ${options.batchSize}. Empty line = flush now. Ctrl+D = done. Ctrl+C = abort.`);
  }

  // Initialise / clear the output file so each interactive session starts fresh.
  if (outputFile) {
    writeFileSync(outputFile, '', 'utf8');
  }

  let csvHeaderWritten = false;
  let batch = [];
  let batchNumber = 0;
  let flushing = false;

  /**
   * Embed the current batch and write its output.
   * The readline interface must be paused before calling this.
   */
  async function flushBatch() {
    if (batch.length === 0) return;
    const texts = [...batch];
    batch = [];
    batchNumber++;

    const embeddings = await embedder.embed(texts, { prefix: options.prefix });
    let content;

    if (options.output === 'csv') {
      const full = formatOutput(texts, embeddings, 'csv', options.withText);
      if (!csvHeaderWritten) {
        content = full;                                // includes header
        csvHeaderWritten = true;
      } else {
        content = full.split('\n').slice(1).join('\n'); // data rows only
      }
    } else {
      content = formatOutput(texts, embeddings, options.output, options.withText);
    }

    if (outputFile) {
      appendFileSync(outputFile, content + '\n', 'utf8');
      console.error(`Batch ${batchNumber}: ${texts.length} record(s) → ${outputFile}`);
    } else {
      process.stdout.write(content + '\n');
    }
  }

  const rl = readline.createInterface({
    input: process.stdin,
    // Route the prompt to stderr so it never pollutes stdout embeddings.
    output: isTTY ? process.stderr : null,
    terminal: isTTY,
  });

  if (isTTY) rl.prompt();

  rl.on('line', (line) => {
    const text = line.trim();

    if (text !== '') {
      batch.push(text);
    }

    const shouldFlush = text === '' || batch.length >= options.batchSize;

    if (shouldFlush && !flushing && batch.length > 0) {
      flushing = true;
      rl.pause();
      flushBatch()
        .then(() => {
          flushing = false;
          rl.resume();
          if (isTTY) rl.prompt();
        })
        .catch((err) => {
          console.error('Error embedding batch:', err.message);
          flushing = false;
          rl.resume();
          if (isTTY) rl.prompt();
        });
    } else if (isTTY) {
      rl.prompt();
    }
  });

  await new Promise((resolve) => {
    rl.on('close', async () => {
      try {
        await flushBatch();
      } catch (err) {
        console.error('Error embedding final batch:', err.message);
      }
      await embedder.destroy();
      if (outputFile) {
        console.error(`Done. ${batchNumber} batch(es) written to ${outputFile}`);
      }
      if (options.timer) {
        const elapsed = ((performance.now() - t0) / 1000).toFixed(3);
        console.error(`Time: ${elapsed}s`);
      }
      resolve();
    });

    // Handle Ctrl+C — flush remaining records then exit cleanly.
    rl.on('SIGINT', () => {
      console.error('\nInterrupted — flushing remaining records…');
      rl.close(); // triggers 'close' event above
    });
  });
}



async function main() {
  const resolvedCacheDir = getCacheDir(options.cacheDir);

  // ── Interactive line-reader mode ─────────────────────────────────────────
  if (options.interactive) {
    return runInteractive(resolvedCacheDir);
  }

  // ── Model-only mode (pull / cache) ──────────────────────────────────────
  const hasDataSource = options.data || options.file || positional.length > 0;
  if (!hasDataSource) {
    const stdinRaw = await readStdin();
    if (!stdinRaw) {
      // No data — download/cache the model and exit.
      const { Embedder } = await import('./embedder.js');
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
    const texts = parseTexts(stdinRaw, options.delimiter);
    return runEmbedding(texts, resolvedCacheDir);
  }

  // ── Collect texts ────────────────────────────────────────────────────────
  let texts = [];

  if (options.file) {
    const raw = readFileSync(options.file, 'utf8').trim();
    texts = parseTexts(raw, options.delimiter);
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
  const t0 = options.timer ? performance.now() : 0;
  const { Embedder } = await import('./embedder.js');

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
    const embeddings = await embedder.embed(texts, { prefix: options.prefix });
    const content = formatOutput(texts, embeddings, options.output, options.withText);
    writeOutput(content, options.dump);
  } finally {
    await embedder.destroy();
    if (options.timer) {
      const elapsed = ((performance.now() - t0) / 1000).toFixed(3);
      console.error(`Time: ${elapsed}s (${texts.length} texts)`);
    }
  }
}

// Only execute main() when this file is run directly, not when imported.
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  main().catch((err) => {
    console.error('Error:', err.message);
    process.exit(1);
  });
}

