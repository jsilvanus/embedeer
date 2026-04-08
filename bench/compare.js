#!/usr/bin/env node
/**
 * Unified embedding benchmark: embedeer vs Ollama
 *
 * Runs the same 100 texts through three approaches and prints a comparison table.
 *
 * Usage:
 *   node bench/compare.js [options]
 *
 * Options:
 *   --model <name>         Model name used for all runners (default: nomic-embed-text)
 *   --ollama-url <url>     Ollama base URL (default: http://localhost:11434)
 *   --batch-size <n>       embedeer batch size (default: 32)
 *   --concurrency <n>      embedeer worker count (default: 2)
 *   --dtype <type>         embedeer quantization: fp32|fp16|q8|q4|auto (default: none)
 *   --skip-embedeer        Skip the embedeer runner
 *   --skip-ollama          Skip both Ollama runners
 */

import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { Embedder } from '../src/embedder.js';
import { getCacheDir } from '../src/model-cache.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

// ── Args ─────────────────────────────────────────────────────────────────────

const args = process.argv.slice(2);
const opts = {
  model:        'nomic-embed-text',
  ollamaUrl:    'http://localhost:11434',
  batchSize:    32,
  concurrency:  2,
  dtype:        undefined,
  skipEmbedeer: false,
  skipOllama:   false,
};

for (let i = 0; i < args.length; i++) {
  const a = args[i];
  if      (a === '--model')        opts.model        = args[++i];
  else if (a === '--ollama-url')   opts.ollamaUrl    = args[++i];
  else if (a === '--batch-size')   opts.batchSize    = parseInt(args[++i], 10);
  else if (a === '--concurrency')  opts.concurrency  = parseInt(args[++i], 10);
  else if (a === '--dtype')        opts.dtype        = args[++i];
  else if (a === '--skip-embedeer') opts.skipEmbedeer = true;
  else if (a === '--skip-ollama')   opts.skipOllama   = true;
}

// ── Texts ─────────────────────────────────────────────────────────────────────

const texts = readFileSync(join(__dirname, 'texts-100.txt'), 'utf8')
  .split('\n').map((l) => l.trim()).filter(Boolean);

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmt(ms)  { return (ms / 1000).toFixed(3) + 's'; }
function tps(ms)  { return (texts.length / (ms / 1000)).toFixed(1); }

/**
 * Run at most `limit` async tasks concurrently.
 * @param {Array<() => Promise<any>>} tasks
 * @param {number} limit
 */
async function pLimit(tasks, limit) {
  const results = new Array(tasks.length);
  let next = 0;
  async function worker() {
    while (next < tasks.length) {
      const i = next++;
      results[i] = await tasks[i]();
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, tasks.length) }, worker));
  return results;
}

// ── Runners ───────────────────────────────────────────────────────────────────

async function runEmbedeer() {
  const cacheDir = getCacheDir();
  const embedder = await Embedder.create(opts.model, {
    batchSize:   opts.batchSize,
    concurrency: opts.concurrency,
    dtype:       opts.dtype,
    cacheDir,
  });
  const t0 = performance.now();
  const embeddings = await embedder.embed(texts);
  const elapsed = performance.now() - t0;
  await embedder.destroy();
  return { elapsed, dims: embeddings[0]?.length, label: `embedeer (concurrency=${opts.concurrency}, batch=${opts.batchSize})` };
}

async function runOllamaLegacy() {
  const url = `${opts.ollamaUrl}/api/embeddings`;
  const tasks = texts.map((text) => async () => {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: opts.model, prompt: text }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
    return (await res.json()).embedding;
  });
  const t0 = performance.now();
  const embeddings = await pLimit(tasks, 1); // sequential — one text at a time
  const elapsed = performance.now() - t0;
  return { elapsed, dims: embeddings[0]?.length, label: 'Ollama /api/embeddings (sequential)' };
}

async function runOllamaBatch() {
  const url = `${opts.ollamaUrl}/api/embed`;
  const t0 = performance.now();
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: opts.model, input: texts }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  const json = await res.json();
  const elapsed = performance.now() - t0;
  return { elapsed, dims: json.embeddings?.[0]?.length, label: 'Ollama /api/embed   (batch)' };
}

// ── Table printer ─────────────────────────────────────────────────────────────

function printTable(results) {
  const COL = { label: 44, time: 10, tps: 12, dims: 6 };
  const line = '─'.repeat(COL.label + COL.time + COL.tps + COL.dims + 9);

  const pad = (s, n) => String(s).padEnd(n);
  const lpad = (s, n) => String(s).padStart(n);

  console.log('');
  console.log(line);
  console.log(
    pad('Runner', COL.label) + ' │ ' +
    lpad('Time', COL.time)   + ' │ ' +
    lpad('texts/s', COL.tps) + ' │ ' +
    lpad('dims', COL.dims)
  );
  console.log(line);

  // Sort by elapsed ascending
  const sorted = [...results].sort((a, b) => a.elapsed - b.elapsed);
  const fastest = sorted[0].elapsed;

  for (const r of sorted) {
    const ratio = r.elapsed / fastest;
    const ratioStr = ratio < 1.005 ? '(fastest)' : `(${ratio.toFixed(2)}x slower)`;
    console.log(
      pad(r.label, COL.label) + ' │ ' +
      lpad(fmt(r.elapsed), COL.time) + ' │ ' +
      lpad(tps(r.elapsed), COL.tps) + ' │ ' +
      lpad(r.dims ?? '?', COL.dims) +
      '   ' + ratioStr
    );
  }
  console.log(line);
  console.log(`Texts: ${texts.length}   Model: ${opts.model}`);
  console.log('');
}

// ── Main ──────────────────────────────────────────────────────────────────────

console.log(`\nembedeer vs Ollama — embedding benchmark`);
console.log(`Model: ${opts.model}   Texts: ${texts.length}\n`);

const results = [];

if (!opts.skipEmbedeer) {
  process.stderr.write('Running embedeer…\n');
  try {
    results.push(await runEmbedeer());
    process.stderr.write('  done.\n');
  } catch (err) {
    console.error(`  embedeer failed: ${err.message}`);
  }
}

if (!opts.skipOllama) {
  process.stderr.write('Running Ollama /api/embeddings…\n');
  try {
    results.push(await runOllamaLegacy());
    process.stderr.write('  done.\n');
  } catch (err) {
    console.error(`  /api/embeddings failed: ${err.message}`);
  }

  process.stderr.write('Running Ollama /api/embed…\n');
  try {
    results.push(await runOllamaBatch());
    process.stderr.write('  done.\n');
  } catch (err) {
    console.error(`  /api/embed failed: ${err.message}`);
  }
}

if (results.length === 0) {
  console.error('No results — all runners failed or were skipped.');
  process.exit(1);
}

printTable(results);
