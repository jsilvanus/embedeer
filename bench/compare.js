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
 *   --mode process|thread  embedeer worker mode (default: thread — avoids a Windows libuv
 *                          assertion crash that affects child-process mode with ONNX runtime)
 *   --dtype <type>         embedeer quantization: fp32|fp16|q8|q4|auto (default: none)
 *   --skip-embedeer        Skip the embedeer runner
 *   --skip-ollama          Skip both Ollama runners
 */

import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import os from 'os';
import { Embedder } from '../src/embedder.js';
import { getCacheDir } from '../src/model-cache.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

// ── Args ─────────────────────────────────────────────────────────────────────

const args = process.argv.slice(2);
const opts = {
  model:        'nomic-ai/nomic-embed-text-v1',
  // Updated default model to the newer Nomic model path
  ollamaUrl:    'http://localhost:11434',
  batchSize:    32,
  concurrency:  2,
  mode:         'thread', // thread avoids Windows libuv UV_HANDLE_CLOSING crash with ONNX in child processes
  dtype:        undefined,
  skipEmbedeer: false,
  skipOllama:   false,
};

for (let i = 0; i < args.length; i++) {
  const a = args[i];
  if      (a === '--model')         opts.model        = args[++i];
  else if (a === '--ollama-url')    opts.ollamaUrl    = args[++i];
  else if (a === '--batch-size')    opts.batchSize    = parseInt(args[++i], 10);
  else if (a === '--concurrency')   opts.concurrency  = parseInt(args[++i], 10);
  else if (a === '--mode')          opts.mode         = args[++i];
  else if (a === '--dtype')         opts.dtype        = args[++i];
  else if (a === '--skip-embedeer') opts.skipEmbedeer = true;
  else if (a === '--skip-ollama')   opts.skipOllama   = true;
}

// ── Texts ─────────────────────────────────────────────────────────────────────

const texts = readFileSync(join(__dirname, 'texts-1000.txt'), 'utf8')
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
    mode:        opts.mode,
    dtype:       opts.dtype,
    cacheDir,
  });
  const t0 = performance.now();
  const embeddings = await embedder.embed(texts);
  const elapsed = performance.now() - t0;
  await embedder.destroy();
  return { elapsed, dims: embeddings[0]?.length, label: `embedeer (${opts.mode}, concurrency=${opts.concurrency}, batch=${opts.batchSize})` };
}

async function runEmbedeerGPU() {
  const cacheDir = getCacheDir();
  const embedder = await Embedder.create(opts.model, {
    batchSize:   opts.batchSize,
    // keep GPU concurrency low to avoid device contention
    concurrency: 1,
    // run GPU worker in a separate process for isolation
    mode:        'process',
    dtype:       opts.dtype,
    cacheDir,
    device:      'gpu',
    // pick provider based on host OS: DirectML on Windows, CUDA elsewhere
    provider:    process.platform === 'win32' ? 'dml' : 'cuda',
  });
  const t0 = performance.now();
  const embeddings = await embedder.embed(texts);
  const elapsed = performance.now() - t0;
  await embedder.destroy();
  return { elapsed, dims: embeddings[0]?.length, label: `embedeer (gpu, provider=cuda, batch=${opts.batchSize})` };
}

async function runOllamaLegacy() {
  const url = `${opts.ollamaUrl}/api/embeddings`;
  const ollamaModel = 'nomic-embed-text';
  const tasks = texts.map((text) => async () => {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: ollamaModel, prompt: text }),
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
  const ollamaModel = 'nomic-embed-text';
  const t0 = performance.now();
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: ollamaModel, input: texts }),
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

async function main() {
  console.log(`\nembedeer vs Ollama — embedding benchmark`);
  console.log(`Model: ${opts.model}   Texts: ${texts.length}   embedeer mode: ${opts.mode}\n`);

  const results = [];

  // Pre-load model into local cache so workers don't have to download it.
  if (!opts.skipEmbedeer) {
    try {
      process.stderr.write('Pre-loading model into cache (this may take a while)…\n');
      await Embedder.loadModel(opts.model, { dtype: opts.dtype, cacheDir: getCacheDir() });
      process.stderr.write('  model pre-loaded.\n');
    } catch (err) {
      console.error(`  model pre-load failed: ${err.message}`);
    }

    // Tune concurrency: default to number of CPU cores if not provided
    const numCores = os.cpus().length || 1;
    const cpuConcurrency = opts.concurrency ?? Math.max(1, numCores);

    // Tune BLAS / threads to avoid oversubscription: set threads per worker
    const threadsPerWorker = Math.max(1, Math.floor(numCores / cpuConcurrency));
    process.env.OMP_NUM_THREADS = String(threadsPerWorker);
    process.env.MKL_NUM_THREADS = String(threadsPerWorker);

    // Create and reuse one Embedder instance for CPU (avoid reinitialization)
    process.stderr.write(`Running embedeer (cpu, concurrency=${cpuConcurrency})…\n`);
    let cpuEmbedder;
    try {
      cpuEmbedder = await Embedder.create(opts.model, {
        batchSize: opts.batchSize,
        concurrency: cpuConcurrency,
        mode: opts.mode,
        dtype: opts.dtype,
        cacheDir: getCacheDir(),
      });
      const t0 = performance.now();
      const embeddings = await cpuEmbedder.embed(texts);
      const elapsed = performance.now() - t0;
      results.push({ elapsed, dims: embeddings[0]?.length, label: `embedeer (cpu, ${opts.mode}, concurrency=${cpuConcurrency}, batch=${opts.batchSize})` });
      process.stderr.write('  done.\n');
    } catch (err) {
      console.error(`  embedeer failed: ${err.message}`);
    } finally {
      if (cpuEmbedder) await cpuEmbedder.destroy();
    }

    // GPU-backed embedeer: prefer small concurrency (1–2) and larger batch sizes
    const gpuConcurrency = 1; // keep single GPU worker to avoid contention
    process.stderr.write('Running embedeer (gpu)…\n');
    let gpuEmbedder;
    try {
      gpuEmbedder = await Embedder.create(opts.model, {
        batchSize: opts.batchSize * 2, // increase batch for GPU
        concurrency: gpuConcurrency,
        mode: 'process',
        dtype: opts.dtype,
        cacheDir: getCacheDir(),
        device: 'gpu',
        provider: process.platform === 'win32' ? 'dml' : 'cuda',
      });
      const t0g = performance.now();
      const embeddingsG = await gpuEmbedder.embed(texts);
      const elapsedG = performance.now() - t0g;
      results.push({ elapsed: elapsedG, dims: embeddingsG[0]?.length, label: `embedeer (gpu, provider=${process.platform === 'win32' ? 'dml' : 'cuda'}, batch=${opts.batchSize * 2})` });
      process.stderr.write('  done.\n');
    } catch (err) {
      console.error(`  embedeer (gpu) failed: ${err.message}`);
    } finally {
      if (gpuEmbedder) await gpuEmbedder.destroy();
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
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
