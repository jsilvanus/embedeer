#!/usr/bin/env node
/**
 * Ollama embedding benchmark
 *
 * Sends the same 100 texts to both Ollama embedding endpoints and reports
 * wall-clock time for each, for comparison against embedeer --timer.
 *
 * Usage:
 *   node bench/ollama-bench.js [--model <name>] [--url <base>] [--concurrency <n>]
 *
 * Defaults:
 *   --model        nomic-embed-text
 *   --url          http://localhost:11434
 *   --concurrency  1  (sequential; increase to test parallel throughput)
 *
 * Endpoints tested:
 *   POST /api/embeddings  — legacy single-text endpoint
 *   POST /api/embed       — newer batch endpoint
 */

import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

// ── CLI args ────────────────────────────────────────────────────────────────

const args = process.argv.slice(2);
const opts = {
  model: 'nomic-embed-text',
  url: 'http://localhost:11434',
  concurrency: 1,
};

for (let i = 0; i < args.length; i++) {
  if (args[i] === '--model')       opts.model       = args[++i];
  else if (args[i] === '--url')    opts.url         = args[++i];
  else if (args[i] === '--concurrency') opts.concurrency = parseInt(args[++i], 10);
}

// ── Load texts ───────────────────────────────────────────────────────────────

const textsFile = join(__dirname, 'texts-100.txt');
const texts = readFileSync(textsFile, 'utf8')
  .split('\n')
  .map((l) => l.trim())
  .filter(Boolean);

console.log(`Texts loaded: ${texts.length}`);
console.log(`Model:        ${opts.model}`);
console.log(`Ollama URL:   ${opts.url}`);
console.log(`Concurrency:  ${opts.concurrency}`);
console.log('');

// ── Helpers ──────────────────────────────────────────────────────────────────

function fmt(ms) {
  return (ms / 1000).toFixed(3) + 's';
}

/**
 * Run tasks with limited concurrency.
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

  const workers = Array.from({ length: Math.min(limit, tasks.length) }, worker);
  await Promise.all(workers);
  return results;
}

// ── /api/embeddings (legacy — one request per text) ─────────────────────────

async function benchLegacy() {
  console.log('── /api/embeddings (one request per text) ──────────────────');
  const url = `${opts.url}/api/embeddings`;

  const tasks = texts.map((text) => async () => {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: opts.model, prompt: text }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
    const json = await res.json();
    return json.embedding;
  });

  const t0 = performance.now();
  let embeddings;
  try {
    embeddings = await pLimit(tasks, opts.concurrency);
  } catch (err) {
    console.error(`  Error: ${err.message}`);
    return;
  }
  const elapsed = performance.now() - t0;

  console.log(`  Texts:      ${embeddings.length}`);
  console.log(`  Dimensions: ${embeddings[0]?.length ?? '?'}`);
  console.log(`  Time:       ${fmt(elapsed)}`);
  console.log(`  Throughput: ${(texts.length / (elapsed / 1000)).toFixed(1)} texts/s`);
  console.log('');
}

// ── /api/embed (newer — sends all texts in one request) ──────────────────────

async function benchBatch() {
  console.log('── /api/embed (batch — all texts in one request) ───────────');
  const url = `${opts.url}/api/embed`;

  const t0 = performance.now();
  let embeddings;
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: opts.model, input: texts }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
    const json = await res.json();
    embeddings = json.embeddings;
  } catch (err) {
    console.error(`  Error: ${err.message}`);
    return;
  }
  const elapsed = performance.now() - t0;

  console.log(`  Texts:      ${embeddings?.length ?? '?'}`);
  console.log(`  Dimensions: ${embeddings?.[0]?.length ?? '?'}`);
  console.log(`  Time:       ${fmt(elapsed)}`);
  console.log(`  Throughput: ${(texts.length / (elapsed / 1000)).toFixed(1)} texts/s`);
  console.log('');
}

// ── Main ─────────────────────────────────────────────────────────────────────

await benchLegacy();
await benchBatch();

console.log('Done. Compare these numbers against:');
console.log(`  node src/cli.js --model ${opts.model} --file bench/texts-100.txt --timer --output txt > /dev/null`);
