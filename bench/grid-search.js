#!/usr/bin/env node
// Grid search for embedeer: batchSize × concurrency × dtype

import { readFileSync, writeFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import os from 'os';
import { Embedder } from '../src/embedder.js';
import { getCacheDir } from '../src/model-cache.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

const argv = process.argv.slice(2);
const arg = (name, def) => {
  const i = argv.indexOf(name);
  if (i === -1) return def;
  return argv[i + 1] === undefined ? def : argv[i + 1];
};

const opts = {
  model: arg('--model', 'nomic-ai/nomic-embed-text-v1'),
  device: arg('--device', 'cpu'), // cpu or gpu
  sampleSize: Number(arg('--sample-size', '200')),
  warmup: arg('--warmup', 'true') !== 'false',
  out: arg('--out', join(__dirname, 'grid-results.json')),
};

function parseList(s, def) {
  if (!s) return def;
  return s.split(',').map((v) => {
    const n = Number(v);
    return Number.isNaN(n) ? v : n;
  });
}

const defaultRanges = {
  cpu: {
    batchSizes: [16, 32, 64],
    concurrencies: [1, 2, 4],
    dtypes: ['fp32', 'fp16', 'q8', 'q4', 'auto'],
  },
  gpu: {
    batchSizes: [32, 64, 128],
    concurrencies: [1, 2],
    dtypes: ['fp32', 'fp16', 'auto'],
  },
};

const ranges = {
  batchSizes: parseList(arg('--batch-sizes', ''), defaultRanges[opts.device].batchSizes),
  concurrencies: parseList(arg('--concurrencies', ''), defaultRanges[opts.device].concurrencies),
  dtypes: parseList(arg('--dtypes', ''), defaultRanges[opts.device].dtypes),
};

const texts = readFileSync(join(__dirname, 'texts-1000.txt'), 'utf8')
  .split('\n').map((l) => l.trim()).filter(Boolean);

async function runConfig(conf) {
  const cacheDir = getCacheDir();
  const result = {
    timestamp: new Date().toISOString(),
    host: os.hostname(),
    platform: process.platform,
    cpus: os.cpus().length,
    model: opts.model,
    device: conf.device,
    provider: conf.provider,
    batchSize: conf.batchSize,
    concurrency: conf.concurrency,
    dtype: conf.dtype,
    success: false,
  };

  let embedder;
  try {
    embedder = await Embedder.create(opts.model, {
      batchSize: conf.batchSize,
      concurrency: conf.concurrency,
      mode: conf.mode ?? (conf.device === 'gpu' ? 'process' : 'thread'),
      dtype: conf.dtype === 'none' ? undefined : conf.dtype,
      cacheDir,
      device: conf.device,
      provider: conf.provider,
    });

    // Warmup
    const warmupCount = Math.min(10, texts.length);
    if (opts.warmup && warmupCount > 0) {
      await embedder.embed(texts.slice(0, warmupCount));
    }

    const sampleCount = Math.min(opts.sampleSize, texts.length);
    const sample = texts.slice(0, sampleCount);
    const t0 = performance.now();
    const embeddings = await embedder.embed(sample);
    const elapsed = performance.now() - t0;

    result.success = true;
    result.elapsedMs = Math.round(elapsed);
    result.textsPerSec = Math.round((sampleCount / (elapsed / 1000)) * 10) / 10;
    result.dims = embeddings[0]?.length ?? null;
    result.heapMB = Math.round(process.memoryUsage().heapUsed / 1024 / 1024);
  } catch (err) {
    result.error = String(err.message ?? err);
  } finally {
    if (embedder) await embedder.destroy();
  }

  return result;
}

async function main() {
  console.log(`Grid search: model=${opts.model} device=${opts.device} sample=${opts.sampleSize}`);
  console.log('Ranges:', ranges);

  // Pre-load model into cache to avoid repeated downloads
  try {
    process.stderr.write('Pre-loading model into cache…\n');
    await Embedder.loadModel(opts.model, { cacheDir: getCacheDir() });
    process.stderr.write('  model pre-loaded.\n');
  } catch (err) {
    console.error('Model pre-load failed:', err?.message ?? err);
  }

  const results = [];

  // Iterate grid
  for (const dtype of ranges.dtypes) {
    for (const batchSize of ranges.batchSizes) {
      for (const concurrency of ranges.concurrencies) {
        const conf = {
          device: opts.device,
          provider: opts.device === 'win32' ? 'dml' : (opts.device === 'gpu' ? (process.platform === 'win32' ? 'dml' : 'cuda') : 'cpu'),
          batchSize,
          concurrency,
          dtype,
        };
        process.stderr.write(`Testing ${JSON.stringify(conf)}…\n`);
        const res = await runConfig(conf);
        results.push(res);
        // append to output file incrementally
        try {
          writeFileSync(opts.out, JSON.stringify({ generated: new Date().toISOString(), ranges, results }, null, 2));
        } catch (err) {
          console.error('Failed to write results:', err?.message ?? err);
        }
        process.stderr.write(`  -> ${res.success ? `${res.textsPerSec} t/s` : 'failed'}\n`);
      }
    }
  }

  console.log('Grid search complete — results written to', opts.out);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
