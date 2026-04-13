#!/usr/bin/env node
// Grid search for embedeer: batchSize × concurrency × dtype

import { readFileSync, writeFileSync, mkdirSync } from 'fs';
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
  out: arg('--out', join(__dirname, 'results', `grid-results-${arg('--device','cpu')}-${Date.now()}.json`)),
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
    concurrencies: [1, 2, 4, 8],
    dtypes: ['fp32', 'fp16', 'q8', 'q4', 'auto'],
    modes: ['thread'], // CPU defaults to thread workers
  },
  gpu: {
    batchSizes: [32, 64, 128],
    concurrencies: [1, 2],
    dtypes: ['fp32', 'fp16', 'auto'],
    modes: ['process'], // GPU defaults to process workers for better isolation
  },
};

const cliRanges = {
  batchSizes: arg('--batch-sizes', ''),
  concurrencies: arg('--concurrencies', ''),
  dtypes: arg('--dtypes', ''),
  // modes: comma-separated list of 'process'|'thread'
  modes: arg('--modes', ''),
};

const deviceArgProvided = argv.indexOf('--device') !== -1;
const devices = deviceArgProvided ? String(arg('--device', '')).split(',').map((s) => s.trim()).filter(Boolean) : ['cpu', 'gpu'];
const modes = ['thread', 'process'];

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
  console.log(`Grid search: model=${opts.model} devices=${devices.join(',')} sample=${opts.sampleSize}`);
  console.log('Ranges (CLI overrides):', cliRanges);

  // Ensure output directory exists
  try {
    mkdirSync(dirname(opts.out), { recursive: true });
  } catch (err) {}

  // Pre-load model into cache to avoid repeated downloads
  try {
    process.stderr.write('Pre-loading model into cache…\n');
    await Embedder.loadModel(opts.model, { cacheDir: getCacheDir() });
    process.stderr.write('  model pre-loaded.\n');
  } catch (err) {
    console.error('Model pre-load failed:', err?.message ?? err);
  }

  const results = [];

  // Iterate grid across devices and modes
  for (const device of devices) {
    const ranges = {
      batchSizes: parseList(cliRanges.batchSizes, defaultRanges[device].batchSizes),
      concurrencies: parseList(cliRanges.concurrencies, defaultRanges[device].concurrencies),
      dtypes: parseList(cliRanges.dtypes, defaultRanges[device].dtypes),
      modes: parseList(cliRanges.modes, defaultRanges[device].modes),
    };

    for (const mode of ranges.modes) {
      for (const dtype of ranges.dtypes) {
        for (const batchSize of ranges.batchSizes) {
          for (const concurrency of ranges.concurrencies) {
            const conf = {
              device,
              mode,
              provider: device === 'gpu' ? (process.platform === 'win32' ? 'dml' : 'cuda') : 'cpu',
              batchSize,
              concurrency,
              dtype,
            };

            process.stderr.write(`Testing ${JSON.stringify(conf)}…\n`);
            const res = await runConfig(conf);
            results.push(res);

            // append to output file incrementally
            try {
              writeFileSync(opts.out, JSON.stringify({ generated: new Date().toISOString(), devices, cliRanges, results }, null, 2));
            } catch (err) {
              console.error('Failed to write results:', err?.message ?? err);
            }

            process.stderr.write(`  -> ${res.success ? `${res.textsPerSec} t/s` : 'failed'}\n`);
            // Also print a compact Result line with the configuration and the measured time
            process.stderr.write(`Result: ${JSON.stringify(conf)} -> ${res.success ? `${res.textsPerSec} t/s` : 'failed'}\n`);
          }
        }
      }
    }
  }

  console.log('Grid search complete — results written to', opts.out);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
