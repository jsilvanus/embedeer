#!/usr/bin/env node
/**
 * Server mode benchmark — compares socket and gRPC server throughput
 * against the baseline process/thread modes.
 *
 * Starts each server as a subprocess, waits for it to signal readiness,
 * runs embeddings via a raw client, then shuts the server down.
 * Measures both startup time (spawn → ready) and embedding throughput.
 *
 * Usage:
 *   node bench/server-bench.js [options]
 *
 * Options:
 *   --model       <name>    HF model identifier (default: Xenova/all-MiniLM-L6-v2)
 *   --batch-size  <n>       Texts per request   (default: 32)
 *   --dtype       <type>    Quantization dtype  (default: none)
 *   --sample-size <n>       Number of texts to embed (default: 200, max: 1000)
 *   --skip-socket           Skip socket server runner
 *   --skip-grpc             Skip gRPC server runner
 *   --skip-baseline         Skip process/thread baseline runners
 */

import net from 'net';
import os from 'os';
import { spawn } from 'child_process';
import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import grpc from '@grpc/grpc-js';
import protoLoader from '@grpc/proto-loader';
import { Embedder } from '../src/embedder.js';
import { getCacheDir } from '../src/model-cache.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const SRC_DIR   = join(__dirname, '..', 'src');
const PROTO_PATH = join(SRC_DIR, 'proto', 'embedder.proto');

// ── Args ──────────────────────────────────────────────────────────────────────

const argv = process.argv.slice(2);
const opts = {
  model:        'Xenova/all-MiniLM-L6-v2',
  batchSize:    32,
  dtype:        undefined,
  sampleSize:   200,
  skipSocket:   false,
  skipGrpc:     false,
  skipBaseline: false,
};

for (let i = 0; i < argv.length; i++) {
  const a = argv[i];
  if      (a === '--model')        opts.model        = argv[++i];
  else if (a === '--batch-size')   opts.batchSize    = parseInt(argv[++i], 10);
  else if (a === '--dtype')        opts.dtype        = argv[++i];
  else if (a === '--sample-size')  opts.sampleSize   = parseInt(argv[++i], 10);
  else if (a === '--skip-socket')  opts.skipSocket   = true;
  else if (a === '--skip-grpc')    opts.skipGrpc     = true;
  else if (a === '--skip-baseline') opts.skipBaseline = true;
}

// ── Texts ─────────────────────────────────────────────────────────────────────

const allTexts = readFileSync(join(__dirname, 'texts-1000.txt'), 'utf8')
  .split('\n').map(l => l.trim()).filter(Boolean);
const texts = allTexts.slice(0, Math.min(opts.sampleSize, allTexts.length));

// ── Helpers ───────────────────────────────────────────────────────────────────

function chunk(arr, size) {
  const out = [];
  for (let i = 0; i < arr.length; i += size) out.push(arr.slice(i, i + size));
  return out;
}

function fmt(ms)  { return (ms / 1000).toFixed(3) + 's'; }
function tps(ms)  { return (texts.length / (ms / 1000)).toFixed(1); }

/**
 * Spawn a server subprocess and wait for {"type":"ready"} on its stdout.
 * Returns { child, startupMs }.
 */
function spawnServer(script, extraArgs = []) {
  return new Promise((resolve, reject) => {
    const t0    = performance.now();
    const child = spawn(process.execPath, [
      join(SRC_DIR, script),
      '--model', opts.model,
      ...(opts.dtype ? ['--dtype', opts.dtype] : []),
      ...extraArgs,
    ], { stdio: ['ignore', 'pipe', 'inherit'] });

    let buf = '';
    child.stdout.on('data', chunk => {
      buf += chunk.toString();
      if (buf.includes('{"type":"ready"}')) {
        const startupMs = performance.now() - t0;
        resolve({ child, startupMs });
      }
    });
    child.on('error', reject);
    child.on('exit', code => {
      if (code !== 0) reject(new Error(`Server exited with code ${code}`));
    });
  });
}

function killServer(child) {
  return new Promise(resolve => {
    child.once('exit', resolve);
    child.kill('SIGTERM');
  });
}

// ── Socket client ─────────────────────────────────────────────────────────────

/**
 * Embed all texts via the socket server using a single connection.
 * Sends all batches up front so the server queue stays full.
 */
function embedViaSocket(socketPath, batchSize) {
  return new Promise((resolve, reject) => {
    const socket  = net.createConnection(socketPath);
    const batches = chunk(texts, batchSize);
    const results = new Map();
    let received  = 0;
    let buf       = '';

    socket.on('connect', () => {
      for (let id = 0; id < batches.length; id++) {
        socket.write(
          JSON.stringify({ type: 'task', id, texts: batches[id] }) + '\n',
        );
      }
    });

    socket.on('data', data => {
      buf += data.toString();
      const lines = buf.split('\n');
      buf = lines.pop();

      for (const line of lines) {
        if (!line.trim()) continue;
        let msg;
        try { msg = JSON.parse(line); } catch { continue; }

        if (msg.type === 'result') {
          results.set(msg.id, msg.embeddings);
          if (++received === batches.length) {
            socket.destroy();
            const ordered = [];
            for (let i = 0; i < batches.length; i++) ordered.push(...results.get(i));
            resolve(ordered);
          }
        } else if (msg.type === 'error') {
          socket.destroy();
          reject(new Error(msg.error));
        }
      }
    });

    socket.on('error', reject);
  });
}

// ── gRPC client ───────────────────────────────────────────────────────────────

function makeGrpcStub(address) {
  const pkgDef = protoLoader.loadSync(PROTO_PATH, {
    longs: String, enums: String, defaults: true, oneofs: true,
  });
  const { embedeer: { EmbedderService } } = grpc.loadPackageDefinition(pkgDef);
  return new EmbedderService(address, grpc.credentials.createInsecure());
}

function grpcEmbed(stub, id, batch) {
  return new Promise((resolve, reject) => {
    stub.embed({ id, texts: batch }, (err, response) => {
      if (err) reject(err);
      else resolve(response.embeddings.map(v => v.values));
    });
  });
}

async function embedViaGrpc(stub, batchSize) {
  const batches = chunk(texts, batchSize);
  // Send all batches concurrently — gRPC multiplexes over HTTP/2.
  const results = await Promise.all(
    batches.map((batch, id) => grpcEmbed(stub, id, batch)),
  );
  return results.flat();
}

// ── Table printer ─────────────────────────────────────────────────────────────

function printTable(results) {
  const COL = { label: 52, startup: 12, time: 10, tps: 12, dims: 6 };
  const line = '─'.repeat(Object.values(COL).reduce((a, b) => a + b) + 14);
  const pad  = (s, n) => String(s).padEnd(n);
  const lpad = (s, n) => String(s).padStart(n);

  console.log('');
  console.log(line);
  console.log(
    pad('Runner', COL.label)            + ' │ ' +
    lpad('Startup', COL.startup)        + ' │ ' +
    lpad('Embed time', COL.time)        + ' │ ' +
    lpad('texts/s', COL.tps)            + ' │ ' +
    lpad('dims', COL.dims),
  );
  console.log(line);

  const sorted  = [...results].sort((a, b) => a.embedMs - b.embedMs);
  const fastest = sorted[0].embedMs;

  for (const r of sorted) {
    const ratio    = r.embedMs / fastest;
    const ratioStr = ratio < 1.005 ? '(fastest)' : `(${ratio.toFixed(2)}x slower)`;
    console.log(
      pad(r.label, COL.label)                         + ' │ ' +
      lpad(r.startupMs != null ? fmt(r.startupMs) : '—', COL.startup) + ' │ ' +
      lpad(fmt(r.embedMs), COL.time)                  + ' │ ' +
      lpad(tps(r.embedMs), COL.tps)                   + ' │ ' +
      lpad(r.dims ?? '?', COL.dims)                   +
      '   ' + ratioStr,
    );
  }

  console.log(line);
  console.log(`Texts: ${texts.length}   Model: ${opts.model}   Batch: ${opts.batchSize}${opts.dtype ? `   dtype: ${opts.dtype}` : ''}`);
  console.log('');
}

// ── Runners ───────────────────────────────────────────────────────────────────

async function runSocket() {
  const socketPath = process.platform === 'win32'
    ? `\\\\.\\pipe\\embedeer-bench-${Date.now()}`
    : join(os.tmpdir(), `embedeer-bench-${Date.now()}.sock`);

  process.stderr.write('Starting socket server…\n');
  const { child, startupMs } = await spawnServer('socket-model-server.js', [
    '--socket', socketPath,
  ]);
  process.stderr.write(`  server ready in ${fmt(startupMs)}\n`);

  process.stderr.write('Running socket embed…\n');
  const t0 = performance.now();
  const embeddings = await embedViaSocket(socketPath, opts.batchSize);
  const embedMs = performance.now() - t0;
  process.stderr.write('  done.\n');

  await killServer(child);

  return {
    label:     `socket server (single connection, batch=${opts.batchSize})`,
    startupMs,
    embedMs,
    dims:      embeddings[0]?.length,
  };
}

async function runGrpc() {
  const port    = 59051;   // Use a non-standard port to avoid conflicts
  const address = `localhost:${port}`;

  process.stderr.write('Starting gRPC server…\n');
  const { child, startupMs } = await spawnServer('grpc-model-server.js', [
    '--address', address,
  ]);
  process.stderr.write(`  server ready in ${fmt(startupMs)}\n`);

  const stub = makeGrpcStub(address);
  await new Promise((resolve, reject) =>
    stub.waitForReady(Date.now() + 5_000, err => err ? reject(err) : resolve()),
  );

  process.stderr.write('Running gRPC embed…\n');
  const t0 = performance.now();
  const embeddings = await embedViaGrpc(stub, opts.batchSize);
  const embedMs = performance.now() - t0;
  process.stderr.write('  done.\n');

  grpc.closeClient(stub);
  await killServer(child);

  return {
    label:     `gRPC server (concurrent batches, batch=${opts.batchSize})`,
    startupMs,
    embedMs,
    dims:      embeddings[0]?.length,
  };
}

async function runBaseline(mode) {
  process.stderr.write(`Running baseline embedeer (${mode})…\n`);
  const t0start = performance.now();
  const embedder = await Embedder.create(opts.model, {
    mode,
    concurrency: 2,
    batchSize:   opts.batchSize,
    dtype:       opts.dtype,
    cacheDir:    getCacheDir(),
    applyPerfProfile: false,
  });
  const startupMs = performance.now() - t0start;

  const t0 = performance.now();
  const embeddings = await embedder.embed(texts);
  const embedMs = performance.now() - t0;
  await embedder.destroy();
  process.stderr.write('  done.\n');

  return {
    label:     `embedeer ${mode} mode (concurrency=2, batch=${opts.batchSize})`,
    startupMs,
    embedMs,
    dims:      embeddings[0]?.length,
  };
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main() {
  console.log(`\nembedeer server benchmark`);
  console.log(`Model: ${opts.model}   Texts: ${texts.length}   Batch size: ${opts.batchSize}\n`);

  // Pre-load model so startup times reflect only process/server init, not download.
  process.stderr.write('Pre-loading model into cache…\n');
  await Embedder.loadModel(opts.model, { dtype: opts.dtype, cacheDir: getCacheDir() });
  process.stderr.write('  model cached.\n\n');

  const results = [];

  if (!opts.skipSocket) {
    try { results.push(await runSocket()); }
    catch (err) { console.error(`  socket runner failed: ${err.message}`); }
  }

  if (!opts.skipGrpc) {
    try { results.push(await runGrpc()); }
    catch (err) { console.error(`  gRPC runner failed: ${err.message}`); }
  }

  if (!opts.skipBaseline) {
    try { results.push(await runBaseline('process')); }
    catch (err) { console.error(`  process baseline failed: ${err.message}`); }

    try { results.push(await runBaseline('thread')); }
    catch (err) { console.error(`  thread baseline failed: ${err.message}`); }
  }

  if (results.length === 0) {
    console.error('No results — all runners failed or were skipped.');
    process.exit(1);
  }

  printTable(results);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
