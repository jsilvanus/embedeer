/**
 * Socket model server — loads a Hugging Face feature-extraction pipeline once
 * and serves embedding requests over a Unix socket (or Windows named pipe)
 * from any number of connected clients.
 *
 * Intended to run as a long-lived daemon shared across multiple OS processes.
 * Each client (SocketWorker) holds a persistent connection; requests from all
 * clients are serialised through an internal FIFO queue because transformers.js
 * is not concurrency-safe.
 *
 * Protocol: newline-delimited JSON (NDJSON).
 *   client → server:  {"type":"task","id":0,"texts":["hello","world"]}\n
 *   server → client:  {"type":"result","id":0,"embeddings":[[...],[...]]}\n
 *                     {"type":"error","id":0,"error":"..."}\n
 *
 * CLI arguments:
 *   --model       <string>   Hugging Face model identifier (required)
 *   --socket      <string>   Unix socket path (default: auto-generated in os.tmpdir())
 *   --pooling     <string>   Pooling strategy: mean|cls|none  (default: mean)
 *   --no-normalize           Disable L2 normalisation (default: enabled)
 *   --dtype       <string>   Quantization: fp32|fp16|q8|q4|q4f16|auto
 *   --device      <string>   Compute device: cpu|gpu|auto  (default: cpu)
 *   --provider    <string>   ONNX provider override: cuda|dml
 *   --token       <string>   Hugging Face API token (overrides HF_TOKEN env var)
 *   --cache-dir   <string>   Model cache directory (default: ~/.embedeer/models)
 *   --idle-timeout <number>  Offload model after N ms of inactivity (default: never)
 */

import net from 'net';
import fs from 'fs';
import os from 'os';
import { join } from 'path';
import { parseArgs } from 'util';
import { pipeline, env } from '@huggingface/transformers';
import { buildPipelineOptions } from './model-cache.js';
import { resolveProvider } from './provider-loader.js';

// ── Argument parsing ──────────────────────────────────────────────────────────

const { values: args } = parseArgs({
  args: process.argv.slice(2),
  options: {
    model:          { type: 'string' },
    socket:         { type: 'string' },
    pooling:        { type: 'string',  default: 'mean' },
    normalize:      { type: 'boolean', default: true },
    dtype:          { type: 'string' },
    device:         { type: 'string',  default: 'cpu' },
    provider:       { type: 'string' },
    token:          { type: 'string' },
    'cache-dir':    { type: 'string' },
    'idle-timeout': { type: 'string' },   // ms as string; parsed to number below
  },
});

if (!args.model) {
  console.error('socket-model-server: --model is required');
  process.exit(1);
}

const MODEL_NAME  = args.model;
const SOCKET_PATH = args.socket ?? join(os.tmpdir(), `embedeer-${MODEL_NAME.replace(/\//g, '-')}.sock`);
const POOLING     = args.pooling;
const NORMALIZE   = args.normalize;
const IDLE_MS     = args['idle-timeout'] ? Number(args['idle-timeout']) : null;

// ── Model state ───────────────────────────────────────────────────────────────

/** @type {import('@huggingface/transformers').FeatureExtractionPipeline|null} */
let extractor     = null;
let pipelineOpts  = null;
let lastRequestTime = Date.now();

async function loadModel() {
  console.error(`socket-model-server: loading model ${MODEL_NAME}`);
  extractor = await pipeline('feature-extraction', MODEL_NAME, pipelineOpts);
  console.error(`socket-model-server: model ready`);
}

async function ensureLoaded() {
  if (extractor) return;
  await loadModel();
}

async function offloadModel() {
  if (!extractor) return;
  console.error('socket-model-server: idle timeout — offloading model');
  await extractor.dispose();   // Pipeline → model.dispose() → session.release()
  extractor = null;
}

// ── Task queue ────────────────────────────────────────────────────────────────

/**
 * @typedef {{ msg: {type:string, id:number, texts:string[]}, socket: net.Socket }} QueueEntry
 * @type {QueueEntry[]}
 */
const queue = [];
let busy = false;

async function processNext() {
  if (busy || queue.length === 0) return;
  busy = true;

  const { msg, socket } = queue.shift();

  if (socket.destroyed) {
    busy = false;
    return processNext();
  }

  try {
    await ensureLoaded();
    lastRequestTime = Date.now();

    const output = await extractor(msg.texts, { pooling: POOLING, normalize: NORMALIZE });
    if (!socket.destroyed) {
      socket.write(
        JSON.stringify({ type: 'result', id: msg.id, embeddings: output.tolist() }) + '\n',
      );
    }
  } catch (err) {
    if (!socket.destroyed) {
      socket.write(
        JSON.stringify({ type: 'error', id: msg.id, error: err.message }) + '\n',
      );
    }
  }

  busy = false;
  processNext();
}

// ── Connection handler ────────────────────────────────────────────────────────

/**
 * @param {net.Socket} socket
 */
function handleConnection(socket) {
  let buf = '';

  socket.on('data', chunk => {
    buf += chunk.toString();
    const lines = buf.split('\n');
    buf = lines.pop();                  // keep any incomplete trailing fragment

    for (const line of lines) {
      if (!line.trim()) continue;
      let msg;
      try { msg = JSON.parse(line); } catch { continue; }

      if (msg.type === 'task' && Array.isArray(msg.texts)) {
        queue.push({ msg, socket });
        processNext();
      }
    }
  });

  // Drop all queued tasks for this socket to avoid writing to a closed connection.
  socket.on('close', () => {
    for (let i = queue.length - 1; i >= 0; i--) {
      if (queue[i].socket === socket) queue.splice(i, 1);
    }
  });

  socket.on('error', () => socket.destroy());
}

// ── Idle timeout ──────────────────────────────────────────────────────────────

if (IDLE_MS !== null) {
  const checkInterval = Math.min(IDLE_MS, 60_000);
  setInterval(async () => {
    if (extractor && !busy && Date.now() - lastRequestTime > IDLE_MS) {
      await offloadModel();
    }
  }, checkInterval).unref();   // unref so the timer doesn't prevent clean exit
}

// ── Startup ───────────────────────────────────────────────────────────────────

async function main() {
  if (args.token)       process.env.HF_TOKEN = args.token;
  if (args['cache-dir']) env.cacheDir = args['cache-dir'];

  const deviceStr = await resolveProvider(args.device, args.provider);
  pipelineOpts = {
    ...buildPipelineOptions(args.dtype),
    ...(deviceStr ? { device: deviceStr } : {}),
  };

  await loadModel();

  // Remove stale socket file left over from a previous crash (Unix only).
  if (process.platform !== 'win32') {
    try { fs.unlinkSync(SOCKET_PATH); } catch { /* not present — that's fine */ }
  }

  const server = net.createServer(handleConnection);

  server.listen(SOCKET_PATH, () => {
    // Signal to WorkerPool (or any parent process) that the server is ready.
    process.stdout.write('{"type":"ready"}\n');
    console.error(`socket-model-server: listening on ${SOCKET_PATH}`);
  });

  server.on('error', err => {
    console.error(`socket-model-server: server error — ${err.message}`);
    process.exit(1);
  });

  // ── Graceful shutdown ───────────────────────────────────────────────────────
  const shutdown = async () => {
    console.error('socket-model-server: shutting down');
    server.close(async () => {
      await offloadModel();
      process.exit(0);
    });
    // Stop accepting new connections immediately; existing ones drain naturally.
  };

  process.on('SIGTERM', shutdown);
  process.on('SIGINT',  shutdown);
}

main().catch(err => {
  console.error(`socket-model-server: fatal — ${err.message}`);
  process.exit(1);
});
