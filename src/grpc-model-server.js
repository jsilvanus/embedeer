/**
 * gRPC model server — loads a Hugging Face feature-extraction pipeline once
 * and serves embedding requests via gRPC (HTTP/2 + Protocol Buffers).
 *
 * Exposes two RPCs defined in src/proto/embedder.proto:
 *   Embed       — unary: send a batch, receive all vectors at once.
 *   EmbedStream — server-streaming: receive one vector per text as computed.
 *
 * Unlike the socket server, gRPC accepts concurrent RPCs. Because
 * transformers.js is not concurrency-safe, each handler awaits ensureLoaded()
 * and then runs inference. Concurrent requests therefore queue implicitly on
 * the Node.js event loop (one inference runs at a time).
 *
 * The idle-timeout option offloads the model (Pipeline.dispose() →
 * model.dispose() → session.release()) after a configurable period of
 * inactivity and reloads it transparently on the next incoming RPC.
 *
 * CLI arguments:
 *   --model        <string>   Hugging Face model identifier (required)
 *   --address      <string>   Bind address (default: localhost:50051)
 *   --pooling      <string>   Pooling strategy: mean|cls|none  (default: mean)
 *   --no-normalize            Disable L2 normalisation (default: enabled)
 *   --dtype        <string>   Quantization: fp32|fp16|q8|q4|q4f16|auto
 *   --device       <string>   Compute device: cpu|gpu|auto  (default: cpu)
 *   --provider     <string>   ONNX provider override: cuda|dml
 *   --token        <string>   Hugging Face API token (overrides HF_TOKEN env var)
 *   --cache-dir    <string>   Model cache directory (default: ~/.embedeer/models)
 *   --idle-timeout <number>   Offload model after N ms of inactivity (default: never)
 */

import grpc from '@grpc/grpc-js';
import protoLoader from '@grpc/proto-loader';
import { parseArgs } from 'util';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { pipeline, env } from '@huggingface/transformers';
import { buildPipelineOptions } from './model-cache.js';
import { resolveProvider } from './provider-loader.js';

const PROTO_PATH = join(dirname(fileURLToPath(import.meta.url)), 'proto', 'embedder.proto');

// ── Argument parsing ──────────────────────────────────────────────────────────

const { values: args } = parseArgs({
  args: process.argv.slice(2),
  options: {
    model:          { type: 'string' },
    address:        { type: 'string',  default: 'localhost:50051' },
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
  console.error('grpc-model-server: --model is required');
  process.exit(1);
}

const MODEL_NAME = args.model;
const ADDRESS    = args.address;
const POOLING    = args.pooling;
const NORMALIZE  = args.normalize;
const IDLE_MS    = args['idle-timeout'] ? Number(args['idle-timeout']) : null;

// ── Model state ───────────────────────────────────────────────────────────────

/** @type {import('@huggingface/transformers').FeatureExtractionPipeline|null} */
let extractor     = null;
let pipelineOpts  = null;
let lastRequestTime = Date.now();

/**
 * If a reload is in progress, subsequent callers await the same promise rather
 * than spawning duplicate pipeline() calls.
 * @type {Promise<void>|null}
 */
let loadingPromise = null;

async function ensureLoaded() {
  if (extractor) return;

  if (!loadingPromise) {
    loadingPromise = (async () => {
      console.error(`grpc-model-server: loading model ${MODEL_NAME}`);
      extractor = await pipeline('feature-extraction', MODEL_NAME, pipelineOpts);
      console.error(`grpc-model-server: model ready`);
      loadingPromise = null;
    })();
  }

  await loadingPromise;
}

async function offloadModel() {
  if (!extractor) return;
  console.error('grpc-model-server: idle timeout — offloading model');
  await extractor.dispose();   // Pipeline → model.dispose() → session.release()
  extractor = null;
}

// ── RPC handlers ──────────────────────────────────────────────────────────────

/**
 * Unary RPC: embed a batch of texts, return all vectors at once.
 * @param {grpc.ServerUnaryCall<{texts: string[], id: number}, object>} call
 * @param {grpc.sendUnaryData<object>} callback
 */
async function handleEmbed(call, callback) {
  try {
    await ensureLoaded();
    lastRequestTime = Date.now();

    const { texts, id } = call.request;
    const output = await extractor(texts, { pooling: POOLING, normalize: NORMALIZE });

    callback(null, {
      id,
      embeddings: output.tolist().map(values => ({ values })),
    });
  } catch (err) {
    callback({ code: grpc.status.INTERNAL, message: err.message });
  }
}

/**
 * Server-streaming RPC: stream one EmbedChunk per text as it is computed.
 * Useful for very large batches where the caller wants progressive results.
 * @param {grpc.ServerWritableStream<{texts: string[], id: number}, object>} call
 */
async function handleEmbedStream(call) {
  try {
    await ensureLoaded();
    lastRequestTime = Date.now();

    const { texts, id } = call.request;

    for (let i = 0; i < texts.length; i++) {
      // Check for client cancellation between iterations.
      if (call.cancelled) break;

      const output = await extractor([texts[i]], { pooling: POOLING, normalize: NORMALIZE });
      call.write({ id, index: i, values: output.tolist()[0] });

      // Update lastRequestTime as each text completes so a long streaming
      // batch doesn't trigger an idle offload mid-stream.
      lastRequestTime = Date.now();
    }

    call.end();
  } catch (err) {
    call.destroy(new Error(err.message));
  }
}

// ── Idle timeout ──────────────────────────────────────────────────────────────

if (IDLE_MS !== null) {
  const checkInterval = Math.min(IDLE_MS, 60_000);
  setInterval(async () => {
    if (extractor && Date.now() - lastRequestTime > IDLE_MS) {
      await offloadModel();
    }
  }, checkInterval).unref();
}

// ── Startup ───────────────────────────────────────────────────────────────────

async function main() {
  if (args.token)        process.env.HF_TOKEN = args.token;
  if (args['cache-dir']) env.cacheDir = args['cache-dir'];

  const deviceStr = await resolveProvider(args.device, args.provider);
  pipelineOpts = {
    ...buildPipelineOptions(args.dtype),
    ...(deviceStr ? { device: deviceStr } : {}),
  };

  // Load model before binding so the server is ready the moment it signals.
  await ensureLoaded();

  // Load proto definition (dynamic — no protoc required).
  const pkgDef = protoLoader.loadSync(PROTO_PATH, {
    longs:    String,
    enums:    String,
    defaults: true,
    oneofs:   true,
  });
  const { embedeer: { EmbedderService } } = grpc.loadPackageDefinition(pkgDef);

  const server = new grpc.Server();
  server.addService(EmbedderService.service, {
    embed:       handleEmbed,
    embedStream: handleEmbedStream,
  });

  server.bindAsync(ADDRESS, grpc.ServerCredentials.createInsecure(), (err, port) => {
    if (err) {
      console.error(`grpc-model-server: bind failed — ${err.message}`);
      process.exit(1);
    }
    server.start();
    // Signal to WorkerPool (or any parent process) that the server is ready.
    process.stdout.write('{"type":"ready"}\n');
    console.error(`grpc-model-server: listening on ${ADDRESS} (port ${port})`);
  });

  // ── Graceful shutdown ───────────────────────────────────────────────────────
  const shutdown = () => {
    console.error('grpc-model-server: shutting down');
    server.tryShutdown(async () => {
      await offloadModel();
      process.exit(0);
    });
  };

  process.on('SIGTERM', shutdown);
  process.on('SIGINT',  shutdown);
}

main().catch(err => {
  console.error(`grpc-model-server: fatal — ${err.message}`);
  process.exit(1);
});
