# gRPC Model Server Plan

## Overview

A new optional `mode: 'grpc'` that exposes the embedding pipeline as a gRPC
service. Like socket mode, one server process loads the model once; client
workers call it via typed RPCs over HTTP/2. Unlike socket mode, gRPC adds a
formal service contract (`.proto`), structured binary serialization (Protocol
Buffers), and first-class server-side streaming — enabling both local
optimization and remote/cross-language deployment.

The multi-server variant mirrors socket mode: multiple servers (e.g. two GPU +
one CPU) are managed by the same `WorkerPool`, with the idle-worker queue
acting as a natural load balancer. Additionally, gRPC's built-in `round_robin`
policy provides a zero-overhead equal-weight alternative for homogeneous
server clusters.

This is purely additive — no existing files change their public API.

---

## Architecture

### Single server

```
Embedder
  └─ WorkerPool (mode: 'grpc', poolSize: N)
        ├─ GrpcWorker 1  ──┐
        ├─ GrpcWorker 2  ──┤── HTTP/2 + protobuf ──► GrpcModelServer
        └─ GrpcWorker N  ──┘  (multiplexed on one channel)  (one model)
```

### Multi-server (WorkerPool-level LB)

```
Embedder
  └─ WorkerPool (mode: 'grpc', servers: [...])
        ├─ GrpcWorker 1  ──┐
        ├─ GrpcWorker 2  ──┤── localhost:50051 ──► GrpcModelServer (CUDA)
        ├─ GrpcWorker 3  ──┘
        ├─ GrpcWorker 4  ──┐
        ├─ GrpcWorker 5  ──┤── localhost:50052 ──► GrpcModelServer (CUDA)
        ├─ GrpcWorker 6  ──┘
        └─ GrpcWorker 7  ──── localhost:50053 ──► GrpcModelServer (CPU)

        All 7 workers share one idleWorkers queue.
        GPU workers finish ~5× faster → naturally receive ~5× more tasks.
```

### Multi-server (gRPC built-in round-robin)

```
Embedder
  └─ WorkerPool (mode: 'grpc', grpcLoadBalancing: 'round_robin')
        ├─ GrpcWorker 1  ──┐
        ├─ GrpcWorker 2  ──┤── single channel ──► round_robin ──► :50051
        └─ GrpcWorker N  ──┘   (gRPC distributes)              ──► :50052
```

N `GrpcWorker` instances each hold a gRPC channel. HTTP/2 multiplexes all N
concurrent RPCs over a small number of TCP connections automatically.

---

## Dependencies

```json
{
  "dependencies": {
    "@grpc/grpc-js":     "^1.12.0",
    "@grpc/proto-loader": "^0.7.15"
  }
}
```

`@grpc/proto-loader` loads `.proto` files dynamically at runtime — no
`protoc` code-generation step, no generated JS to commit.

---

## Proto Definition

### `src/proto/embedder.proto`

```protobuf
syntax = "proto3";

package embedeer;

service EmbedderService {
  // Unary: embed a batch of texts, return all vectors at once.
  rpc Embed (EmbedRequest) returns (EmbedResponse);

  // Server-streaming: stream one vector per text as it is computed.
  // Useful for very large batches and CLI progress display.
  rpc EmbedStream (EmbedRequest) returns (stream EmbedChunk);
}

message EmbedRequest {
  repeated string texts = 1;
  int32           id    = 2;   // echoed back for correlation
}

message EmbedResponse {
  int32             id         = 1;
  repeated FloatVec embeddings = 2;
}

// One embedding vector streamed per text.
message EmbedChunk {
  int32          id     = 1;
  int32          index  = 2;   // position in the original texts array
  repeated float values = 3;
}

message FloatVec {
  repeated float values = 1;
}
```

---

## New Files

### `src/grpc-model-server.js`

The server entry point. Same model-loading pattern as `worker.js`; serves
requests via gRPC instead of IPC.

**Responsibilities:**
1. Parse config from `process.argv`: `modelName`, `pooling`, `normalize`,
   `token`, `dtype`, `cacheDir`, `device`, `provider`, `grpcAddress`.
2. Load the model via `@huggingface/transformers` pipeline.
3. Write `{"type":"ready"}\n` to stdout once the model is loaded and the
   server is bound — `WorkerPool` waits for this before creating workers.
4. Register the `EmbedderService` implementation and start listening.

**`Embed` handler (unary):**
```js
handleEmbed(call, callback) {
  const { texts, id } = call.request;
  extractor(texts, { pooling, normalize })
    .then(output => callback(null, {
      id,
      embeddings: output.tolist().map(v => ({ values: v })),
    }))
    .catch(err => callback({ code: grpc.status.INTERNAL, message: err.message }));
}
```

**`EmbedStream` handler (server-streaming):**
```js
handleEmbedStream(call) {
  const { texts, id } = call.request;
  // Process texts in sub-batches; call.write({ id, index, values })
  // for each completed vector; call.end() when all are done.
}
```

Protobuf `repeated float` maps directly to a JS plain array — no conversion
beyond the `.tolist()` call already used in `worker.js`.

**Graceful shutdown:** On `SIGTERM`/`SIGINT` call `server.tryShutdown(cb)`,
which drains in-flight RPCs before closing.

---

### `src/grpc-worker.js`

Drop-in replacement for `ChildProcessWorker`. Implements
`postMessage / terminate / EventEmitter('message'|'error'|'exit')`.

**Constructor:**
```js
new GrpcWorker(scriptPath, { workerData })
```
- `scriptPath` is ignored — the server is managed by `WorkerPool`.
- `workerData.grpcAddress` is the server address.
- `workerData.grpcLoadBalancing` is an optional policy name (e.g.
  `'round_robin'`).

**Lifecycle:**
```js
// 1. Load proto and create stub.
const pkgDef = protoLoader.loadSync(PROTO_PATH, { longs: String, enums: String, defaults: true });
const { embedeer: { EmbedderService } } = grpc.loadPackageDefinition(pkgDef);
const channelOpts = workerData.grpcLoadBalancing
  ? { 'grpc.lb_policy_name': workerData.grpcLoadBalancing }
  : {};
this._stub = new EmbedderService(address, grpc.credentials.createInsecure(), channelOpts);

// 2. Wait for server readiness, then signal WorkerPool.
this._stub.waitForReady(Date.now() + 10_000, err => {
  if (err) this.emit('error', err);
  else     this.emit('message', { type: 'ready' });
});

// 3. postMessage routes to the appropriate RPC.
postMessage({ type: 'task', id, texts }) {
  this._stub.embed({ texts, id }, (err, response) => {
    if (err) this.emit('message', { type: 'error', id, error: err.message });
    else     this.emit('message', {
      type: 'result', id,
      embeddings: response.embeddings.map(v => v.values),
    });
  });
}

// 4. terminate() closes the channel.
terminate() { grpc.closeClient(this._stub); return Promise.resolve(); }
```

**Error handling:**
- `waitForReady` timeout → emit `'error'`.
- RPC status `UNAVAILABLE` → emit `'error'`; `WorkerPool` treats it as a
  worker crash and rejects the in-flight task.

---

## Changes to Existing Files

### `src/worker-pool.js`

**Constructor — `mode: 'grpc'` branch:**
```js
} else if (mode === 'grpc') {
  this._WorkerClass = GrpcWorker;
  this._workerScript = null;
  // Normalise single-server shorthand to the internal servers array.
  // dtype/pooling/normalize are top-level — they must be uniform across all
  // servers so every server produces comparable embedding vectors.
  this._servers = options.servers ?? [{
    address:  options.grpcAddress ?? 'localhost:50051',
    workers:  options.concurrency ?? poolSize,
    device:   options.device,
    provider: options.provider,
  }];
  this._grpcLoadBalancing = options.grpcLoadBalancing ?? null;
  this._autoStartServer   = options.autoStartServer ?? true;
}
```

**`initialize()` — `_startServers()`:**
```js
async _startServers() {
  // For each entry in this._servers:
  //   1. Attempt a quick gRPC health probe (waitForReady with 200 ms deadline).
  //   2. If it fails (and autoStartServer is true), spawn grpc-model-server.js
  //      with the entry's device/provider config as argv flags, plus the
  //      top-level dtype/pooling/normalize (uniform across all servers).
  //   3. Wait for {"type":"ready"} on the child's stdout.
  //   4. Push the child into this._serverProcesses[].
  // All spawns run concurrently (Promise.all) — multiple GPU servers load
  // their models in parallel.
}
```

**`_createWorker()` — distribute workers across servers:**
```js
// Iterate this._servers; for each entry create entry.workers GrpcWorker
// instances, each with workerData = {
//   grpcAddress: entry.address,
//   grpcLoadBalancing: this._grpcLoadBalancing,
// }.
// All workers land in the shared idleWorkers pool.
```

**`destroy()`:**
```js
// After terminating all workers:
await Promise.all(this._serverProcesses.map(p => killAndWait(p)));
```

### `src/embedder.js`

Add to JSDoc and forward to `WorkerPool`. No logic changes.

```js
// @param {string}   [options.grpcAddress]       gRPC address (single-server shorthand)
// @param {object[]} [options.servers]            Multi-server list; each entry has
//                                                address, workers, device, provider, dtype
// @param {string}   [options.grpcLoadBalancing]  gRPC LB policy ('round_robin')
// @param {boolean}  [options.autoStartServer=true]  Spawn servers if not reachable
```

---

## WorkerPool Interface — No Changes

`GrpcWorker` emits the same events and accepts the same method calls as
existing workers. `WorkerPool._dispatch()` and `_removeWorker()` work without
modification.

---

## Multi-Server Load Balancing

### WorkerPool-level (recommended for heterogeneous servers)

The `servers` array assigns each server a `workers` count. All workers land in
the shared `idleWorkers` queue. Faster servers (GPU, ~100 ms/batch) return
their workers to the queue ~5× more often than slower ones (CPU, ~500 ms),
so they naturally receive ~5× more tasks — no extra routing logic required.

Use the `workers` count as the weight:

```
GPU server:  workers: 6  (high weight — fast, receives most work)
CPU server:  workers: 2  (low weight — acts as overflow)
```

### gRPC built-in `round_robin` (equal-weight homogeneous clusters)

For a cluster of identical GPU servers where you want equal distribution,
gRPC's built-in `round_robin` policy handles it with zero application-level
bookkeeping. Pass all addresses in a single URI; set `grpcLoadBalancing`:

```js
// All servers receive equal share; no per-server workers config needed.
grpcAddress: 'ipv4:///localhost:50051,localhost:50052,localhost:50053',
grpcLoadBalancing: 'round_robin',
autoStartServer: false,   // servers must be started separately for this mode
```

### When a dedicated LB process is worthwhile

The WorkerPool-level approach covers the common cases. A standalone LB
(e.g. Envoy, nginx stream, custom Node proxy) only adds value when you need:
- Dynamic server registration / deregistration at runtime
- Health checking with automatic failover
- Routing by request content (long texts → GPU, short texts → CPU)
- Sharing one server pool across multiple `Embedder` instances in separate OS
  processes

---

## Usage: All Modes

### `'process'` — default, full isolation

Each worker is an OS child process with its own model copy. A crash rejects
only its in-flight task; the pool keeps running.

```js
import { Embedder } from 'embedeer';

// Defaults: mode='process', concurrency=numCores, batchSize=32
const embedder = await Embedder.create('Xenova/all-MiniLM-L6-v2');
const vectors = await embedder.embed(['Hello world', 'Foo bar']);
await embedder.destroy();

// Explicit options
const embedder = await Embedder.create('Xenova/all-MiniLM-L6-v2', {
  mode: 'process',
  concurrency: 4,
  batchSize: 32,
  dtype: 'q8',
  pooling: 'mean',
  normalize: true,
});
```

---

### `'thread'` — lower memory, same process

Workers are `worker_threads` threads. Lower startup time and memory vs
`'process'`; a thread crash can propagate to the parent.

```js
const embedder = await Embedder.create('Xenova/all-MiniLM-L6-v2', {
  mode: 'thread',
  concurrency: 4,
  batchSize: 32,
});
```

---

### `'grpc'` — single server, auto-start

`WorkerPool` spawns one `GrpcModelServer`, waits for it to load the model,
then opens `concurrency` gRPC channels to it.

```js
const embedder = await Embedder.create('Xenova/all-MiniLM-L6-v2', {
  mode: 'grpc',
  concurrency: 4,
  grpcAddress: 'localhost:50051',   // omit to use default port
  autoStartServer: true,            // default
});
const vectors = await embedder.embed(['Hello world', 'Foo bar']);
await embedder.destroy();           // shuts down server + closes channels
```

---

### `'grpc'` — single server, pre-running

Connect to a server started externally — useful for a shared inference daemon
or for connecting to a Python-based model server.

```js
// Terminal 1:
// node src/grpc-model-server.js \
//   --model Xenova/all-MiniLM-L6-v2 --address localhost:50051

// Terminal 2:
const embedder = await Embedder.create('Xenova/all-MiniLM-L6-v2', {
  mode: 'grpc',
  grpcAddress: 'localhost:50051',
  autoStartServer: false,
  concurrency: 8,
});
```

---

### `'grpc'` — remote server

The server runs on another machine; the API is identical to local.

```js
const embedder = await Embedder.create('Xenova/all-MiniLM-L6-v2', {
  mode: 'grpc',
  grpcAddress: '10.0.1.42:50051',
  autoStartServer: false,
  concurrency: 8,
});
```

---

### `'grpc'` — multiple servers, auto-start (heterogeneous)

`WorkerPool` starts each server concurrently. Workers are distributed by
`workers` count and all share one `idleWorkers` queue.

```js
const embedder = await Embedder.create('Xenova/all-MiniLM-L6-v2', {
  mode: 'grpc',
  dtype: 'fp16',        // uniform — all servers load the same quantization
  pooling: 'mean',      // uniform
  normalize: true,      // uniform
  servers: [
    { address: 'localhost:50051', workers: 6, device: 'cuda', provider: 'cuda' },
    { address: 'localhost:50052', workers: 6, device: 'cuda', provider: 'cuda' },
    { address: 'localhost:50053', workers: 2, device: 'cpu' },
  ],
  autoStartServer: true,
});
// 14 workers total. GPU workers pick up ~5× more tasks than CPU workers.
```

---

### `'grpc'` — multiple servers, pre-running (heterogeneous)

```js
const embedder = await Embedder.create('Xenova/all-MiniLM-L6-v2', {
  mode: 'grpc',
  autoStartServer: false,
  servers: [
    { address: 'localhost:50051', workers: 6 },
    { address: 'localhost:50052', workers: 6 },
    { address: 'localhost:50053', workers: 2 },
  ],
});
```

---

### `'grpc'` — built-in round-robin (homogeneous cluster)

For identically-configured servers, use gRPC's built-in `round_robin` policy.
Servers must be started separately; auto-start is not supported in this mode.

```js
const embedder = await Embedder.create('Xenova/all-MiniLM-L6-v2', {
  mode: 'grpc',
  grpcAddress: 'ipv4:///localhost:50051,localhost:50052,localhost:50053',
  grpcLoadBalancing: 'round_robin',
  autoStartServer: false,
  concurrency: 8,
});
```

---

## Options Reference

**Top-level (uniform — all servers must produce identical vectors):**

| Option | Type | Default | Description |
|---|---|---|---|
| `mode` | string | `'process'` | `'process'` \| `'thread'` \| `'socket'` \| `'grpc'` |
| `concurrency` | number | numCores | Workers for single-server shorthand |
| `batchSize` | number | 32 | Max texts per worker task |
| `dtype` | string | `'fp32'` | Quantization applied to every server (`'fp32'`\|`'fp16'`\|`'q8'`\|`'q4'`\|`'q4f16'`\|`'auto'`) |
| `pooling` | string | `'mean'` | Pooling strategy — must match across servers |
| `normalize` | boolean | `true` | L2 normalisation — must match across servers |

**Routing (per-server hardware config):**

| Option | Type | Default | Description |
|---|---|---|---|
| `grpcAddress` | string | `'localhost:50051'` | Single-server shorthand address |
| `servers` | object[] | — | Multi-server list; each entry has `address`, `workers`, `device`, `provider` |
| `grpcLoadBalancing` | string | — | gRPC LB policy — `'round_robin'` for built-in equal-weight distribution |
| `autoStartServer` | boolean | `true` | Spawn server process(es) if not reachable |

---

## Key Trade-offs

| | `'grpc'` | `'socket'` | `'process'` | `'thread'` |
|---|---|---|---|---|
| Model copies in RAM | 1 per server | 1 per server | 1 per worker | 1 per worker |
| Protocol | HTTP/2 + protobuf | NDJSON/Unix socket | Node IPC | SharedArrayBuffer |
| Remote serving | Yes — TCP natively | Manual | No | No |
| Cross-language clients | Yes | No | No | No |
| Streaming | First-class (`EmbedStream`) | Manual framing | No | No |
| Schema / tooling | `.proto`, `grpcurl`, Postman | Informal JSON, `nc` | Informal JSON | — |
| Dependency overhead | `@grpc/grpc-js` ~2 MB | None (built-in `net`) | None | None |
| Crash isolation | Server crash affects its workers | Same | Per-worker isolation | Thread crash propagates |

---

## File List

| File | Status |
|------|--------|
| `src/proto/embedder.proto` | **New** |
| `src/grpc-model-server.js` | **New** |
| `src/grpc-worker.js` | **New** |
| `src/worker-pool.js` | **Modify** — `mode:'grpc'` branch, `_startServers()`, `_serverProcesses`, multi-server `_createWorker()`, `grpcLoadBalancing` |
| `src/embedder.js` | **Modify** — JSDoc + forward `grpcAddress`/`servers`/`grpcLoadBalancing`/`autoStartServer` |
| `test/grpc-worker.test.js` | **New** — unit tests for `GrpcWorker` against a mock server |
| `test/grpc-integration.test.js` | **New** — end-to-end: single server, multi-server, round-robin |

---

## Implementation Order

1. Write `src/proto/embedder.proto`; verify dynamic loading with a scratch
   script using `proto-loader` before writing service code.
2. `src/grpc-model-server.js` — implement `Embed` (unary) first; test with
   `grpcurl localhost:50051 embedeer.EmbedderService/Embed` before writing
   any client code.
3. `src/grpc-worker.js` — implement and unit-test against the real server.
4. `src/worker-pool.js` — single-server `mode:'grpc'`; integration-test
   end-to-end.
5. `src/worker-pool.js` — extend `_startServers()` and `_createWorker()` to
   support the `servers` array; test with two mock gRPC servers.
6. `src/grpc-worker.js` — add `grpcLoadBalancing` support; test round-robin
   against two real servers.
7. `src/grpc-model-server.js` — implement `EmbedStream`; test streaming path.
8. `src/embedder.js` — expose all new options; verify with existing test suite.
9. Update `README.md` and `CLI.md` with `--mode grpc`, `--grpc-address`,
   `--grpc-lb`, and multi-server examples.
