# gRPC Model Server Plan

## Overview

A new optional `mode: 'grpc'` that exposes the embedding pipeline as a gRPC
service. Like socket mode, one server process loads the model once; client
workers call it via typed RPCs over HTTP/2. Unlike socket mode, gRPC adds a
formal service contract (`.proto`), structured binary serialization (Protocol
Buffers), and first-class server-side streaming — enabling both local
optimization and remote/cross-language deployment.

This is purely additive — no existing files change their public API.

---

## Architecture

```
Embedder
  └─ WorkerPool (mode: 'grpc', poolSize: N)
        ├─ GrpcWorker 1  ──┐
        ├─ GrpcWorker 2  ──┤── HTTP/2 + protobuf ──► GrpcModelServer
        └─ GrpcWorker N  ──┘  (multiplexed on one addr)  (one model in memory)
```

N `GrpcWorker` instances each hold a gRPC channel to the same server address.
HTTP/2 multiplexes all N concurrent RPCs over a small number of TCP connections
automatically — no manual connection management needed.

---

## Dependencies

```json
{
  "dependencies": {
    "@grpc/grpc-js":    "^1.12.0",
    "@grpc/proto-loader": "^0.7.15"
  }
}
```

`@grpc/proto-loader` loads `.proto` files dynamically at runtime — no
code-generation step (no `protoc` required, no generated JS to commit).

---

## Proto Definition

### `src/proto/embedder.proto`

```protobuf
syntax = "proto3";

package embedeer;

service EmbedderService {
  // Unary: embed a batch of texts, return all vectors at once.
  rpc Embed (EmbedRequest) returns (EmbedResponse);

  // Server-streaming: embed a large batch; stream one vector per text
  // as it is computed. Useful for very large inputs and CLI progress.
  rpc EmbedStream (EmbedRequest) returns (stream EmbedChunk);
}

message EmbedRequest {
  repeated string texts = 1;
  int32           id    = 2;   // echoed back in the response for correlation
}

message EmbedResponse {
  int32            id         = 1;
  repeated FloatVec embeddings = 2;
}

// A single embedding vector (one per text in stream mode).
message EmbedChunk {
  int32           id     = 1;
  int32           index  = 2;   // position in original texts array
  repeated float  values = 3;
}

message FloatVec {
  repeated float values = 1;
}
```

---

## New Files

### `src/grpc-model-server.js`

The server entry point. Mirrors `socket-model-server.js` but uses gRPC
instead of raw sockets.

**Responsibilities:**
1. Parse config from `process.argv` (same fields as existing `init` message
   plus `grpcAddress` e.g. `'localhost:50051'` or `'unix:///tmp/embedeer.sock'`).
2. Load the model via `@huggingface/transformers` pipeline.
3. Write `{"type":"ready"}\n` to stdout once the model is loaded and the server
   is bound, so the spawning `WorkerPool` knows when to proceed.
4. Register the `EmbedderService` implementation (see below) and start
   listening.

**Service handler — `Embed` (unary):**
```
handleEmbed(call, callback) {
  const { texts, id } = call.request;
  extractor(texts, { pooling, normalize })
    .then(output => callback(null, { id, embeddings: output.tolist().map(v => ({ values: v })) }))
    .catch(err  => callback({ code: grpc.status.INTERNAL, message: err.message }));
}
```

**Service handler — `EmbedStream` (server-streaming):**
```
handleEmbedStream(call) {
  const { texts, id } = call.request;
  // For each text individually (or each sub-batch), call extractor and
  // stream back EmbedChunk messages as they complete.
  // call.write({ id, index, values }) for each result.
  // call.end() when done.
}
```

**Serialization note:** protobuf `repeated float` maps directly to a JS
`Float32Array` / plain array — no manual conversion beyond the `.tolist()`
call already used in `worker.js`.

**Graceful shutdown:** On `SIGTERM` / `SIGINT` call `server.tryShutdown(cb)`,
which drains in-flight RPCs before closing.

---

### `src/grpc-worker.js`

Drop-in replacement for `ChildProcessWorker`. Implements
`postMessage / terminate / EventEmitter('message'|'error'|'exit')`.

**Constructor signature (matches existing workers):**
```js
new GrpcWorker(scriptPath, { workerData })
```
- `scriptPath` is ignored (server is managed by `WorkerPool`).
- `workerData.grpcAddress` is the server address (e.g. `'localhost:50051'`).

**Lifecycle:**
1. `constructor` — load the proto and create a gRPC client stub:
   ```js
   const pkgDef = protoLoader.loadSync(PROTO_PATH, { ...protoLoaderOpts });
   const { embedeer: { EmbedderService } } = grpc.loadPackageDefinition(pkgDef);
   this._stub = new EmbedderService(address, grpc.credentials.createInsecure());
   ```
2. Call `this._stub.waitForReady(deadline, err => ...)` then emit
   `message { type: 'ready' }` — this is what `WorkerPool` listens for in
   `_createWorker()`.
3. `postMessage({ type: 'task', id, texts })` — call `this._stub.embed(...)`:
   ```js
   this._stub.embed({ texts, id }, (err, response) => {
     if (err) this.emit('message', { type: 'error', id, error: err.message });
     else     this.emit('message', { type: 'result', id,
                                     embeddings: response.embeddings.map(v => v.values) });
   });
   ```
4. `terminate()` — call `grpc.closeClient(this._stub)`, resolve immediately.

**Error handling:**
- `waitForReady` timeout → emit `'error'`.
- RPC status `UNAVAILABLE` → emit `'error'` (server crashed); WorkerPool
  catches this as a worker crash.

---

## Changes to Existing Files

### `src/worker-pool.js`

Add `mode: 'grpc'` handling in the constructor:

```js
} else if (mode === 'grpc') {
  this._WorkerClass = GrpcWorker;
  this._workerScript = null;
  this._grpcAddress = options.grpcAddress ?? 'localhost:50051';
  this._autoStartServer = options.autoStartServer ?? true;
}
```

Add `_startGrpcServer()` called at the top of `initialize()` when
`mode === 'grpc'` and `_autoStartServer` is true (same pattern as the socket
server helper):

```js
async _startGrpcServer() {
  // Spawn grpc-model-server.js as a plain child process.
  // Pass all model config + grpcAddress as argv flags.
  // Wait for {"type":"ready"} on stdout before resolving.
  // Store child as this._serverProcess.
}
```

`_createWorker()` passes `{ grpcAddress: this._grpcAddress }` inside
`workerData` so `GrpcWorker` knows where to connect.

`destroy()` additionally calls `server.tryShutdown` / kills
`this._serverProcess` after workers are terminated.

### `src/embedder.js`

Add `grpcAddress` and `autoStartServer` to the options JSDoc and forward them
to `WorkerPool`. No logic changes.

```js
// New options:
// @param {string}  [options.grpcAddress]            gRPC server address for mode:'grpc'
// @param {boolean} [options.autoStartServer=true]   Spawn server if not reachable
```

---

## WorkerPool Interface — No Changes

`GrpcWorker` emits the same events and accepts the same method calls as the
existing workers. `WorkerPool._createWorker()`, `_dispatch()`, and
`_removeWorker()` work without modification.

---

## Options Summary

```js
await Embedder.create('Xenova/all-MiniLM-L6-v2', {
  mode: 'grpc',
  concurrency: 4,            // 4 parallel in-flight RPCs, one model copy
  grpcAddress: 'localhost:50051',   // optional; defaults to 50051
  autoStartServer: true,     // default; set false to connect to pre-running server
});
```

### Connecting to a pre-running remote server

```js
await Embedder.create('Xenova/all-MiniLM-L6-v2', {
  mode: 'grpc',
  grpcAddress: '192.168.1.100:50051',
  autoStartServer: false,    // do not spawn locally
});
```

---

## Key Trade-offs vs Socket Mode

| | gRPC mode | Socket mode |
|---|---|---|
| Protocol | HTTP/2 + protobuf (binary, typed) | NDJSON over Unix socket (text) |
| Remote serving | Yes — works over TCP natively | Manual (needs TCP socket + security) |
| Cross-language | Yes — Python/Go/Rust can call the server | No — server is Node.js only |
| Streaming | First-class (`EmbedStream`) | Manual chunking |
| Schema / docs | `.proto` is self-documenting | Informal JSON convention |
| Dependency overhead | `@grpc/grpc-js` (~2 MB) | None (Node `net` built-in) |
| Debugging | `grpcurl`, Postman, Buf Studio | `nc`, plain JSON |
| Complexity | Medium | Low |

---

## File List

| File | Status |
|------|--------|
| `src/proto/embedder.proto` | **New** |
| `src/grpc-model-server.js` | **New** |
| `src/grpc-worker.js` | **New** |
| `src/worker-pool.js` | **Modify** — add `mode:'grpc'` branch, `_startGrpcServer()`, server cleanup |
| `src/embedder.js` | **Modify** — JSDoc + forward `grpcAddress`/`autoStartServer` |
| `test/grpc-worker.test.js` | **New** — unit tests for `GrpcWorker` (mock server) |
| `test/grpc-integration.test.js` | **New** — end-to-end with real gRPC server |

---

## Implementation Order

1. Write `src/proto/embedder.proto` and verify it loads with `proto-loader`
   in a scratch script before writing any service code.
2. `src/grpc-model-server.js` — implement `Embed` (unary) first; test with
   `grpcurl` before writing any client.
3. `src/grpc-worker.js` — implement and unit-test against the server.
4. Add `_startGrpcServer()` to `WorkerPool`; integration-test end-to-end.
5. Implement `EmbedStream` handler and add a streaming path to `GrpcWorker`
   (opt-in, triggered by a `stream: true` option on `WorkerPool`).
6. `src/embedder.js` — expose options; verify with existing test suite.
7. Update `README.md` and `CLI.md` (`--mode grpc`, `--grpc-address`).
