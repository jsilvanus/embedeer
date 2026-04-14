# Socket-Based Model Server Plan

## Overview

A new optional `mode: 'socket'` that runs a **persistent model server daemon**
shared across multiple independent OS processes. This is the primary use case:
a machine running several services (a web server, a background worker, a CLI
tool) all connecting to one loaded model rather than each paying the memory and
startup cost of loading it separately.

```
OS Process A (web server)          ┐
  └─ Embedder (mode:'socket') ─────┤
                                   ├── /tmp/emb.sock ──► socket-model-server.js
OS Process B (background worker)   │                     (one model in memory)
  └─ Embedder (mode:'socket') ─────┘
```

`mode: 'process'` with `concurrency: 1` already gives you a single model copy
and serial execution within one process — socket mode adds nothing there.
The distinction is cross-process sharing: any number of independent OS processes
can connect to the same server using only the socket path.

Two secondary benefits:
- **Server outlives the client** — the model stays loaded between
  `Embedder.create()` / `destroy()` cycles; subsequent clients connect
  instantly with no reload.
- **Multi-server load balancing** — multiple servers (e.g. two GPU + one CPU)
  managed by the same `WorkerPool`, with the idle-worker queue acting as a
  natural load balancer.

This is purely additive — no existing files change their public API.

---

## Architecture

### Single server

```
Embedder
  └─ WorkerPool (mode: 'socket', poolSize: N)
        ├─ SocketWorker 1  ──┐
        ├─ SocketWorker 2  ──┤── Unix socket ──► SocketModelServer
        └─ SocketWorker N  ──┘                    (one model in memory)
```

### Multi-server

```
Embedder
  └─ WorkerPool (mode: 'socket', servers: [...])
        ├─ SocketWorker 1  ──┐
        ├─ SocketWorker 2  ──┤── /tmp/embedeer-gpu0.sock ──► SocketModelServer (CUDA)
        ├─ SocketWorker 3  ──┘
        ├─ SocketWorker 4  ──┐
        ├─ SocketWorker 5  ──┤── /tmp/embedeer-gpu1.sock ──► SocketModelServer (CUDA)
        ├─ SocketWorker 6  ──┘
        └─ SocketWorker 7  ──── /tmp/embedeer-cpu.sock  ──► SocketModelServer (CPU)

        All 7 workers share one idleWorkers queue.
        GPU workers finish ~5× faster → naturally receive ~5× more tasks.
```

- **SocketModelServer** — loads the model once, listens on a Unix socket (or
  named pipe on Windows), accepts N persistent connections, and processes
  incoming task messages via an internal serial FIFO queue (transformers.js is
  not concurrency-safe).
- **SocketWorker** — implements the same `postMessage / on / terminate`
  interface as `ChildProcessWorker` and `ThreadWorker`. Holds one persistent
  `net.Socket` connection per instance.

---

## New Files

### `src/socket-model-server.js`

The server entry point. Mirrors `worker.js` but with `net.createServer`
instead of `process.on('message')`.

**Responsibilities:**
1. Parse config from `process.argv`: `modelName`, `pooling`, `normalize`,
   `token`, `dtype`, `cacheDir`, `device`, `provider`, `socketPath`,
   `idleTimeout`.
2. Load the model via `@huggingface/transformers` pipeline (same logic as
   `worker.js`).
3. Write `{"type":"ready"}\n` to stdout so the spawning `WorkerPool` knows
   when to proceed.
4. Start a `net.Server` on the given `socketPath` (auto-generated under
   `os.tmpdir()` if omitted).
5. Accept multiple simultaneous client connections; each connection gets its
   own NDJSON (newline-delimited JSON) parser.
6. Maintain an internal FIFO task queue — pull the next task when idle, run
   the extractor, write the result back to the originating connection.
7. On connection close, reject any queued tasks from that connection rather
   than writing results to a closed socket.
8. If `idleTimeout` is set, offload the model after that many milliseconds
   of inactivity and reload it transparently on the next incoming task.

**Idle timeout — model offload/reload:**

`Pipeline.dispose()` cascades through `model.dispose()` to
`session.release()` on every ONNX Runtime session, cleanly freeing GPU VRAM
and CPU RAM. Reload uses the same `pipeline()` call as startup.

```js
let extractor = null;
let lastRequestTime = Date.now();

async function ensureLoaded() {
  if (extractor) return;
  console.error('socket-model-server: reloading model after idle offload');
  extractor = await pipeline('feature-extraction', modelName, pipelineOpts);
}

if (args['idle-timeout']) {
  const idleMs = Number(args['idle-timeout']);
  setInterval(async () => {
    if (extractor && Date.now() - lastRequestTime > idleMs) {
      console.error('socket-model-server: idle timeout — offloading model');
      await extractor.dispose();   // Pipeline → model.dispose() → session.release()
      extractor = null;
    }
  }, Math.min(idleMs, 60_000));   // check at most every minute
}

// In processNext(), before running the extractor:
async function processNext() {
  if (busy || queue.length === 0) return;
  busy = true;
  await ensureLoaded();        // no-op if already loaded; reloads if offloaded
  lastRequestTime = Date.now();
  // ... run task as normal
}
```

Requests that arrive while reloading queue normally — the first caller
after an idle period sees reload latency; all others are unaffected.

```
client → server:  {"type":"task","id":0,"texts":["hello","world"]}\n
server → client:  {"type":"result","id":0,"embeddings":[[...],[...]]}\n
                  {"type":"error","id":0,"error":"..."}\n
```

`id` is echoed back unchanged so `SocketWorker` can match responses to
pending promises — same convention as the existing IPC protocol.

**Graceful shutdown:** On `SIGTERM`/`SIGINT`, stop accepting new connections,
drain the in-progress task, call `extractor?.dispose()` if loaded, then
`server.close()` and `process.exit(0)`.

---

### `src/socket-worker.js`

Drop-in replacement for `ChildProcessWorker`. Implements
`postMessage / terminate / EventEmitter('message'|'error'|'exit')`.

**Constructor:**
```js
new SocketWorker(scriptPath, { workerData })
```
- `scriptPath` is ignored — the server is managed by `WorkerPool`.
- `workerData.socketPath` is the path to connect to.

**Lifecycle:**
1. Connect to `workerData.socketPath`. Buffer incoming data; split on `\n`
   to reconstruct complete JSON messages; emit `'message'` for each.
2. Emit `message { type: 'ready' }` once the socket `'connect'` event fires.
3. `postMessage(msg)` — write `JSON.stringify(msg) + '\n'` to the socket.
4. `terminate()` — call `socket.destroy()`, resolve immediately.

**Error handling:**
- Socket `'error'` event → emit `'error'`.
- Socket `'close'` event → emit `'exit', 1` so `WorkerPool` treats it as a
  crash and rejects the in-flight task.

---

## Changes to Existing Files

### `src/worker-pool.js`

**Constructor — single-server shorthand:**

`SocketWorker` is **not** imported at the top of `worker-pool.js`. The
constructor only stores config; the actual `import('./socket-worker.js')`
happens in `initialize()`. This means the `net` module is never loaded for
users who only use `'process'` or `'thread'` mode.

```js
} else if (mode === 'socket') {
  // _WorkerClass intentionally left null — lazy-loaded in initialize().
  this._WorkerClass = null;
  this._workerScript = null;
  // Normalise to the servers array format used internally.
  // dtype/pooling/normalize are top-level — they must be uniform across all
  // servers so every server produces comparable embedding vectors.
  this._servers = options.servers ?? [{
    socketPath: options.socketPath ?? defaultSocketPath(modelName),
    workers:    options.concurrency ?? poolSize,
    device:     options.device,
    provider:   options.provider,
  }];
  this._autoStartServer = options.autoStartServer ?? true;
}
```

**`initialize()` — lazy-load then start servers:**
```js
async initialize() {
  if (this._initialized) return;

  // Lazy-load: import only when mode:'socket' is actually used.
  if (this.mode === 'socket' && !this._WorkerClass) {
    const { SocketWorker } = await import('./socket-worker.js');
    this._WorkerClass = SocketWorker;
  }

  await this._startServers();
  // ... rest of initialize
}
```

**`initialize()` — replace `_startSocketServer()` with `_startServers()`:**
```js
async _startServers() {
  // For each entry in this._servers:
  //   1. Check if socketPath is already accepting connections.
  //   2. If not (and autoStartServer is true), spawn socket-model-server.js
  //      with the entry's device/provider config as argv flags, plus the
  //      top-level dtype/pooling/normalize (uniform across all servers).
  //   3. Wait for {"type":"ready"} on the child's stdout.
  //   4. Push the child into this._serverProcesses[].
  // Runs all spawns concurrently (Promise.all) so multiple servers
  // load their models in parallel.
}
```

**`_createWorker()` — distribute workers across servers:**
```js
// Build the flat worker list by iterating this._servers.
// For each server entry, create entry.workers SocketWorker instances,
// each with workerData = { socketPath: entry.socketPath }.
// All workers land in the same this.workers / idleWorkers arrays.
```

**`destroy()`:**
```js
// After terminating all workers, kill every process in this._serverProcesses.
await Promise.all(this._serverProcesses.map(p => killAndWait(p)));
```

### `src/embedder.js`

Add to JSDoc and forward to `WorkerPool`. No logic changes.

```js
// @param {string}   [options.socketPath]      Unix socket path (single-server shorthand)
// @param {object[]} [options.servers]         Multi-server list (see WorkerPool)
// @param {boolean}  [options.autoStartServer=true]  Spawn servers if not running
```

---

## WorkerPool Interface — No Changes

`SocketWorker` emits the same events and accepts the same method calls as the
existing workers. `WorkerPool._dispatch()` and `_removeWorker()` work without
modification — the idle-worker queue is inherently server-agnostic.

---

## Multi-Server Load Balancing

No dedicated LB process or extra routing logic is needed. The existing
`idleWorkers` FIFO queue distributes work organically:

- A GPU server processing a batch in ~100 ms returns its workers to the idle
  pool 5× more frequently than a CPU server taking ~500 ms.
- Assigning more workers to a server is equivalent to increasing its weight.
- The CPU server acts as automatic overflow when all GPU workers are busy.

A dedicated LB process only becomes worthwhile when you need capabilities
outside this model: dynamic server registration, sticky routing, or health
checking with automatic failover.

---

## Usage: All Modes

### `'process'` — default, full isolation

Each worker is an OS child process with its own model copy. A worker crash
rejects only its in-flight task; the pool keeps running.

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

Workers are `worker_threads` threads. Lower startup time and memory overhead;
a thread crash can propagate to the parent. Use in trusted environments.

```js
const embedder = await Embedder.create('Xenova/all-MiniLM-L6-v2', {
  mode: 'thread',
  concurrency: 4,
  batchSize: 32,
});
```

---

### `'socket'` — shared daemon across multiple OS processes

The primary use case. Start the server once; any number of independent
processes connect to it by socket path. One model copy serves them all.

```bash
# Start the daemon once (e.g. on machine boot / in a tmux pane)
node src/socket-model-server.js \
  --model Xenova/all-MiniLM-L6-v2 \
  --socket /tmp/embedeer.sock \
  --dtype fp16
```

```js
// process-a.js — web server
const embedder = await Embedder.create('Xenova/all-MiniLM-L6-v2', {
  mode: 'socket',
  socketPath: '/tmp/embedeer.sock',
  autoStartServer: false,
});

// process-b.js — background worker (separate OS process, same machine)
const embedder = await Embedder.create('Xenova/all-MiniLM-L6-v2', {
  mode: 'socket',
  socketPath: '/tmp/embedeer.sock',
  autoStartServer: false,
});
```

Both processes share the one loaded model. `embed()` is identical in both.

---

### `'socket'` — single owner, auto-start

One process owns and manages the server. `WorkerPool` spawns it on
`initialize()` and kills it on `destroy()`. Simpler than a separate daemon
but does not survive `embedder.destroy()`.

```js
const embedder = await Embedder.create('Xenova/all-MiniLM-L6-v2', {
  mode: 'socket',
  concurrency: 4,
  socketPath: '/tmp/embedeer-main.sock',  // omit to auto-generate
  autoStartServer: true,                  // default
});
const vectors = await embedder.embed(['Hello world', 'Foo bar']);
await embedder.destroy();   // kills server + closes connections
```

---

### `'socket'` — server outlives the client

The server stays loaded between `Embedder` lifecycles — useful when the
model is large and you want zero reload cost across multiple short-lived runs.

```js
// First run — server starts, model loads
const e1 = await Embedder.create('Xenova/all-MiniLM-L6-v2', {
  mode: 'socket', socketPath: '/tmp/embedeer.sock', autoStartServer: true,
});
await e1.embed([...]);
await e1.destroy();   // closes connections; server keeps running

// Second run — connects instantly, no model reload
const e2 = await Embedder.create('Xenova/all-MiniLM-L6-v2', {
  mode: 'socket', socketPath: '/tmp/embedeer.sock', autoStartServer: false,
});
```

---

### `'socket'` — legacy: single server, pre-running (manual start)

// Terminal 2 — connect:
const embedder = await Embedder.create('Xenova/all-MiniLM-L6-v2', {
  mode: 'socket',
  socketPath: '/tmp/embedeer-main.sock',
  autoStartServer: false,   // do not spawn; connect to running server
  concurrency: 8,           // multiple connections to the same server
});
```

---

### `'socket'` — multiple servers, auto-start

`WorkerPool` starts each server concurrently (parallel model loads), then
creates the specified number of workers per server. All workers share one
`idleWorkers` queue.

```js
const embedder = await Embedder.create('Xenova/all-MiniLM-L6-v2', {
  mode: 'socket',
  dtype: 'fp16',         // uniform — all servers load the same quantization
  pooling: 'mean',       // uniform
  normalize: true,       // uniform
  servers: [
    { socketPath: '/tmp/embedeer-gpu0.sock', workers: 6, device: 'cuda', provider: 'cuda' },
    { socketPath: '/tmp/embedeer-gpu1.sock', workers: 6, device: 'cuda', provider: 'cuda' },
    { socketPath: '/tmp/embedeer-cpu.sock',  workers: 2, device: 'cpu' },
  ],
  autoStartServer: true,
});
// 14 workers total; GPU workers pick up ~5× more tasks than CPU workers.
```

---

### `'socket'` — multiple servers, pre-running

```js
const embedder = await Embedder.create('Xenova/all-MiniLM-L6-v2', {
  mode: 'socket',
  autoStartServer: false,
  servers: [
    { socketPath: '/tmp/embedeer-gpu0.sock', workers: 6 },
    { socketPath: '/tmp/embedeer-gpu1.sock', workers: 6 },
    { socketPath: '/tmp/embedeer-cpu.sock',  workers: 2 },
  ],
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
| `socketPath` | string | auto | Single-server shorthand socket path |
| `servers` | object[] | — | Multi-server list; each entry has `socketPath`, `workers`, `device`, `provider` |
| `autoStartServer` | boolean | `true` | Spawn server process(es) if not reachable |
| `idleTimeout` | number | — | Milliseconds of inactivity before offloading the model (`--idle-timeout` on the server CLI). `null`/omitted = never offload |

---

## Key Trade-offs

| | `'socket'` | `'process'` | `'thread'` |
|---|---|---|---|
| Model copies in RAM | 1 per server | 1 per worker | 1 per worker |
| Crash isolation | Server crash kills its clients | Per-worker isolation | Thread crash propagates |
| Startup time | One load per server | One load per worker | One load per worker |
| Throughput | Serial per server (one inference at a time) | True parallel | True parallel |
| Cross-instance sharing | Yes — connect multiple Embedders to one server | No | No |
| GPU support | Yes — per-server `device`/`provider` | Yes | Yes |
| Windows | Named pipe instead of Unix socket | Same as today | Same as today |

---

## File List

| File | Status |
|------|--------|
| `src/socket-model-server.js` | **New** |
| `src/socket-worker.js` | **New** |
| `src/worker-pool.js` | **Modify** — `mode:'socket'` branch, `_startServers()`, `_serverProcesses`, multi-server `_createWorker()` |
| `src/embedder.js` | **Modify** — JSDoc + forward `socketPath`/`servers`/`autoStartServer` |
| `test/socket-worker.test.js` | **New** — unit tests for `SocketWorker` against a mock server |
| `test/socket-integration.test.js` | **New** — end-to-end: single server, multi-server |

---

## Implementation Order

1. `src/socket-model-server.js` — implement and test manually with `nc` before
   writing any client code.
2. `src/socket-worker.js` — unit-test with a mock `net.Server` that echoes
   messages.
3. `src/worker-pool.js` — single-server `mode:'socket'` branch; integration-test
   end-to-end.
4. `src/worker-pool.js` — extend to `servers` array; test multi-server with
   two mock servers.
5. `src/embedder.js` — expose options; verify with existing test suite.
6. Update `README.md` and `CLI.md` with `--mode socket` and multi-server docs.
