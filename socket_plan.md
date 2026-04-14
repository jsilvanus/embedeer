# Socket-Based Model Server Plan

## Overview

A new optional `mode: 'socket'` that runs **one model server process** shared
by N lightweight client workers. Unlike `'process'` and `'thread'` modes where
each worker loads its own model copy, socket mode loads the model once and
serves all client connections through a single process.

This is purely additive — no existing files change their public API.

---

## Architecture

```
Embedder
  └─ WorkerPool (mode: 'socket', poolSize: N)
        ├─ SocketWorker 1  ──┐
        ├─ SocketWorker 2  ──┤── Unix socket ──► SocketModelServer
        └─ SocketWorker N  ──┘                    (one model in memory)
```

- **SocketModelServer** — a Node.js script that loads the model once and
  listens on a Unix socket (or named pipe on Windows). It accepts N persistent
  connections and processes incoming task messages from any of them using an
  internal serial queue (transformers.js is not concurrency-safe).
- **SocketWorker** — implements the same `postMessage / on / terminate`
  interface as `ChildProcessWorker` and `ThreadWorker`. Internally it holds a
  persistent `net.Socket` connection to the server.

---

## New Files

### `src/socket-model-server.js`

The server entry point. Mirrors `worker.js` but with `net.createServer`
instead of `process.on('message')`.

**Responsibilities:**
1. Parse config from `process.argv` (same fields as the existing `init` IPC
   message: `modelName`, `pooling`, `normalize`, `token`, `dtype`, `cacheDir`,
   `device`, `provider`, `socketPath`).
2. Load the model via `@huggingface/transformers` pipeline (same logic as
   `worker.js`).
3. Write `{"type":"ready"}\n` to stdout so the spawning process knows when to
   proceed.
4. Start a `net.Server` on the given `socketPath` (or a default auto-generated
   path under `os.tmpdir()`).
5. Accept multiple simultaneous client connections. Each connection gets a
   NDJSON (newline-delimited JSON) parser.
6. Maintain an internal FIFO task queue. When idle, pull the next task, run
   the extractor, and write the result back to the originating connection.
7. On connection close, flush any queued tasks from that connection (reject
   them) rather than running them and writing to a closed socket.

**Message protocol (NDJSON, `\n`-delimited):**

```
client → server:  {"type":"task","id":0,"texts":["hello","world"]}\n
server → client:  {"type":"result","id":0,"embeddings":[[...],[...]]}\n
                  {"type":"error","id":0,"error":"..."}\n
```

The `id` field is echoed back unchanged so `SocketWorker` can match responses
to pending promises (mirrors the existing IPC protocol).

**Graceful shutdown:**
- On `SIGTERM` / `SIGINT`, stop accepting new connections, drain the queue,
  then call `server.close()` and `process.exit(0)`.

---

### `src/socket-worker.js`

Drop-in replacement for `ChildProcessWorker`. Implements
`postMessage / terminate / EventEmitter('message'|'error'|'exit')`.

**Constructor signature (matches existing workers):**
```js
new SocketWorker(scriptPath, { workerData })
```
- `scriptPath` is ignored (the server is already running or will be spawned by
  `WorkerPool`).
- `workerData` is stored but not sent (model config lives in the server).

**Lifecycle:**
1. `constructor` — connect to the socket path stored in `workerData.socketPath`.
   Buffer incoming data and split on `\n` to reconstruct JSON messages.
   Emit `message` for each parsed object.
2. Emit `message { type: 'ready' }` once the TCP/socket `connect` event fires.
3. `postMessage(msg)` — serialize as `JSON.stringify(msg) + '\n'` and write to
   the socket.
4. `terminate()` — call `socket.destroy()`, resolve immediately.

**Error handling:**
- Socket `error` event → emit `'error'`.
- Socket `close` event → emit `'exit', 1` (non-zero so WorkerPool treats it as
  a crash and rejects any in-flight task).

---

## Changes to Existing Files

### `src/worker-pool.js`

Add `mode: 'socket'` handling in the constructor:

```js
} else if (mode === 'socket') {
  this._WorkerClass = SocketWorker;
  this._workerScript = null;            // server is external
  this._socketPath = options.socketPath ?? defaultSocketPath(modelName);
  this._autoStartServer = options.autoStartServer ?? true;
}
```

Add a `_startSocketServer()` helper called at the top of `initialize()` when
`mode === 'socket'` and `_autoStartServer` is true:

```js
async _startSocketServer() {
  // Check if socket path is already accepting connections.
  // If not, spawn socket-model-server.js as a child process (not a worker),
  // wait for {"type":"ready"} on its stdout, then continue.
  // Store the server child as this._serverProcess for cleanup in destroy().
}
```

`_createWorker()` passes `{ socketPath: this._socketPath }` inside `workerData`
when mode is `'socket'`, so `SocketWorker` knows where to connect.

`destroy()` additionally kills `this._serverProcess` (if present) after all
workers have terminated.

### `src/embedder.js`

Add `socketPath` and `autoStartServer` to the options JSDoc and forward them
to `WorkerPool`. No logic changes.

```js
// New options:
// @param {string}  [options.socketPath]       Unix socket path for mode:'socket'
// @param {boolean} [options.autoStartServer=true]  Spawn server if not running
```

---

## WorkerPool Interface — No Changes

`SocketWorker` emits the same events (`message`, `error`, `exit`) and accepts
the same method calls (`postMessage`, `terminate`) as the existing workers.
`WorkerPool._createWorker()`, `_dispatch()`, and `_removeWorker()` work without
modification.

---

## Options Summary

```js
await Embedder.create('Xenova/all-MiniLM-L6-v2', {
  mode: 'socket',
  concurrency: 4,          // 4 connections to the server, still one model copy
  socketPath: '/tmp/embedeer-mymodel.sock',  // optional; auto-generated if omitted
  autoStartServer: true,   // default; set false to connect to a pre-running server
});
```

---

## Key Trade-offs

| | Socket mode | Process mode |
|---|---|---|
| Model copies in RAM | 1 | N (one per worker) |
| Crash isolation | Partial — server crash kills all clients | Full — per-worker isolation |
| Startup time | Slower first time (one model load) then instant | Each worker loads independently |
| Throughput | Serial queue in server (one inference at a time) | True parallel (N inferences) |
| Cross-instance sharing | Yes — multiple Embedder instances share one server | No |
| Windows support | Named pipe (`\\.\pipe\embedeer-*`) instead of Unix socket | Same as today |

---

## File List

| File | Status |
|------|--------|
| `src/socket-model-server.js` | **New** |
| `src/socket-worker.js` | **New** |
| `src/worker-pool.js` | **Modify** — add `mode:'socket'` branch, `_startSocketServer()`, server cleanup |
| `src/embedder.js` | **Modify** — JSDoc + forward `socketPath`/`autoStartServer` |
| `test/socket-worker.test.js` | **New** — unit tests for `SocketWorker` |
| `test/socket-integration.test.js` | **New** — end-to-end with real model server |

---

## Implementation Order

1. `src/socket-model-server.js` — implement and test manually with `nc` /
   `netcat` before writing any client code.
2. `src/socket-worker.js` — unit-test with a mock server that echoes messages.
3. `src/worker-pool.js` — add `mode:'socket'` branch; integration-test with
   the real server.
4. `src/embedder.js` — expose options; verify end-to-end with existing test
   suite.
5. Update `README.md` and `CLI.md` with `--mode socket` documentation.
