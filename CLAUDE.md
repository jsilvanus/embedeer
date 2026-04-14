# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
npm install

# Run tests
npm run test                 # node --test test/*.test.js

# Run a single test file
node --test test/embedder.test.js

# Run tests with coverage
npm run coverage             # c8 node --test test/*.test.js

# Lint
npx eslint src/ test/

# Benchmarks
npm run benchmark            # compare modes (process/thread/socket/gRPC)
npm run server-bench         # server startup and throughput benchmarks

# Start socket daemon (shared model via Unix socket/named pipe)
npm run daemon               # node src/socket-model-server.js

# Start gRPC server (shared model via HTTP/2, default :50051)
npm run server               # node src/grpc-model-server.js
```

## Architecture

Embedeer uses a **WorkerPool pattern** to distribute embedding work across configurable worker types:

```
embed(texts)
  └─ split into batches → WorkerPool
       ├─ process  → ChildProcessWorker (isolated child processes, each owns model copy)
       ├─ thread   → ThreadWorker (worker_threads, each owns model copy)
       ├─ socket   → SocketWorker → socket-model-server (one shared model via IPC)
       └─ grpc     → GrpcWorker → grpc-model-server (one shared model via HTTP/2)
```

**Key files:**
- `src/index.js` — Public API exports
- `src/index.d.ts` — TypeScript type definitions
- `src/embedder.js` — High-level `Embedder` class (main user-facing API)
- `src/worker-pool.js` — Pool orchestration; selects and manages workers
- `src/worker.js` — Base Worker class
- `src/child-process-worker.js` — Spawns isolated child processes
- `src/thread-worker.js` / `src/thread-worker-script.js` — Worker thread implementation
- `src/socket-model-server.js` — Unix socket / named pipe daemon
- `src/grpc-model-server.js` — gRPC server (uses `@grpc/grpc-js`, lazy-loaded)
- `src/model-cache.js` — Cache directory resolution and pipeline options
- `src/model-management.js` — Model download, list, delete, import
- `src/provider-loader.js` — GPU/execution provider (CUDA, DirectML, CPU) resolution
- `src/cli.js` — CLI entry point

**Worker mode tradeoffs:**
- `process` (default): safest isolation, each process loads its own model
- `thread`: faster startup, same memory space, each thread loads its own model
- `socket`/`grpc`: one model instance shared across all workers — optimal for GPU memory

**Multi-server load balancing:** Multiple socket/gRPC servers can be configured (e.g., one GPU + one CPU) and the pool distributes work across them.

**GPU support:** CUDA (Linux x64) and DirectML (Windows x64) are available through `onnxruntime-node`. Device selection via `device: 'gpu'|'cpu'|'auto'` and provider override via `provider: 'cuda'|'dml'|'cpu'`.

**Model quantization:** Supported dtypes are `fp32`, `fp16`, `q8`, `q4`, `q4f16`, `auto`.

## Module System & Testing

- ES modules (`"type": "module"` in package.json); use `import`/`export` throughout
- Uses Node.js built-in test runner (`node --test`), no external test framework
- `@grpc/grpc-js` and `@grpc/proto-loader` are **optional** dependencies — lazy-loaded only when gRPC mode is used
- ESLint config in `eslint.config.cjs`; no lint script in package.json, run manually with `npx eslint`
