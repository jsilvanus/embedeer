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

## Package & Publishing

- **Package manager:** npm (`package-lock.json`)
- **Registry:** npm public (`publishConfig.access: "public"`, with npm provenance enabled)

### Releases (Changesets)

Versioning, changelog, and publishing are automated via [Changesets](https://github.com/changesets/changesets) and `.github/workflows/release.yml`:

1. When making a user-facing change, run `npx changeset` and describe the change (patch/minor/major). Commit the generated `.changeset/*.md` file with your PR.
   - **Agent convention:** when an agent opens a PR with a user-facing change, it should create the changeset file itself (writing `.changeset/*.md` directly with an appropriate bump type and summary, rather than running the interactive `npx changeset` command) and include it in the PR, then explicitly tell the user it added a changeset and what bump type it chose.
2. On merge to `main`, the release workflow installs deps and runs `npm run test`. If they pass and there are pending changesets, it opens/updates a "Version Packages" PR that bumps `package.json`, updates `CHANGELOG.md`, and consumes the changeset files.
3. Merging the "Version Packages" PR triggers the release workflow again; with no pending changesets and a version bump present, it runs `npx changeset publish` to publish to npm.
4. Publishing uses npm's OIDC **Trusted Publishing** — no `NPM_TOKEN` secret is stored. The npm package must have this repo's `release.yml` workflow registered as a Trusted Publisher (npmjs.com → package → Settings → Trusted Publisher).

**Do not run `npm version` or manually edit the `version` field in `package.json`.** Do not run `npm publish` locally either. All versioning and publishing is handled by Changesets and the release workflow as described above — the only manual step for a contributor is `npx changeset`.
