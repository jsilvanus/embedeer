/**
 * Install script for @embedeer/ort-linux-x64-cuda
 *
 * Downloads (or builds) a CUDA-enabled ONNX Runtime Node.js binding for
 * Linux x64 and places it under vendor/ so the package can activate it at
 * runtime.
 *
 * This script runs automatically via the "install" lifecycle hook:
 *   npm install @embedeer/ort-linux-x64-cuda
 *
 * ── Current status ────────────────────────────────────────────────────────
 * STUB — the actual binary download / build is not yet implemented.
 * The structure and hooks are in place; see the TODOs below.
 * ─────────────────────────────────────────────────────────────────────────
 *
 * Expected artifact layout after install:
 *   packages/ort-linux-x64-cuda/
 *   └── vendor/
 *       ├── onnxruntime_binding.node   ← CUDA-enabled ORT Node binding
 *       └── libonnxruntime_providers_cuda.so  ← shared lib (may be bundled in .node)
 *
 * TODO:
 *   1. Build or obtain a CUDA-enabled onnxruntime-node binding.
 *      Options:
 *        (a) Build from source: https://onnxruntime.ai/docs/build/inferencing.html
 *            cmake flags: --use_cuda --cuda_home /usr/local/cuda
 *        (b) Download a prebuilt binary from a GitHub Release in this repo.
 *            See: https://github.com/jsilvanus/embedeer/releases
 *   2. Upload the binary as a GitHub Release asset tagged by version + platform.
 *   3. Replace the stub below with actual download logic using the fetch API
 *      (or the 'node-fetch' package for older Node versions).
 *   4. Verify the binary checksum (SHA-256) before using it.
 *
 * CUDA compatibility:
 *   The binary must be compiled against the same CUDA major version as the
 *   host system (e.g. CUDA 12.x). Consider publishing multiple binaries:
 *     ort-linux-x64-cuda12, ort-linux-x64-cuda11, etc.
 *
 * onnxruntime version:
 *   Must match the version that @huggingface/transformers depends on.
 *   Check: node -e "require('onnxruntime-node/package.json').version"
 */

import { mkdirSync, writeFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const VENDOR_DIR = join(__dirname, 'vendor');

// Only run on Linux x64; other platforms should not have installed this package,
// but guard anyway.
if (process.platform !== 'linux' || process.arch !== 'x64') {
  console.warn(
    `[embedeer] @embedeer/ort-linux-x64-cuda: skipping install on ${process.platform}/${process.arch}`,
  );
  process.exit(0);
}

console.log('[embedeer] @embedeer/ort-linux-x64-cuda: running install script...');

mkdirSync(VENDOR_DIR, { recursive: true });

// ── TODO: replace this stub with real binary download ─────────────────────
//
// Example skeleton for a real download (requires Node 18+ built-in fetch):
//
//   const VERSION = '1.0.0';
//   const BASE_URL = `https://github.com/jsilvanus/embedeer/releases/download/ort-linux-x64-cuda-${VERSION}`;
//   const BINARY_NAME = 'onnxruntime_binding.node';
//   const CHECKSUM_NAME = 'onnxruntime_binding.node.sha256';
//
//   const res = await fetch(`${BASE_URL}/${BINARY_NAME}`);
//   if (!res.ok) throw new Error(`Download failed: ${res.status} ${res.statusText}`);
//   const buf = Buffer.from(await res.arrayBuffer());
//
//   // TODO: verify SHA-256 checksum here
//
//   writeFileSync(join(VENDOR_DIR, BINARY_NAME), buf);
//   console.log(`[embedeer] Installed CUDA ORT binding → ${join(VENDOR_DIR, BINARY_NAME)}`);
// ──────────────────────────────────────────────────────────────────────────

// For now write a placeholder so the package directory is not empty.
writeFileSync(
  join(VENDOR_DIR, 'README.txt'),
  'This directory will contain the CUDA-enabled ONNX Runtime native binding.\n' +
  'See packages/ort-linux-x64-cuda/install.js for the download TODO.\n',
);

console.warn(
  '[embedeer] @embedeer/ort-linux-x64-cuda: STUB install complete. ' +
  'No real CUDA binary was downloaded yet — GPU execution is not available. ' +
  'See packages/ort-linux-x64-cuda/install.js for the implementation TODO.',
);
