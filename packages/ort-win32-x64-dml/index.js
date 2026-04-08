/**
 * @embedeer/ort-win32-x64-dml
 *
 * DirectML execution provider for embedeer on Windows x64.
 *
 * DirectML supports NVIDIA, AMD, and Intel GPUs on Windows via the
 * Direct3D 12 API — no CUDA installation required.
 *
 * @see packages/ort-linux-x64-cuda/index.js for full documentation.
 */

import { existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

const BINARY_PATH = join(__dirname, 'vendor', 'onnxruntime_binding.node');

/**
 * Activate the DirectML provider on Windows x64.
 * @returns {Promise<void>}
 * @throws {Error} If the native binary is not present.
 */
export async function activate() {
  if (!existsSync(BINARY_PATH)) {
    throw new Error(
      `@embedeer/ort-win32-x64-dml: native DirectML binary not found at ${BINARY_PATH}. ` +
      `Re-run: npm install @embedeer/ort-win32-x64-dml`,
    );
  }
  // TODO: wire up the custom ORT binary to onnxruntime-node resolution.
}

/**
 * @returns {string}
 */
export function getDevice() {
  return 'dml';
}
