/**
 * @embedeer/ort-linux-x64-cuda
 *
 * CUDA execution provider for embedeer on Linux x64.
 *
 * This package activates a CUDA-enabled ONNX Runtime build so that
 * @huggingface/transformers pipeline() runs inference on the GPU.
 *
 * Usage (automatic via embedeer):
 *   // Install this package and embedeer will use it when device='gpu' or 'auto'
 *   // npm install @embedeer/ort-linux-x64-cuda
 *
 * Manual usage:
 *   import { activate, getDevice } from '@embedeer/ort-linux-x64-cuda';
 *   await activate();
 *   // then pass getDevice() as the device option to pipeline()
 */

import { existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

/**
 * Path where the install script places the CUDA-enabled ORT native binding.
 * TODO: update this path once actual binary distribution is implemented.
 */
const BINARY_PATH = join(__dirname, 'vendor', 'onnxruntime_binding.node');

/**
 * Activate the CUDA provider.
 *
 * Verifies that the native CUDA-enabled ONNX Runtime binary is present.
 * In the future this hook can also set environment variables or call into
 * the ORT C API to configure the CUDA execution provider.
 *
 * TODO: When distributing real binaries, also configure the ORT env to
 * point at the custom binary path so that onnxruntime-node loads it.
 *
 * @returns {Promise<void>}
 * @throws {Error} If the native binary is not present (install.js was not run).
 */
export async function activate() {
  if (!existsSync(BINARY_PATH)) {
    throw new Error(
      `@embedeer/ort-linux-x64-cuda: native CUDA binary not found at ${BINARY_PATH}. ` +
      `Re-run: npm install @embedeer/ort-linux-x64-cuda`,
    );
  }
  // TODO: wire up the custom ORT binary to onnxruntime-node resolution.
  // This requires either:
  //   (a) patching the onnxruntime-node module resolution to load from BINARY_PATH, or
  //   (b) using ORT's env.ortModuleUrl / similar API once the JS library exposes it.
  // For now the binary presence check above is sufficient to confirm installation.
}

/**
 * Returns the device string that @huggingface/transformers pipeline() should
 * use with this provider.
 *
 * @returns {string}
 */
export function getDevice() {
  return 'cuda';
}
