/**
 * @embedeer/ort-win32-x64-cuda
 *
 * CUDA execution provider for embedeer on Windows x64.
 *
 * IMPORTANT: onnxruntime-node does not currently ship CUDA support for Windows
 * in its prebuilt binaries. CUDA on Windows requires a custom ORT build or a
 * future official release that includes Windows CUDA prebuilts.
 *
 * Use @embedeer/ort-win32-x64-dml for GPU acceleration on Windows instead —
 * DirectML supports NVIDIA, AMD, and Intel GPUs on Windows 10/11 without
 * requiring a CUDA installation.
 *
 * @see packages/ort-win32-x64-dml
 * @see https://github.com/microsoft/onnxruntime/releases for CUDA Windows builds
 */

/**
 * Activate the CUDA execution provider on Windows x64.
 *
 * @returns {Promise<void>}
 * @throws {Error} Always — CUDA is not currently supported via standard onnxruntime-node
 *   prebuilts on Windows. Use @embedeer/ort-win32-x64-dml for DirectML GPU acceleration.
 */
export async function activate() {
  throw new Error(
    '@embedeer/ort-win32-x64-cuda: CUDA is not currently available in onnxruntime-node\n' +
    'prebuilt binaries for Windows.\n\n' +
    'For GPU acceleration on Windows, use DirectML instead:\n' +
    '  npm install @embedeer/ort-win32-x64-dml\n' +
    '  npx embedeer --provider dml --data "Hello"\n\n' +
    'DirectML supports NVIDIA, AMD, and Intel GPUs on Windows 10/11 without CUDA.\n\n' +
    'For Windows CUDA support, a custom onnxruntime build is required.\n' +
    'See: https://onnxruntime.ai/docs/build/inferencing.html',
  );
}

/**
 * @returns {string}
 */
export function getDevice() {
  return 'cuda';
}
