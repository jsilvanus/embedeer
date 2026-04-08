/**
 * Install / post-install for @embedeer/ort-win32-x64-cuda
 *
 * IMPORTANT: onnxruntime-node does not currently ship CUDA prebuilts for Windows.
 * This package is a placeholder for future Windows CUDA support.
 *
 * For GPU acceleration on Windows, use DirectML instead:
 *   npm install @embedeer/ort-win32-x64-dml
 */

console.warn(
  '\n[embedeer] WARNING: @embedeer/ort-win32-x64-cuda — CUDA is not currently available\n' +
  '  in onnxruntime-node prebuilt binaries for Windows.\n\n' +
  '  For GPU acceleration on Windows, use DirectML instead:\n' +
  '    npm install @embedeer/ort-win32-x64-dml\n\n' +
  '  DirectML supports NVIDIA, AMD, and Intel GPUs on Windows 10/11.\n',
);
