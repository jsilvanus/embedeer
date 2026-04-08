/**
 * Install / post-install check for @jsilvanus/embedeer-ort-win32-x64-dml
 *
 * onnxruntime-node ships DirectML support bundled on Windows x64.
 * No additional binary download is required.
 *
 * DirectML is part of Windows 10/11 and supports all DirectX 12 GPUs:
 * NVIDIA, AMD, Intel, Qualcomm, etc. No CUDA installation needed.
 *
 * This script just confirms the environment is suitable.
 */

if (process.platform !== 'win32') {
  console.warn(
    `[embedeer] @jsilvanus/embedeer-ort-win32-x64-dml: skipping checks on ${process.platform}/${process.arch} (this package is for Windows x64 only)`,
  );
  process.exit(0);
}

console.log(
  '[embedeer] @jsilvanus/embedeer-ort-win32-x64-dml: DirectML is bundled with onnxruntime-node on Windows.\n' +
  '  No additional binary download is required.\n' +
  '  GPU acceleration via DirectML is available on Windows 10 (1903+) / Windows 11\n' +
  '  with any DirectX 12-capable GPU.\n' +
  '  Make sure your GPU drivers are up to date.\n',
);
