/**
 * @jsilvanus/ort-win32-x64-dml
 *
 * DirectML execution provider for embedeer on Windows x64.
 *
 * How it works:
 *   onnxruntime-node ships DirectML support on Windows x64 out of the box.
 *   No additional binary download is required — DirectML is bundled with
 *   the standard onnxruntime-node package and comes with Windows 10/11.
 *
 * Hardware:
 *   Supports NVIDIA, AMD, Intel, and Qualcomm GPUs via Direct3D 12.
 *   No CUDA installation required.
 *
 * System requirements:
 *   - Windows 10 (1903+) or Windows 11
 *   - Any DirectX 12-capable GPU (most GPUs from 2016+)
 *   - Up-to-date GPU drivers (from your GPU vendor)
 */

/**
 * Activate the DirectML execution provider.
 *
 * DirectML is bundled with onnxruntime-node on Windows and available natively
 * on Windows 10/11. No system library installation is required.
 *
 * @returns {Promise<void>}
 * @throws {Error} If not running on Windows.
 */
export async function activate() {
  if (process.platform !== 'win32') {
    throw new Error(
      `@jsilvanus/ort-win32-x64-dml: DirectML is only available on Windows (current platform: ${process.platform}).`,
    );
  }
  // DirectML is natively available via onnxruntime-node on Windows 10/11.
  // onnxruntime will load the DirectML EP automatically when device='dml' is requested.
}

/**
 * Returns the device string passed to @huggingface/transformers pipeline().
 * @returns {string}
 */
export function getDevice() {
  return 'dml';
}
