/**
 * @jsilvanus/ort-linux-x64-cuda
 *
 * CUDA execution provider for embedeer on Linux x64.
 *
 * How it works:
 *   onnxruntime-node v1.20+ ships libonnxruntime_providers_cuda.so on Linux x64.
 *   No additional binary download is required — the CUDA execution provider is
 *   already bundled with the standard onnxruntime-node package.
 *   (@huggingface/transformers@4.x requires onnxruntime-node@1.24.x which ships CUDA.)
 *
 *   This package verifies that the required CUDA 12 system libraries are
 *   available before attempting to use the CUDA execution provider.
 *
 * System requirements:
 *   - NVIDIA GPU with driver ≥ 525 (CUDA 12 compatible)
 *   - CUDA 12 Toolkit: libcudart.so.12, libcublas.so.12, libcublasLt.so.12,
 *                      libcurand.so.10, libcufft.so.11
 *   - cuDNN 9: libcudnn.so.9
 *
 *   Install CUDA 12:  https://developer.nvidia.com/cuda-downloads
 *   Install cuDNN 9:  https://developer.nvidia.com/cudnn-downloads
 *   Or via apt (Ubuntu/Debian):
 *     sudo apt install cuda-toolkit-12-x libcudnn9-cuda-12
 */

import { execSync } from 'child_process';
import { existsSync } from 'fs';

/**
 * Shared libraries required by libonnxruntime_providers_cuda.so (CUDA 12 / cuDNN 9).
 * These are system-installed libraries; they are NOT bundled with onnxruntime-node.
 */
const REQUIRED_LIBS = [
  'libcudart.so.12',
  'libcublas.so.12',
  'libcublasLt.so.12',
  'libcurand.so.10',
  'libcufft.so.11',
  'libcudnn.so.9',
];

/**
 * Common directories where CUDA libraries may be installed.
 * Includes entries from LD_LIBRARY_PATH so custom installs are detected.
 */
function cudaSearchDirs() {
  const extra = (process.env.LD_LIBRARY_PATH ?? '').split(':').filter(Boolean);
  return [
    '/usr/local/cuda/lib64',
    '/usr/local/cuda-12/lib64',
    '/usr/local/cuda-12.0/lib64',
    '/usr/local/cuda-12.1/lib64',
    '/usr/local/cuda-12.2/lib64',
    '/usr/local/cuda-12.3/lib64',
    '/usr/local/cuda-12.4/lib64',
    '/usr/local/cuda-12.5/lib64',
    '/usr/local/cuda-12.6/lib64',
    '/usr/lib/x86_64-linux-gnu',
    '/usr/lib64',
    ...extra,
  ];
}

/**
 * Find a shared library by name. Checks common CUDA paths then falls back to
 * `ldconfig -p` for libraries registered in the dynamic linker cache.
 *
 * @param {string} libName  e.g. 'libcudart.so.12'
 * @returns {string|null}   Path to the library, or null if not found.
 */
function findLib(libName) {
  for (const dir of cudaSearchDirs()) {
    const fullPath = `${dir}/${libName}`;
    if (existsSync(fullPath)) return fullPath;
  }

  // Use ldconfig cache as a fallback
  try {
    const output = execSync('ldconfig -p', {
      stdio: ['ignore', 'pipe', 'ignore'],
      encoding: 'utf8',
      timeout: 3000,
    });
    for (const line of output.split('\n')) {
      if (line.includes(libName) && line.includes('=>')) {
        const match = line.match(/=>\s*(.+)/);
        if (match) return match[1].trim();
      }
    }
  } catch {
    // ldconfig not available in all environments; that's ok
  }

  return null;
}

/**
 * Activate the CUDA execution provider.
 *
 * Checks that all required CUDA 12 / cuDNN 9 system libraries are present.
 * onnxruntime-node v1.20+ bundles libonnxruntime_providers_cuda.so on Linux x64
 * (@huggingface/transformers@4.x requires onnxruntime-node@1.24.x which ships CUDA),
 * so no additional binary download is needed — only system CUDA libraries are required.
 *
 * @returns {Promise<void>}
 * @throws {Error} If NVIDIA GPU is not detected or required CUDA libraries are missing.
 */
export async function activate() {
  // 1. Check for NVIDIA GPU / driver
  if (!existsSync('/dev/nvidiactl')) {
    throw new Error(
      '@jsilvanus/ort-linux-x64-cuda: No NVIDIA GPU detected (/dev/nvidiactl not found).\n' +
      'Ensure NVIDIA drivers are installed.\n' +
      'Verify with: nvidia-smi',
    );
  }

  // 2. Check required CUDA / cuDNN system libraries
  const missing = REQUIRED_LIBS.filter((lib) => findLib(lib) === null);

  if (missing.length > 0) {
    throw new Error(
      `@jsilvanus/ort-linux-x64-cuda: Missing CUDA system libraries: ${missing.join(', ')}\n\n` +
      'onnxruntime-node CUDA requires CUDA 12 + cuDNN 9. Install them:\n\n' +
      '  # Option A — CUDA 12 + cuDNN 9 via apt (Ubuntu/Debian)\n' +
      '  sudo apt install cuda-toolkit-12-6 libcudnn9-cuda-12\n\n' +
      '  # Option B — CUDA Toolkit installer from NVIDIA\n' +
      '  https://developer.nvidia.com/cuda-downloads\n' +
      '  https://developer.nvidia.com/cudnn-downloads\n\n' +
      '  # After installing, make sure libraries are on LD_LIBRARY_PATH if non-standard:\n' +
      '  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH',
    );
  }

  // onnxruntime-node will dynamically load libonnxruntime_providers_cuda.so at
  // runtime when device='cuda' is passed to pipeline(). No further action needed here.
}

/**
 * Returns the device string passed to @huggingface/transformers pipeline().
 * @returns {string}
 */
export function getDevice() {
  return 'cuda';
}
