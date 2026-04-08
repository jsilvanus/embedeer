/**
 * Install / post-install check for @jsilvanus/ort-linux-x64-cuda
 *
 * onnxruntime-node v1.20+ ships libonnxruntime_providers_cuda.so on Linux x64.
 * (@huggingface/transformers@4.x requires onnxruntime-node@1.24.x which ships CUDA.)
 * No additional binary download is required. This script just verifies that
 * the necessary CUDA 12 system libraries are present, and prints actionable
 * install instructions if they are not.
 *
 * System requirements verified here:
 *   - NVIDIA GPU with CUDA 12-compatible driver (≥ 525)
 *   - CUDA 12 Toolkit:   libcudart.so.12, libcublas.so.12, libcublasLt.so.12,
 *                        libcurand.so.10, libcufft.so.11
 *   - cuDNN 9:           libcudnn.so.9
 */

import { execSync } from 'child_process';
import { existsSync } from 'fs';

if (process.platform !== 'linux' || process.arch !== 'x64') {
  console.warn(
    `[embedeer] @jsilvanus/ort-linux-x64-cuda: skipping checks on ${process.platform}/${process.arch} (this package is for Linux x64 only)`,
  );
  process.exit(0);
}

console.log('[embedeer] @jsilvanus/ort-linux-x64-cuda: checking system CUDA requirements...');

const REQUIRED_LIBS = [
  'libcudart.so.12',
  'libcublas.so.12',
  'libcublasLt.so.12',
  'libcurand.so.10',
  'libcufft.so.11',
  'libcudnn.so.9',
];

const CUDA_SEARCH_DIRS = [
  '/usr/local/cuda/lib64',
  '/usr/local/cuda-12/lib64',
  '/usr/lib/x86_64-linux-gnu',
  '/usr/lib64',
  ...(process.env.LD_LIBRARY_PATH ?? '').split(':').filter(Boolean),
];

function findLib(libName) {
  for (const dir of CUDA_SEARCH_DIRS) {
    if (existsSync(`${dir}/${libName}`)) return `${dir}/${libName}`;
  }
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
  } catch { /* ldconfig not available */ }
  return null;
}

// Check NVIDIA GPU / driver
const hasGpu = existsSync('/dev/nvidiactl');
if (!hasGpu) {
  console.warn(
    '\n[embedeer] WARNING: No NVIDIA GPU detected (/dev/nvidiactl not found).\n' +
    '  @jsilvanus/ort-linux-x64-cuda requires an NVIDIA GPU with CUDA 12 drivers.\n' +
    '  GPU acceleration will not be available until drivers are installed.\n',
  );
} else {
  console.log('[embedeer] ✓ NVIDIA GPU detected');
}

// Check CUDA libraries
const missing = REQUIRED_LIBS.filter((lib) => findLib(lib) === null);
const found = REQUIRED_LIBS.filter((lib) => findLib(lib) !== null);

for (const lib of found) {
  console.log(`[embedeer] ✓ ${lib}`);
}

if (missing.length > 0) {
  console.warn(
    `\n[embedeer] WARNING: Missing CUDA system libraries: ${missing.join(', ')}\n\n` +
    '  onnxruntime-node CUDA EP requires CUDA 12 + cuDNN 9.\n\n' +
    '  Install on Ubuntu/Debian:\n' +
    '    sudo apt install cuda-toolkit-12-6 libcudnn9-cuda-12\n\n' +
    '  Or download from NVIDIA:\n' +
    '    https://developer.nvidia.com/cuda-downloads\n' +
    '    https://developer.nvidia.com/cudnn-downloads\n\n' +
    '  After installing, if libraries are not on the default path:\n' +
    '    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH\n',
  );
  // Exit 0 so npm install doesn't fail — the user may install CUDA later.
  process.exit(0);
}

console.log(
  '\n[embedeer] @jsilvanus/ort-linux-x64-cuda: all CUDA requirements satisfied.\n' +
  '  GPU acceleration is available. Use device="gpu" or device="auto" in embedeer.\n',
);
