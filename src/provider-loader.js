/**
 * Provider loader — selects and activates an ONNX Runtime execution provider
 * before @huggingface/transformers creates its pipeline.
 *
 * onnxruntime-node (a transitive dependency of @huggingface/transformers@4.x)
 * already ships the CUDA provider on Linux x64 and DirectML on Windows x64 with
 * no additional packages needed.  This module performs the necessary system checks
 * (NVIDIA driver, CUDA libraries) and returns the device string to pass to
 * pipeline().
 *
 * Usage:
 *   import { resolveProvider } from './provider-loader.js';
 *   const deviceStr = await resolveProvider(device, provider);
 *   // pass deviceStr to pipeline() if truthy
 */

import { execSync } from 'child_process';
import fs from 'fs';

/** Cached output of `ldconfig -p` to avoid repeated subprocess calls. */
let _ldconfigCache = null;

// ── CUDA (linux/x64) ─────────────────────────────────────────────────────────

/**
 * Shared libraries required by libonnxruntime_providers_cuda.so (CUDA 12 / cuDNN 9).
 * These are system-installed libraries; they are NOT bundled with onnxruntime-node.
 */
const REQUIRED_CUDA_LIBS = [
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
 * @returns {string[]}
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
    if (fs.existsSync(`${dir}/${libName}`)) return `${dir}/${libName}`;
  }
  try {
    if (_ldconfigCache === null) {
      _ldconfigCache = execSync('ldconfig -p', {
        stdio: ['ignore', 'pipe', 'ignore'],
        encoding: 'utf8',
        timeout: 3000,
      });
    }
    for (const line of _ldconfigCache.split('\n')) {
      if (line.includes(libName) && line.includes('=>')) {
        const match = line.match(/=>\s*(.+)/);
        if (match) return match[1].trim();
      }
    }
  } catch {
    // ldconfig not available in all environments
  }
  return null;
}

/**
 * Activate the CUDA execution provider.
 * Checks for NVIDIA GPU driver and required CUDA 12 / cuDNN 9 system libraries.
 *
 * @returns {Promise<void>}
 * @throws {Error} If NVIDIA GPU is not detected or required CUDA libraries are missing.
 */
function activateCuda() {
  if (!fs.existsSync('/dev/nvidiactl')) {
    throw new Error(
      'No NVIDIA GPU detected (/dev/nvidiactl not found).\n' +
      'Ensure NVIDIA drivers are installed. Verify with: nvidia-smi',
    );
  }

  const missing = REQUIRED_CUDA_LIBS.filter((lib) => findLib(lib) === null);
  if (missing.length > 0) {
    throw new Error(
      `Missing CUDA system libraries: ${missing.join(', ')}\n\n` +
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
}

// ── DirectML (win32/x64) ─────────────────────────────────────────────────────

/**
 * Activate the DirectML execution provider.
 * DirectML is bundled with onnxruntime-node on Windows. Just verifies platform.
 *
 * @returns {Promise<void>}
 * @throws {Error} If not running on Windows.
 */
function activateDml() {
  if (process.platform !== 'win32') {
    throw new Error(
      `DirectML is only available on Windows (current platform: ${process.platform}).`,
    );
  }
}

// ── Internal provider map ────────────────────────────────────────────────────

/**
 * Internal map of "<platform>-<arch>-<provider>" to inline activation logic.
 * Replacing the old external-package-per-provider pattern since onnxruntime-node
 * already bundles the CUDA and DirectML providers.
 *
 * @type {Record<string, { activate: () => Promise<void>, getDevice: () => string }>}
 */
const PROVIDER_IMPLS = {
  'linux-x64-cuda': { activate: activateCuda, getDevice: () => 'cuda' },
  'win32-x64-dml':  { activate: activateDml,  getDevice: () => 'dml'  },
};

// ── Public API ───────────────────────────────────────────────────────────────

/**
 * Returns the ordered list of preferred GPU providers for the current platform.
 * @returns {string[]}
 */
export function getPlatformDefaultProviders() {
  const platform = process.platform;
  const arch = process.arch;
  if (platform === 'linux' && arch === 'x64') return ['cuda'];
  if (platform === 'win32' && arch === 'x64') return ['cuda', 'dml'];
  return [];
}

/**
 * Attempt to activate a specific provider. Returns a result object:
 *   - { loaded: true, deviceStr, error: null }   — provider ready
 *   - { loaded: false, deviceStr: null, error }  — provider unavailable
 *
 * @param {string} provider  e.g. 'cuda' or 'dml'
 * @returns {Promise<{loaded: boolean, deviceStr: string|null, error: Error|null}>}
 */
export async function tryLoadProvider(provider) {
  const key = `${process.platform}-${process.arch}-${provider}`;
  const impl = PROVIDER_IMPLS[key];
  if (!impl) {
    return { loaded: false, deviceStr: null, error: null };
  }
  try {
    await impl.activate();
    const deviceStr = impl.getDevice();
    return { loaded: true, deviceStr, error: null };
  } catch (err) {
    return { loaded: false, deviceStr: null, error: err };
  }
}

/**
 * Resolve and activate the appropriate execution provider, returning the
 * device string to pass to `@huggingface/transformers` pipeline().
 *
 * @param {'auto'|'cpu'|'gpu'|undefined} device
 * @param {'cpu'|'cuda'|'dml'|undefined} provider  Optional explicit override
 * @returns {Promise<string|undefined>}  Device string or undefined (CPU default)
 *
 * @throws {Error} When an explicit provider is requested but not available.
 * @throws {Error} When device='gpu' and no GPU provider is available.
 */
export async function resolveProvider(device, provider) {
  const dev = (device ?? 'cpu').toLowerCase();
  const prov = provider ? provider.toLowerCase() : undefined;

  // --- Explicit CPU ---
  if (dev === 'cpu' && !prov) return undefined;
  if (prov === 'cpu') return undefined;

  // --- Explicit provider ---
  if (prov && prov !== 'cpu') {
    const key = `${process.platform}-${process.arch}-${prov}`;
    if (!PROVIDER_IMPLS[key]) {
      const supportedPlatforms = Object.keys(PROVIDER_IMPLS)
        .filter((k) => k.endsWith(`-${prov}`))
        .map((k) => k.replace(`-${prov}`, ''));
      throw new Error(
        `Provider '${prov}' is not supported on ${process.platform}/${process.arch}. ` +
        `Supported platforms: ${supportedPlatforms.join(', ') || 'none'}.`,
      );
    }

    const { loaded, deviceStr, error } = await tryLoadProvider(prov);
    if (!loaded) {
      if (error) throw error;
      throw new Error(
        `Provider '${prov}' is not available on ${process.platform}/${process.arch}.`,
      );
    }
    return deviceStr ?? undefined;
  }

  // --- device='gpu' or device='auto': try platform defaults in order ---
  const candidates = getPlatformDefaultProviders();
  let lastError = null;

  for (const candidate of candidates) {
    const { loaded, deviceStr, error } = await tryLoadProvider(candidate);
    if (loaded) return deviceStr ?? candidate;
    if (error) lastError = error;
  }

  if (dev === 'gpu') {
    if (lastError) throw lastError;
    throw new Error(
      `device='gpu' was requested but no GPU provider is available ` +
      `for ${process.platform}/${process.arch}. ` +
      `Supported: linux/x64 (CUDA 12 + cuDNN 9), win32/x64 (DirectML).`,
    );
  }

  // device='auto' and no GPU available → silently fall back to CPU
  return undefined;
}
