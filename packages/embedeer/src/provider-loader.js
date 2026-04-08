/**
 * Provider loader — dynamically selects and activates an ONNX Runtime
 * execution-provider package before @huggingface/transformers creates its
 * pipeline.
 *
 * Provider packages are published as separate optional npm packages:
 *   @jsilvanus/embedeer-ort-linux-x64-cuda   — CUDA on Linux x64
 *   @jsilvanus/embedeer-ort-win32-x64-dml    — DirectML on Windows x64
 *
 * Each provider package exports:
 *   activate(): Promise<void>   — runs any setup needed before pipeline()
 *   getDevice(): string         — the device string to pass to pipeline()
 *                                 e.g. 'cuda', 'dml'
 *
 * Usage:
 *   import { resolveProvider } from './provider-loader.js';
 *   const deviceStr = await resolveProvider(device, provider);
 *   // pass deviceStr to pipeline() if truthy
 */

/**
 * Map of "<platform>-<arch>-<provider>" to package name.
 * @type {Record<string, string>}
 */
export const PROVIDER_PACKAGES = {
  'linux-x64-cuda':  '@jsilvanus/embedeer-ort-linux-x64-cuda',
  'win32-x64-dml':   '@jsilvanus/embedeer-ort-win32-x64-dml',
};

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
 * Attempt to load a specific provider package.  Returns a result object
 * that distinguishes between:
 *   - package not installed (ERR_MODULE_NOT_FOUND)
 *   - package installed but activation failed (e.g. native binary missing)
 *   - package loaded successfully
 *
 * @param {string} provider  e.g. 'cuda' or 'dml'
 * @returns {Promise<{loaded: boolean, deviceStr: string|null, error: Error|null}>}
 */
export async function tryLoadProvider(provider) {
  const key = `${process.platform}-${process.arch}-${provider}`;
  const packageName = PROVIDER_PACKAGES[key];
  if (!packageName) {
    return { loaded: false, deviceStr: null, error: null };
  }
  try {
    const mod = await import(packageName);
    if (typeof mod.activate === 'function') {
      await mod.activate();
    }
    const deviceStr = typeof mod.getDevice === 'function' ? mod.getDevice() : provider;
    return { loaded: true, deviceStr, error: null };
  } catch (err) {
    // Any error (package not installed, binary missing, etc.) → not loaded
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
  // Normalise to lower-case strings for consistent comparison
  const dev = (device ?? 'cpu').toLowerCase();
  const prov = provider ? provider.toLowerCase() : undefined;

  // --- Explicit CPU ---
  if (dev === 'cpu' && !prov) return undefined;
  if (prov === 'cpu') return undefined;

  // --- Explicit provider ---
  if (prov && prov !== 'cpu') {
    const key = `${process.platform}-${process.arch}-${prov}`;
    const packageName = PROVIDER_PACKAGES[key];

    if (!packageName) {
      const supportedPlatforms = Object.entries(PROVIDER_PACKAGES)
        .filter(([k]) => k.endsWith(`-${prov}`))
        .map(([k]) => k.replace(`-${prov}`, ''));
      throw new Error(
        `Provider '${prov}' is not supported on ${process.platform}/${process.arch}. ` +
        `Supported platforms: ${supportedPlatforms.join(', ') || 'none'}.`,
      );
    }

    const { loaded, deviceStr, error } = await tryLoadProvider(prov);
    if (!loaded) {
      // If error is NOT a "package not found" error, re-throw original (e.g. binary missing)
      if (error && error.code !== 'ERR_MODULE_NOT_FOUND') {
        throw error;
      }
      throw new Error(
        `Provider '${prov}' was requested but its package '${packageName}' is not installed. ` +
        `Run: npm install ${packageName}`,
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
    // If a package was found but activate() failed with a non-not-found error,
    // re-throw that error as it contains useful diagnostic information.
    if (lastError && lastError.code !== 'ERR_MODULE_NOT_FOUND') {
      throw lastError;
    }
    const packageNames = candidates
      .map((p) => PROVIDER_PACKAGES[`${process.platform}-${process.arch}-${p}`])
      .filter(Boolean);
    throw new Error(
      `device='gpu' was requested but no GPU provider packages are installed ` +
      `for ${process.platform}/${process.arch}. ` +
      `Install one of: ${packageNames.join(', ') || '(none available for this platform)'}.`,
    );
  }

  // device='auto' and no GPU provider found → silently fall back to CPU
  return undefined;
}
