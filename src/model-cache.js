/**
 * Model cache directory management.
 *
 * Models are stored under ~/.embedeer/models by default, similar in spirit
 * to how Ollama stores models under ~/.ollama/models. This means the same
 * cached model is reused across multiple embedeer invocations without
 * re-downloading from Hugging Face.
 */

import { homedir } from 'os';
import { join } from 'path';
import { mkdirSync } from 'fs';

/** Default cache root: ~/.embedeer/models */
export const DEFAULT_CACHE_DIR = join(homedir(), '.embedeer', 'models');

/**
 * Return the effective cache directory, creating it if it does not exist.
 * @param {string} [dir]  Custom directory; falls back to DEFAULT_CACHE_DIR.
 * @returns {string} Absolute path to the cache directory.
 */
export function getCacheDir(dir) {
  const resolved = dir ?? DEFAULT_CACHE_DIR;
  mkdirSync(resolved, { recursive: true });
  return resolved;
}

/**
 * Build the options object to pass to @huggingface/transformers pipeline()
 * for quantization. Returns `{ dtype }` when dtype is truthy, or `{}`.
 *
 * @param {string|undefined} dtype  e.g. 'q8', 'fp16', 'auto'
 * @returns {{dtype?: string}}
 */
export function buildPipelineOptions(dtype) {
  return dtype ? { dtype } : {};
}
