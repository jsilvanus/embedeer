/**
 * Public entry point for the embedeer package.
 *
 * @example
 * import { Embedder } from 'embedeer';
 *
 * const embedder = await Embedder.create('Xenova/all-MiniLM-L6-v2');
 * const vectors = await embedder.embed(['Hello', 'World']);
 * await embedder.destroy();
 *
 * @example — pre-pull a model (like `ollama pull`)
 * import { loadModel } from 'embedeer';
 * await loadModel('Xenova/all-MiniLM-L6-v2');
 */

export { Embedder } from './embedder.js';
export { WorkerPool } from './worker-pool.js';
export { ChildProcessWorker } from './child-process-worker.js';
export { ThreadWorker } from './thread-worker.js';
export { getCacheDir, DEFAULT_CACHE_DIR, buildPipelineOptions } from './model-cache.js';

/**
 * Download and cache a model without creating any workers.
 * Convenience re-export of Embedder.loadModel().
 *
 * @param {string} modelName
 * @param {object} [options]  { token?, dtype?, cacheDir? }
 * @returns {Promise<{modelName: string, cacheDir: string}>}
 */
export async function loadModel(modelName, options) {
  const { Embedder } = await import('./embedder.js');
  return Embedder.loadModel(modelName, options);
}
