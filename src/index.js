/**
 * Public entry point for the embedeer package.
 *
 * @example
 * import { Embedder } from 'embedeer';
 *
 * const embedder = await Embedder.create('Xenova/all-MiniLM-L6-v2');
 * const vectors = await embedder.embed(['Hello', 'World']);
 * await embedder.destroy();
 */

export { Embedder } from './embedder.js';
export { WorkerPool } from './worker-pool.js';
