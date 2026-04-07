/**
 * Embedder — high-level API for batched, parallel text embeddings.
 *
 * Uses a WorkerPool so the model is loaded once per worker and reused
 * across many `embed()` calls.
 *
 * @example
 * import { Embedder } from 'embedeer';
 *
 * const embedder = await Embedder.create('Xenova/all-MiniLM-L6-v2');
 * const vectors = await embedder.embed(['Hello world', 'Foo bar']);
 * console.log(vectors); // [[0.12, ...], [0.34, ...]]
 * await embedder.destroy();
 */

import { WorkerPool } from './worker-pool.js';

export class Embedder {
  /**
   * @param {string} modelName  Hugging Face model identifier
   * @param {object} [options]
   * @param {number} [options.batchSize=32]   Max texts per worker task
   * @param {number} [options.concurrency=2]  Number of parallel workers
   * @param {string} [options.pooling='mean'] Pooling strategy ('mean'|'cls'|'none')
   * @param {boolean}[options.normalize=true] Whether to L2-normalise embeddings
   */
  constructor(modelName = 'Xenova/all-MiniLM-L6-v2', options = {}) {
    this.modelName = modelName;
    this.batchSize = options.batchSize ?? 32;
    this._pool = new WorkerPool(modelName, {
      poolSize: options.concurrency ?? 2,
      pooling: options.pooling ?? 'mean',
      normalize: options.normalize ?? true,
    });
  }

  /**
   * Factory helper — creates and initializes an Embedder in one call.
   *
   * @param {string} modelName
   * @param {object} [options]
   * @returns {Promise<Embedder>}
   */
  static async create(modelName, options = {}) {
    const embedder = new Embedder(modelName, options);
    await embedder.initialize();
    return embedder;
  }

  /**
   * Initialize the underlying worker pool (download / cache the model).
   * Called automatically by `Embedder.create()`.
   */
  async initialize() {
    await this._pool.initialize();
    return this;
  }

  /**
   * Embed one or more texts.
   *
   * Texts are split into batches of `batchSize` and all batches are
   * submitted to the worker pool concurrently, so multiple workers run
   * in parallel whenever there are more batches than workers.
   *
   * @param {string | string[]} texts
   * @returns {Promise<number[][]>} One embedding vector per input text
   */
  async embed(texts) {
    const input = Array.isArray(texts) ? texts : [texts];
    if (input.length === 0) return [];

    // Split into batches
    const batches = [];
    for (let i = 0; i < input.length; i += this.batchSize) {
      batches.push(input.slice(i, i + this.batchSize));
    }

    // Submit all batches to the pool simultaneously — workers pick them
    // up in parallel up to the pool's concurrency limit.
    const results = await Promise.all(batches.map((batch) => this._pool.run(batch)));

    // results is Array<number[][]>; flatten one level to get number[][]
    return results.flat();
  }

  /**
   * Shut down all worker threads and free resources.
   * The Embedder instance must not be used after this call.
   */
  async destroy() {
    await this._pool.destroy();
  }
}
