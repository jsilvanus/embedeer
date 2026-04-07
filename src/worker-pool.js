/**
 * WorkerPool manages a fixed-size pool of persistent worker threads.
 * Each worker loads the model once and reuses it across many batches,
 * avoiding repeated model-download overhead.
 */

import { Worker } from 'worker_threads';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const WORKER_PATH = join(__dirname, 'worker.js');

export class WorkerPool {
  /**
   * @param {string} modelName  Hugging Face model identifier
   * @param {object} [options]
   * @param {number}   [options.poolSize=2]     Number of parallel workers
   * @param {string}   [options.pooling='mean'] Pooling strategy ('mean'|'cls'|'none')
   * @param {boolean}  [options.normalize=true] Whether to L2-normalise embeddings
   * @param {Function} [options._WorkerClass]   Override Worker class (for testing)
   */
  constructor(modelName, { poolSize = 2, pooling = 'mean', normalize = true, _WorkerClass } = {}) {
    this.modelName = modelName;
    this.poolSize = poolSize;
    this.pooling = pooling;
    this.normalize = normalize;
    this._WorkerClass = _WorkerClass ?? Worker;

    /** @type {Worker[]} */
    this.workers = [];
    /** @type {Worker[]} idle workers ready to accept tasks */
    this.idleWorkers = [];
    /** @type {Array<{id:number, texts:string[]}>} queued tasks waiting for a free worker */
    this.pendingTasks = [];
    /** @type {Map<number, {resolve:Function, reject:Function}>} */
    this.taskCallbacks = new Map();
    this._nextId = 0;
    this._initialized = false;
  }

  /**
   * Start all workers and wait until each has loaded its model.
   * Must be called before `run()`.
   */
  async initialize() {
    if (this._initialized) return;

    const readyPromises = [];
    for (let i = 0; i < this.poolSize; i++) {
      const { worker, readyPromise } = this._createWorker();
      this.workers.push(worker);
      readyPromises.push(readyPromise);
    }

    await Promise.all(readyPromises);
    this._initialized = true;
  }

  /**
   * Submit a batch of texts for embedding.
   * @param {string[]} texts
   * @returns {Promise<number[][]>} Array of embedding vectors
   */
  run(texts) {
    if (!this._initialized) {
      return Promise.reject(
        new Error('WorkerPool must be initialized before calling run(). Call initialize() first.'),
      );
    }
    return new Promise((resolve, reject) => {
      const id = this._nextId++;
      this.taskCallbacks.set(id, { resolve, reject });
      this.pendingTasks.push({ id, texts });
      this._dispatch();
    });
  }

  /** Terminate all workers and release resources. */
  async destroy() {
    await Promise.all(this.workers.map((w) => w.terminate()));
    this.workers = [];
    this.idleWorkers = [];
    this.pendingTasks = [];
    this.taskCallbacks.clear();
    this._initialized = false;
  }

  // ── Private ──────────────────────────────────────────────────────────────

  _createWorker() {
    let resolveReady;
    const readyPromise = new Promise((resolve) => {
      resolveReady = resolve;
    });

    const worker = new this._WorkerClass(WORKER_PATH, {
      workerData: {
        modelName: this.modelName,
        pooling: this.pooling,
        normalize: this.normalize,
      },
    });

    worker.on('message', ({ type, id, embeddings, error }) => {
      if (type === 'ready') {
        this.idleWorkers.push(worker);
        resolveReady();
        this._dispatch();
      } else if (type === 'result') {
        const cb = this.taskCallbacks.get(id);
        this.taskCallbacks.delete(id);
        this.idleWorkers.push(worker);
        cb.resolve(embeddings);
        this._dispatch();
      } else if (type === 'error') {
        const cb = this.taskCallbacks.get(id);
        if (cb) {
          this.taskCallbacks.delete(id);
          cb.reject(new Error(error));
        }
        this.idleWorkers.push(worker);
        this._dispatch();
      }
    });

    worker.on('error', (err) => {
      // Surface the error to all pending tasks on this worker.
      for (const [id, cb] of this.taskCallbacks) {
        cb.reject(err);
        this.taskCallbacks.delete(id);
      }
    });

    return { worker, readyPromise };
  }

  /** Assign queued tasks to idle workers. */
  _dispatch() {
    while (this.pendingTasks.length > 0 && this.idleWorkers.length > 0) {
      const task = this.pendingTasks.shift();
      const worker = this.idleWorkers.shift();
      worker.postMessage({ id: task.id, texts: task.texts });
    }
  }
}
