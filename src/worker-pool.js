/**
 * WorkerPool manages a fixed-size pool of persistent child processes.
 * Each child loads the model once and reuses it across many batches.
 *
 * Workers run as isolated OS processes (via ChildProcessWorker / fork) so a
 * crash in any worker rejects only that worker's in-flight task; the pool and
 * the calling program continue unaffected.
 */

import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { ChildProcessWorker } from './child-process-worker.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const WORKER_PATH = join(__dirname, 'worker.js');

export class WorkerPool {
  /**
   * @param {string}   modelName  Hugging Face model identifier
   * @param {object} [options]
   * @param {number}   [options.poolSize=2]     Number of parallel workers
   * @param {string}   [options.pooling='mean'] Pooling strategy ('mean'|'cls'|'none')
   * @param {boolean}  [options.normalize=true] Whether to L2-normalise embeddings
   * @param {Function} [options._WorkerClass]   Override worker class (for testing)
   */
  constructor(modelName, { poolSize = 2, pooling = 'mean', normalize = true, _WorkerClass } = {}) {
    this.modelName = modelName;
    this.poolSize = poolSize;
    this.pooling = pooling;
    this.normalize = normalize;
    this._WorkerClass = _WorkerClass ?? ChildProcessWorker;

    /** @type {Array} all workers (active + idle) */
    this.workers = [];
    /** @type {Array} idle workers ready to accept tasks */
    this.idleWorkers = [];
    /** @type {Array<{id:number, texts:string[]}>} queued tasks waiting for a free worker */
    this.pendingTasks = [];
    /** @type {Map<number, {resolve:Function, reject:Function}>} */
    this.taskCallbacks = new Map();
    /**
     * Maps each worker to the task ID it is currently processing.
     * Used to reject the right task when a worker crashes.
     * @type {Map<object, number>}
     */
    this._workerToTaskId = new Map();
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
    this._workerToTaskId.clear();
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
        this._workerToTaskId.delete(worker);
        this.idleWorkers.push(worker);
        cb.resolve(embeddings);
        this._dispatch();
      } else if (type === 'error') {
        const cb = this.taskCallbacks.get(id);
        if (cb) {
          this.taskCallbacks.delete(id);
          this._workerToTaskId.delete(worker);
          cb.reject(new Error(error));
        }
        this.idleWorkers.push(worker);
        this._dispatch();
      }
    });

    worker.on('error', (err) => {
      // OS-level error (e.g. failed to spawn). Reject the in-flight task.
      const taskId = this._workerToTaskId.get(worker);
      if (taskId !== undefined) {
        const cb = this.taskCallbacks.get(taskId);
        if (cb) {
          this.taskCallbacks.delete(taskId);
          cb.reject(err);
        }
        this._workerToTaskId.delete(worker);
      }
      this._removeWorker(worker);
    });

    // Crash isolation: a non-zero exit rejects only this worker's task.
    worker.on('exit', (code) => {
      if (code !== 0) {
        const taskId = this._workerToTaskId.get(worker);
        if (taskId !== undefined) {
          const cb = this.taskCallbacks.get(taskId);
          if (cb) {
            this.taskCallbacks.delete(taskId);
            cb.reject(new Error(`Worker process exited unexpectedly (code ${code})`));
          }
          this._workerToTaskId.delete(worker);
        }
        this._removeWorker(worker);
      }
    });

    return { worker, readyPromise };
  }

  /** Assign queued tasks to idle workers. */
  _dispatch() {
    while (this.pendingTasks.length > 0 && this.idleWorkers.length > 0) {
      const task = this.pendingTasks.shift();
      const worker = this.idleWorkers.shift();
      this._workerToTaskId.set(worker, task.id);
      worker.postMessage({ type: 'task', id: task.id, texts: task.texts });
    }
  }

  /** Remove a worker from both the active and idle lists. */
  _removeWorker(worker) {
    this.workers = this.workers.filter((w) => w !== worker);
    this.idleWorkers = this.idleWorkers.filter((w) => w !== worker);
  }
}
