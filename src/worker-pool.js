/**
 * WorkerPool manages a fixed-size pool of persistent workers.
 * Each worker loads the model once and reuses it across many batches.
 *
 * Two execution modes are supported:
 *   'process' (default) — each worker is an isolated OS child process
 *      (fork). A crash rejects only the in-flight task; the pool and the
 *      calling program continue unaffected.
 *   'thread' — each worker is a worker_threads thread in the same process.
 *      Lower memory overhead and startup time, but a thread crash can
 *      propagate to the parent. Use for trusted environments.
 *
 * Concurrency 1 produces sequential (single-worker) execution.
 */

import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { ChildProcessWorker } from './child-process-worker.js';
import { ThreadWorker } from './thread-worker.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROCESS_WORKER_PATH = join(__dirname, 'worker.js');
const THREAD_WORKER_PATH  = join(__dirname, 'thread-worker-script.js');

export class WorkerPool {
  /**
   * @param {string}   modelName  Hugging Face model identifier
   * @param {object} [options]
   * @param {number}   [options.poolSize=2]         Number of parallel workers
   * @param {string}   [options.mode='process']     'process' (isolated) or 'thread' (same process)
   * @param {string}   [options.pooling='mean']     Pooling strategy ('mean'|'cls'|'none')
   * @param {boolean}  [options.normalize=true]     Whether to L2-normalise embeddings
   * @param {string}   [options.token]              Hugging Face API token (overrides HF_TOKEN env var)
   * @param {string}   [options.dtype]              Quantization dtype ('fp32'|'fp16'|'q8'|'q4'|'q4f16'|'auto')
   * @param {string}   [options.cacheDir]           Custom model cache directory
   * @param {string}   [options.device]             Compute device: 'auto'|'cpu'|'gpu' (default: 'cpu')
   * @param {string}   [options.provider]           Execution provider override: 'cpu'|'cuda'|'dml'
   * @param {Function} [options._WorkerClass]       Override worker class (for testing)
   */
  constructor(modelName, {
    poolSize = 2,
    mode = 'process',
    pooling = 'mean',
    normalize = true,
    token,
    dtype,
    cacheDir,
    device,
    provider,
    _WorkerClass,
  } = {}) {
    this.modelName = modelName;
    this.poolSize = poolSize;
    this.mode = mode;
    this.pooling = pooling;
    this.normalize = normalize;
    this.token = token;
    this.dtype = dtype;
    this.cacheDir = cacheDir;
    this.device = device;
    this.provider = provider;

    // Pick defaults based on mode; can be overridden for testing.
    if (_WorkerClass) {
      this._WorkerClass = _WorkerClass;
      this._workerScript = PROCESS_WORKER_PATH; // doesn't matter for tests
    } else if (mode === 'thread') {
      this._WorkerClass = ThreadWorker;
      this._workerScript = THREAD_WORKER_PATH;
    } else {
      this._WorkerClass = ChildProcessWorker;
      this._workerScript = PROCESS_WORKER_PATH;
    }

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
    console.error(`Starting ${this.poolSize} worker(s); loading model: ${this.modelName} (may download)`);

    const readyPromises = [];
    for (let i = 0; i < this.poolSize; i++) {
      const { worker, readyPromise } = this._createWorker();
      this.workers.push(worker);
      readyPromises.push(readyPromise);
    }

    await Promise.all(readyPromises);
    console.error(`Worker pool initialized — model should be loaded.`);
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

    const worker = new this._WorkerClass(this._workerScript, {
      workerData: {
        modelName: this.modelName,
        pooling: this.pooling,
        normalize: this.normalize,
        token: this.token,
        dtype: this.dtype,
        cacheDir: this.cacheDir,
        device: this.device,
        provider: this.provider,
      },
    });

    worker.on('message', ({ type, id, embeddings, error }) => {
      if (type === 'ready') {
        console.error(`Worker ready (model loaded)`);
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

