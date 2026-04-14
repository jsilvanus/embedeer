/**
 * WorkerPool manages a fixed-size pool of persistent workers.
 * Each worker loads the model once and reuses it across many batches.
 *
 * Execution modes:
 *   'process' (default) — each worker is an isolated OS child process
 *      (fork). A crash rejects only the in-flight task; the pool and the
 *      calling program continue unaffected.
 *   'thread' — each worker is a worker_threads thread in the same process.
 *      Lower memory overhead and startup time, but a thread crash can
 *      propagate to the parent. Use for trusted environments.
 *   'socket' — workers are lightweight socket connections to a shared
 *      SocketModelServer process. Worker class is lazy-loaded on first
 *      initialize() call so the 'net' module is not imported unnecessarily.
 *   'grpc'   — workers are gRPC stubs connecting to a GrpcModelServer.
 *      Worker class and @grpc/grpc-js are lazy-loaded on first initialize()
 *      so users who never use gRPC pay zero import cost.
 *
 * Concurrency 1 produces sequential (single-worker) execution.
 */

import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { ChildProcessWorker } from './child-process-worker.js';
import { ThreadWorker } from './thread-worker.js';
import { registerLoadedModel, unregisterLoadedModel } from './runtime-models.js';

// Small O(1) FIFO queue to avoid repeated `Array.shift()` costs when the
// pending task queue grows large.
class FIFOQueue {
  constructor() {
    this._arr = [];
    this._head = 0;
  }
  push(item) { this._arr.push(item); }
  shift() {
    if (this._head >= this._arr.length) return undefined;
    const item = this._arr[this._head++];
    // Periodically trim the backing array to avoid unbounded growth.
    if (this._head > 1024 && this._head * 2 >= this._arr.length) {
      this._arr = this._arr.slice(this._head);
      this._head = 0;
    }
    return item;
  }
  get length() { return this._arr.length - this._head; }
  drain() { const rest = this._arr.slice(this._head); this._arr = []; this._head = 0; return rest; }
}

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
    // Optional overrides for worker script paths (generation vs embedder)
    workerScript,
    threadWorkerScript,
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
    // 'socket' and 'grpc' worker classes are NOT imported here — they are
    // lazy-loaded in initialize() so that @grpc/grpc-js and socket deps are
    // never loaded for users who only use 'process' or 'thread' mode.
    if (_WorkerClass) {
      this._WorkerClass = _WorkerClass;
      this._workerScript = workerScript ?? PROCESS_WORKER_PATH; // tests may override
    } else if (mode === 'thread') {
      this._WorkerClass = ThreadWorker;
      this._workerScript = threadWorkerScript ?? THREAD_WORKER_PATH;
    } else if (mode === 'socket' || mode === 'grpc') {
      // Deferred — resolved in initialize() via dynamic import.
      this._WorkerClass = null;
    } else {
      this._WorkerClass = ChildProcessWorker;
      this._workerScript = workerScript ?? PROCESS_WORKER_PATH;
    }

    /** @type {Array} all workers (active + idle) */
    this.workers = [];
    /** @type {Array} idle workers ready to accept tasks */
    this.idleWorkers = [];
    /** Pending task queue (FIFO) */
    this.pendingTasks = new FIFOQueue();
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
    this._registered = false;
  }

  /**
   * Start all workers and wait until each has loaded its model.
   * Must be called before `run()`.
   */
  async initialize() {
    if (this._initialized) return;

    // Lazy-load worker class for server modes. Dynamic import means
    // @grpc/grpc-js is never touched unless mode:'grpc' is actually used.
    if (this.mode === 'socket' && !this._WorkerClass) {
      const { SocketWorker } = await import('./socket-worker.js');
      this._WorkerClass = SocketWorker;
    } else if (this.mode === 'grpc' && !this._WorkerClass) {
      try {
        const { GrpcWorker } = await import('./grpc-worker.js');
        this._WorkerClass = GrpcWorker;
      } catch (err) {
        if (err.code === 'ERR_MODULE_NOT_FOUND' && err.message.includes('grpc')) {
          throw new Error(
            'mode:\'grpc\' requires optional peer packages that are not installed.\n' +
            'Install them with: npm install @grpc/grpc-js @grpc/proto-loader',
          );
        }
        throw err;
      }
    }

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
    // Register this model as loaded in the runtime registry.
    if (!this._registered) {
      registerLoadedModel(this.modelName);
      this._registered = true;
    }
  }

  /**
   * Submit a batch of texts for embedding.
   * @param {string[]} texts
   * @returns {Promise<number[][]>} Array of embedding vectors
   */
  run(payload) {
    if (!this._initialized) {
      return Promise.reject(
        new Error('WorkerPool must be initialized before calling run(). Call initialize() first.'),
      );
    }
    return new Promise((resolve, reject) => {
      const id = this._nextId++;
      this.taskCallbacks.set(id, { resolve, reject });
      this.pendingTasks.push({ id, payload });
      this._dispatch();
    });
  }

  /** Terminate all workers and release resources. */
  async destroy() {
    await Promise.all(this.workers.map((w) => w.terminate()));
    this.workers = [];
    this.idleWorkers = [];
    this.pendingTasks = new FIFOQueue();
    this.taskCallbacks.clear();
    this._workerToTaskId.clear();
    this._initialized = false;
    if (this._registered) {
      unregisterLoadedModel(this.modelName);
      this._registered = false;
    }
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

    worker.on('message', (msg) => {
      const { type, id } = msg;
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

        // Prefer explicit embeddings/text/result fields. Support transferred
        // ArrayBuffers / TypedArrays coming from worker_threads by reshaping
        // into the public `number[][]` format before resolving.
        let resultPayload = msg.embeddings ?? msg.text ?? msg.result ?? msg;
        const emb = msg.embeddings;
        if (emb !== undefined) {
          if (ArrayBuffer.isView(emb) || emb instanceof ArrayBuffer) {
            const ab = ArrayBuffer.isView(emb) ? emb.buffer : emb;
            const shape = msg.shape;
            if (shape && Array.isArray(shape) && shape.length >= 2) {
              const [rows, cols] = shape;
              const floatView = new Float32Array(ab);
              const out = new Array(rows);
              for (let i = 0; i < rows; i++) {
                const row = new Array(cols);
                const offset = i * cols;
                for (let j = 0; j < cols; j++) row[j] = floatView[offset + j];
                out[i] = row;
              }
              resultPayload = out;
            } else {
              // Fallback — expose as regular Array
              resultPayload = Array.from(new Float32Array(ab));
            }
          } else {
            // embeddings is likely already nested arrays or other type — use as-is
            resultPayload = emb;
          }
        } else {
          resultPayload = msg.text ?? msg.result ?? msg;
        }

        cb.resolve(resultPayload);
        this._dispatch();
      } else if (type === 'error') {
        const cb = this.taskCallbacks.get(id);
        if (cb) {
          this.taskCallbacks.delete(id);
          this._workerToTaskId.delete(worker);
          cb.reject(new Error(msg.error));
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
      // Support legacy embedding API where run(texts) passes an array of texts,
      // as well as newer generic payload objects for generation tasks.
      if (Array.isArray(task.payload)) {
        // legacy embeddings path
        worker.postMessage({ type: 'task', id: task.id, texts: task.payload });
      } else if (task.payload && typeof task.payload === 'object') {
        // generic payload: spread fields (e.g. prompt, genOpts)
        worker.postMessage({ type: 'task', id: task.id, ...task.payload });
      } else {
        // fallback: send raw payload under `payload` key
        worker.postMessage({ type: 'task', id: task.id, payload: task.payload });
      }
    }
  }

  /** Remove a worker from both the active and idle lists. */
  _removeWorker(worker) {
    this.workers = this.workers.filter((w) => w !== worker);
    this.idleWorkers = this.idleWorkers.filter((w) => w !== worker);
    // If all workers have crashed, reject any pending tasks rather than hanging.
    if (this.workers.length === 0 && this._initialized && this.pendingTasks.length > 0) {
      console.error('WorkerPool: all workers have crashed — rejecting all pending tasks');
      const pending = this.pendingTasks.drain();
      for (const task of pending) {
        const cb = this.taskCallbacks.get(task.id);
        if (cb) {
          this.taskCallbacks.delete(task.id);
          cb.reject(new Error('WorkerPool: all workers have crashed'));
        }
      }
    }
  }
}

