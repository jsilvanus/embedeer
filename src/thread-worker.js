/**
 * ThreadWorker — wraps a worker_threads.Worker in the same EventEmitter
 * interface used by WorkerPool and ChildProcessWorker.
 *
 * Use this when you want parallel execution within the same OS process
 * (lower memory overhead, shared heap) rather than full process isolation.
 * A thread crash will propagate to the parent process; use ChildProcessWorker
 * if crash isolation is required.
 *
 * Interface (mirrors ChildProcessWorker):
 *   constructor(scriptPath, { workerData })
 *   postMessage(msg)
 *   terminate()           — returns Promise<void>
 *   on('message'|'error'|'exit', fn)
 */

import { Worker } from 'worker_threads';
import { EventEmitter } from 'events';

export class ThreadWorker extends EventEmitter {
  /**
   * @param {string} scriptPath  Absolute path to the worker thread script.
   * @param {object} [opts]
   * @param {object} [opts.workerData]  Passed directly to the worker thread
   *                                    so the thread can read it via workerData.
   */
  constructor(scriptPath, { workerData } = {}) {
    super();

    this._worker = new Worker(scriptPath, { workerData });

    this._worker.on('message', (msg) => this.emit('message', msg));
    this._worker.on('error', (err) => this.emit('error', err));
    this._worker.on('exit', (code) => this.emit('exit', code));
  }

  /**
   * Send a message to the worker thread.
   * @param {object} msg
   */
  postMessage(msg) {
    this._worker.postMessage(msg);
  }

  /**
   * Terminate the worker thread and resolve when it has exited.
   * @returns {Promise<void>}
   */
  async terminate() {
    await this._worker.terminate();
  }
}
