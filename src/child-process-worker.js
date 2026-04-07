/**
 * ChildProcessWorker — wraps a forked Node.js child process in the same
 * EventEmitter interface used by the WorkerPool, making it a drop-in
 * replacement for worker_threads.Worker.
 *
 * Running the model in a separate OS process means:
 *   - A crash (segfault, OOM, unhandled exception) in the child does NOT
 *     propagate to the parent; the parent process keeps running.
 *   - Each worker has its own heap and cannot corrupt the parent's memory.
 *
 * Interface (mirrors worker_threads.Worker):
 *   constructor(scriptPath, { workerData })  — starts the child and sends init
 *   postMessage(msg)                          — sends a message via IPC
 *   terminate()                               — kills the child, returns Promise
 *   on('message' | 'error' | 'exit', fn)     — standard EventEmitter
 */

import { fork } from 'child_process';
import { EventEmitter } from 'events';

export class ChildProcessWorker extends EventEmitter {
  /**
   * @param {string} scriptPath  Absolute path to the worker script.
   * @param {object} [opts]
   * @param {object} [opts.workerData]  Config object forwarded to the child
   *                                    as an IPC `init` message.
   */
  constructor(scriptPath, { workerData } = {}) {
    super();

    this._child = fork(scriptPath);

    // Forward IPC messages to callers as 'message' events.
    this._child.on('message', (msg) => this.emit('message', msg));

    // Forward OS-level errors (e.g. ENOENT).
    this._child.on('error', (err) => this.emit('error', err));

    // Forward exit so the pool can detect crashes (non-zero code).
    this._child.on('exit', (code, signal) => {
      // Normalise: signal-killed processes have code=null; treat as code 1.
      this.emit('exit', code ?? (signal ? 1 : 0));
    });

    // Bootstrap the child with its configuration.
    if (workerData) {
      this._child.send({ type: 'init', ...workerData });
    }
  }

  /**
   * Send a message to the child process via IPC.
   * @param {object} msg
   */
  postMessage(msg) {
    this._child.send(msg);
  }

  /**
   * Terminate the child process and resolve when it has exited.
   * @returns {Promise<void>}
   */
  terminate() {
    return new Promise((resolve) => {
      if (!this._child.connected && this._child.exitCode !== null) {
        resolve();
        return;
      }
      this._child.once('exit', () => resolve());
      this._child.kill('SIGTERM');
    });
  }
}
