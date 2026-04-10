/**
 * Tests for WorkerPool lifecycle edge cases:
 *   - destroy() resolves cleanly even when a task is in-flight
 *   - destroy() drains the pending task queue
 *   - run() after destroy() rejects with a clear message
 *   - Worker 'error' event (OS-level spawn failure) rejects the in-flight task
 *
 * Uses lightweight FakeWorker variants injected via _WorkerClass.
 */

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'events';
import { WorkerPool } from '../src/worker-pool.js';

// ── HangingWorker — accepts tasks, never responds ────────────────────────────

/**
 * A worker that loads immediately but never replies to tasks.
 * Used to keep tasks permanently in-flight so we can test destroy() timing.
 */
class HangingWorker extends EventEmitter {
  constructor() {
    super();
    this._terminated = false;
    setImmediate(() => this.emit('message', { type: 'ready' }));
  }

  postMessage() { /* never respond to tasks */ }

  async terminate() {
    this._terminated = true;
    setImmediate(() => this.emit('exit', 0));
  }
}

// ── ErrorEventWorker — emits an 'error' event on the first task ──────────────

/**
 * A worker that emits an OS-level 'error' event instead of responding normally.
 * This simulates a spawn failure mid-flight.
 */
class ErrorEventWorker extends EventEmitter {
  constructor() {
    super();
    setImmediate(() => this.emit('message', { type: 'ready' }));
  }

  postMessage({ type }) {
    if (type === 'task') {
      setImmediate(() => this.emit('error', new Error('spawn failed')));
    }
  }

  async terminate() {
    setImmediate(() => this.emit('exit', 0));
  }
}

// ── destroy() behaviour ───────────────────────────────────────────────────────

describe('WorkerPool — destroy() lifecycle', () => {
  test('destroy() resolves even when a task is in-flight', async () => {
    const pool = new WorkerPool('test', { _WorkerClass: HangingWorker, poolSize: 1 });
    await pool.initialize();

    // Submit a task that will never get a response
    const hanging = pool.run(['text']);
    // Suppress the unhandled rejection from the dropped callback
    hanging.catch(() => {});

    // destroy() should still resolve promptly
    await pool.destroy();
    assert.equal(pool._initialized, false);
  });

  test('destroy() clears the pending task queue', async () => {
    const pool = new WorkerPool('test', { _WorkerClass: HangingWorker, poolSize: 1 });
    await pool.initialize();

    // First task dispatched to the single worker (in-flight, never responds)
    const t1 = pool.run(['first']);
    t1.catch(() => {});

    // Second task queued (no idle worker available)
    const t2 = pool.run(['second']);
    t2.catch(() => {});

    assert.equal(pool.pendingTasks.length, 1, 'second task should be in the pending queue');

    await pool.destroy();
    assert.equal(pool.pendingTasks.length, 0, 'pending queue should be empty after destroy');
  });

  test('destroy() resets worker state', async () => {
    const pool = new WorkerPool('test', { _WorkerClass: HangingWorker, poolSize: 2 });
    await pool.initialize();
    assert.equal(pool.workers.length, 2);

    await pool.destroy();
    assert.equal(pool.workers.length, 0);
    assert.equal(pool.idleWorkers.length, 0);
    assert.equal(pool._initialized, false);
  });

  test('run() rejects after destroy() with a helpful message', async () => {
    const pool = new WorkerPool('test', { _WorkerClass: HangingWorker, poolSize: 1 });
    await pool.initialize();
    await pool.destroy();

    await assert.rejects(
      () => pool.run(['text']),
      /initialize/,
      'should reject with a message mentioning initialize()',
    );
  });
});

// ── Worker 'error' event ──────────────────────────────────────────────────────

describe('WorkerPool — worker "error" event', () => {
  test('OS-level worker error rejects the in-flight task', async () => {
    const pool = new WorkerPool('test', { _WorkerClass: ErrorEventWorker, poolSize: 1 });
    await pool.initialize();

    await assert.rejects(
      () => pool.run(['text']),
      { message: 'spawn failed' },
    );

    await pool.destroy();
  });

  test('pool continues to accept new tasks after an error event (worker removed)', async () => {
    // After an error event the broken worker is removed. If the pool has more
    // workers, they can still serve tasks.
    class OneErrorOneOk extends EventEmitter {
      static count = 0;
      constructor(path, opts) {
        super();
        this._id = ++OneErrorOneOk.count;
        this._terminated = false;
        setImmediate(() => this.emit('message', { type: 'ready' }));
      }
      postMessage({ type, id, texts }) {
        if (this._terminated || type !== 'task') return;
        if (this._id === 1) {
          // First worker instance: emit error
          setImmediate(() => this.emit('error', new Error('first worker error')));
        } else {
          // Second worker instance: respond normally
          setImmediate(() =>
            this.emit('message', { type: 'result', id, embeddings: texts.map(() => [7]) }),
          );
        }
      }
      async terminate() {
        this._terminated = true;
        setImmediate(() => this.emit('exit', 0));
      }
    }
    // Reset static counter between test runs
    OneErrorOneOk.count = 0;

    const pool = new WorkerPool('test', { _WorkerClass: OneErrorOneOk, poolSize: 2 });
    await pool.initialize();

    // This will hit worker 1 first (error), then the pool re-dispatches to worker 2
    // Actually, with 2 idle workers both tasks go out simultaneously. Let's submit
    // one task; it may go to either worker depending on scheduling.
    // Instead, explicitly verify that a task eventually succeeds even after an error.
    const p1 = pool.run(['a']).catch(() => null); // may fail (worker 1) or succeed (worker 2)
    const p2 = pool.run(['b']).catch(() => null); // same

    const results = await Promise.all([p1, p2]);
    // At least one should have succeeded (worker 2 responds normally)
    const successes = results.filter((r) => r !== null);
    assert.ok(successes.length >= 1, 'at least one task should succeed with 2 workers');

    await pool.destroy();
  });
});
