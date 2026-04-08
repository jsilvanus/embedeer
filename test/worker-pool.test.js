/**
 * Unit tests for WorkerPool
 *
 * A lightweight FakeWorker is injected via the _WorkerClass option so no
 * real model is loaded and no actual worker threads are spawned.
 */

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'events';
import { WorkerPool } from '../src/worker-pool.js';

// ── FakeWorker ───────────────────────────────────────────────────────────────

/**
 * A minimal Worker substitute.
 *
 * Behaviour:
 *   - Emits { type: 'ready' } after construction (via setImmediate).
 *   - Responds to postMessage({ id, texts }) with
 *     { type: 'result', id, embeddings: [[0.1, 0.2, 0.3], ...] }.
 *   - terminate() resolves immediately.
 */
class FakeWorker extends EventEmitter {
  constructor(_path, _opts) {
    super();
    this._terminated = false;
    // Simulate async model loading then ready signal
    setImmediate(() => this.emit('message', { type: 'ready' }));
  }

  postMessage({ type, id, texts }) {
    if (this._terminated) return;
    if (type === 'task') {
      setImmediate(() => {
        const embeddings = texts.map(() => [0.1, 0.2, 0.3]);
        this.emit('message', { type: 'result', id, embeddings });
      });
    }
  }

  async terminate() {
    this._terminated = true;
    setImmediate(() => this.emit('exit', 0));
  }
}

/** Creates a WorkerPool wired to FakeWorker. */
function makePool(opts = {}) {
  return new WorkerPool('test-model', { _WorkerClass: FakeWorker, ...opts });
}

// ── Tests ────────────────────────────────────────────────────────────────────

describe('WorkerPool', async () => {
  test('initialize() starts the requested number of workers', async () => {
    const pool = makePool({ poolSize: 3 });
    await pool.initialize();
    assert.equal(pool.workers.length, 3);
    assert.equal(pool._initialized, true);
    await pool.destroy();
  });

  test('initialize() is idempotent', async () => {
    const pool = makePool({ poolSize: 1 });
    await pool.initialize();
    await pool.initialize(); // second call should be a no-op
    assert.equal(pool.workers.length, 1);
    await pool.destroy();
  });

  test('run() rejects before initialize()', async () => {
    const pool = makePool({ poolSize: 1 });
    await assert.rejects(() => pool.run(['hello']), /initialize/);
  });

  test('run() returns one embedding vector per text', async () => {
    const pool = makePool({ poolSize: 1 });
    await pool.initialize();
    const result = await pool.run(['hello', 'world', 'foo']);
    assert.equal(result.length, 3);
    for (const vec of result) {
      assert.deepEqual(vec, [0.1, 0.2, 0.3]);
    }
    await pool.destroy();
  });

  test('run() handles multiple concurrent tasks', async () => {
    const pool = makePool({ poolSize: 2 });
    await pool.initialize();
    // Submit 4 tasks simultaneously
    const results = await Promise.all([
      pool.run(['a', 'b']),
      pool.run(['c']),
      pool.run(['d', 'e', 'f']),
      pool.run(['g']),
    ]);
    assert.equal(results[0].length, 2);
    assert.equal(results[1].length, 1);
    assert.equal(results[2].length, 3);
    assert.equal(results[3].length, 1);
    await pool.destroy();
  });

  test('destroy() terminates all workers and resets state', async () => {
    const pool = makePool({ poolSize: 2 });
    await pool.initialize();
    await pool.destroy();
    assert.equal(pool.workers.length, 0);
    assert.equal(pool._initialized, false);
  });

  test('worker error is surfaced as a rejected promise', async () => {
    class ErrorWorker extends EventEmitter {
      constructor() {
        super();
        setImmediate(() => this.emit('message', { type: 'ready' }));
      }
      postMessage({ type, id }) {
        if (type === 'task') {
          setImmediate(() =>
            this.emit('message', { type: 'error', id, error: 'kaboom' }),
          );
        }
      }
      async terminate() {}
    }

    const pool = new WorkerPool('test-model', { _WorkerClass: ErrorWorker, poolSize: 1 });
    await pool.initialize();
    await assert.rejects(() => pool.run(['text']), { message: 'kaboom' });
    await pool.destroy();
  });

  test('a crashing worker rejects only its task; the pool keeps running', async () => {
    // CrashWorker: completes the first task normally, crashes on the second.
    let taskCount = 0;
    class CrashWorker extends EventEmitter {
      constructor() {
        super();
        setImmediate(() => this.emit('message', { type: 'ready' }));
      }
      postMessage({ type, id, texts }) {
        if (type !== 'task') return;
        taskCount++;
        if (taskCount === 1) {
          // First task succeeds
          setImmediate(() =>
            this.emit('message', { type: 'result', id, embeddings: texts.map(() => [1]) }),
          );
        } else {
          // Second task: worker crashes
          setImmediate(() => this.emit('exit', 1));
        }
      }
      async terminate() { this.emit('exit', 0); }
    }

    const pool = new WorkerPool('test-model', { _WorkerClass: CrashWorker, poolSize: 1 });
    await pool.initialize();

    // First task should succeed
    const first = await pool.run(['hello']);
    assert.deepEqual(first, [[1]]);

    // Second task triggers the worker crash — must reject, not crash the process
    await assert.rejects(
      () => pool.run(['world']),
      /Worker process exited unexpectedly/,
    );

    // Pool is still usable (no exception thrown, process still alive)
    assert.equal(pool._initialized, true);
    await pool.destroy();
  });
});

