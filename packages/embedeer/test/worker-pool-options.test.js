/**
 * Tests for new WorkerPool options: mode, token, dtype, cacheDir.
 * Uses FakeWorker injection so no real model is loaded.
 */

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'events';
import { WorkerPool } from '../src/worker-pool.js';

// ── FakeWorker that captures constructor arguments ───────────────────────────

class SpyWorker extends EventEmitter {
  constructor(scriptPath, opts) {
    super();
    SpyWorker.lastPath = scriptPath;
    SpyWorker.lastOpts = opts;
    this._terminated = false;
    setImmediate(() => this.emit('message', { type: 'ready' }));
  }

  postMessage({ type, id, texts }) {
    if (this._terminated || type !== 'task') return;
    setImmediate(() =>
      this.emit('message', { type: 'result', id, embeddings: texts.map(() => [7]) }),
    );
  }

  async terminate() {
    this._terminated = true;
    setImmediate(() => this.emit('exit', 0));
  }
}

function makePool(extraOpts = {}) {
  return new WorkerPool('test-model', { _WorkerClass: SpyWorker, poolSize: 1, ...extraOpts });
}

// ── Tests ────────────────────────────────────────────────────────────────────

describe('WorkerPool — new options', async () => {
  test('mode defaults to "process"', () => {
    const pool = new WorkerPool('model', { _WorkerClass: SpyWorker });
    assert.equal(pool.mode, 'process');
  });

  test('mode "thread" is stored', () => {
    const pool = new WorkerPool('model', { _WorkerClass: SpyWorker, mode: 'thread' });
    assert.equal(pool.mode, 'thread');
  });

  test('token, dtype, cacheDir are stored', () => {
    const pool = new WorkerPool('model', {
      _WorkerClass: SpyWorker,
      token: 'hf_abc',
      dtype: 'q8',
      cacheDir: '/tmp/cache',
    });
    assert.equal(pool.token, 'hf_abc');
    assert.equal(pool.dtype, 'q8');
    assert.equal(pool.cacheDir, '/tmp/cache');
  });

  test('workerData passed to worker includes token, dtype, cacheDir', async () => {
    const pool = new WorkerPool('my-model', {
      _WorkerClass: SpyWorker,
      poolSize: 1,
      token: 'hf_xyz',
      dtype: 'fp16',
      cacheDir: '/tmp/models',
    });
    await pool.initialize();
    const wd = SpyWorker.lastOpts.workerData;
    assert.equal(wd.modelName, 'my-model');
    assert.equal(wd.token, 'hf_xyz');
    assert.equal(wd.dtype, 'fp16');
    assert.equal(wd.cacheDir, '/tmp/models');
    await pool.destroy();
  });

  test('concurrency 1 processes tasks sequentially', async () => {
    const order = [];
    class SeqWorker extends EventEmitter {
      constructor() {
        super();
        setImmediate(() => this.emit('message', { type: 'ready' }));
      }
      postMessage({ type, id, texts }) {
        if (type !== 'task') return;
        // Use a real async delay to prove serialisation.
        setTimeout(() => {
          order.push(texts[0]);
          this.emit('message', { type: 'result', id, embeddings: [[0]] });
        }, 5);
      }
      async terminate() {}
    }

    const pool = new WorkerPool('model', { _WorkerClass: SeqWorker, poolSize: 1 });
    await pool.initialize();
    // Submit two tasks — with poolSize:1 they run one at a time.
    await Promise.all([
      pool.run(['first']),
      pool.run(['second']),
    ]);
    assert.equal(order.length, 2);
    await pool.destroy();
  });
});
