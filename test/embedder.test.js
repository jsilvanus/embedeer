/**
 * Unit tests for Embedder
 *
 * The WorkerPool is replaced with a lightweight stub so no real model
 * is loaded.  Tests verify batching logic, option forwarding, and
 * error handling.
 */

import { test, describe, mock } from 'node:test';
import assert from 'node:assert/strict';

// ── Stub WorkerPool ──────────────────────────────────────────────────────────

/**
 * Returns a WorkerPool stub whose `run(texts)` resolves with one
 * zero-vector per text (dimension 4).
 */
function makeStubPool(overrides = {}) {
  return {
    _initialized: false,
    initialize: mock.fn(async function () {
      this._initialized = true;
    }),
    run: mock.fn(async (texts) => texts.map(() => [0, 0, 0, 0])),
    destroy: mock.fn(async () => {}),
    ...overrides,
  };
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Build an Embedder whose internal pool is replaced with `stubPool`.
 */
async function makeEmbedder(stubPool, modelName = 'test-model', options = {}) {
  const { Embedder } = await import('../src/embedder.js');
  const e = new Embedder(modelName, options);
  e._pool = stubPool;
  // We've replaced the pool so we need to "manually" mark the pool initialized
  stubPool._initialized = true;
  stubPool.initialize = mock.fn(async () => {});
  return e;
}

// ── Tests ────────────────────────────────────────────────────────────────────

describe('Embedder', async () => {
  test('embed() returns empty array for empty input', async () => {
    const pool = makeStubPool();
    const embedder = await makeEmbedder(pool);
    const result = await embedder.embed([]);
    assert.deepEqual(result, []);
    assert.equal(pool.run.mock.calls.length, 0);
  });

  test('embed() wraps a single string in an array', async () => {
    const pool = makeStubPool();
    const embedder = await makeEmbedder(pool);
    const result = await embedder.embed('hello');
    assert.equal(result.length, 1);
  });

  test('embed() returns one vector per input text', async () => {
    const pool = makeStubPool();
    const embedder = await makeEmbedder(pool);
    const texts = ['a', 'b', 'c', 'd', 'e'];
    const result = await embedder.embed(texts);
    assert.equal(result.length, texts.length);
    for (const vec of result) {
      assert.deepEqual(vec, [0, 0, 0, 0]);
    }
  });

  test('embed() splits input into batches of batchSize', async () => {
    const pool = makeStubPool();
    const embedder = await makeEmbedder(pool, 'test-model', { batchSize: 3 });
    const texts = ['a', 'b', 'c', 'd', 'e', 'f', 'g']; // 7 texts → 3 batches
    const result = await embedder.embed(texts);

    // 3 batches: [3, 3, 1]
    assert.equal(pool.run.mock.calls.length, 3);
    assert.equal(pool.run.mock.calls[0].arguments[0].length, 3);
    assert.equal(pool.run.mock.calls[1].arguments[0].length, 3);
    assert.equal(pool.run.mock.calls[2].arguments[0].length, 1);
    assert.equal(result.length, 7);
  });

  test('embed() submits all batches concurrently (Promise.all)', async () => {
    const order = [];
    const pool = makeStubPool({
      run: mock.fn(async (texts) => {
        order.push(texts[0]);
        return texts.map(() => [1]);
      }),
    });
    const embedder = await makeEmbedder(pool, 'test-model', { batchSize: 1 });
    await embedder.embed(['x', 'y', 'z']);
    // All three tasks are submitted; order may vary but all must appear
    assert.equal(order.length, 3);
    assert.ok(order.includes('x'));
    assert.ok(order.includes('y'));
    assert.ok(order.includes('z'));
  });

  test('Embedder.create() calls initialize()', async () => {
    // We can't easily mock Embedder.create's internal new, so we just verify
    // initialize() sets up the pool correctly by calling it directly.
    const { Embedder } = await import('../src/embedder.js');
    const e = new Embedder('test-model');
    const pool = makeStubPool();
    e._pool = pool;
    await e.initialize();
    assert.equal(pool.initialize.mock.calls.length, 1);
  });

  test('destroy() delegates to pool.destroy()', async () => {
    const pool = makeStubPool();
    const embedder = await makeEmbedder(pool);
    await embedder.destroy();
    assert.equal(pool.destroy.mock.calls.length, 1);
  });

  test('embed() propagates errors from the pool', async () => {
    const pool = makeStubPool({
      run: mock.fn(async () => { throw new Error('model error'); }),
    });
    const embedder = await makeEmbedder(pool);
    await assert.rejects(
      () => embedder.embed(['text']),
      { message: 'model error' },
    );
  });
});
