/**
 * Tests for Embedder.getBestConfig() fallback chain and embed() prefix option.
 *
 * getBestConfig() has three levels of fallback:
 *   1. Explicit profilePath → applyPerfProfile
 *   2. Latest bench/grid-results*.json → applyPerfProfile
 *   3. Neither present → initialPerformanceCheckup
 *
 * Static methods are stubbed via direct assignment and restored in finally
 * blocks so tests remain self-contained.
 */

import { test, describe, mock } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs/promises';
import fsSync from 'fs';
import { join } from 'path';
import { tmpdir } from 'os';
import { Embedder } from '../src/embedder.js';

// ── getBestConfig — fallback chain ────────────────────────────────────────────

describe('Embedder.getBestConfig — fallback chain', () => {
  test('uses an explicit valid profilePath when provided', async () => {
    const tmp = await fs.mkdtemp(join(tmpdir(), 'embedeer-cfg-'));
    try {
      const profile = {
        results: [
          {
            success: true, textsPerSec: 150, batchSize: 48, concurrency: 3,
            dtype: 'q8', provider: 'cpu', device: 'cpu',
          },
        ],
      };
      const p = join(tmp, 'profile.json');
      fsSync.writeFileSync(p, JSON.stringify(profile));

      const cfg = await Embedder.getBestConfig({ profilePath: p, device: 'cpu' });
      assert.equal(cfg.batchSize, 48);
      assert.equal(cfg.concurrency, 3);
      assert.equal(cfg.dtype, 'q8');
    } finally {
      await fs.rm(tmp, { recursive: true, force: true });
    }
  });

  test('falls back to grid result file when no profilePath given', async () => {
    const tmp = await fs.mkdtemp(join(tmpdir(), 'embedeer-cfg-'));
    const origFindLatest = Embedder.findLatestGridResult;
    try {
      const profile = {
        results: [
          {
            success: true, textsPerSec: 200, batchSize: 77, concurrency: 2,
            dtype: 'fp16', provider: 'cpu', device: 'cpu',
          },
        ],
      };
      const gridFile = join(tmp, 'grid-results-cpu-123.json');
      fsSync.writeFileSync(gridFile, JSON.stringify(profile));

      // Point findLatestGridResult at our synthetic file
      Embedder.findLatestGridResult = () => gridFile;

      const cfg = await Embedder.getBestConfig({ device: 'cpu' });
      assert.equal(cfg.batchSize, 77);
      assert.equal(cfg.concurrency, 2);
    } finally {
      Embedder.findLatestGridResult = origFindLatest;
      await fs.rm(tmp, { recursive: true, force: true });
    }
  });

  test('falls through to initialPerformanceCheckup when no profile and no grid result', async () => {
    const origFindLatest = Embedder.findLatestGridResult;
    const origCheckup = Embedder.initialPerformanceCheckup;
    try {
      Embedder.findLatestGridResult = () => null;
      Embedder.initialPerformanceCheckup = async () => ({
        device: 'cpu', batchSize: 9, concurrency: 1, dtype: 'fp32',
      });

      const cfg = await Embedder.getBestConfig({ device: 'cpu' });
      assert.equal(cfg.batchSize, 9);
      assert.equal(cfg.concurrency, 1);
    } finally {
      Embedder.findLatestGridResult = origFindLatest;
      Embedder.initialPerformanceCheckup = origCheckup;
    }
  });

  test('falls through to initialPerformanceCheckup when profilePath is invalid', async () => {
    const origFindLatest = Embedder.findLatestGridResult;
    const origCheckup = Embedder.initialPerformanceCheckup;
    let checkupCalled = false;
    try {
      Embedder.findLatestGridResult = () => null;
      Embedder.initialPerformanceCheckup = async () => {
        checkupCalled = true;
        return { device: 'cpu', batchSize: 32, concurrency: 1, dtype: 'fp32' };
      };

      // Non-existent path — applyPerfProfile will throw
      await Embedder.getBestConfig({
        profilePath: '/nonexistent/profile.json',
        device: 'cpu',
      });
      assert.ok(checkupCalled, 'should have fallen through to initialPerformanceCheckup');
    } finally {
      Embedder.findLatestGridResult = origFindLatest;
      Embedder.initialPerformanceCheckup = origCheckup;
    }
  });

  test('picks highest textsPerSec from grid file when multiple results exist', async () => {
    const tmp = await fs.mkdtemp(join(tmpdir(), 'embedeer-cfg-'));
    const origFindLatest = Embedder.findLatestGridResult;
    try {
      const profile = {
        results: [
          { success: true, textsPerSec: 50,  batchSize: 16, concurrency: 1, dtype: 'fp32', provider: 'cpu', device: 'cpu' },
          { success: true, textsPerSec: 300, batchSize: 64, concurrency: 4, dtype: 'fp16', provider: 'cpu', device: 'cpu' },
          { success: true, textsPerSec: 100, batchSize: 32, concurrency: 2, dtype: 'fp32', provider: 'cpu', device: 'cpu' },
        ],
      };
      const gridFile = join(tmp, 'grid-results.json');
      fsSync.writeFileSync(gridFile, JSON.stringify(profile));
      Embedder.findLatestGridResult = () => gridFile;

      const cfg = await Embedder.getBestConfig({ device: 'cpu' });
      assert.equal(cfg.batchSize, 64, 'should pick the fastest config');
      assert.equal(cfg.concurrency, 4);
    } finally {
      Embedder.findLatestGridResult = origFindLatest;
      await fs.rm(tmp, { recursive: true, force: true });
    }
  });
});

// ── embed() — prefix option ───────────────────────────────────────────────────

describe('Embedder.embed — prefix option', () => {
  /** Stub pool that records what texts it receives. */
  function makeRecordingPool() {
    const received = [];
    return {
      _initialized: true,
      initialize: async () => {},
      run: async (texts) => {
        received.push([...texts]);
        return texts.map(() => [0]);
      },
      destroy: async () => {},
      received,
    };
  }

  async function makeEmbedder(pool, options = {}) {
    const e = new Embedder('test-model', { batchSize: 100, ...options });
    e._pool = pool;
    return e;
  }

  test('prepends prefix to every text when prefix is provided', async () => {
    const pool = makeRecordingPool();
    const embedder = await makeEmbedder(pool);
    await embedder.embed(['hello', 'world'], { prefix: 'search_query: ' });

    assert.equal(pool.received.length, 1);
    assert.deepEqual(pool.received[0], ['search_query: hello', 'search_query: world']);
  });

  test('passes texts unchanged when no prefix is given', async () => {
    const pool = makeRecordingPool();
    const embedder = await makeEmbedder(pool);
    await embedder.embed(['hello', 'world']);

    assert.deepEqual(pool.received[0], ['hello', 'world']);
  });

  test('passes texts unchanged when prefix is empty string', async () => {
    const pool = makeRecordingPool();
    const embedder = await makeEmbedder(pool);
    await embedder.embed(['hello', 'world'], { prefix: '' });

    // Empty string is falsy — treated same as no prefix
    assert.deepEqual(pool.received[0], ['hello', 'world']);
  });

  test('prefix is applied before batching', async () => {
    const pool = makeRecordingPool();
    const embedder = await makeEmbedder(pool, { batchSize: 2 });
    await embedder.embed(['a', 'b', 'c'], { prefix: 'p:' });

    // 3 texts with batchSize 2 → 2 batches, all with prefix
    const allTexts = pool.received.flat();
    assert.deepEqual(allTexts, ['p:a', 'p:b', 'p:c']);
  });
});
