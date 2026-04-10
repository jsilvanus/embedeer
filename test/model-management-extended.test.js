/**
 * Extended tests for model-management.js covering:
 *   - isModelDownloaded() fuzzy (substring) fallback
 *   - getCachedModels() size and mtime computation
 *   - listModels() directory-only filtering
 *
 * All tests use temporary directories and do not touch the real cache.
 */

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs/promises';
import { join } from 'path';
import { tmpdir } from 'os';
import {
  isModelDownloaded,
  listModels,
  getCachedModels,
} from '../src/model-management.js';

// ── isModelDownloaded — fuzzy fallback ───────────────────────────────────────

describe('isModelDownloaded — fuzzy fallback', () => {
  test('returns true when a directory name contains the model name', async () => {
    const tmp = await fs.mkdtemp(join(tmpdir(), 'embedeer-mm-'));
    try {
      // Exact dir "nomic-embed-text" does NOT exist, but a variant does.
      await fs.mkdir(join(tmp, 'nomic-embed-text-q4'), { recursive: true });
      const found = await isModelDownloaded('nomic-embed-text', { cacheDir: tmp });
      assert.equal(found, true);
    } finally {
      await fs.rm(tmp, { recursive: true, force: true });
    }
  });

  test('returns true for exact directory name match', async () => {
    const tmp = await fs.mkdtemp(join(tmpdir(), 'embedeer-mm-'));
    try {
      await fs.mkdir(join(tmp, 'all-MiniLM-L6-v2'), { recursive: true });
      const found = await isModelDownloaded('all-MiniLM-L6-v2', { cacheDir: tmp });
      assert.equal(found, true);
    } finally {
      await fs.rm(tmp, { recursive: true, force: true });
    }
  });

  test('returns false when no directory name matches', async () => {
    const tmp = await fs.mkdtemp(join(tmpdir(), 'embedeer-mm-'));
    try {
      await fs.mkdir(join(tmp, 'unrelated-model'), { recursive: true });
      const found = await isModelDownloaded('no-such-model', { cacheDir: tmp });
      assert.equal(found, false);
    } finally {
      await fs.rm(tmp, { recursive: true, force: true });
    }
  });

  test('returns false when the cache directory is empty', async () => {
    const tmp = await fs.mkdtemp(join(tmpdir(), 'embedeer-mm-'));
    try {
      const found = await isModelDownloaded('any-model', { cacheDir: tmp });
      assert.equal(found, false);
    } finally {
      await fs.rm(tmp, { recursive: true, force: true });
    }
  });
});

// ── getCachedModels — size and mtime ─────────────────────────────────────────

describe('getCachedModels — size and mtime', () => {
  test('reports total size including all files in the model directory', async () => {
    const tmp = await fs.mkdtemp(join(tmpdir(), 'embedeer-mm-'));
    try {
      const modelDir = join(tmp, 'size-test-model');
      await fs.mkdir(modelDir, { recursive: true });
      // Write files with known sizes
      await fs.writeFile(join(modelDir, 'weights.bin'), Buffer.alloc(1024));
      await fs.writeFile(join(modelDir, 'config.json'), Buffer.alloc(256));

      const models = await getCachedModels({ cacheDir: tmp });
      const model = models.find((m) => m.name === 'size-test-model');
      assert.ok(model, 'model should appear in getCachedModels()');
      assert.ok(
        model.size >= 1024 + 256,
        `expected size >= 1280, got ${model.size}`,
      );
    } finally {
      await fs.rm(tmp, { recursive: true, force: true });
    }
  });

  test('accumulates size recursively across nested subdirectories', async () => {
    const tmp = await fs.mkdtemp(join(tmpdir(), 'embedeer-mm-'));
    try {
      const modelDir = join(tmp, 'nested-model');
      await fs.mkdir(join(modelDir, 'onnx'), { recursive: true });
      await fs.writeFile(join(modelDir, 'config.json'), Buffer.alloc(100));
      await fs.writeFile(join(modelDir, 'onnx', 'model.onnx'), Buffer.alloc(500));

      const models = await getCachedModels({ cacheDir: tmp });
      const model = models.find((m) => m.name === 'nested-model');
      assert.ok(model);
      assert.ok(model.size >= 600, `expected size >= 600, got ${model.size}`);
    } finally {
      await fs.rm(tmp, { recursive: true, force: true });
    }
  });

  test('reports mtime as an ISO 8601 string', async () => {
    const tmp = await fs.mkdtemp(join(tmpdir(), 'embedeer-mm-'));
    try {
      await fs.mkdir(join(tmp, 'mtime-model'), { recursive: true });

      const models = await getCachedModels({ cacheDir: tmp });
      const model = models.find((m) => m.name === 'mtime-model');
      assert.ok(model);
      assert.equal(typeof model.mtime, 'string', 'mtime should be a string');
      // ISO 8601 strings contain 'T' between date and time parts
      assert.ok(model.mtime.includes('T'), 'mtime should be ISO 8601');
      // Must be parseable as a real date
      const parsed = new Date(model.mtime);
      assert.ok(!isNaN(parsed.getTime()), 'mtime should parse to a valid Date');
    } finally {
      await fs.rm(tmp, { recursive: true, force: true });
    }
  });

  test('returns empty array for an empty cache directory', async () => {
    const tmp = await fs.mkdtemp(join(tmpdir(), 'embedeer-mm-'));
    try {
      const models = await getCachedModels({ cacheDir: tmp });
      assert.deepEqual(models, []);
    } finally {
      await fs.rm(tmp, { recursive: true, force: true });
    }
  });

  test('includes path property pointing inside the cache dir', async () => {
    const tmp = await fs.mkdtemp(join(tmpdir(), 'embedeer-mm-'));
    try {
      await fs.mkdir(join(tmp, 'path-model'), { recursive: true });
      const models = await getCachedModels({ cacheDir: tmp });
      const model = models.find((m) => m.name === 'path-model');
      assert.ok(model);
      assert.ok(model.path.startsWith(tmp), 'path should be inside cacheDir');
    } finally {
      await fs.rm(tmp, { recursive: true, force: true });
    }
  });
});

// ── listModels — directory filtering ─────────────────────────────────────────

describe('listModels — directory filtering', () => {
  test('returns only directory names, not files', async () => {
    const tmp = await fs.mkdtemp(join(tmpdir(), 'embedeer-mm-'));
    try {
      await fs.mkdir(join(tmp, 'model-a'), { recursive: true });
      await fs.mkdir(join(tmp, 'model-b'), { recursive: true });
      await fs.writeFile(join(tmp, 'readme.txt'), 'not a model');
      await fs.writeFile(join(tmp, 'config.json'), '{}');

      const models = await listModels({ cacheDir: tmp });
      assert.ok(Array.isArray(models));
      assert.ok(models.includes('model-a'), 'should include model-a directory');
      assert.ok(models.includes('model-b'), 'should include model-b directory');
      assert.ok(!models.includes('readme.txt'), 'should not include files');
      assert.ok(!models.includes('config.json'), 'should not include files');
    } finally {
      await fs.rm(tmp, { recursive: true, force: true });
    }
  });

  test('returns empty array when cache contains no directories', async () => {
    const tmp = await fs.mkdtemp(join(tmpdir(), 'embedeer-mm-'));
    try {
      await fs.writeFile(join(tmp, 'just-a-file.txt'), 'data');
      const models = await listModels({ cacheDir: tmp });
      assert.deepEqual(models, []);
    } finally {
      await fs.rm(tmp, { recursive: true, force: true });
    }
  });
});
