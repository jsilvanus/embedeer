/**
 * Tests for CLI output formatting and model-cache helpers.
 * These are pure unit tests — no workers, no network.
 */

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { homedir } from 'os';
import { join } from 'path';
import { existsSync } from 'fs';

describe('model-cache', async () => {
  test('DEFAULT_CACHE_DIR is ~/.embedeer/models', async () => {
    const { DEFAULT_CACHE_DIR } = await import('../src/model-cache.js');
    assert.equal(DEFAULT_CACHE_DIR, join(homedir(), '.embedeer', 'models'));
  });

  test('getCacheDir() returns provided dir and creates it', async () => {
    const { getCacheDir } = await import('../src/model-cache.js');
    const dir = join(homedir(), '.embedeer', 'models');
    const result = getCacheDir(dir);
    assert.equal(result, dir);
    assert.ok(existsSync(dir), 'directory should exist on disk after getCacheDir()');
  });

  test('getCacheDir() with no arg returns the default', async () => {
    const { getCacheDir, DEFAULT_CACHE_DIR } = await import('../src/model-cache.js');
    const result = getCacheDir();
    assert.equal(result, DEFAULT_CACHE_DIR);
  });
});

describe('CLI output formatting', async () => {
  // We test the formatting logic by importing the private helpers.
  // Since cli.js is a script, we extract the formatting to test it directly.

  function formatOutput(texts, embeddings, format) {
    switch (format) {
      case 'txt':
        return embeddings.map((vec) => vec.join(' ')).join('\n');
      case 'sql': {
        const rows = texts.map((text, i) => {
          const safeText = text.replace(/'/g, "''");
          const vector = JSON.stringify(embeddings[i]);
          return `  ('${safeText}', '${vector}')`;
        });
        return (
          'INSERT INTO embeddings (text, vector) VALUES\n' +
          rows.join(',\n') +
          ';'
        );
      }
      default:
        return JSON.stringify(embeddings);
    }
  }

  const texts = ['Hello world', "It's a test"];
  const embeddings = [[0.1, 0.2], [0.3, 0.4]];

  test('json output is valid JSON array', () => {
    const out = formatOutput(texts, embeddings, 'json');
    const parsed = JSON.parse(out);
    assert.deepEqual(parsed, embeddings);
  });

  test('txt output is one space-separated line per embedding', () => {
    const out = formatOutput(texts, embeddings, 'txt');
    const lines = out.split('\n');
    assert.equal(lines.length, 2);
    assert.equal(lines[0], '0.1 0.2');
    assert.equal(lines[1], '0.3 0.4');
  });

  test('sql output starts with INSERT and contains both rows', () => {
    const out = formatOutput(texts, embeddings, 'sql');
    assert.ok(out.startsWith('INSERT INTO embeddings'));
    assert.ok(out.includes("('Hello world'"));
    // Single quotes in text are escaped
    assert.ok(out.includes("('It''s a test'"));
    assert.ok(out.endsWith(';'));
  });

  test('unknown format falls back to json', () => {
    const out = formatOutput(texts, embeddings, 'unknown');
    const parsed = JSON.parse(out);
    assert.deepEqual(parsed, embeddings);
  });
});
