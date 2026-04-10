/**
 * Tests for CLI output formatting, input parsing, and model-cache helpers.
 * These are pure unit tests — no workers, no network.
 *
 * All helpers are imported directly from cli.js now that it is guarded by an
 * ESM entry-point check and will not invoke main() when imported.
 */

import { test, describe, mock } from 'node:test';
import assert from 'node:assert/strict';
import { homedir } from 'os';
import { join } from 'path';
import { existsSync, readFileSync, mkdtempSync, rmSync } from 'fs';
import { tmpdir } from 'os';

// Import the real helpers from cli.js instead of duplicating them here.
// The entry guard in cli.js ensures main() is not called on import.
const { parseDelimiter, parseTexts, formatOutput, writeOutput } = await import('../src/cli.js');

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

// ── parseDelimiter ───────────────────────────────────────────────────────────

describe('parseDelimiter', () => {
  test('leaves plain string unchanged', () => {
    assert.equal(parseDelimiter('|||'), '|||');
  });

  test('\\0 becomes null byte', () => {
    assert.equal(parseDelimiter('\\0'), '\0');
  });

  test('\\n becomes newline', () => {
    assert.equal(parseDelimiter('\\n'), '\n');
  });

  test('\\t becomes tab', () => {
    assert.equal(parseDelimiter('\\t'), '\t');
  });

  test('\\r becomes carriage return', () => {
    assert.equal(parseDelimiter('\\r'), '\r');
  });

  test('multiple escape sequences in one string', () => {
    assert.equal(parseDelimiter('\\r\\n'), '\r\n');
  });
});

// ── parseTexts ───────────────────────────────────────────────────────────────

describe('parseTexts', () => {
  test('JSON array is parsed directly', () => {
    const result = parseTexts('["a","b","c"]');
    assert.deepEqual(result, ['a', 'b', 'c']);
  });

  test('defaults to newline delimiter', () => {
    const result = parseTexts('foo\nbar\nbaz');
    assert.deepEqual(result, ['foo', 'bar', 'baz']);
  });

  test('custom delimiter (pipe)', () => {
    const result = parseTexts('foo|bar|baz', '|');
    assert.deepEqual(result, ['foo', 'bar', 'baz']);
  });

  test('custom delimiter (null byte)', () => {
    const result = parseTexts('foo\0bar\0baz', '\0');
    assert.deepEqual(result, ['foo', 'bar', 'baz']);
  });

  test('custom delimiter (tab)', () => {
    const result = parseTexts('foo\tbar\tbaz', '\t');
    assert.deepEqual(result, ['foo', 'bar', 'baz']);
  });

  test('filters empty strings after split', () => {
    const result = parseTexts('\nfoo\n\nbar\n', '\n');
    assert.deepEqual(result, ['foo', 'bar']);
  });

  test('JSON array takes precedence over delimiter parsing', () => {
    const result = parseTexts('["x","y"]', '|');
    assert.deepEqual(result, ['x', 'y']);
  });

  test('non-array JSON falls through to delimiter splitting', () => {
    // A JSON number is valid JSON but not an array — treated as raw text.
    const result = parseTexts('42');
    assert.deepEqual(result, ['42']);
  });

  test('JSON string (not array) falls through to delimiter splitting', () => {
    // '"hello"' is valid JSON but not an array.
    const result = parseTexts('"hello"');
    assert.deepEqual(result, ['"hello"']);
  });
});

// ── CLI output formatting ────────────────────────────────────────────────────

describe('CLI output formatting', () => {
  const texts = ['Hello world', "It's a test"];
  const embeddings = [[0.1, 0.2], [0.3, 0.4]];

  test('json output is valid JSON array of vectors', () => {
    const out = formatOutput(texts, embeddings, 'json');
    const parsed = JSON.parse(out);
    assert.deepEqual(parsed, embeddings);
  });

  test('json --with-text wraps each item with text field', () => {
    const out = formatOutput(texts, embeddings, 'json', true);
    const parsed = JSON.parse(out);
    assert.equal(parsed.length, 2);
    assert.equal(parsed[0].text, 'Hello world');
    assert.deepEqual(parsed[0].embedding, [0.1, 0.2]);
    assert.equal(parsed[1].text, "It's a test");
    assert.deepEqual(parsed[1].embedding, [0.3, 0.4]);
  });

  test('jsonl produces one JSON object per line', () => {
    const out = formatOutput(texts, embeddings, 'jsonl');
    const lines = out.split('\n');
    assert.equal(lines.length, 2);
    const first = JSON.parse(lines[0]);
    assert.equal(first.text, 'Hello world');
    assert.deepEqual(first.embedding, [0.1, 0.2]);
    const second = JSON.parse(lines[1]);
    assert.equal(second.text, "It's a test");
    assert.deepEqual(second.embedding, [0.3, 0.4]);
  });

  test('csv produces header row and data rows', () => {
    const out = formatOutput(texts, embeddings, 'csv');
    const lines = out.split('\n');
    assert.equal(lines[0], 'text,dim_0,dim_1');
    assert.equal(lines[1], '"Hello world",0.1,0.2');
    assert.equal(lines[2], '"It\'s a test",0.3,0.4');
  });

  test('csv escapes double-quotes in text', () => {
    const out = formatOutput(['say "hi"'], [[1, 2]], 'csv');
    const lines = out.split('\n');
    assert.equal(lines[1], '"say ""hi""",1,2');
  });

  test('csv returns empty string for zero embeddings', () => {
    assert.equal(formatOutput([], [], 'csv'), '');
  });

  test('csv with a single embedding still includes header', () => {
    const out = formatOutput(['one'], [[0.5, 0.6, 0.7]], 'csv');
    const lines = out.split('\n');
    assert.equal(lines[0], 'text,dim_0,dim_1,dim_2');
    assert.equal(lines[1], '"one",0.5,0.6,0.7');
    assert.equal(lines.length, 2);
  });

  test('txt output is one space-separated line per embedding', () => {
    const out = formatOutput(texts, embeddings, 'txt');
    const lines = out.split('\n');
    assert.equal(lines.length, 2);
    assert.equal(lines[0], '0.1 0.2');
    assert.equal(lines[1], '0.3 0.4');
  });

  test('txt --with-text prefixes each line with text and tab', () => {
    const out = formatOutput(texts, embeddings, 'txt', true);
    const lines = out.split('\n');
    assert.equal(lines[0], 'Hello world\t0.1 0.2');
    assert.equal(lines[1], "It's a test\t0.3 0.4");
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

// ── writeOutput ──────────────────────────────────────────────────────────────

describe('writeOutput', () => {
  test('writes content + newline to a file when dumpPath is provided', () => {
    const tmp = mkdtempSync(join(tmpdir(), 'embedeer-test-'));
    try {
      const dumpPath = join(tmp, 'out.json');
      writeOutput('[1,2,3]', dumpPath);
      const content = readFileSync(dumpPath, 'utf8');
      assert.equal(content, '[1,2,3]\n');
    } finally {
      rmSync(tmp, { recursive: true, force: true });
    }
  });

  test('logs content to console when no dumpPath given', () => {
    const logged = [];
    const origLog = console.log;
    console.log = (...args) => logged.push(args);
    try {
      writeOutput('hello output', null);
      assert.equal(logged.length, 1);
      assert.equal(logged[0][0], 'hello output');
    } finally {
      console.log = origLog;
    }
  });

  test('logs to console when dumpPath is undefined', () => {
    const logged = [];
    const origLog = console.log;
    console.log = (...args) => logged.push(args);
    try {
      writeOutput('no path', undefined);
      assert.equal(logged.length, 1);
      assert.equal(logged[0][0], 'no path');
    } finally {
      console.log = origLog;
    }
  });
});
