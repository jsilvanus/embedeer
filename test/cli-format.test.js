/**
 * Tests for CLI output formatting, input parsing, and model-cache helpers.
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

// ── Inline helpers mirroring cli.js (cli.js runs main() on import) ──────────

function parseDelimiter(str) {
  return str
    .replace(/\\0/g, '\0')
    .replace(/\\n/g, '\n')
    .replace(/\\t/g, '\t')
    .replace(/\\r/g, '\r');
}

function parseTexts(raw, delimiter = '\n') {
  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) throw new Error('Expected a JSON array');
    return parsed;
  } catch {
    return raw.split(delimiter).filter(Boolean);
  }
}

function formatOutput(texts, embeddings, format, withText = false) {
  switch (format) {
    case 'jsonl':
      return texts
        .map((text, i) => JSON.stringify({ text, embedding: embeddings[i] }))
        .join('\n');

    case 'csv': {
      if (embeddings.length === 0) return '';
      const dims = embeddings[0].length;
      const header = ['text', ...Array.from({ length: dims }, (_, k) => `dim_${k}`)].join(',');
      const rows = texts.map((text, i) => {
        const safeText = '"' + text.replace(/"/g, '""') + '"';
        return [safeText, ...embeddings[i]].join(',');
      });
      return [header, ...rows].join('\n');
    }

    case 'txt':
      if (withText) {
        return texts.map((text, i) => `${text}\t${embeddings[i].join(' ')}`).join('\n');
      }
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

    default: // json
      if (withText) {
        return JSON.stringify(
          texts.map((text, i) => ({ text, embedding: embeddings[i] }))
        );
      }
      return JSON.stringify(embeddings);
  }
}

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
