import { strict as assert } from 'assert';
import test from 'node:test';
import { Explainer } from '../src/explainer.js';

test('parses valid JSON output from adapter', async () => {
  const mockAdapter = {
    generate: async (prompt, opts) => {
      return { text: JSON.stringify({
        explanation: 'Test explanation [1]',
        labels: [],
        references: [{ id: 1, source: 'src/foo.js', claim: 'rewrite' }],
        meta: {}
      }) };
    }
  };

  const expl = new Explainer(mockAdapter, { deterministic: true });

  const req = {
    task: 'narrate',
    domain: 'evolution',
    context: {},
    evidence: [{ id: 1, source: 'src/foo.js', excerpt: 'foo changed' }],
    maxTokens: 128,
  };

  const res = await expl.explain(req);
  assert.equal(res.explanation, 'Test explanation [1]');
  assert.equal(res.references.length, 1);
  assert.equal(res.labels.length, 0);
});

test('repairs non-JSON output by asking adapter to return JSON', async () => {
  let calls = 0;
  const mockAdapter = {
    generate: async (prompt, opts) => {
      calls++;
      if (calls === 1) {
        return { text: 'Note: summary follows\nNO JSON HERE' };
      }
      // repair call
      return { text: JSON.stringify({
        explanation: 'Repaired [1]',
        labels: [],
        references: [{ id: 1, source: 'src/foo.js' }],
        meta: {}
      }) };
    }
  };

  const expl = new Explainer(mockAdapter, { deterministic: true });
  const req = {
    task: 'narrate',
    domain: 'evolution',
    context: {},
    evidence: [{ id: 1, source: 'src/foo.js', excerpt: 'foo changed' }],
    maxTokens: 128,
  };

  const res = await expl.explain(req);
  assert.equal(res.explanation, 'Repaired [1]');
});

test('invalid citations produce INSUFFICIENT_EVIDENCE', async () => {
  const mockAdapter = {
    generate: async (prompt, opts) => {
      return { text: JSON.stringify({
        explanation: 'Claim [99]',
        labels: [],
        references: [{ id: 99, source: 'unknown' }],
        meta: {}
      }) };
    }
  };

  const expl = new Explainer(mockAdapter, { deterministic: true });
  const req = {
    task: 'narrate',
    domain: 'evolution',
    context: {},
    evidence: [{ id: 1, source: 'src/foo.js', excerpt: 'foo changed' }],
    maxTokens: 128,
  };

  const res = await expl.explain(req);
  assert.equal(res.explanation, 'INSUFFICIENT_EVIDENCE');
});
