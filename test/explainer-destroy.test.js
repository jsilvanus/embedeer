import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Explainer } from '../src/explainer.js';

test('Explainer.destroy forwards to adapter.destroy', async () => {
  let destroyed = false;
  const mockAdapter = {
    generate: async () => ({ text: '{}' }),
    destroy: async () => { destroyed = true; },
  };

  const expl = new Explainer(mockAdapter, { deterministic: true });
  await expl.destroy();
  assert.equal(destroyed, true);
});
