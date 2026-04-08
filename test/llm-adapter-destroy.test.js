import { test } from 'node:test';
import assert from 'node:assert/strict';
import { LLMAdapter } from '../src/llm-adapter.js';

test('LLMAdapter.destroy forwards to pool.destroy when pool present', async () => {
  let destroyed = false;
  const fakeGen = async () => ({ text: 'ok', raw: null });
  const adapter = new LLMAdapter(fakeGen, 'model', true);
  adapter._pool = { destroy: async () => { destroyed = true; } };
  await adapter.destroy();
  assert.equal(destroyed, true);
});

test('LLMAdapter.destroy is safe when no pool present', async () => {
  const fakeGen = async () => ({ text: 'ok', raw: null });
  const adapter = new LLMAdapter(fakeGen, 'model', true);
  await adapter.destroy(); // should not throw
});
