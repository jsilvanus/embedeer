/**
 * Integration tests for ThreadWorker.
 *
 * Uses real worker threads (the helpers in test/helpers/) to verify
 * that in-process thread workers communicate correctly.
 */

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { ThreadWorker } from '../src/thread-worker.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ECHO_THREAD_WORKER = join(__dirname, 'helpers/echo-thread-worker.js');

describe('ThreadWorker', async () => {
  test('emits ready message after construction', async () => {
    const worker = new ThreadWorker(ECHO_THREAD_WORKER, {
      workerData: { modelName: 'test-model', pooling: 'mean', normalize: true },
    });

    const msg = await new Promise((resolve) => worker.once('message', resolve));
    assert.equal(msg.type, 'ready');
    assert.equal(msg.modelName, 'test-model');

    await worker.terminate();
  });

  test('postMessage delivers task results', async () => {
    const worker = new ThreadWorker(ECHO_THREAD_WORKER, {
      workerData: { modelName: 'test', pooling: 'mean', normalize: true },
    });

    // Wait for ready
    await new Promise((resolve) => worker.once('message', resolve));

    // Send a task
    worker.postMessage({ type: 'task', id: 5, texts: ['alpha', 'beta'] });

    const result = await new Promise((resolve) =>
      worker.on('message', (m) => { if (m.type === 'result') resolve(m); }),
    );

    assert.equal(result.id, 5);
    assert.equal(result.embeddings.length, 2);
    assert.deepEqual(result.embeddings[0], [99]);

    await worker.terminate();
  });

  test('terminate() resolves once the thread exits', async () => {
    const worker = new ThreadWorker(ECHO_THREAD_WORKER, {
      workerData: { modelName: 'test', pooling: 'mean', normalize: true },
    });

    await new Promise((resolve) => worker.once('message', resolve));

    const terminated = await worker.terminate().then(() => true, () => false);
    assert.equal(terminated, true);
  });

  test('workerData is forwarded to the thread script', async () => {
    const worker = new ThreadWorker(ECHO_THREAD_WORKER, {
      workerData: { modelName: 'my-model', pooling: 'cls', normalize: false },
    });

    const msg = await new Promise((resolve) => worker.once('message', resolve));
    // The echo thread script echoes modelName back in the ready message.
    assert.equal(msg.modelName, 'my-model');

    await worker.terminate();
  });
});
