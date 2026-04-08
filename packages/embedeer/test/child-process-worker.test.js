/**
 * Integration tests for ChildProcessWorker.
 *
 * Uses real forked child processes (the helpers in test/helpers/) to
 * verify that the worker is truly isolated from the parent.
 */

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { ChildProcessWorker } from '../src/child-process-worker.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ECHO_WORKER   = join(__dirname, 'helpers/echo-worker.js');
const CRASH_WORKER  = join(__dirname, 'helpers/crash-worker.js');

describe('ChildProcessWorker', async () => {
  test('emits ready message after init', async () => {
    const worker = new ChildProcessWorker(ECHO_WORKER, {
      workerData: { modelName: 'test', pooling: 'mean', normalize: true },
    });

    const msg = await new Promise((resolve) => worker.once('message', resolve));
    assert.equal(msg.type, 'ready');

    await worker.terminate();
  });

  test('postMessage delivers task results', async () => {
    const worker = new ChildProcessWorker(ECHO_WORKER, {
      workerData: { modelName: 'test', pooling: 'mean', normalize: true },
    });

    // Wait for ready
    await new Promise((resolve) => worker.once('message', resolve));

    // Send a task
    worker.postMessage({ type: 'task', id: 7, texts: ['hello', 'world'] });

    const result = await new Promise((resolve) =>
      worker.on('message', (m) => { if (m.type === 'result') resolve(m); }),
    );

    assert.equal(result.id, 7);
    assert.equal(result.embeddings.length, 2);
    assert.deepEqual(result.embeddings[0], [42]);

    await worker.terminate();
  });

  test('terminate() resolves once the process has exited', async () => {
    const worker = new ChildProcessWorker(ECHO_WORKER, {
      workerData: { modelName: 'test', pooling: 'mean', normalize: true },
    });

    await new Promise((resolve) => worker.once('message', resolve));

    // terminate() must resolve — we don't assert a specific code because
    // SIGTERM on Linux yields code=null / signal='SIGTERM'.
    const terminated = await worker.terminate().then(() => true, () => false);
    assert.equal(terminated, true);
  });

  test('a crashing child emits exit without crashing the parent', async () => {
    const worker = new ChildProcessWorker(CRASH_WORKER, {
      workerData: { modelName: 'test', pooling: 'mean', normalize: true },
    });

    // The child will exit(1) — we should get an 'exit' event, not an exception.
    const exitCode = await new Promise((resolve) => worker.once('exit', resolve));

    // Non-zero exit code is reported; this process (the test) is still alive.
    assert.equal(exitCode, 1);
  });
});
