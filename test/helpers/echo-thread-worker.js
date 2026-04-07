/**
 * A minimal echo worker for worker_threads tests.
 *
 * Reads configuration from workerData, sends { type: 'ready' } immediately,
 * then echoes each text as an embedding of [99].
 */

import { workerData, parentPort } from 'worker_threads';

// Validate that workerData is forwarded correctly.
const { modelName } = workerData ?? {};

// Signal ready
parentPort.postMessage({ type: 'ready', modelName });

parentPort.on('message', (msg) => {
  if (msg.type === 'task') {
    parentPort.postMessage({
      type: 'result',
      id: msg.id,
      embeddings: msg.texts.map(() => [99]),
    });
  }
});
