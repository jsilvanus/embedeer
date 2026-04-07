/**
 * Worker thread: loads a Hugging Face feature-extraction pipeline once
 * and processes embedding batches sent from the main thread.
 *
 * Protocol (messages from main → worker):
 *   { id: number, texts: string[] }
 *
 * Protocol (messages from worker → main):
 *   { type: 'ready' }
 *   { type: 'result', id: number, embeddings: number[][] }
 *   { type: 'error',  id: number | null, error: string }
 */

import { workerData, parentPort } from 'worker_threads';
import { pipeline } from '@huggingface/transformers';

const { modelName, pooling, normalize } = workerData;

let extractor;

async function init() {
  extractor = await pipeline('feature-extraction', modelName);
  parentPort.postMessage({ type: 'ready' });
}

parentPort.on('message', async ({ id, texts }) => {
  try {
    const output = await extractor(texts, { pooling, normalize });
    parentPort.postMessage({ type: 'result', id, embeddings: output.tolist() });
  } catch (err) {
    parentPort.postMessage({ type: 'error', id, error: err.message });
  }
});

init().catch((err) => {
  parentPort.postMessage({ type: 'error', id: null, error: err.message });
  process.exit(1);
});
