/**
 * Worker thread script: loads a Hugging Face feature-extraction pipeline
 * once and processes embedding batches received from the parent thread.
 *
 * Configuration is received via workerData (set at thread creation time).
 * This makes thread workers slightly lighter than child-process workers
 * because there is no separate IPC init round-trip.
 *
 * Protocol (messages from parent → thread):
 *   { type: 'task', id: number, texts: string[] }
 *
 * Protocol (messages from thread → parent):
 *   { type: 'ready' }
 *   { type: 'result', id: number, embeddings: number[][] }
 *   { type: 'error',  id: number | null, error: string }
 */

import { workerData, parentPort } from 'worker_threads';
import { pipeline, env } from '@huggingface/transformers';
import { buildPipelineOptions } from './model-cache.js';

const { modelName, pooling, normalize, token, dtype, cacheDir } = workerData;

// Apply configuration before loading the model.
if (token) process.env.HF_TOKEN = token;
if (cacheDir) env.cacheDir = cacheDir;

let extractor;

async function init() {
  extractor = await pipeline('feature-extraction', modelName, buildPipelineOptions(dtype));
  parentPort.postMessage({ type: 'ready' });
}

parentPort.on('message', async (msg) => {
  if (msg.type !== 'task') return;
  try {
    const output = await extractor(msg.texts, { pooling, normalize });
    parentPort.postMessage({ type: 'result', id: msg.id, embeddings: output.tolist() });
  } catch (err) {
    parentPort.postMessage({ type: 'error', id: msg.id, error: err.message });
  }
});

init().catch((err) => {
  parentPort.postMessage({ type: 'error', id: null, error: err.message });
  process.exit(1);
});
