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
import { resolveProvider } from './provider-loader.js';

const { modelName, pooling, normalize, token, dtype, cacheDir, device, provider } = workerData;

// Apply configuration before loading the model.
if (token) process.env.HF_TOKEN = token;
if (cacheDir) env.cacheDir = cacheDir;

let extractor;

async function init() {
  // Activate GPU provider (if requested) before creating the pipeline.
  const deviceStr = await resolveProvider(device, provider);
  const pipelineOpts = {
    ...buildPipelineOptions(dtype),
    ...(deviceStr ? { device: deviceStr } : {}),
  };
  console.error(`Thread worker: loading model ${modelName} into ${cacheDir || 'default cache'} (this may download)`);
  extractor = await pipeline('feature-extraction', modelName, pipelineOpts);
  console.error(`Thread worker: model ${modelName} loaded`);
  parentPort.postMessage({ type: 'ready' });
}

parentPort.on('message', async (msg) => {
  if (msg.type !== 'task') return;
  try {
    const output = await extractor(msg.texts, { pooling, normalize });
    // Prefer a JS nested-array representation if that's what the extractor
    // exposes via `tolist()`. To avoid expensive structured-clone copies
    // across worker thread boundaries we flatten into a Float32Array here
    // and transfer the underlying ArrayBuffer to the parent (zero-copy).
    const list = (typeof output?.tolist === 'function') ? output.tolist() : output;

    // If extractor already returned a typed view / ArrayBuffer, transfer it.
    if (ArrayBuffer.isView(list) || list instanceof ArrayBuffer) {
      const ab = ArrayBuffer.isView(list) ? list.buffer : list;
      parentPort.postMessage({ type: 'result', id: msg.id, embeddings: ab, shape: [list.length] }, [ab]);
      return;
    }

    // Expect nested arrays: [rows][cols]
    const rows = Array.isArray(list) ? list.length : 0;
    const cols = rows > 0 && Array.isArray(list[0]) ? list[0].length : 0;
    const flat = new Float32Array(rows * cols);
    let k = 0;
    for (let i = 0; i < rows; i++) {
      const row = list[i];
      for (let j = 0; j < cols; j++) {
        flat[k++] = Number(row[j]);
      }
    }
    parentPort.postMessage({ type: 'result', id: msg.id, embeddings: flat.buffer, shape: [rows, cols] }, [flat.buffer]);
  } catch (err) {
    parentPort.postMessage({ type: 'error', id: msg.id, error: err.message });
  }
});

init().catch((err) => {
  parentPort.postMessage({ type: 'error', id: null, error: err.message });
  process.exit(1);
});
