/**
 * Worker child process: loads a Hugging Face feature-extraction pipeline
 * once and processes embedding batches sent from the parent process via IPC.
 *
 * Running as a forked child process (not a worker thread) ensures full OS-level
 * isolation — a crash here cannot take down the parent.
 *
 * Protocol (messages from parent → worker):
 *   { type: 'init',   modelName: string, pooling: string, normalize: boolean,
 *                     token?: string, dtype?: string, cacheDir?: string }
 *   { type: 'task',   id: number, texts: string[] }
 *
 * Protocol (messages from worker → parent):
 *   { type: 'ready' }
 *   { type: 'result', id: number, embeddings: number[][] }
 *   { type: 'error',  id: number | null, error: string }
 */

import { pipeline, env } from '@huggingface/transformers';
import { buildPipelineOptions } from './model-cache.js';
import { resolveProvider } from './provider-loader.js';

let extractor;
let pooling;
let normalize;

process.on('message', async (msg) => {
  if (msg.type === 'init') {
    try {
      ({ pooling, normalize } = msg);
      // Apply auth and cache config before loading the model.
      if (msg.token) process.env.HF_TOKEN = msg.token;
      if (msg.cacheDir) env.cacheDir = msg.cacheDir;
      // Activate GPU provider (if requested) before creating the pipeline.
      const deviceStr = await resolveProvider(msg.device, msg.provider);
      const pipelineOpts = {
        ...buildPipelineOptions(msg.dtype),
        ...(deviceStr ? { device: deviceStr } : {}),
      };
      console.error(`Worker: loading model ${msg.modelName} into ${msg.cacheDir || 'default cache'} (this may download)`);
      extractor = await pipeline('feature-extraction', msg.modelName, pipelineOpts);
      console.error(`Worker: model ${msg.modelName} loaded`);
      process.send({ type: 'ready' });
    } catch (err) {
      process.send({ type: 'error', id: null, error: err.message });
      process.exit(1);
    }
  } else if (msg.type === 'task') {
    try {
      const output = await extractor(msg.texts, { pooling, normalize });
      process.send({ type: 'result', id: msg.id, embeddings: output.tolist() });
    } catch (err) {
      process.send({ type: 'error', id: msg.id, error: err.message });
    }
  }
});
