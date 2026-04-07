/**
 * Worker child process: loads a Hugging Face feature-extraction pipeline
 * once and processes embedding batches sent from the parent process via IPC.
 *
 * Running as a forked child process (not a worker thread) ensures full OS-level
 * isolation — a crash here cannot take down the parent.
 *
 * Protocol (messages from parent → worker):
 *   { type: 'init',   modelName: string, pooling: string, normalize: boolean }
 *   { type: 'task',   id: number, texts: string[] }
 *
 * Protocol (messages from worker → parent):
 *   { type: 'ready' }
 *   { type: 'result', id: number, embeddings: number[][] }
 *   { type: 'error',  id: number | null, error: string }
 */

import { pipeline } from '@huggingface/transformers';

let extractor;
let pooling;
let normalize;

process.on('message', async (msg) => {
  if (msg.type === 'init') {
    try {
      ({ pooling, normalize } = msg);
      extractor = await pipeline('feature-extraction', msg.modelName);
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
