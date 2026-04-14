import { parentPort, workerData } from 'worker_threads';

async function init() {
  // small startup delay
  setTimeout(() => {
    parentPort.postMessage({ type: 'ready' });
  }, 10);
}

parentPort.on('message', (msg) => {
  if (!msg) return;
  if (msg.type === 'task') {
    const texts = Array.isArray(msg.texts) ? msg.texts : (msg.payload && msg.payload.texts) || [];
    const rows = texts.length || 1;
    const cols = 64;
    const flat = new Float32Array(rows * cols);
    for (let i = 0; i < flat.length; i++) flat[i] = Math.random();
    parentPort.postMessage({ type: 'result', id: msg.id, embeddings: flat.buffer, shape: [rows, cols] }, [flat.buffer]);
  }
});

init();
