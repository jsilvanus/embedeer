#!/usr/bin/env node
// Mock child-process worker for micro-bench: supports init and task messages.
process.on('message', (msg) => {
  if (!msg) return;
  if (msg.type === 'init') {
    // small startup delay
    setTimeout(() => process.send({ type: 'ready' }), 10);
  } else if (msg.type === 'task') {
    const texts = Array.isArray(msg.texts) ? msg.texts : (msg.payload && msg.payload.texts) || [];
    const rows = texts.length || 1;
    const cols = 64;
    const out = new Array(rows);
    for (let i = 0; i < rows; i++) {
      const row = new Array(cols);
      for (let j = 0; j < cols; j++) row[j] = Math.random();
      out[i] = row;
    }
    process.send({ type: 'result', id: msg.id, embeddings: out });
  }
});
