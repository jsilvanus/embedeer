#!/usr/bin/env node
import { WorkerPool } from '../src/worker-pool.js';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const procScript = join(__dirname, 'mock-worker-process.js');
const threadScript = join(__dirname, 'mock-worker-thread.js');

async function benchMode(mode, poolSize = 4, numTasks = 200, textsPerTask = 8) {
  console.log(`\nBenchmarking mode=${mode} poolSize=${poolSize} tasks=${numTasks} textsPerTask=${textsPerTask}`);
  const pool = new WorkerPool('dummy-model', {
    poolSize,
    mode,
    workerScript: procScript,
    threadWorkerScript: threadScript,
  });
  await pool.initialize();
  const sample = Array.from({ length: textsPerTask }, (_, i) => `text-${i}`);
  const tasks = [];
  const t0 = performance.now();
  for (let i = 0; i < numTasks; i++) tasks.push(pool.run(sample));
  await Promise.all(tasks);
  const elapsed = performance.now() - t0;
  await pool.destroy();
  console.log(`mode=${mode} elapsed=${(elapsed / 1000).toFixed(3)}s texts/s=${(numTasks * textsPerTask / (elapsed / 1000)).toFixed(1)}`);
  return elapsed;
}

(async () => {
  await benchMode('thread');
  await benchMode('process');
})().catch((err) => { console.error(err); process.exit(1); });
