#!/usr/bin/env node
import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { Embedder } from '../src/embedder.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

const texts = readFileSync(join(__dirname, 'texts-100.txt'), 'utf8')
  .split('\n').map((l) => l.trim()).filter(Boolean);

function cosine(a, b) {
  if (a.length !== b.length) throw new Error('dim mismatch');
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

async function main() {
  console.log('Loading embedeer model...');
  const embedder = await Embedder.create('nomic-ai/nomic-embed-text-v1', { batchSize: 32, concurrency: 2, mode: 'thread' });
  const embE = await embedder.embed(texts);
  await embedder.destroy();
  console.log('Got embedeer embeddings:', embE.length, 'vectors');

  console.log('Requesting Ollama batch embeddings...');
  const url = 'http://localhost:11434/api/embed';
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: 'nomic-embed-text', input: texts }),
  });
  if (!res.ok) {
    console.error('Ollama error:', await res.text());
    process.exit(1);
  }
  const j = await res.json();
  const embO = j.embeddings;
  console.log('Got Ollama embeddings:', embO.length, 'vectors');

  if (embE.length !== embO.length) {
    console.error('Different numbers of vectors:', embE.length, embO.length);
    process.exit(1);
  }

  const sims = [];
  for (let i = 0; i < embE.length; i++) {
    sims.push(cosine(embE[i], embO[i]));
  }

  const sum = sims.reduce((a,b)=>a+b,0);
  const avg = sum / sims.length;
  const min = Math.min(...sims);
  const max = Math.max(...sims);

  console.log(`\nSimilarity stats (cosine): avg=${avg.toFixed(6)} min=${min.toFixed(6)} max=${max.toFixed(6)}`);
  console.log('\nFirst 10 similarities:');
  sims.slice(0,10).forEach((s,i)=>console.log(`${i}: ${s.toFixed(6)}`));
}

main().catch((err)=>{ console.error(err); process.exit(1); });
