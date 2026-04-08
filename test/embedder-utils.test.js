import { test, describe } from 'node:test'
import assert from 'node:assert/strict'
import fs from 'fs/promises'
import fsSync from 'fs'
import os from 'os'
import { join } from 'path'
import { Embedder } from '../src/embedder.js'

const tmpBase = os.tmpdir()

describe('Embedder utilities', () => {
  test('applyPerfProfile picks highest textsPerSec result', async () => {
    const tmp = await fs.mkdtemp(join(tmpBase, 'embedeer-test-'))
    try {
      const profile = { results: [
        { success: true, textsPerSec: 100, batchSize: 32, concurrency: 1, dtype: 'fp32', provider: 'cpu', device: 'cpu' },
        { success: true, textsPerSec: 200, batchSize: 64, concurrency: 2, dtype: 'fp16', provider: 'cpu', device: 'cpu' }
      ] }
      const p = join(tmp, 'profile.json')
      await fs.writeFile(p, JSON.stringify(profile))
      const best = await Embedder.applyPerfProfile(p, 'cpu')
      assert.deepEqual(best, { batchSize: 64, concurrency: 2, dtype: 'fp16', provider: 'cpu', device: 'cpu' })
    } finally {
      await fs.rm(tmp, { recursive: true, force: true })
    }
  })

  test('applyPerfProfile filters by device when provided', async () => {
    const tmp = await fs.mkdtemp(join(tmpBase, 'embedeer-test-'))
    try {
      const profile = { results: [
        { success: true, textsPerSec: 100, batchSize: 32, concurrency: 1, dtype: 'fp32', provider: 'cpu', device: 'cpu' },
        { success: true, textsPerSec: 300, batchSize: 128, concurrency: 1, dtype: 'fp32', provider: 'cuda', device: 'gpu' }
      ] }
      const p = join(tmp, 'profile.json')
      await fs.writeFile(p, JSON.stringify(profile))
      const bestGpu = await Embedder.applyPerfProfile(p, 'gpu')
      assert.deepEqual(bestGpu, { batchSize: 128, concurrency: 1, dtype: 'fp32', provider: 'cuda', device: 'gpu' })
    } finally {
      await fs.rm(tmp, { recursive: true, force: true })
    }
  })

  test('applyPerfProfile throws when no successful results', async () => {
    const tmp = await fs.mkdtemp(join(tmpBase, 'embedeer-test-'))
    try {
      const profile = { results: [ { success: false } ] }
      const p = join(tmp, 'profile.json')
      await fs.writeFile(p, JSON.stringify(profile))
      await assert.rejects(async () => { await Embedder.applyPerfProfile(p) })
    } finally {
      await fs.rm(tmp, { recursive: true, force: true })
    }
  })

  test('findLatestGridResult finds most recent grid-results file', async () => {
    const tmp = await fs.mkdtemp(join(tmpBase, 'embedeer-bench-'))
    try {
      const bench = join(tmp, 'bench')
      await fs.mkdir(bench)
      await fs.writeFile(join(bench, 'grid-results-1.json'), '{}')
      await fs.writeFile(join(bench, 'grid-results-2.json'), '{}')
      const latest = Embedder.findLatestGridResult(bench)
      assert.ok(latest.endsWith('grid-results-2.json'))
    } finally {
      await fs.rm(tmp, { recursive: true, force: true })
    }
  })

  test('generateAndSaveProfile (quick) writes profile to user profile path', async () => {
    const tmp = await fs.mkdtemp(join(tmpBase, 'embedeer-save-'))
    const origInitial = Embedder.initialPerformanceCheckup
    const origHomedir = os.homedir
    try {
      // stub the expensive performance check
      Embedder.initialPerformanceCheckup = async () => ({ device: 'cpu', batchSize: 16, concurrency: 1, dtype: 'fp32' })
      // force home dir to our temp location so we don't write to user home
      os.homedir = () => tmp

      const res = await Embedder.generateAndSaveProfile({ mode: 'quick', device: 'cpu', sampleSize: 1, modelName: 'nomic-embed-text' })
      assert.ok(res.best)
      assert.ok(res.savedPath)
      const content = fsSync.readFileSync(res.savedPath, 'utf8')
      const parsed = JSON.parse(content)
      assert.ok(parsed.best)
    } finally {
      // restore
      Embedder.initialPerformanceCheckup = origInitial
      os.homedir = origHomedir
      await fs.rm(tmp, { recursive: true, force: true })
    }
  })
})
