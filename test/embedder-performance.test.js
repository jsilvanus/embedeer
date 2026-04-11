import { test, describe } from 'node:test'
import assert from 'node:assert/strict'
import os from 'os'
import { Embedder } from '../src/embedder.js'

// `performance.now` lives on a non-writable prototype property in Node 18,
// so plain assignment (`performance.now = ...`) throws. Installing an own
// property via defineProperty shadows it; deleting the own property restores
// access to the original prototype method. Works on Node 18, 20, 22.
function stubPerformanceNow(impl) {
  Object.defineProperty(performance, 'now', {
    configurable: true,
    writable: true,
    value: impl,
  })
  return () => { delete performance.now }
}

describe('Embedder performance heuristics', () => {
  test('initialPerformanceCheckup (CPU) low TPS -> batchSize 32, concurrency 1', async () => {
    const origInit = Embedder.prototype.initialize
    const origEmbed = Embedder.prototype.embed
    const origDestroy = Embedder.prototype.destroy
    let restorePerf
    try {
      Embedder.prototype.initialize = async function () {}
      Embedder.prototype.embed = async function (sample) { return sample.map(() => [0]) }
      Embedder.prototype.destroy = async function () {}

      let calls = 0
      restorePerf = stubPerformanceNow(() => {
        calls += 1
        return calls === 1 ? 0 : 1000
      })

      const cfg = await Embedder.initialPerformanceCheckup({ device: 'cpu', sampleSize: 50, modelName: 'test-model' })
      assert.equal(cfg.batchSize, 32)
      assert.equal(cfg.concurrency, 1)
      assert.equal(cfg.dtype, 'fp32')
    } finally {
      Embedder.prototype.initialize = origInit
      Embedder.prototype.embed = origEmbed
      Embedder.prototype.destroy = origDestroy
      restorePerf?.()
    }
  })

  test('initialPerformanceCheckup (CPU) high TPS -> batchSize 64, concurrency <= 2', async () => {
    const origInit = Embedder.prototype.initialize
    const origEmbed = Embedder.prototype.embed
    const origDestroy = Embedder.prototype.destroy
    let restorePerf
    try {
      Embedder.prototype.initialize = async function () {}
      Embedder.prototype.embed = async function (sample) { return sample.map(() => [0]) }
      Embedder.prototype.destroy = async function () {}

      let calls = 0
      restorePerf = stubPerformanceNow(() => {
        calls += 1
        return calls === 1 ? 0 : 1 // tiny elapsed -> very high TPS
      })

      const cfg = await Embedder.initialPerformanceCheckup({ device: 'cpu', sampleSize: 50, modelName: 'test-model' })
      assert.equal(cfg.batchSize, 64)
      // concurrency should be at most 2
      assert.ok(Number.isInteger(cfg.concurrency) && cfg.concurrency >= 1 && cfg.concurrency <= 2)
    } finally {
      Embedder.prototype.initialize = origInit
      Embedder.prototype.embed = origEmbed
      Embedder.prototype.destroy = origDestroy
      restorePerf?.()
    }
  })

  test('initialPerformanceCheckup (GPU) high TPS -> batchSize 128, concurrency 1, dtype fp32', async () => {
    const origInit = Embedder.prototype.initialize
    const origEmbed = Embedder.prototype.embed
    const origDestroy = Embedder.prototype.destroy
    let restorePerf
    try {
      Embedder.prototype.initialize = async function () {}
      Embedder.prototype.embed = async function (sample) { return sample.map(() => [0]) }
      Embedder.prototype.destroy = async function () {}

      let calls = 0
      restorePerf = stubPerformanceNow(() => {
        calls += 1
        return calls === 1 ? 0 : 1
      })

      const cfg = await Embedder.initialPerformanceCheckup({ device: 'gpu', sampleSize: 50, modelName: 'test-model' })
      assert.equal(cfg.batchSize, 128)
      assert.equal(cfg.concurrency, 1)
      assert.equal(cfg.dtype, 'fp32')
    } finally {
      Embedder.prototype.initialize = origInit
      Embedder.prototype.embed = origEmbed
      Embedder.prototype.destroy = origDestroy
      restorePerf?.()
    }
  })

  test('getUserProfilePath returns perf-profile.json', () => {
    const p = Embedder.getUserProfilePath()
    assert.ok(p.endsWith('perf-profile.json'))
  })
})
