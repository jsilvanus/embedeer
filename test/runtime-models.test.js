import { test } from 'node:test'
import assert from 'node:assert/strict'
import { EventEmitter } from 'events'
import { WorkerPool } from '../src/worker-pool.js'
import { getLoadedModels } from '../src/index.js'

test('WorkerPool registers and unregisters loaded models (mock worker)', async () => {
  class MockWorker extends EventEmitter {
    constructor(scriptPath, { workerData } = {}) {
      super()
      this._script = scriptPath
      this._workerData = workerData
      // simulate async model load
      setImmediate(() => this.emit('message', { type: 'ready' }))
    }

    postMessage(msg) {
      if (msg && msg.type === 'task') {
        // echo a simple numeric embedding
        setImmediate(() => this.emit('message', { type: 'result', id: msg.id, embeddings: [[0.1, 0.2]] }))
      }
    }

    terminate() {
      // simulate clean exit
      setImmediate(() => this.emit('exit', 0))
      return Promise.resolve()
    }
  }

  const pool = new WorkerPool('test-model', { poolSize: 1, _WorkerClass: MockWorker })
  try {
    await pool.initialize()
    const loaded = getLoadedModels()
    assert.ok(Array.isArray(loaded))
    assert.ok(loaded.includes('test-model'))
  } finally {
    await pool.destroy()
  }

  const loadedAfter = getLoadedModels()
  assert.ok(!loadedAfter.includes('test-model'))
})
