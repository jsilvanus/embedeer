import { test, describe } from 'node:test'
import assert from 'node:assert/strict'
import { EventEmitter } from 'events'
import { WorkerPool } from '../src/worker-pool.js'

describe('WorkerPool — OS-level errors and workerData', () => {
  test('OS-level error rejects in-flight task and removes worker', async () => {
    class ErrorEventWorker extends EventEmitter {
      constructor() {
        super()
        // simulate async ready
        setImmediate(() => this.emit('message', { type: 'ready' }))
      }
      postMessage({ type }) {
        if (type === 'task') {
          // simulate an OS-level error (emitted as 'error')
          setImmediate(() => this.emit('error', new Error('spawn failed')))
        }
      }
      async terminate() { setImmediate(() => this.emit('exit', 0)) }
    }

    const pool = new WorkerPool('test-model', { _WorkerClass: ErrorEventWorker, poolSize: 1 })
    await pool.initialize()
    await assert.rejects(() => pool.run(['a']), { message: 'spawn failed' })
    // worker should be removed from pool after OS-level error
    assert.equal(pool.workers.length, 0)
    await pool.destroy()
  })

  test('Worker constructor receives expected workerData', async () => {
    class InspectWorker extends EventEmitter {
      constructor(_path, { workerData } = {}) {
        super()
        this._workerData = workerData
        setImmediate(() => this.emit('message', { type: 'ready' }))
      }
      postMessage({ type, id, texts }) {
        if (type === 'task') {
          setImmediate(() => this.emit('message', { type: 'result', id, embeddings: texts.map(() => [0]) }))
        }
      }
      async terminate() { setImmediate(() => this.emit('exit', 0)) }
    }

    const opts = { poolSize: 1, _WorkerClass: InspectWorker, pooling: 'cls', normalize: false, token: 'tok', dtype: 'q4', cacheDir: '/tmp/cache', device: 'cpu', provider: 'cpu' }
    const pool = new WorkerPool('modelX', opts)
    await pool.initialize()
    const w = pool.workers[0]
    assert.deepEqual(w._workerData, { modelName: 'modelX', pooling: 'cls', normalize: false, token: 'tok', dtype: 'q4', cacheDir: '/tmp/cache', device: 'cpu', provider: 'cpu' })

    // sanity-check that run still works with this fake worker
    const res = await pool.run(['x', 'y'])
    assert.equal(res.length, 2)
    await pool.destroy()
  })
})
