import { test, describe } from 'node:test'
import assert from 'node:assert/strict'
import fs from 'fs/promises'
import { join } from 'path'
import os from 'os'
import { ensureModel } from '../src/model-management.js'
import { getCacheDir } from '../src/model-cache.js'

const tmpBase = os.tmpdir()

describe('model-management ensureModel', () => {
  test('returns info when model present', async () => {
    const tmp = await fs.mkdtemp(join(tmpBase, 'embedeer-mm-'))
    try {
      const modelName = 'present-model'
      const cacheDir = getCacheDir(tmp)
      await fs.mkdir(join(cacheDir, modelName), { recursive: true })

      const res = await ensureModel(modelName, { cacheDir: tmp, downloadIfMissing: true, prepare: false })
      assert.equal(res.modelName, modelName)
      assert.equal(res.cacheDir, getCacheDir(tmp))
    } finally {
      await fs.rm(tmp, { recursive: true, force: true })
    }
  })

  test('throws when model missing and downloadIfMissing=false', async () => {
    const tmp = await fs.mkdtemp(join(tmpBase, 'embedeer-mm-'))
    try {
      await assert.rejects(async () => {
        await ensureModel('no-such-model', { cacheDir: tmp, downloadIfMissing: false })
      })
    } finally {
      await fs.rm(tmp, { recursive: true, force: true })
    }
  })
})
