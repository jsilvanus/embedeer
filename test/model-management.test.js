import { test, describe } from 'node:test'
import assert from 'node:assert/strict'
import fs from 'fs/promises'
import os from 'os'
import { join } from 'path'
import { isModelDownloaded, listModels } from '../src/model-management.js'

const tmpBase = os.tmpdir()

describe('model-management', () => {
  test('detects model directories and lists them', async () => {
    const tmp = await fs.mkdtemp(join(tmpBase, 'embedeer-test-'))
    try {
      const modelName = 'my-test-model'
      await fs.mkdir(join(tmp, modelName), { recursive: true })
      const found = await isModelDownloaded(modelName, { cacheDir: tmp })
      assert.equal(found, true)
      const list = await listModels({ cacheDir: tmp })
      assert.ok(Array.isArray(list))
      assert.ok(list.includes(modelName))
    } finally {
      await fs.rm(tmp, { recursive: true, force: true })
    }
  })
})
