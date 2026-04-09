import { test } from 'node:test'
import assert from 'node:assert/strict'
import fs from 'fs/promises'
import os from 'os'
import { join } from 'path'
import { getCachedModels } from '../src/model-management.js'

test('getCachedModels returns metadata for cached models', async () => {
  const tmpBase = os.tmpdir()
  const tmp = await fs.mkdtemp(join(tmpBase, 'embedeer-test-'))
  try {
    const modelName = 'model-a'
    const modelDir = join(tmp, modelName)
    await fs.mkdir(modelDir, { recursive: true })
    await fs.writeFile(join(modelDir, 'a.txt'), 'hello')
    await fs.writeFile(join(modelDir, 'b.bin'), Buffer.alloc(512))

    const models = await getCachedModels({ cacheDir: tmp })
    assert.ok(Array.isArray(models))
    const found = models.find((m) => m.name === modelName)
    assert.ok(found, 'expected cached model to be present')
    assert.ok(typeof found.path === 'string')
    assert.ok(typeof found.size === 'number')
    assert.ok(found.size >= 512)
    assert.ok(found.mtime === null || typeof found.mtime === 'string')
  } finally {
    await fs.rm(tmp, { recursive: true, force: true })
  }
})
