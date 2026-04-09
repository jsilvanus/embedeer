import { test } from 'node:test'
import assert from 'node:assert/strict'
import fs from 'fs/promises'
import os from 'os'
import { join } from 'path'
import { deleteModel } from '../src/model-management.js'

test('deleteModel removes matching directories from cache', async () => {
  const tmpBase = os.tmpdir()
  const tmp = await fs.mkdtemp(join(tmpBase, 'embedeer-test-'))
  try {
    const modelName = 'my-model'
    const d1 = join(tmp, modelName)
    const d2 = join(tmp, modelName + '-v2')
    await fs.mkdir(d1, { recursive: true })
    await fs.mkdir(d2, { recursive: true })

    // First call removes the exact directory name if present
    const removed = await deleteModel(modelName, { cacheDir: tmp })
    assert.equal(removed, true)

    // After first removal the '-v2' variant should still exist
    let entries = await fs.readdir(tmp)
    assert.equal(entries.includes(modelName), false)
    assert.equal(entries.includes(modelName + '-v2'), true)

    // Second call should use the fallback and remove the remaining matching directory
    const removedAgain = await deleteModel(modelName, { cacheDir: tmp })
    assert.equal(removedAgain, true)
    entries = await fs.readdir(tmp)
    assert.equal(entries.includes(modelName + '-v2'), false)

    // deleting a non-existent model now returns false
    const removed2 = await deleteModel('no-such-model', { cacheDir: tmp })
    assert.equal(removed2, false)
  } finally {
    await fs.rm(tmp, { recursive: true, force: true })
  }
})
