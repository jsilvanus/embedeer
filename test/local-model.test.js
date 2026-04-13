import { test } from 'node:test'
import assert from 'node:assert/strict'
import fs from 'fs/promises'
import { mkdtempSync } from 'fs'
import os from 'os'
import { join } from 'path'

const tmpdir = os.tmpdir()

async function exists(path) {
  try {
    await fs.access(path)
    return true
  } catch {
    return false
  }
}

test('importLocalModel copies directory into cache with given name', async () => {
  const tmp = mkdtempSync(join(tmpdir, 'embedeer-test-'))
  const src = join(tmp, 'src-model')
  const cache = join(tmp, 'cache')
  try {
    await fs.mkdir(src, { recursive: true })
    const modelFile = join(src, 'model.bin')
    await fs.writeFile(modelFile, 'dummy')

    const mod = await import('../src/index.js')
    const { importLocalModel, deleteModel } = mod

    const res = await importLocalModel(src, { name: 'my-local-model', cacheDir: cache })
    assert.equal(res.modelName, 'my-local-model')
    assert.ok(await exists(res.path), 'cached path should exist')
    assert.ok(await exists(join(res.path, 'model.bin')))

    // cleanup via API
    const deleted = await deleteModel('my-local-model', { cacheDir: cache })
    assert.equal(deleted, true)
  } finally {
    await fs.rm(tmp, { recursive: true, force: true })
  }
})

test('importLocalModel rejects when provided name already exists', async () => {
  const tmp = mkdtempSync(join(tmpdir, 'embedeer-test-'))
  const src = join(tmp, 'src-model')
  const cache = join(tmp, 'cache')
  try {
    await fs.mkdir(src, { recursive: true })
    await fs.writeFile(join(src, 'a.txt'), 'x')

    // create pre-existing cache entry with same name
    const existing = join(cache, 'already')
    await fs.mkdir(existing, { recursive: true })
    await fs.writeFile(join(existing, 'b.txt'), 'y')

    const mod = await import('../src/index.js')
    const { importLocalModel } = mod

    await assert.rejects(
      importLocalModel(src, { name: 'already', cacheDir: cache }),
      /already exists/
    )
  } finally {
    await fs.rm(tmp, { recursive: true, force: true })
  }
})
