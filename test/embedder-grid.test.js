import { test, describe } from 'node:test'
import assert from 'node:assert/strict'
import { spawnSync } from 'child_process'
import fs from 'fs/promises'
import os from 'os'
import { join } from 'path'
import { pathToFileURL } from 'url'

describe('Embedder.generateAndSaveProfile (grid mode)', () => {
  test('grid mode uses provided out file and saves best profile', async () => {
    const tmp = await fs.mkdtemp(join(os.tmpdir(), 'embedeer-grid-'))
    try {
      const outPath = join(tmp, 'grid-out.json')
      const results = {
        results: [
          { success: true, textsPerSec: 50, batchSize: 32, concurrency: 1, dtype: 'fp32', provider: 'cpu', device: 'cpu' },
          { success: true, textsPerSec: 80, batchSize: 64, concurrency: 2, dtype: 'fp16', provider: 'cpu', device: 'cpu' }
        ]
      }
      await fs.writeFile(outPath, JSON.stringify(results))

      const runnerPath = join(tmp, 'run-grid.mjs')
      const embedderPath = join(process.cwd(), 'src', 'embedder.js')
      const embedderFileUrl = pathToFileURL(embedderPath).href

      const code = `
import child_process from 'child_process'
import os from 'os'
import fs from 'fs'
// Avoid running the real benchmark script (execFileSync after copilot security fix)
child_process.execFileSync = () => {}
child_process.execSync = () => {}
// Use tmp homedir so we don't write to the user's real home
os.homedir = () => ${JSON.stringify(tmp)}
const mod = await import(${JSON.stringify(embedderFileUrl)})
const res = await mod.Embedder.generateAndSaveProfile({ mode: 'grid', device: 'cpu', sampleSize: 1, profileOut: ${JSON.stringify(outPath)}, modelName: 'test-model' })
console.log(JSON.stringify(res))
`

      await fs.writeFile(runnerPath, code)

      const proc = spawnSync(process.execPath, [runnerPath], { encoding: 'utf8' })
      if (proc.status !== 0) throw new Error(`runner failed: ${proc.stderr}`)
      const outAll = proc.stdout.trim()
      const lines = outAll.split(/\r?\n/).filter(Boolean)
      if (lines.length === 0) throw new Error('no stdout from runner')
      const last = lines[lines.length - 1]
      const obj = JSON.parse(last)
      assert.ok(obj.best)
      assert.ok(obj.savedPath)
      const saved = await fs.readFile(obj.savedPath, 'utf8')
      const parsed = JSON.parse(saved)
      assert.ok(parsed.best)
    } finally {
      await fs.rm(tmp, { recursive: true, force: true })
    }
  })
})
