import { test, describe } from 'node:test'
import assert from 'node:assert/strict'
import { spawnSync } from 'child_process'
import fs from 'fs/promises'
import os from 'os'
import { join } from 'path'
import { pathToFileURL } from 'url'

// Matches the list in src/provider-loader.js
const REQUIRED_CUDA_LIBS = [
  'libcudart.so.12',
  'libcublas.so.12',
  'libcublasLt.so.12',
  'libcurand.so.10',
  'libcufft.so.11',
  'libcudnn.so.9',
]

describe('provider-loader (CUDA activation via runner)', () => {
  test('tryLoadProvider("cuda") succeeds when /dev/nvidiactl and libs exist (linux/x64 only)', async () => {
    if (process.platform !== 'linux' || process.arch !== 'x64') {
      test.skip('linux/x64 only')
      return
    }

    const tmp = await fs.mkdtemp(join(os.tmpdir(), 'embedeer-cuda-'))
    try {
      // create dummy library files in a temp dir and point LD_LIBRARY_PATH at it
      for (const lib of REQUIRED_CUDA_LIBS) {
        await fs.writeFile(join(tmp, lib), '')
      }

      const runnerPath = join(tmp, 'run-provider.mjs')
      const providerLoaderPath = join(process.cwd(), 'src', 'provider-loader.js')
      const providerLoaderFileUrl = pathToFileURL(providerLoaderPath).href

      const code = `
import { createRequire } from 'module'
import process from 'process'
const req = createRequire(import.meta.url)
const fsModule = req('fs')
const origExists = fsModule.existsSync
// pretend /dev/nvidiactl exists on this test runner
fsModule.existsSync = (p) => (p === '/dev/nvidiactl') || origExists(p)
// point LD_LIBRARY_PATH to our temp dir so findLib sees the dummy libs
process.env.LD_LIBRARY_PATH = ${JSON.stringify(tmp)}
const mod = await import(${JSON.stringify(providerLoaderFileUrl)})
const res = await mod.tryLoadProvider('cuda')
console.log(JSON.stringify(res))
`

      await fs.writeFile(runnerPath, code)

      const proc = spawnSync(process.execPath, [runnerPath], { encoding: 'utf8' })
      if (proc.status !== 0) {
        // surface stderr for debugging
        throw new Error(`runner failed: ${proc.stderr}`)
      }
      const out = proc.stdout.trim()
      const obj = JSON.parse(out)
      assert.equal(obj.loaded, true)
      assert.equal(obj.deviceStr, 'cuda')
    } finally {
      await fs.rm(tmp, { recursive: true, force: true })
    }
  })
})
