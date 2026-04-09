import { getCacheDir } from './model-cache.js'
import fs from 'fs'
import { join } from 'path'

async function exists(path) {
  try {
    await fs.promises.access(path)
    return true
  } catch {
    return false
  }
}

/**
 * Check whether a model appears to be present in the embedeer cache.
 * This is a best-effort check that looks for an entry matching the
 * requested model name under the cache directory.
 *
 * @param {string} modelName
 * @param {{cacheDir?: string}} opts
 * @returns {Promise<boolean>}
 */
export async function isModelDownloaded(modelName, opts = {}) {
  const dir = getCacheDir(opts.cacheDir)
  const target = join(dir, modelName)
  if (await exists(target)) return true

  // Fallback: look for any directory whose name contains the modelName
  try {
    const entries = await fs.promises.readdir(dir, { withFileTypes: true })
    for (const e of entries) {
      if (e.isDirectory() && e.name.includes(modelName)) return true
    }
  } catch (err) {
    // ignore and return false
  }
  return false
}

/**
 * List cached model directories under the embedeer cache dir.
 *
 * @param {{cacheDir?: string}} opts
 * @returns {Promise<string[]>}
 */
export async function listModels(opts = {}) {
  const dir = getCacheDir(opts.cacheDir)
  try {
    const entries = await fs.promises.readdir(dir, { withFileTypes: true })
    return entries.filter((e) => e.isDirectory()).map((d) => d.name)
  } catch {
    return []
  }
}

/**
 * Recursively compute total size (bytes) of a directory.
 * @param {string} p
 * @returns {Promise<number>}
 */
async function _dirSize(p) {
  let total = 0
  try {
    const entries = await fs.promises.readdir(p, { withFileTypes: true })
    for (const e of entries) {
      const sub = join(p, e.name)
      if (e.isFile()) {
        try {
          const st = await fs.promises.stat(sub)
          total += st.size
        } catch {}
      } else if (e.isDirectory()) {
        total += await _dirSize(sub)
      }
    }
  } catch {
    // ignore and return whatever we've accumulated
  }
  return total
}

/**
 * Return cached model metadata (name, path, size, mtime) under the cache dir.
 * @param {{cacheDir?: string}} opts
 * @returns {Promise<Array<{name:string,path:string,size:number,mtime:string}>>}
 */
export async function getCachedModels(opts = {}) {
  const dir = getCacheDir(opts.cacheDir)
  try {
    const entries = await fs.promises.readdir(dir, { withFileTypes: true })
    const out = []
    for (const e of entries) {
      if (!e.isDirectory()) continue
      const path = join(dir, e.name)
      let size = 0
      try { size = await _dirSize(path) } catch {}
      let mtime = null
      try { mtime = (await fs.promises.stat(path)).mtime.toISOString() } catch { mtime = null }
      out.push({ name: e.name, path, size, mtime })
    }
    return out
  } catch {
    return []
  }
}

/**
 * Download or ensure a model is cached. This delegates to the existing
 * `loadModel`/`Embedder.loadModel` implementation where available.
 *
 * @param {string} modelName
 * @param {{token?:string,dtype?:string,cacheDir?:string}} opts
 */
export async function downloadModel(modelName, opts = {}) {
  // Prefer the simple public helper if present
  try {
    const mod = await import('./index.js')
    if (typeof mod.loadModel === 'function') return mod.loadModel(modelName, opts)
  } catch (err) {
    // fallthrough to try Embedder directly
  }

  const embedderMod = await import('./embedder.js')
  if (embedderMod && embedderMod.Embedder && typeof embedderMod.Embedder.loadModel === 'function') {
    return embedderMod.Embedder.loadModel(modelName, opts)
  }
  throw new Error('No download API found (expected loadModel/downloadModel)')
}

/**
 * Prepare a model for use. This is a convenience wrapper that currently
 * delegates to `downloadModel` with optional `dtype` to influence loading.
 * If `quantize` is requested we re-run a download with a quantized dtype
 * as a best-effort convenience (full quantization pipelines are out of
 * scope for this helper and should be added as explicit tooling).
 */
export async function prepareModel(modelName, { quantize = false, dtype, cacheDir } = {}) {
  await downloadModel(modelName, { dtype, cacheDir })
  if (quantize) {
    const qd = dtype ?? 'q4'
    await downloadModel(modelName, { dtype: qd, cacheDir })
  }
  return { modelName, cacheDir: getCacheDir(cacheDir) }
}

/**
 * Ensure a model is present, optionally preparing it.
 *
 * @param {string} modelName
 * @param {{downloadIfMissing?:boolean,prepare?:boolean,quantize?:boolean,dtype?:string,cacheDir?:string}} opts
 */
export async function ensureModel(modelName, { downloadIfMissing = true, prepare = false, quantize = false, dtype, cacheDir } = {}) {
  const present = await isModelDownloaded(modelName, { cacheDir })
  if (present) {
    if (prepare) await prepareModel(modelName, { quantize, dtype, cacheDir })
    return { modelName, cacheDir: getCacheDir(cacheDir) }
  }
  if (!downloadIfMissing) throw new Error(`Model '${modelName}' not found in cache`)
  const res = await downloadModel(modelName, { dtype, cacheDir })
  if (prepare) await prepareModel(modelName, { quantize, dtype, cacheDir })
  return { modelName, cacheDir: res.cacheDir ?? getCacheDir(cacheDir) }
}

/**
 * Delete a cached model directory (or directories matching `modelName`).
 * Returns `true` if at least one directory was removed, `false` if none found.
 *
 * @param {string} modelName
 * @param {{cacheDir?: string}} opts
 * @returns {Promise<boolean>}
 */
export async function deleteModel(modelName, opts = {}) {
  const dir = getCacheDir(opts.cacheDir);
  const target = join(dir, modelName);
  if (await exists(target)) {
    await fs.promises.rm(target, { recursive: true, force: true });
    return true;
  }

  // Fallback: remove any directory whose name contains the modelName string
  try {
    const entries = await fs.promises.readdir(dir, { withFileTypes: true });
    const matches = entries.filter((e) => e.isDirectory() && e.name.includes(modelName));
    if (matches.length === 0) return false;
    for (const m of matches) {
      await fs.promises.rm(join(dir, m.name), { recursive: true, force: true });
    }
    return true;
  } catch (err) {
    throw err;
  }
}

export default {
  isModelDownloaded,
  listModels,
  downloadModel,
  prepareModel,
  ensureModel,
  deleteModel,
}
