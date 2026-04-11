/**
 * Embedder — high-level API for batched, parallel text embeddings.
 *
 * Uses a WorkerPool so the model is loaded once per worker and reused
 * across many `embed()` calls.
 *
 * @example
 * import { Embedder } from 'embedeer';
 *
 * const embedder = await Embedder.create('Xenova/all-MiniLM-L6-v2');
 * const vectors = await embedder.embed(['Hello world', 'Foo bar']);
 * console.log(vectors); // [[0.12, ...], [0.34, ...]]
 * await embedder.destroy();
 */

import { pipeline, env } from '@huggingface/transformers';
import { WorkerPool } from './worker-pool.js';
import { getCacheDir, buildPipelineOptions } from './model-cache.js';
import os from 'os';
import fs from 'fs';
import { join } from 'path';
import childProcess from 'child_process';

export class Embedder {
  /**
   * @param {string} modelName  Hugging Face model identifier
   * @param {object} [options]
   * @param {number} [options.batchSize=32]      Max texts per worker task
   * @param {number} [options.concurrency=2]     Number of parallel workers
   * @param {string} [options.mode='process']    'process' (isolated) or 'thread' (same process)
   * @param {string} [options.pooling='mean']    Pooling strategy ('mean'|'cls'|'none')
   * @param {boolean}[options.normalize=true]    Whether to L2-normalise embeddings
   * @param {string} [options.token]             Hugging Face API token (overrides HF_TOKEN env)
   * @param {string} [options.dtype]             Quantization dtype ('fp32'|'fp16'|'q8'|'q4'|'q4f16'|'auto')
   * @param {string} [options.cacheDir]          Custom model cache directory
   * @param {string} [options.device]            Compute device: 'auto'|'cpu'|'gpu' (default: 'cpu')
   * @param {string} [options.provider]          Execution provider override: 'cpu'|'cuda'|'dml'
   */
  constructor(modelName = 'nomic-embed-text', options = {}) {
    this.modelName = modelName;
    // Determine sensible defaults based on host machine and requested device.
    const numCores = os.cpus()?.length ?? 1;

    // If user didn't provide a device, default to CPU.
    const device = options.device ?? 'cpu';

    // Default concurrency: for GPU prefer a small pool (1), for CPU default to numCores.
    const defaultConcurrency = device === 'gpu' ? 1 : Math.max(1, numCores);
    const concurrency = options.concurrency ?? defaultConcurrency;

    // Tune BLAS / thread usage to avoid oversubscription when using multiple workers.
    // Set threads-per-worker to floor(numCores / concurrency) (at least 1).
    const threadsPerWorker = Math.max(1, Math.floor(numCores / concurrency));
    if (!process.env.OMP_NUM_THREADS) process.env.OMP_NUM_THREADS = String(threadsPerWorker);
    if (!process.env.MKL_NUM_THREADS) process.env.MKL_NUM_THREADS = String(threadsPerWorker);

    this.batchSize = options.batchSize ?? (device === 'gpu' ? 64 : 32);
    // Default provider selection for GPU if not provided
    const provider = options.provider ?? (device === 'gpu' ? (process.platform === 'win32' ? 'dml' : 'cuda') : undefined);

    this._pool = new WorkerPool(modelName, {
      poolSize: concurrency,
      mode: options.mode ?? 'process',
      pooling: options.pooling ?? 'mean',
      normalize: options.normalize ?? true,
      token: options.token,
      dtype: options.dtype,
      cacheDir: options.cacheDir ?? getCacheDir(),
      device,
      provider,
    });
  }

  /**
   * Factory helper — creates and initializes an Embedder in one call.
   *
   * @param {string} modelName
   * @param {object} [options]
   * @returns {Promise<Embedder>}
   */
  static async create(modelName, options = {}) {
    // Auto-load per-user saved profile by default unless explicitly disabled.
    // If `options.applyPerfProfile` is a string it's treated as a profile path.
    // If it's `false` we skip auto-loading. If it's `true` or `undefined`
    // we attempt to load the user's saved profile at ~/.embedeer/perf-profile.json
    // and apply it if present. We do NOT run an automatic checkup here.
    if (options.applyPerfProfile !== false) {
      try {
        let profilePath = null;
        if (typeof options.applyPerfProfile === 'string') {
          profilePath = options.applyPerfProfile;
        } else {
          // use saved user profile if available
          const userPath = Embedder.getUserProfilePath();
          try { if (fs.existsSync(userPath)) profilePath = userPath; } catch {}
        }
        if (profilePath) {
          const recommended = await Embedder.applyPerfProfile(profilePath, options.device);
          // Merge recommended settings unless explicitly provided by caller
          options = Object.assign({}, recommended, options);
        }
      } catch (err) {
        // Non-fatal: fall back to provided options
        console.error('Perf profile application failed:', err?.message ?? err);
      }
    }

    const embedder = new Embedder(modelName, options);
    await embedder.initialize();
    return embedder;
  }

  /**
   * Find the most recent grid-results file under `bench/`.
   * Returns a full path or null.
   */
  static findLatestGridResult(benchDir = join(process.cwd(), 'bench')) {
    try {
      const files = fs.readdirSync(benchDir).filter((f) => f.startsWith('grid-results') && f.endsWith('.json'));
      if (files.length === 0) return null;
      files.sort();
      return join(benchDir, files[files.length - 1]);
    } catch {
      return null;
    }
  }

  /**
   * Read a bench/grid-results JSON file and pick the best config for the
   * requested device/platform. Returns an options object suitable for
   * `Embedder.create()` (batchSize, concurrency, dtype, provider, device).
   */
  static async applyPerfProfile(profilePath, device) {
    const data = JSON.parse(fs.readFileSync(profilePath, 'utf8'));
    let results = (data.results || []).filter((r) => r.success);
    if (device) {
      const byDevice = results.filter((r) => r.device === device);
      if (byDevice.length) {
        results = byDevice;
      }
    }
    if (results.length === 0) throw new Error('No successful results in profile');
    // pick highest textsPerSec
    results.sort((a, b) => (b.textsPerSec || 0) - (a.textsPerSec || 0));
    const best = results[0];
    return {
      batchSize: best.batchSize,
      concurrency: best.concurrency,
      dtype: best.dtype,
      provider: best.provider,
      device: best.device,
    };
  }

  /**
   * Run a short performance check using a temporary Embedder instance and
   * return a recommended config. This avoids relying on a precomputed
   * profile file and is useful on first-run or when profiles are absent.
   */
  static async initialPerformanceCheckup({ device = 'cpu', sampleSize = 50, modelName = 'nomic-embed-text' } = {}) {
    const numCores = os.cpus()?.length ?? 1;
    const tempOpts = { device, concurrency: 1, batchSize: device === 'gpu' ? 32 : 16, mode: device === 'gpu' ? 'process' : 'thread', applyPerfProfile: false };
    const embedder = new Embedder(modelName, tempOpts);
    await embedder.initialize();
    const sample = Array.from({ length: Math.max(1, Math.min(sampleSize, 200)) }, (_, i) => `perf-check-${i}`);
    const t0 = performance.now();
    await embedder.embed(sample);
    const elapsed = performance.now() - t0;
    await embedder.destroy();
    const tps = (sample.length / (elapsed / 1000));

    if (device === 'gpu') {
      const batchSize = tps >= 300 ? 128 : 64;
      return { device, batchSize, concurrency: 1, dtype: 'fp32' };
    }

    // CPU heuristics
    const batchSize = tps >= 70 ? 64 : 32;
    const concurrency = tps >= 70 ? Math.min(numCores, 2) : 1;
    return { device, batchSize, concurrency, dtype: 'fp32' };
  }

  /**
   * Return the best config for the current host.
   * Preference order:
   * 1) explicit `profilePath` if provided and valid
   * 2) latest `bench/grid-results*.json` file
   * 3) runtime `initialPerformanceCheckup`
   *
   * Returns an object suitable to pass into `Embedder.create()`.
   */
  static async getBestConfig({ profilePath = null, device = 'cpu', sampleSize = 50 } = {}) {
    if (profilePath) {
      try {
        return await Embedder.applyPerfProfile(profilePath, device);
      } catch (err) {
        console.error('applyPerfProfile failed for provided path:', err?.message ?? err);
      }
    }

    const latest = Embedder.findLatestGridResult();
    if (latest) {
      try {
        return await Embedder.applyPerfProfile(latest, device);
      } catch (err) {
        console.error('applyPerfProfile failed for latest profile:', err?.message ?? err);
      }
    }

    return await Embedder.initialPerformanceCheckup({ device, sampleSize });
  }

  /**
   * Return a path where per-user saved profile should be stored.
   */
  static getUserProfilePath() {
    const dir = join(os.homedir(), '.embedeer');
    return join(dir, 'perf-profile.json');
  }

  /**
   * Run a resource check and save the recommended profile for the current
   * user. Modes:
   * - 'quick': run a short initialPerformanceCheckup and save result
   * - 'grid': run bench/grid-search.js (may take longer) and pick best
   *
   * Returns { best, savedPath }.
   */
  static async generateAndSaveProfile({ mode = 'quick', device = 'cpu', sampleSize = 100, profileOut = null, modelName = 'nomic-embed-text' } = {}) {
    let best;
    if (mode === 'grid') {
      // Run the bench/grid-search.js script and wait for it to finish.
      const script = join(process.cwd(), 'bench', 'grid-search.js');
      const out = profileOut ?? join(process.cwd(), 'bench', `grid-results-${device}-${Date.now()}.json`);
      try {
        childProcess.execFileSync(process.execPath, [script, '--device', device, '--sample-size', String(sampleSize), '--out', out], { stdio: 'inherit' });
      } catch (err) {
        throw new Error(`Grid search failed: ${err.message}`);
      }
      // pick best from generated file
      best = await Embedder.applyPerfProfile(out, device);
    } else {
      best = await Embedder.initialPerformanceCheckup({ device, sampleSize, modelName });
    }

    const saveDir = join(os.homedir(), '.embedeer');
    try { fs.mkdirSync(saveDir, { recursive: true }); } catch {}
    const savePath = Embedder.getUserProfilePath();
    const payload = { generated: new Date().toISOString(), host: os.hostname(), best };
    fs.writeFileSync(savePath, JSON.stringify(payload, null, 2));
    return { best, savedPath: savePath };
  }

  /**
   * Download and cache a Hugging Face model without creating workers.
   *
   * Useful for pre-warming the local cache (similar to `ollama pull`) before
   * running `Embedder.create()` — subsequent worker startups will load the
   * model instantly from disk.
   *
   * @param {string} modelName  Hugging Face model identifier
   * @param {object} [options]
   * @param {string} [options.token]     Hugging Face API token
   * @param {string} [options.dtype]     Quantization dtype
   * @param {string} [options.cacheDir]  Custom cache directory
   * @returns {Promise<{modelName: string, cacheDir: string}>}
   */
  static async loadModel(modelName, { token, dtype, cacheDir } = {}) {
    const resolvedCacheDir = getCacheDir(cacheDir);
    if (token) process.env.HF_TOKEN = token;
    env.cacheDir = resolvedCacheDir;
    console.error(`Downloading / loading model: ${modelName} → ${resolvedCacheDir} (this may take a while)`);
    await pipeline('feature-extraction', modelName, buildPipelineOptions(dtype));
    console.error(`Model ${modelName} downloaded/loaded into ${resolvedCacheDir}`);
    return { modelName, cacheDir: resolvedCacheDir };
  }

  /**
   * Initialize the underlying worker pool (download / cache the model).
   * Called automatically by `Embedder.create()`.
   */
  async initialize() {
    await this._pool.initialize();
    return this;
  }

  /**
   * Embed one or more texts.
   *
   * Texts are split into batches of `batchSize` and all batches are
   * submitted to the worker pool concurrently, so multiple workers run
   * in parallel whenever there are more batches than workers.
   *
   * @param {string | string[]} texts
   * @param {object} [options]
   * @param {string} [options.prefix]  Task prefix prepended to every text before
   *   embedding (e.g. `'search_query: '` for nomic-embed-text). When omitted or
   *   empty the texts are passed through unchanged — useful when the caller has
   *   already added its own prefix.
   * @returns {Promise<number[][]>} One embedding vector per input text
   */
  async embed(texts, { prefix } = {}) {
    const input = Array.isArray(texts) ? texts : [texts];
    if (input.length === 0) return [];

    const prepared = prefix ? input.map((t) => prefix + t) : input;

    // Split into batches
    const batches = [];
    for (let i = 0; i < prepared.length; i += this.batchSize) {
      batches.push(prepared.slice(i, i + this.batchSize));
    }

    // Submit all batches to the pool simultaneously — workers pick them
    // up in parallel up to the pool's concurrency limit.
    const results = await Promise.all(batches.map((batch) => this._pool.run(batch)));

    // results is Array<number[][]>; flatten one level to get number[][]
    return results.flat();
  }

  /**
   * Shut down all worker processes and free resources.
   * The Embedder instance must not be used after this call.
   */
  async destroy() {
    await this._pool.destroy();
  }
}

