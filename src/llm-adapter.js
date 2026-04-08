/**
 * LLMAdapter — small abstraction over a text-generation backend.
 *
 * It supports two modes:
 *  - a provided `generateFn(prompt, opts)` for testing or custom backends
 *  - a Hugging Face `pipeline('text-generation', model)` when no generateFn is given
 */

import { pipeline, env } from '@huggingface/transformers';
import { getCacheDir, buildPipelineOptions } from './model-cache.js';
import { resolveProvider } from './provider-loader.js';
import { WorkerPool } from './worker-pool.js';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

export class LLMAdapter {
  /**
   * Create an adapter.
   * @param {string} modelName
   * @param {object} opts
   */
  static async create(modelName, opts = {}) {
    const { token, dtype, cacheDir, device, provider, generateFn, deterministic, useWorkerPool, poolSize, mode } = opts;

    if (generateFn) {
      return new LLMAdapter(generateFn, modelName, !!deterministic);
    }

    if (token) process.env.HF_TOKEN = token;
    const resolvedCache = getCacheDir(cacheDir);
    env.cacheDir = resolvedCache;

    // WorkerPool-backed generation (isolated processes) — optional
    if (useWorkerPool) {
      const __dirname = dirname(fileURLToPath(import.meta.url));
      const workerScript = join(__dirname, 'worker-gen.js');
      const threadWorkerScript = join(__dirname, 'thread-worker-gen-script.js');

      const pool = new WorkerPool(modelName, {
        poolSize: poolSize ?? 1,
        mode: mode ?? 'process',
        token,
        dtype,
        cacheDir: resolvedCache,
        device,
        provider,
        workerScript,
        threadWorkerScript,
      });
      await pool.initialize();

      const generateFnPool = async (prompt, genOpts = {}) => {
        const res = await pool.run({ prompt, genOpts });
        return { text: String(res ?? ''), raw: null };
      };

      const adapter = new LLMAdapter(generateFnPool, modelName, !!deterministic);
      adapter._pool = pool;
      return adapter;
    }

    const deviceStr = await resolveProvider(device, provider);
    const pipelineOpts = {
      ...buildPipelineOptions(dtype),
      ...(deviceStr ? { device: deviceStr } : {}),
    };

    const gen = await pipeline('text-generation', modelName, pipelineOpts);

    const generateFnLocal = async (prompt, genOpts = {}) => {
      // Hugging Face pipeline usually returns an array of outputs.
      const out = await gen(prompt, genOpts);
      let text = '';
      if (Array.isArray(out) && out.length > 0) {
        text = out[0].generated_text ?? out[0].text ?? JSON.stringify(out);
      } else {
        text = String(out);
      }
      return { text, raw: out };
    };

    // Keep a reference to the underlying pipeline if available so callers
    // can attempt to clean it up via `destroy()` later (best-effort).
    const adapter = new LLMAdapter(generateFnLocal, modelName, !!deterministic);
    adapter._pipeline = gen;
    return adapter;
  }

  constructor(generateFn, modelName = 'local', deterministic = true) {
    if (typeof generateFn !== 'function') throw new Error('generateFn must be a function');
    this.generateFn = generateFn;
    this.modelName = modelName;
    this.deterministic = deterministic;
  }

  /**
   * Generate text for the given prompt.
   * Returns { text, raw, meta }
   */
  async generate(prompt, { maxTokens = 256, temperature = 0, top_k = 1, top_p = 1, do_sample = false, ...rest } = {}) {
    const genOpts = {
      max_new_tokens: maxTokens,
      temperature,
      top_k,
      top_p,
      do_sample,
      ...rest,
    };
    const res = await this.generateFn(prompt, genOpts);
    return { text: res.text ?? '', raw: res.raw ?? null, meta: { model: this.modelName } };
  }

  /**
   * Clean up any resources used by the adapter (worker pools, pipelines).
   * This is a best-effort cleanup: pipelines may not expose an explicit
   * disposal API; the method safely ignores missing hooks.
   */
  async destroy() {
    // Destroy worker pool if present
    if (this._pool && typeof this._pool.destroy === 'function') {
      try { await this._pool.destroy(); } catch (err) { console.error('LLMAdapter: error destroying pool', err); }
      this._pool = undefined;
    }

    // Attempt to clean up an attached pipeline if it exposes a cleanup API
    if (this._pipeline) {
      try {
        if (typeof this._pipeline.cleanup === 'function') await this._pipeline.cleanup();
        else if (typeof this._pipeline.destroy === 'function') await this._pipeline.destroy();
      } catch (err) {
        // Non-fatal
      }
      this._pipeline = undefined;
    }
  }
}

export default LLMAdapter;
