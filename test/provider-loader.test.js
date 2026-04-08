/**
 * Unit tests for provider-loader.js
 *
 * Tests verify provider selection logic and error messages when a GPU provider
 * is unavailable or unsupported on the current platform.
 *
 * All tests use process.platform/arch overrides to isolate platform logic
 * without requiring real GPU hardware.
 */

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import {
  getPlatformDefaultProviders,
  tryLoadProvider,
  resolveProvider,
} from '../src/provider-loader.js';

// ── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Temporarily override process.platform and process.arch, restore after fn().
 */
async function withPlatform(platform, arch, fn) {
  const origPlatform = Object.getOwnPropertyDescriptor(process, 'platform');
  const origArch = Object.getOwnPropertyDescriptor(process, 'arch');
  Object.defineProperty(process, 'platform', { value: platform, configurable: true });
  Object.defineProperty(process, 'arch', { value: arch, configurable: true });
  try {
    await fn();
  } finally {
    if (origPlatform) Object.defineProperty(process, 'platform', origPlatform);
    if (origArch) Object.defineProperty(process, 'arch', origArch);
  }
}

// ── getPlatformDefaultProviders() ────────────────────────────────────────────

describe('getPlatformDefaultProviders()', () => {
  test('returns [cuda] on linux/x64', async () => {
    await withPlatform('linux', 'x64', () => {
      assert.deepEqual(getPlatformDefaultProviders(), ['cuda']);
    });
  });

  test('returns [cuda, dml] on win32/x64 (CUDA preferred over DML)', async () => {
    await withPlatform('win32', 'x64', () => {
      assert.deepEqual(getPlatformDefaultProviders(), ['cuda', 'dml']);
    });
  });

  test('returns [] on unsupported platforms (e.g. darwin/arm64)', async () => {
    await withPlatform('darwin', 'arm64', () => {
      assert.deepEqual(getPlatformDefaultProviders(), []);
    });
  });
});

// ── tryLoadProvider() ────────────────────────────────────────────────────────

describe('tryLoadProvider()', () => {
  test('returns { loaded: false } when provider is not supported on platform', async () => {
    await withPlatform('darwin', 'arm64', async () => {
      const result = await tryLoadProvider('cuda');
      assert.equal(result.loaded, false);
      assert.equal(result.deviceStr, null);
    });
  });

  test('returns { loaded: false } when GPU hardware or system libs are missing', async () => {
    // In a typical CI environment there is no NVIDIA GPU, so activateCuda()
    // throws when /dev/nvidiactl is missing. tryLoadProvider must catch it
    // and return { loaded: false }.
    await withPlatform('linux', 'x64', async () => {
      const result = await tryLoadProvider('cuda');
      assert.equal(result.loaded, false);
      assert.equal(result.deviceStr, null);
      // error may be set (GPU not found) or null (provider not implemented)
    });
  });
});

// ── resolveProvider() ────────────────────────────────────────────────────────

describe('resolveProvider()', () => {
  // ── CPU paths ─────────────────────────────────────────────────────────────

  test('returns undefined when device=cpu', async () => {
    const result = await resolveProvider('cpu', undefined);
    assert.equal(result, undefined);
  });

  test('returns undefined when provider=cpu', async () => {
    const result = await resolveProvider('auto', 'cpu');
    assert.equal(result, undefined);
  });

  test('returns undefined when device and provider are both undefined', async () => {
    const result = await resolveProvider(undefined, undefined);
    assert.equal(result, undefined);
  });

  // ── device=auto with no GPU available ────────────────────────────────────

  test('device=auto returns undefined (CPU fallback) when GPU provider fails to activate', async () => {
    await withPlatform('linux', 'x64', async () => {
      // No NVIDIA GPU in CI; device='auto' must silently fall back to CPU.
      const result = await resolveProvider('auto', undefined);
      assert.equal(result, undefined);
    });
  });

  test('device=auto returns undefined on unsupported platform (no GPU providers)', async () => {
    await withPlatform('darwin', 'arm64', async () => {
      const result = await resolveProvider('auto', undefined);
      assert.equal(result, undefined);
    });
  });

  // ── device=gpu with no GPU available ─────────────────────────────────────

  test('device=gpu throws with GPU-related error when no GPU available (linux/x64)', async () => {
    await withPlatform('linux', 'x64', async () => {
      // No NVIDIA GPU in CI; resolveProvider should throw with a diagnostic
      // message about the GPU or CUDA requirements.
      await assert.rejects(
        () => resolveProvider('gpu', undefined),
        (err) => {
          assert.ok(
            err.message.toLowerCase().includes('nvidia') ||
            err.message.toLowerCase().includes('cuda') ||
            err.message.toLowerCase().includes('gpu'),
            `Expected GPU-related context in error, got: ${err.message}`,
          );
          return true;
        },
      );
    });
  });

  test('device=gpu throws on unsupported platform with informative message', async () => {
    await withPlatform('darwin', 'arm64', async () => {
      await assert.rejects(
        () => resolveProvider('gpu', undefined),
        (err) => {
          assert.ok(
            err.message.includes("device='gpu'"),
            `Expected GPU error message, got: ${err.message}`,
          );
          return true;
        },
      );
    });
  });

  // ── explicit provider not available ──────────────────────────────────────

  test('explicit provider=cuda throws with diagnostic error when GPU hardware is missing', async () => {
    await withPlatform('linux', 'x64', async () => {
      // No NVIDIA GPU in CI; activate() throws a diagnostic error about the
      // missing hardware or CUDA libraries. resolveProvider re-throws it.
      await assert.rejects(
        () => resolveProvider('cpu', 'cuda'),
        (err) => {
          assert.ok(
            err.message.toLowerCase().includes('nvidia') ||
            err.message.toLowerCase().includes('cuda') ||
            err.message.toLowerCase().includes('gpu'),
            `Expected GPU-related context in error, got: ${err.message}`,
          );
          return true;
        },
      );
    });
  });

  test('explicit provider=dml succeeds on win32 when platform is Windows', async () => {
    await withPlatform('win32', 'x64', async () => {
      // activateDml() checks process.platform === 'win32' (mocked here) and succeeds.
      const result = await resolveProvider('cpu', 'dml');
      assert.equal(result, 'dml');
    });
  });

  // ── unsupported provider on platform ──────────────────────────────────────

  test('explicit provider=dml throws "not supported" on linux', async () => {
    await withPlatform('linux', 'x64', async () => {
      await assert.rejects(
        () => resolveProvider('gpu', 'dml'),
        (err) => {
          assert.ok(
            err.message.toLowerCase().includes('not supported'),
            `Expected "not supported" in error, got: ${err.message}`,
          );
          return true;
        },
      );
    });
  });

  test('explicit provider=cuda throws "not supported" on darwin/arm64', async () => {
    await withPlatform('darwin', 'arm64', async () => {
      await assert.rejects(
        () => resolveProvider('gpu', 'cuda'),
        (err) => {
          assert.ok(
            err.message.toLowerCase().includes('not supported'),
            `Expected "not supported" in error, got: ${err.message}`,
          );
          return true;
        },
      );
    });
  });
});

// ── WorkerPool device/provider options ───────────────────────────────────────

describe('WorkerPool — device and provider options', async () => {
  const { WorkerPool } = await import('../src/worker-pool.js');
  const { EventEmitter } = await import('events');

  class SpyWorker extends EventEmitter {
    constructor(scriptPath, opts) {
      super();
      SpyWorker.lastOpts = opts;
      setImmediate(() => this.emit('message', { type: 'ready' }));
    }
    postMessage() {}
    async terminate() { setImmediate(() => this.emit('exit', 0)); }
  }

  test('device and provider are stored in WorkerPool', () => {
    const pool = new WorkerPool('model', {
      _WorkerClass: SpyWorker,
      device: 'gpu',
      provider: 'cuda',
    });
    assert.equal(pool.device, 'gpu');
    assert.equal(pool.provider, 'cuda');
  });

  test('workerData includes device and provider', async () => {
    const pool = new WorkerPool('model', {
      _WorkerClass: SpyWorker,
      poolSize: 1,
      device: 'auto',
      provider: 'cuda',
    });
    await pool.initialize();
    const wd = SpyWorker.lastOpts.workerData;
    assert.equal(wd.device, 'auto');
    assert.equal(wd.provider, 'cuda');
    await pool.destroy();
  });
});
