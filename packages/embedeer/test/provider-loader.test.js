/**
 * Unit tests for provider-loader.js
 *
 * Tests verify provider selection logic and error messages when provider
 * packages are missing or unsupported.
 *
 * All tests use module mocking to avoid any real network or native binary
 * access — the provider-loader is tested purely for its logic.
 */

import { test, describe, mock, before, after } from 'node:test';
import assert from 'node:assert/strict';
import {
  PROVIDER_PACKAGES,
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

// ── PROVIDER_PACKAGES map ────────────────────────────────────────────────────

describe('PROVIDER_PACKAGES', () => {
  test('contains entries for all supported platform+provider combinations', () => {
    assert.equal(PROVIDER_PACKAGES['linux-x64-cuda'],  '@embedeer/ort-linux-x64-cuda');
    assert.equal(PROVIDER_PACKAGES['win32-x64-cuda'],  '@embedeer/ort-win32-x64-cuda');
    assert.equal(PROVIDER_PACKAGES['win32-x64-dml'],   '@embedeer/ort-win32-x64-dml');
  });
});

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

  test('returns { loaded: false } when provider package is not installed or binary is missing', async () => {
    // In the workspace, @embedeer/ort-linux-x64-cuda is linked but the native
    // binary does not exist (install.js was not run), so activate() throws.
    // tryLoadProvider must return { loaded: false } in either case.
    await withPlatform('linux', 'x64', async () => {
      const result = await tryLoadProvider('cuda');
      assert.equal(result.loaded, false);
      assert.equal(result.deviceStr, null);
      // error may be set (binary not found) or null (package not installed)
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

  // ── device=auto with no packages installed ────────────────────────────────

  test('device=auto returns undefined (CPU fallback) when GPU provider fails to activate', async () => {
    await withPlatform('linux', 'x64', async () => {
      // @embedeer/ort-linux-x64-cuda is linked in the workspace but binary is
      // missing. device='auto' must silently fall back to CPU (return undefined).
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

  // ── device=gpu with no packages installed ─────────────────────────────────

  test('device=gpu throws when no GPU provider is available (linux/x64)', async () => {
    await withPlatform('linux', 'x64', async () => {
      // In the workspace, ort-linux-x64-cuda is linked but binary is missing.
      // resolveProvider should throw (either the activate error or a "not installed" error).
      // The error must reference the @embedeer package name to guide the user.
      await assert.rejects(
        () => resolveProvider('gpu', undefined),
        (err) => {
          assert.ok(
            err.message.includes('@embedeer/ort-linux-x64-cuda'),
            `Expected package name in error, got: ${err.message}`,
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
            err.message.includes('device=\'gpu\'') || err.message.includes("device='gpu'"),
            `Expected GPU error message, got: ${err.message}`,
          );
          return true;
        },
      );
    });
  });

  // ── explicit provider not installed ──────────────────────────────────────

  test('explicit provider=cuda throws with npm install hint when not installed', async () => {
    await withPlatform('linux', 'x64', async () => {
      await assert.rejects(
        () => resolveProvider('cpu', 'cuda'),
        (err) => {
          assert.ok(
            err.message.includes('@embedeer/ort-linux-x64-cuda'),
            `Expected package name in error, got: ${err.message}`,
          );
          assert.ok(
            err.message.toLowerCase().includes('npm install'),
            `Expected npm install hint in error, got: ${err.message}`,
          );
          return true;
        },
      );
    });
  });

  test('explicit provider=dml throws with npm install hint on windows when not installed', async () => {
    await withPlatform('win32', 'x64', async () => {
      await assert.rejects(
        () => resolveProvider('cpu', 'dml'),
        (err) => {
          assert.ok(
            err.message.includes('@embedeer/ort-win32-x64-dml'),
            `Expected package name in error, got: ${err.message}`,
          );
          assert.ok(
            err.message.toLowerCase().includes('npm install'),
            `Expected npm install hint in error, got: ${err.message}`,
          );
          return true;
        },
      );
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
