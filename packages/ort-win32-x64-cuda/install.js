/**
 * Install script for @embedeer/ort-win32-x64-cuda
 *
 * Downloads a CUDA-enabled ONNX Runtime Node.js binding for Windows x64.
 *
 * STUB — see packages/ort-linux-x64-cuda/install.js for full documentation
 * and TODO list. This file follows the same pattern.
 */

import { mkdirSync, writeFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const VENDOR_DIR = join(__dirname, 'vendor');

if (process.platform !== 'win32' || process.arch !== 'x64') {
  console.warn(
    `[embedeer] @embedeer/ort-win32-x64-cuda: skipping install on ${process.platform}/${process.arch}`,
  );
  process.exit(0);
}

console.log('[embedeer] @embedeer/ort-win32-x64-cuda: running install script...');

mkdirSync(VENDOR_DIR, { recursive: true });

// TODO: replace with real binary download (see ort-linux-x64-cuda/install.js)
writeFileSync(
  join(VENDOR_DIR, 'README.txt'),
  'This directory will contain the CUDA-enabled ONNX Runtime native binding for Windows x64.\n',
);

console.warn(
  '[embedeer] @embedeer/ort-win32-x64-cuda: STUB install complete. ' +
  'No real CUDA binary was downloaded yet.',
);
