/**
 * Install script for @embedeer/ort-win32-x64-dml
 *
 * Downloads a DirectML-enabled ONNX Runtime Node.js binding for Windows x64.
 *
 * STUB — see packages/ort-linux-x64-cuda/install.js for full documentation.
 *
 * DirectML note: onnxruntime already ships a DirectML provider on Windows.
 * This package may only need to configure the execution provider order rather
 * than download a full custom binary. See the TODO below.
 */

import { mkdirSync, writeFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const VENDOR_DIR = join(__dirname, 'vendor');

if (process.platform !== 'win32' || process.arch !== 'x64') {
  console.warn(
    `[embedeer] @embedeer/ort-win32-x64-dml: skipping install on ${process.platform}/${process.arch}`,
  );
  process.exit(0);
}

console.log('[embedeer] @embedeer/ort-win32-x64-dml: running install script...');

mkdirSync(VENDOR_DIR, { recursive: true });

// TODO:
//   - Verify onnxruntime-node ships with DirectML support on Windows.
//     If it does, activate() may only need to set the execution provider preference.
//   - If a separate DML-enabled binary is needed, download it here.
//     See ort-linux-x64-cuda/install.js for the download skeleton.
writeFileSync(
  join(VENDOR_DIR, 'README.txt'),
  'This directory will contain the DirectML-enabled ONNX Runtime native binding for Windows x64.\n',
);

console.warn(
  '[embedeer] @embedeer/ort-win32-x64-dml: STUB install complete. ' +
  'No real DirectML binary was downloaded yet.',
);
