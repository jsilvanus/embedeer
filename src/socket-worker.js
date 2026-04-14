import net from 'net';
import os from 'os';
import { join } from 'path';
import { EventEmitter } from 'events';

/**
 * SocketWorker — connects to a socket-model-server (NDJSON protocol) and
 * exposes the same EventEmitter interface used by WorkerPool.
 *
 * constructor(scriptPath, { workerData }) — workerData must include `modelName`.
 */
export class SocketWorker extends EventEmitter {
  constructor(_scriptPath, { workerData } = {}) {
    super();
    const modelName = workerData?.modelName ?? 'model';
    // Use the same socket path convention as socket-model-server.js
    this._socketPath = (process.platform === 'win32')
      ? `\\\\.\\\\pipe\\embedeer-${modelName.replace(/\//g, '-')}`
      : join(os.tmpdir(), `embedeer-${modelName.replace(/\//g, '-')}.sock`);

    this._buf = '';
    this._socket = null;
    this._connected = false;

    // Start connecting immediately.
    this._connect();
  }

  _connect() {
    const s = net.createConnection(this._socketPath);
    this._socket = s;

    s.on('connect', () => {
      this._connected = true;
      // Signal ready to the pool
      this.emit('message', { type: 'ready' });
    });

    s.on('data', (chunk) => {
      this._buf += chunk.toString();
      const lines = this._buf.split('\n');
      this._buf = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        let msg;
        try { msg = JSON.parse(line); } catch (err) { continue; }
        // Emit raw message objects so WorkerPool can handle result/error
        this.emit('message', msg);
      }
    });

    s.on('error', (err) => {
      this.emit('error', err);
    });

    s.on('close', (hadError) => {
      this._connected = false;
      // Normalise exit code: 0 on clean close, 1 on error
      this.emit('exit', hadError ? 1 : 0);
    });
  }

  postMessage(msg) {
    if (!this._socket || !this._connected) {
      // Report an error back via an async microtask so callers see the failure
      process.nextTick(() => this.emit('error', new Error('Socket not connected')));
      return;
    }
    try {
      this._socket.write(JSON.stringify(msg) + '\n');
    } catch (err) {
      this.emit('error', err);
    }
  }

  terminate() {
    return new Promise((resolve) => {
      if (!this._socket) return resolve();
      this._socket.end(() => resolve());
      // safety: destroy if not closed after short timeout
      setTimeout(() => { try { this._socket.destroy(); } catch {} ; resolve(); }, 2500).unref();
    });
  }
}
