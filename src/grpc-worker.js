import grpc from '@grpc/grpc-js';
import protoLoader from '@grpc/proto-loader';
import { EventEmitter } from 'events';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const PROTO_PATH = join(dirname(fileURLToPath(import.meta.url)), 'proto', 'embedder.proto');

/**
 * GrpcWorker — gRPC client wrapper exposing a Worker-like API for WorkerPool.
 *
 * Usage: new GrpcWorker(scriptPath, { workerData })
 * - `workerData.grpcAddress` optional override (default: localhost:50051)
 * - `workerData.grpcLoadBalancing` optional gRPC LB policy
 */
export class GrpcWorker extends EventEmitter {
  constructor(_scriptPath, { workerData } = {}) {
    super();

    this._address = workerData?.grpcAddress ?? process.env.EMBEDEER_GRPC_ADDRESS ?? 'localhost:50051';
    const channelOpts = workerData?.grpcLoadBalancing ? { 'grpc.lb_policy_name': workerData.grpcLoadBalancing } : {};

    // Load proto definition and create stub.
    const pkgDef = protoLoader.loadSync(PROTO_PATH, {
      longs: String,
      enums: String,
      defaults: true,
      oneofs: true,
    });
    const { embedeer: { EmbedderService } } = grpc.loadPackageDefinition(pkgDef);

    this._stub = new EmbedderService(this._address, grpc.credentials.createInsecure(), channelOpts);

    // Wait for the server to be ready and signal WorkerPool.
    const deadline = Date.now() + 10_000;
    this._stub.waitForReady(deadline, (err) => {
      if (err) this.emit('error', err);
      else this.emit('message', { type: 'ready' });
    });
  }

  /**
   * Accept a `{ type: 'task', id, texts }` message and call the unary Embed RPC.
   */
  postMessage(msg) {
    if (!msg || msg.type !== 'task') return;
    const id = msg.id;
    const texts = msg.texts ?? (Array.isArray(msg.payload) ? msg.payload : (msg.payload?.texts ?? []));

    try {
      this._stub.embed({ texts, id }, (err, response) => {
        if (err) {
          this.emit('message', { type: 'error', id, error: err.message });
          return;
        }

        // response.embeddings -> Array<{ values: number[] }>
        const embeddings = (response.embeddings ?? []).map((v) => v.values ?? []);
        this.emit('message', { type: 'result', id, embeddings });
      });
    } catch (err) {
      this.emit('message', { type: 'error', id, error: err?.message ?? String(err) });
    }
  }

  /**
   * Close the gRPC channel.
   * @returns {Promise<void>}
   */
  terminate() {
    return new Promise((resolve) => {
      try {
        if (this._stub && typeof this._stub.close === 'function') {
          this._stub.close();
        } else if (this._stub && typeof grpc.closeClient === 'function') {
          grpc.closeClient(this._stub);
        }
      } catch (err) {
        // ignore
      }
      resolve();
    });
  }
}
