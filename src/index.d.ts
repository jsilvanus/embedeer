// Type definitions for @jsilvanus/embedeer
// Minimal, focused declarations covering public API for TypeScript users.

import type { EventEmitter } from 'events';

export type DType = 'fp32' | 'fp16' | 'q8' | 'q4' | 'q4f16' | 'auto' | string;
export type Mode = 'process' | 'thread' | string;
export type Pooling = 'mean' | 'cls' | 'none' | string;
export type Device = 'auto' | 'cpu' | 'gpu' | string;
export type Provider = 'cpu' | 'cuda' | 'dml' | string;

export interface EmbedderOptions {
  batchSize?: number;
  concurrency?: number;
  mode?: Mode;
  pooling?: Pooling;
  normalize?: boolean;
  token?: string;
  dtype?: DType;
  cacheDir?: string;
  device?: Device;
  provider?: Provider;
  applyPerfProfile?: boolean | string;
}

export interface PerfProfile {
  batchSize?: number;
  concurrency?: number;
  dtype?: string;
  provider?: string;
  device?: string;
}

export class Embedder {
  constructor(modelName?: string, options?: EmbedderOptions);
  static create(modelName: string, options?: EmbedderOptions): Promise<Embedder>;
  static loadModel(modelName: string, options?: { token?: string; dtype?: string; cacheDir?: string }): Promise<{ modelName: string; cacheDir: string }>;
  static findLatestGridResult(benchDir?: string): string | null;
  static applyPerfProfile(profilePath: string, device?: string): Promise<PerfProfile>;
  static initialPerformanceCheckup(opts?: { device?: string; sampleSize?: number; modelName?: string }): Promise<PerfProfile>;
  static getBestConfig(opts?: { profilePath?: string | null; device?: string; sampleSize?: number }): Promise<PerfProfile>;
  static getUserProfilePath(): string;
  static generateAndSaveProfile(opts?: { mode?: 'quick' | 'grid'; device?: string; sampleSize?: number; profileOut?: string; modelName?: string }): Promise<{ best: PerfProfile; savedPath: string }>;

  initialize(): Promise<this>;
  embed(texts: string | string[], options?: { prefix?: string }): Promise<number[][]>;
  destroy(): Promise<void>;
}

export interface WorkerPoolOptions {
  poolSize?: number;
  mode?: Mode;
  pooling?: Pooling;
  normalize?: boolean;
  token?: string;
  dtype?: string;
  cacheDir?: string;
  device?: Device;
  provider?: Provider;
  _WorkerClass?: any;
  workerScript?: string;
  threadWorkerScript?: string;
}

export class WorkerPool {
  constructor(modelName: string, options?: WorkerPoolOptions);
  initialize(): Promise<void>;
  run(payload: string[] | Record<string, any>): Promise<number[][]>;
  destroy(): Promise<void>;
}

export class ChildProcessWorker extends EventEmitter {
  constructor(scriptPath: string, opts?: { workerData?: any });
  postMessage(msg: any): void;
  terminate(): Promise<void>;
}

export class ThreadWorker extends EventEmitter {
  constructor(scriptPath: string, opts?: { workerData?: any });
  postMessage(msg: any): void;
  terminate(): Promise<void>;
}


export const DEFAULT_CACHE_DIR: string;
export function getCacheDir(dir?: string): string;
export function buildPipelineOptions(dtype?: string): { dtype?: string };

export function loadModel(modelName: string, options?: { token?: string; dtype?: string; cacheDir?: string }): Promise<{ modelName: string; cacheDir: string }>;

export function isModelDownloaded(modelName: string, opts?: { cacheDir?: string }): Promise<boolean>;
export function listModels(opts?: { cacheDir?: string }): Promise<string[]>;
export function downloadModel(modelName: string, opts?: { token?: string; dtype?: string; cacheDir?: string }): Promise<{ modelName: string; cacheDir: string } | any>;
export function prepareModel(modelName: string, opts?: { quantize?: boolean; dtype?: string; cacheDir?: string }): Promise<{ modelName: string; cacheDir: string }>;
export function ensureModel(modelName: string, opts?: { downloadIfMissing?: boolean; prepare?: boolean; quantize?: boolean; dtype?: string; cacheDir?: string }): Promise<{ modelName: string; cacheDir: string }>;
export function deleteModel(modelName: string, opts?: { cacheDir?: string }): Promise<boolean>;

// Runtime helpers
export function getLoadedModels(): string[];
export function getCachedModels(opts?: { cacheDir?: string }): Promise<Array<{ name: string; path: string; size: number; mtime: string | null }>>;

export function resolveProvider(device?: Device, provider?: Provider): Promise<string | undefined>;
