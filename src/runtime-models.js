/**
 * Runtime registry of loaded models.
 * WorkerPool instances register/unregister models when initialized/destroyed.
 */

const _counts = new Map();

export function registerLoadedModel(modelName) {
  const c = _counts.get(modelName) || 0;
  _counts.set(modelName, c + 1);
}

export function unregisterLoadedModel(modelName) {
  const c = _counts.get(modelName) || 0;
  if (c <= 1) _counts.delete(modelName);
  else _counts.set(modelName, c - 1);
}

export function getLoadedModels() {
  return Array.from(_counts.keys());
}

export function getLoadedModelCounts() {
  return new Map(_counts);
}
