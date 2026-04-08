import { Explainer } from './explainer.js';

/**
 * Minimal adapter to map a gitsema-style payload into an Explainer request.
 * This is intentionally small — it provides the function expected by the
 * public index and can be expanded with full integration behavior later.
 */
export async function explainForGitsema(payload = {}, opts = {}) {
  const { autoDestroy = false } = opts;
  const modelName = payload.model || opts.model || 'default-model';
  const explainer = await Explainer.create(modelName, opts);
  const request = {
    task: payload.task || 'explain',
    domain: payload.domain || 'general',
    context: payload.context || {},
    evidence: payload.evidence || [],
    maxTokens: payload.maxTokens || 256,
  };

  let res;
  try {
    res = await explainer.explain(request);
  } finally {
    if (autoDestroy && typeof explainer.destroy === 'function') {
      try { await explainer.destroy(); } catch (err) { /* ignore */ }
    }
  }

  return res;
}

export default { explainForGitsema };
