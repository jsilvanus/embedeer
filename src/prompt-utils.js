/**
 * Prompt utilities: token estimation and simple evidence trimming.
 *
 * These implementations are intentionally lightweight: they provide a
 * conservative char->token estimate and a simple trimming heuristic. For
 * production use, replace with a tokenizer that matches the target model.
 */

export function estimateTokensFromChars(chars) {
  // Very rough heuristic: ~4 characters per token on average.
  return Math.max(1, Math.ceil((chars || '').length / 4));
}

/**
 * Trim evidence blocks to fit within an allowed character/token budget.
 *
 * @param {string} prelude  Prompt text that precedes evidence
 * @param {Array<{id:number,source:string,excerpt:string}>} evidence
 * @param {number} allowedContext  Approximate allowed characters (conservative)
 * @returns {Array} subset of evidence that fits the budget (or all if fits)
 */
export function trimEvidenceForBudget(prelude, evidence = [], allowedContext = 2048) {
  if (!Array.isArray(evidence) || evidence.length === 0) return [];
  const preludeLen = String(prelude || '').length;
  // Compute a simple per-evidence serialized length and include until budget
  const serialized = evidence.map((e) => {
    const excerpt = String(e.excerpt || '');
    const src = String(e.source || '');
    return { e, len: 3 + String(e.id).length + src.length + excerpt.length };
  });

  const out = [];
  let running = preludeLen;
  for (const item of serialized) {
    // conservative addition
    running += item.len;
    if (running > allowedContext) break;
    out.push(item.e);
  }
  // If nothing fit, include the last evidence truncated to a short excerpt
  if (out.length === 0 && evidence.length > 0) {
    const first = { ...evidence[evidence.length - 1] };
    first.excerpt = String(first.excerpt || '').slice(0, Math.max(20, allowedContext / 4));
    return [first];
  }
  return out;
}

export default { estimateTokensFromChars, trimEvidenceForBudget };
