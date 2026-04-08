/**
 * Explainer — lightweight wrapper that builds deterministic prompts,
 * calls an LLM adapter, parses/repairs JSON output and validates citations.
 */

import LLMAdapter from './llm-adapter.js';
import { renderTemplate } from './explainer-templates.js';
import { trimEvidenceForBudget } from './prompt-utils.js';

export class Explainer {
  /**
   * Create an Explainer bound to a model or adapter.
   * opts may include: adapter, generateFn, deterministic, token, cacheDir, dtype, device, provider
   */
  static async create(modelName, opts = {}) {
    if (opts.adapter) return new Explainer(opts.adapter, opts);
    const adapter = await LLMAdapter.create(modelName, opts);
    return new Explainer(adapter, opts);
  }

  constructor(adapter, { deterministic = true } = {}) {
    if (!adapter || typeof adapter.generate !== 'function') {
      throw new Error('adapter must implement generate(prompt, opts)');
    }
    this.adapter = adapter;
    this.deterministic = deterministic;
  }

  buildPrompt(request, maxTokens = 256) {
    const { task, domain, context = {}, evidence = [] } = request;
    const header = 'You are an expert code explainer. Use ONLY the evidence blocks below.\n' +
      'Output VALID JSON only with keys: explanation, labels, references, meta.\n';

    const domainPrelude = renderTemplate(domain, { task, context });

    const taskSection = `Task: ${task}\nDomain: ${domain}\nContext: ${JSON.stringify(context)}\n\n`;

    const instructions = `\n\nINSTRUCTIONS:\n` +
      `1) Use ONLY the evidence above; cite claims using exact IDs like [1].\n` +
      `2) Do not invent or reference IDs that are not present.\n` +
      `3) If you cannot answer from the evidence, set explanation to "INSUFFICIENT_EVIDENCE" and set labels and references to [].\n` +
      `4) Keep explanation concise (<= 200 words).\n` +
      `5) RETURN JSON ONLY — no extra text.`;

    // Estimate a conservative context window and trim evidence to fit
    const CONTEXT_WINDOW = request.contextWindow ?? 2048;
    const allowedContext = Math.max(256, CONTEXT_WINDOW - (maxTokens || 256) - 64);
    const prelude = header + domainPrelude + '\n\n' + taskSection + instructions;
    const trimmedEvidence = trimEvidenceForBudget(prelude, evidence, allowedContext);

    const evidenceText = trimmedEvidence.map((e) => {
      return `[${e.id}] ${e.source}\n${e.excerpt}`;
    }).join('\n\n');

    return prelude + '\n\nEvidence:\n' + evidenceText;
  }

  tryParseJson(text) {
    try {
      return JSON.parse(text);
    } catch {
      return null;
    }
  }

  extractFirstJson(text) {
    const m = text.match(/\{[\s\S]*\}/);
    if (!m) return null;
    try { return JSON.parse(m[0]); } catch { return null; }
  }

  async repairToJson(rawText) {
    const repairPrompt = `The previous output was not valid JSON. Extract and return VALID JSON only that matches the schema: { explanation, labels, references, meta }.\n` +
      `Previous output:\n${rawText}\nReturn valid JSON only.`;
    const gen = await this.adapter.generate(repairPrompt, { maxTokens: 256, temperature: 0, do_sample: false, top_k: 1, top_p: 1 });
    return this.tryParseJson(gen.text) || this.extractFirstJson(gen.text) || null;
  }

  validateReferences(parsed, evidence) {
    const ids = new Set((evidence || []).map((e) => e.id));
    const refs = Array.isArray(parsed.references) ? parsed.references : [];
    for (const r of refs) {
      if (typeof r !== 'object' || r === null || !('id' in r)) return false;
      if (!ids.has(r.id)) return false;
    }
    // Also check that explanation contains only IDs that exist (basic check)
    const explanation = String(parsed.explanation || '');
    const cited = [...explanation.matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1]));
    for (const c of cited) if (!ids.has(c)) return false;
    return true;
  }

  /**
   * Explain request: { task, domain, context, evidence, maxTokens }
   */
  async explain(request = {}) {
    const maxTokens = request.maxTokens ?? 256;

    // Build prompt and generate (trim evidence to fit maxTokens)
    const prompt = this.buildPrompt(request, maxTokens);
    const genOpts = this.deterministic
      ? { maxTokens, temperature: 0, do_sample: false, top_k: 1, top_p: 1 }
      : { maxTokens };

    let genRes;
    try {
      genRes = await this.adapter.generate(prompt, genOpts);
    } catch (err) {
      // Infrastructure-level failures should bubble as Errors per contract
      throw err;
    }

    let parsed = this.tryParseJson(genRes.text) || this.extractFirstJson(genRes.text);

    if (!parsed) {
      // Attempt repair once
      parsed = await this.repairToJson(genRes.text);
    }

    if (!parsed) {
      return {
        explanation: 'INSUFFICIENT_EVIDENCE',
        labels: [],
        references: [],
        meta: { model: this.adapter.modelName, tokensUsed: 0, deterministic: this.deterministic },
      };
    }

    // Validate references map back to evidence IDs
    const ok = this.validateReferences(parsed, request.evidence || []);
    if (!ok) {
      return {
        explanation: 'INSUFFICIENT_EVIDENCE',
        labels: [],
        references: [],
        meta: { model: this.adapter.modelName, tokensUsed: 0, deterministic: this.deterministic },
      };
    }

    return {
      explanation: String(parsed.explanation ?? 'INSUFFICIENT_EVIDENCE'),
      labels: Array.isArray(parsed.labels) ? parsed.labels : [],
      references: Array.isArray(parsed.references) ? parsed.references : [],
      meta: { model: this.adapter.modelName, tokensUsed: 0, deterministic: this.deterministic },
    };
  }

  /**
   * Clean up underlying adapter resources if supported (e.g. worker pool).
   */
  async destroy() {
    if (this.adapter && typeof this.adapter.destroy === 'function') {
      await this.adapter.destroy();
    }
  }
}

export default Explainer;
