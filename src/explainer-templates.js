/**
 * Minimal explainer templates used by the Explainer class.
 *
 * This module provides a small set of domain preludes. Tests only require
 * a working `renderTemplate(domain, vars)` function; richer domain
 * templates can be added later.
 */

const TEMPLATES = {
  general: 'You are an expert explainer. Provide a concise, factual explanation using only provided evidence.',
  evolution: 'Explain code changes and their consequences concisely, using only the evidence blocks below.',
  security: 'Analyze the evidence for security implications, vulnerabilities, and mitigations.',
  performance: 'Summarize performance impacts and potential optimizations based on the evidence.',
  docs: 'Produce a short, clear explanation suitable for documentation purposes.',
  api: 'Describe API behavior, inputs, outputs, and side-effects based on the evidence.',
  ux: 'Explain user-facing behavior and UX implications using the evidence provided.',
  infra: 'Summarize infrastructure and operational concerns from the evidence.',
  legal: 'Provide a concise legal/risk summary using only the evidence.',
  compliance: 'Assess compliance concerns and relevant controls from the evidence.',
  data: 'Explain data flow, schemas, and privacy implications from the evidence.',
  testing: 'Suggest test cases and testing concerns derived from the evidence.',
  architecture: 'Describe high-level architecture and component interactions using the evidence.',
};

export function renderTemplate(domain, vars = {}) {
  if (!domain) return TEMPLATES.general;
  const key = String(domain).toLowerCase();
  return TEMPLATES[key] || TEMPLATES.general;
}

export default { renderTemplate };
