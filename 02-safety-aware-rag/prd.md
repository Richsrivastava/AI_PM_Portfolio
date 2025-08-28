# PRD: Safety-aware RAG
**Owner:** Richa Srivastava  
**Date:** 2025-08-17  
**Status:** Draft

## 1) Problem & Goals
Users need reliable answers on semi-regulated content. We will add confidence gating, citations, and safe fallbacks to reduce hallucinations and increase trust.
- **Goal 1:** Accuracy ≥ 80% on offline eval set for target tasks.
- **Goal 2:** Hallucination rate ≤ 2% (no fabricated citations).
- **Goal 3:** Maintain p95 latency ≤ 3s with confidence checks.

## 2) Success Metrics (Anthropic-aligned)
**Quality & Safety**
- **Answer accuracy (rubric/keyphrase):** ≥ 0.80
- **Hallucination rate:** ≤ 2%
- **Refusal appropriateness:** ≥ 98% on sampled sensitive prompts
- **Calibration proxy:** high-confidence wrong answers ≤ 1%

**Steerability & UX**
- **Citations present when confidence ≥ threshold:** 100%
- **Safe fallback rate (clarify, show snippets) for low confidence:** ≥ 90%
- **User trust CSAT (qual sample):** ≥ 4.3/5

**Cost & Latency**
- **p95 latency:** ≤ 3s (RAG + generation)
- **Index compute cost:** track; budget under baseline

## 3) Users & Scope
- Prosumer users querying public policy/benefits-like docs.
- In scope: retrieval, confidence gating, refusal & clarify flows, citations.
- Out of scope: enterprise auth, doc ingestion pipelines at scale.

## 4) Requirements
- Guardrails: PII redaction, banned patterns, safe system prompts.
- Confidence thresholds & UX variations.
- Offline eval set (50–100 Qs) with rubric answers.

## 5) Experiments
- Threshold sweeps on confidence vs. accuracy/latency tradeoffs.
- Chunk size / top-k sweeps; prompt variants for answer style.
- Eval cadence with dashboards and weekly decisions.

## 6) Risks & Mitigations
- Over-refusal → better thresholds, clarifying questions.
- Latency creep → cache retrieval, prompt compression, limit context.
- Stale sources → timestamp checks, doc refresh policy.
