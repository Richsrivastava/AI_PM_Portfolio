# PRD: First 5 Minutes to Value
**Owner:** Richa Srivastava  
**Date:** 2025-08-17  
**Status:** Draft

## 1) Problem & Goals
New users struggle to discover what an AI assistant can do. We will reduce time-to-first-success and boost early retention via opinionated onboarding patterns.
- **Goal 1:** Help a new user accomplish a task in < 120s (p50) without prior instruction.
- **Goal 2:** Increase first-session task success rate to ≥ 65%.


## 2) Success Metrics 
**Activation Measures**
- **Time-to-first-success (TTV):** p50 < 120s, p95 < 300s
- **First-impression-metric:** ≥ 65% (An evaluation set of tasks is created. Success is scored according to a rubric that defines clear success criteria + real user sessions are sampled to ensure metrics reflect actual user behaviour)

**Quality & Reliability**
- **Task success (offline):** ≥ 0.75 keyphrase/rubric score on evals for top use-cases. Controlled evaluation of system performance using evaluation datasets. Success if system has expected keywords or phrases (keyphrase scoring) more than 75% of tests, leveraging a detailed grading rubric to evaluate correctness, relevance or quality
- **Refusal appropriateness:** ≥ 95% (manual spot-checks on sampled refusals) ensuring atleast 95% of refusals are correct or justified (asking for sensitive information, requests that violate policy or questions that it cannot answer reliably)
- **Hallucination rate:** ≤ 3% on eval set (citation/verification when applicable). 

**Safety & Steerability**
- **Guardrail trigger rate:** track; aim for high precision on harmful patterns. Gaurdrail trigger rate is tracked as a diagnostic signal. The quality of triggers (precision) is the main optimization goal- 
**Confidence-aware UX:** ≥ 90% of low-confidence answers use clarify/fallback

**Cost & Latency**
- **p95 latency budget:** ≤ 2.5s for generation step for 95% of all requests
- **Cost/user (tokens):** track as this is the infrastructure cost per user. It should NOT creep up as features evolve. As we track over time it should hold flat or improve vs. baseline with caching

## 3) Users & Scope
- New users (consumer/prosumer) arriving via homepage or invite.
- In scope: onboarding UI variants (Blank Chat, Wizard, Gallery), example library, prompt chips.
- Out of scope: account management, billing/subscription.

## 4) Requirements
- Instrumentation: event logs must capture key metrics (TTV, tokens, latency), session sampling for qualitative review (assess task success, refusals, hallucinations).
- A/B/C experiments with rollout guardrails.
- Accessibility (keyboard nav, readable defaults).

## 5) Experiments
- **A/B/C:** Blank chat vs. Wizard vs. Gallery; primary KPI = task_success; secondary = TTV, tokens/user.
- **Prompt scaffolds:** chips vs. inline hints.
- **Results cadence:** weekly readout with decision.

## 6) Risks & Mitigations
- Wizard fatigue → keep under 3 clicks; escape hatch to chat.
- Overfitting to eval set → rotate holdout set monthly. Ensure evals cover a broad range of top use-cases
- Cost creep → prompt compression, caching, example re-use.
