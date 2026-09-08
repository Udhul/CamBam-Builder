# Model routing and session sizing

Status: reusable project guidance. Decision recorded 2026-09-08.

This document selects a main-thread model and bounded worker roles from the
task's reasoning risk. It supplements the execution rules in
[WORKFLOW.md](WORKFLOW.md); [PROGRESS.md](PROGRESS.md) remains the only owner of
priority and active work.

## Decision model

Score the proposed work before choosing a model. Use the highest applicable
route when a hard gate applies; otherwise add the factors below.

| Factor | Score |
| --- | ---: |
| Architectural or user-impact consequence | 0-3 |
| Contract ambiguity | 0-3 |
| Cross-layer coupling | 0-3 |
| Numerical or algorithmic risk | 0-2 |
| Manual, irreversible or external acceptance | 0-2 |

The total is a routing signal, not a promise about model quality:

| Total | Main route | Typical work |
| ---: | --- | --- |
| 0-3 | Luna | Retrieval, classification, documentation, mechanical edits and focused tests with a settled contract |
| 4-6 | Terra | Bounded implementation with moderate coupling and clear acceptance |
| 7-9 | Sol | Shared contracts, integration, numerical correctness and substantive review |
| 10-13 | Astra | Foundational design, unresolved cross-layer semantics, broad orchestration or repeated failed attempts |

Hard gates override the total. Use Astra when a foundational contract is still
unresolved and the lead must make a high-impact decision in the same pass, or
when earlier bounded attempts have failed. Use Sol when the contract is settled
enough to divide the work and the lead still owns integration. Use Terra or
Luna only when the contract and acceptance are already explicit.

The score must include coordination cost. Delegation is useful only when the
worker can own a separable result with a compact packet and executable checks;
for a tiny task, the setup and review overhead can cost more than it saves.

## Current model guidance

OpenAI's current model overview describes GPT-6 Astra as the starting point for
the hardest reasoning and coding, GPT-5.6 Sol for complex professional work,
GPT-5.6 Terra as the intelligence/cost balance, and GPT-5.6 Luna for
cost-sensitive workloads. The [Astra guidance](https://developers.openai.com/api/docs/guides/latest-model)
also supports multi-agent orchestration and recommends making delegation
behavior explicit when it matters.

The listed API prices are useful only as a relative signal: the overview lists
Astra at $10/$50 and Sol at $4/$20 per million input/output tokens, with the
same listed 1.05M context and 128K maximum output limits. These prices are not
Codex UI usage telemetry and must not be treated as a direct budget forecast.

## Configured role mapping

The native roles currently available in this environment map as follows:

- `project_explorer` / Luna medium: bounded, read-only repository retrieval.
- `focused_implementer` / Luna high: one exclusive implementation or test slice
  whose contract is already fixed.
- `deep_reviewer` / Luna xhigh: difficult numerical, regression or invariant
  review with a defined target.
- The main Sol or Astra thread: architecture, task division, ambiguous
  semantics, integration, final verification and acceptance judgment.

External GLM roles may be useful for authorized retrieval or small bounded work,
but private repository content, user assets, generated reports and secrets must
not be sent to an external provider without explicit authorization. If provider
availability or telemetry is unknown, prefer the native role and state the
uncertainty.

## Re-evaluation triggers

Re-score when the task crosses a module or ownership boundary, when a worker
reports unresolved semantics, when a regression affects identity or serialized
relationships, or when manual CamBam acceptance becomes part of the slice.
Record a new dated decision in [REVIEW.md](REVIEW.md) when the route changes for
one of those reasons.

References: [OpenAI model overview](https://developers.openai.com/api/docs/models/gpt),
[GPT-6 Astra](https://developers.openai.com/api/docs/models/gpt-6-astra),
[GPT-5.6 Sol](https://developers.openai.com/api/docs/models/gpt-5.6-sol),
[WORKFLOW.md](WORKFLOW.md), and [AGENTS.md](../AGENTS.md).
