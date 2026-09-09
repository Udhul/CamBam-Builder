# Model routing

Choose the first obvious capable fit; do not score routes or spend a long reasoning
trace on delegation. Usage impact is perceptual and relative per task. Include packet,
review and integration overhead, and delegate only when the likely benefit is clear.

| Role | Best fit | Relative usage/tradeoff |
| --- | --- | --- |
| `project_explorer` (Luna high) | Narrow native repository discovery | Low native impact; keep the question specific |
| `focused_implementer` (Luna xhigh) | Clear bounded edit with settled acceptance | Moderate native impact; lead reviews integration |
| `deep_reviewer` (Luna max) | Difficult bounded correctness or invariant review | Higher Luna effort for risk-sensitive checking |
| `glm_retriever` (GLM high) | Retrieval, classification, synthesis and larger-context scans | Separate OpenRouter allowance; only packet/result consume native context |
| `glm_focused_worker` (GLM high) | Clear bounded code/test/docs slice | Separate allowance; requires explicit ownership and checks |
| `glm_deep_reviewer` (GLM max) | Independent bounded second opinion | Separate allowance; use when independence adds value |
| `sol_scoped_worker` (Sol xhigh) | Bounded work needing materially stronger reasoning or carrying higher impact | Higher native impact; justified by difficulty or risk |
| Main thread | Architecture, ambiguity, cross-layer contracts, integration and final judgment | Preserve its context for decisions only it should own |

Astra is stronger than Sol: an Astra lead may delegate a settled difficult slice to
Sol, while a Terra/Luna lead may escalate one to Sol and then review it. If no row is
an obvious fit, keep the task with the capable lead. Quality dominates usage savings.

## Launching GLM

Codex 0.150.1 cannot attach an OpenRouter child directly to a ChatGPT-backed native
parent. From a native session, launch a bounded isolated worker with:

```powershell
codex exec --profile openrouter-glm "<compact task packet>"
```

An OpenRouter-backed parent may use the `glm_*` roles directly. If GLM is unavailable,
fall back once to the matching native role. See [WORKFLOW.md](WORKFLOW.md#delegation-execution)
for packet and integration mechanics and [REVIEW.md](REVIEW.md#delegation-routing-correction---2026-09-09)
for the provider test evidence.
