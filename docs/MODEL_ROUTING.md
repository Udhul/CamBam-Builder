# Model routing

Choose the first obvious capable fit; include packet, checking and integration
overhead. Delegate when independent progress or saved lead context clearly exceeds
that overhead. Keep the immediate blocking step with the lead. Quality dominates
usage savings; a smaller model at maximum effort is not a substitute for a stronger
model on ambiguous or consequential work.

## Current native strategy

Revision: **2026-10-03**. Use `gpt-6-astra`, `gpt-6.1-sol` and `gpt-6-luna`.
**GPT-5.6 models are retired from this project's active routes**, including fallbacks.
Prefer 6.1 Sol over the superseded 6 Sol. Historical evidence retains its original
model names; it is not current selection guidance.

| Lead | Focused, clear work | Difficult independent slice | Lead-owned work |
| --- | --- | --- | --- |
| GPT-6 Astra | GPT-6 Luna | GPT-6.1 Sol; Astra only when the slice needs frontier judgment or an independent peer opinion | Architecture, unsettled semantics, cross-layer decisions, integration and final judgment |
| GPT-6.1 Sol | GPT-6 Luna | Another 6.1 Sol for parallelism/context isolation, not a cheaper tier; Astra only for a named capability gap | Decisions and integration within its capability; escalate unresolved high-impact reasoning |
| GPT-6 Luna | Another Luna only when parallelism/context isolation pays | Escalate to 6.1 Sol; Astra for the hardest unresolved problem | Routine scoped coordination; stronger output does not make Luna the final arbiter of decisions it cannot assess |

The lead remains accountable but must be capable of judging the result. Escalate
the deciding work too when that condition fails. Same-model workers need an
independence or context benefit; do not spawn merely because a slot is available.

| Native role | Model / starting effort | Best fit and boundary |
| --- | --- | --- |
| `project_explorer` | `gpt-6-luna` / high | Read-only targeted discovery, caller/owner tracing, extraction and fact gathering; return file/symbol evidence |
| `focused_implementer` | `gpt-6-luna` / high | Bounded code/test/docs work with clear requirements and settled acceptance; no architecture decisions |
| `sol_scoped_worker` | `gpt-6.1-sol` / high | Nontrivial debugging, algorithms, multi-file implementation or synthesis of conflicting evidence within a defined contract |
| `deep_reviewer` | `gpt-6.1-sol` / high | Read-only difficult correctness, invariant or contract analysis; findings with evidence, not a mandatory review step |

Select model strength from ambiguity, reasoning difficulty, consequence of error
and how reliably the lead can check the output, rather than file count or task
label. Documentation synthesis can need Sol; a mechanical code edit can fit Luna.
A routine narrow check can fit Luna, but difficult review belongs to Sol or Astra.

Effort is a separate choice: low/medium for mechanical work, high for logic tracing
or bounded implementation, xhigh for a named difficult reasoning gap. Max is
exceptional when its expected quality benefit justifies latency and usage. Do not
default every worker to xhigh/max, or repeatedly retry Luna on an unresolved
semantic problem; return it to the lead or promote to Sol.

## Evidence and limits

The [official model guide](https://developers.openai.com/api/docs/guides/latest-model)
positions Astra as highest capability, 6.1 Sol for complex work at lower cost, and
Luna for focused high-volume tasks. Current Standard API input/output prices per
million tokens are **Astra $10/$50, Sol 6.1 $2/$10, Luna $0.10/$0.50**;
see the [model comparison](https://developers.openai.com/api/docs/models/compare).
Those rates do not predict Codex subscription allowance or total task cost:
effort, retries, context transfer, caching and service tier also matter.

The [Sol 6.1 launch evaluations](https://openai.com/index/introducing-gpt-6-1-sol/)
report DeepSWE v1.1 matching Astra at roughly one-fifth the task cost and exceeding
6 Sol's best score by 6.4 percentage points at lower effort. AutomationBench gains
4.8 points over 6 Sol at medium effort. Astra still leads the reported hardest
scientific tasks. The [Sol/Luna launch evaluations](https://openai.com/index/introducing-gpt-6-sol-and-luna/)
report Luna at 66.6% on DeepSWE v1.1 at max effort, and a 5.4-point AutomationBench
gain over its predecessor at high effort with 58% lower task cost. New Sol/Luna
token prices are lower than their 5.6 predecessors.

These are publisher-reported, benchmark-specific results, not a CamBam task
evaluation or a claim that a model wins every task. Harnesses and effort settings
differ; do not rank models by mixing scores from separate evaluation revisions.
The mapping is an engineering recommendation supported by those results and the
user's retirement policy. No model trials, runtime tests or code review were
required for this policy/configuration revision. Reopen on a new release, a routing
quality failure, unavailable model, or coordination overhead erasing the benefit.

## Codex configuration and invocation

Personal worker definitions live in `~/.codex/agents/<role>.toml`; the current
[custom-agent schema](https://learn.chatgpt.com/docs/agent-configuration/subagents#custom-agents)
requires `name`, `description` and `developer_instructions`. Each native role above
sets both model and effort. Do not recreate the old role-registration tables.

The local main default remains 6.1 Sol/high. The generic subagent default is
6.1 Sol/medium, a capable fallback for unclassified work. Choose the explicit Luna
role for lighter work instead of relying on inheritance. Local Codex 0.160.0 and
the cached catalog list all three target models; this is configuration/catalog
evidence, not a successful invocation or guaranteed future account access.

Custom role files override explicit spawn model/effort settings. To use another
effort or Astra for a particular slice, use a generic agent with explicit model and
effort rather than a role pinned to another setting. This session's collaboration
tool supports overrides with `fork_turns="none"` or a bounded turn count, not
`"all"`; send the compact contract packet rather than the full history.
Use the capabilities actually exposed by the current client.

The local catalog supports low through max for Luna, and additionally ultra for
Sol/Astra. API reasoning support differs; do not copy client-only ultra into API
settings. No global concurrency, permission, provider or service-tier changes are
needed to implement this mapping. Settings and role discovery apply to fresh agent
sessions; do not assume an already-running parent reloads them.

For the next model revision, invoke the installed **`$revise-model-routing`** skill.
Its procedure researches current evidence and updates existing owners; it does not
freeze this model list into the reusable skill.

## Optional GLM route

This project continues to authorize task-scoped code, tests and documentation via
GLM/OpenRouter when a configured route is available and its separate allowance
makes it worthwhile. It is an optional provider alternative, not the default
native tier. Use retrieval/clear bounded work; keep architecture, ambiguous
semantics and final judgment with a capable lead.

The [2026-09-09 provider evidence](REVIEW.md#delegation-routing-correction---2026-09-09)
established an isolated `openrouter-glm` profile and `glm_*` roles on Codex 0.150.1,
but native-parent cross-provider spawning failed. The profile and roles are absent
from the current personal configuration; historical success does not establish
current access. Do not recreate credentials or change providers as part of native
model selection.

If that profile is configured again, use the evidenced isolated route:

```powershell
codex exec --profile openrouter-glm "<compact task packet>"
```

Recheck provider compatibility only for an authorized GLM task or a relevant client
change. If unavailable, use the matching current native tier once; never fall back
to 5.6. [WORKFLOW.md](WORKFLOW.md#delegation-execution) owns packets, disjoint edit
ownership, proportional checking and integration for every route.
